#!/usr/bin/env python3
"""Record sustained-kernel profiles with Instruments or macOS sample.

  python scripts/profile.py --backend sample --implementations naive reordered blocked --duration 5s
  python scripts/profile.py --backend instruments --implementation parallel --duration 15s

Inputs and warmups precede a readiness handshake. No correctness oracle runs in
this process: run matmul verify separately. sample records stack samples,
not CPU hardware counters or an Instruments trace.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import select
import shlex
import shutil
import subprocess
import sys
import time

from run_sweep import collect_environment, positive


def duration_seconds(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m)?", value)
    if not match:
        raise argparse.ArgumentTypeError("duration must be a positive number followed optionally by ms, s, or m")
    seconds = float(match[1]) * {None: 1, "ms": 0.001, "s": 1, "m": 60}[match[2]]
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be finite and positive")
    return seconds


def resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        probe = subprocess.run(["xcrun", "--find", "xctrace"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "sample"
    return "instruments" if probe.returncode == 0 else "sample"


def recording_command(backend: str, pid: str, output: Path, duration: float,
                      interval: int, template: str) -> list[str]:
    if backend == "sample":
        return ["sample", pid, str(math.ceil(duration)), str(interval), "-file", str(output.resolve())]
    return ["xcrun", "xctrace", "record", "--template", template, "--time-limit", f"{duration:g}s",
            "--output", str(output.resolve()), "--attach", pid]


def wait_ready(process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    signal = bytearray()
    while b"\n" not in signal:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
            raise RuntimeError("profile target did not become ready before the startup timeout")
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            raise RuntimeError("profile target exited before its readiness signal")
        signal.extend(chunk)
        if len(signal) > 4096:
            raise RuntimeError("profile target returned an oversized readiness signal")
    line = bytes(signal).split(b"\n", 1)[0].decode("utf-8").strip()
    if line != "PROFILE_READY":
        raise RuntimeError(f"profile target exited or returned an invalid readiness signal: {line!r}")


def workload_command(args: argparse.Namespace, backend: str, implementation: str) -> list[str]:
    window = math.ceil(args.duration) if backend == "sample" else args.duration
    seconds = window + (2 if backend == "sample" else 15)
    return [str(args.binary.resolve()), "profile", "--implementation", implementation,
        "--size", str(args.size), "--block-size", str(args.block_size), "--threads", args.threads,
        "--warmups", "1", "--duration", str(seconds), "--wait-for-profiler"]


def stop_owned(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()


def record(args: argparse.Namespace, backend: str, implementation: str, output: Path) -> dict:
    log_path = output.with_name(output.name + ".recorder.log")
    metadata_path = output.with_name(output.name + ".metadata.json")
    if any(path.exists() for path in (output, log_path, metadata_path)):
        raise ValueError(f"profile output already exists; choose a new path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    window = math.ceil(args.duration) if backend == "sample" else args.duration
    workload_seconds = window + (2 if backend == "sample" else 15)
    target_command = workload_command(args, backend, implementation)
    metadata = {"backend": backend, "implementation": implementation,
        "recorded_utc": datetime.now(timezone.utc).isoformat(), "matrix_size": args.size,
        "block_size": args.block_size, "requested_threads": args.threads,
        "capture_seconds": window, "sampling_interval_ms": args.sampling_interval_ms if backend == "sample" else None,
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "target_command": target_command, "output": str(output.resolve()),
        "environment_before": collect_environment(), "status": "started",
        "limitations": ["Stack samples are not function-call counts or hardware counters.",
            "A short readiness-wait interval can appear before kernel execution.",
            "Correctness verification runs separately and is excluded from this target."]}
    target = recorder = None
    try:
        target = subprocess.Popen(target_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        wait_ready(target, args.startup_timeout)
        command = recording_command(backend, str(target.pid), output, args.duration,
                                    args.sampling_interval_ms, args.template)
        metadata["recording_command"] = command
        print(shlex.join(command), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            recorder = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
            time.sleep(0.25)
            if recorder.poll() is not None:
                raise RuntimeError(f"recorder exited before kernel execution; see {log_path}")
            target.stdin.write("start\n")
            target.stdin.flush()
            if recorder.wait(timeout=window + 90) != 0:
                raise RuntimeError(f"recorder failed; see {log_path}")
        target_stdout, target_stderr = target.communicate(timeout=workload_seconds + 30)
        if target.returncode != 0:
            raise RuntimeError(f"profile target failed: {target_stderr.strip()}")
        result = json.loads(target_stdout)
        if not isinstance(result, dict) or type(result.get("iterations")) is not int or result["iterations"] < 1:
            raise RuntimeError("profile target did not report completed kernel iterations")
        for field in ("elapsed_time_ms", "checksum"):
            value = result.get(field)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise RuntimeError(f"profile target returned an invalid {field}")
        if result["elapsed_time_ms"] <= 0:
            raise RuntimeError("profile target returned a nonpositive duration")
        if not output.exists() or (output.is_file() and output.stat().st_size == 0):
            raise RuntimeError("recorder reported success without a profile artifact")
        metadata.update(status="complete", kernel_workload=result)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        stop_owned(recorder)
        stop_owned(target)
        metadata["environment_after"] = collect_environment()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=Path("build/profile/matmul"))
    parser.add_argument("--backend", choices=("instruments", "sample", "auto"), default="instruments")
    parser.add_argument("--implementation", default="naive")
    parser.add_argument("--implementations", nargs="+")
    parser.add_argument("--size", type=positive, default=1024)
    parser.add_argument("--block-size", type=positive, default=64)
    parser.add_argument("--threads", default="auto")
    parser.add_argument("--template", default="Time Profiler")
    parser.add_argument("--duration", type=duration_seconds, default=15.0)
    parser.add_argument("--sampling-interval-ms", type=positive, default=1)
    parser.add_argument("--startup-timeout", type=duration_seconds, default=120.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("traces/profiles"))
    parser.add_argument("--list-templates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.threads != "auto":
        try:
            positive(args.threads)
        except (ValueError, argparse.ArgumentTypeError):
            parser.error("--threads must be auto or a positive integer")
    if args.list_templates:
        command = ["xcrun", "xctrace", "list", "templates"]
        print(shlex.join(command), flush=True)
        return 0 if args.dry_run else subprocess.run(command, check=False).returncode
    names = args.implementations or [args.implementation]
    if any(not re.fullmatch(r"[a-z][a-z_-]*", name) for name in names):
        parser.error("implementation names must contain only lowercase letters, underscores, and hyphens")
    if args.output and len(names) != 1:
        parser.error("--output requires exactly one implementation; use --output-dir for several")
    try:
        backend = resolve_backend(args.backend)
        for implementation in names:
            suffix = ".sample.txt" if backend == "sample" else ".trace"
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            output = args.output or args.output_dir / f"{implementation}-{stamp}{suffix}"
            if args.dry_run:
                print(shlex.join(workload_command(args, backend, implementation)))
                print(shlex.join(recording_command(backend, "<PID>", output, args.duration,
                                                  args.sampling_interval_ms, args.template)))
                continue
            if platform.system() != "Darwin":
                raise RuntimeError("native profiling requires macOS; --dry-run works on other platforms")
            if not args.binary.is_file():
                raise ValueError(f"binary does not exist: {args.binary}")
            if backend == "sample" and shutil.which("sample") is None:
                raise RuntimeError("macOS sample utility is unavailable")
            record(args, backend, implementation, output)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Profiling failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
