#!/usr/bin/env python3
"""Run deterministic, timeout-bounded MatMul experiments.

Examples:
  python scripts/run_sweep.py --binary build/release/matmul --preset quick
  python scripts/run_sweep.py --binary build/release/matmul --preset full \
      --output results/raw/O3.csv --build-label O3 --timeout 120

Existing successful configurations are resumed by default. --fresh overwrites the
selected CSV and its skip log. Timeouts, unavailable implementations, and failures
are recorded in OUTPUT.skipped.jsonl; they are never fabricated as measurements.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any


SIZES = (32, 64, 128, 256, 512, 768, 1024, 1536, 2048)
BLOCKS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256)
LOOP_ORDERS = ("naive", "reordered", "jik", "jki", "kij", "kji")
RECTANGLES = ((128, 1024, 64), (1024, 128, 1024), (64, 4096, 64))
KEY_FIELDS = (
    "build_label", "binary_sha256", "implementation", "rows_a", "cols_a", "cols_b", "block_size",
    "requested_threads", "warmups", "repetitions", "seed",
    "sweep_index", "machine_id", "environment_id",
)


@dataclass(frozen=True)
class Configuration:
    implementation: str
    rows_a: int
    cols_a: int
    cols_b: int
    block_size: int = 64
    requested_threads: int = 1


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def configurations(preset: str, cpu_count: int, sizes: list[int] | None = None) -> list[Configuration]:
    """Vary one experimental dimension at a time; avoid a huge Cartesian product."""
    selected_sizes = sizes or (list(SIZES) if preset in ("full", "comparison") else [32, 64, 128])
    blocks = BLOCKS if preset == "full" else (16, 32, 64)
    threads = range(1, cpu_count + 1) if preset == "full" else sorted({1, min(2, cpu_count), min(4, cpu_count)})
    configs: set[Configuration] = set()
    if preset == "comparison":
        for rows, inner, cols in [(n, n, n) for n in selected_sizes] + list(RECTANGLES):
            for implementation in ("naive", "reordered", "blocked", "parallel", "accelerate"):
                count = min(8, cpu_count) if implementation == "parallel" else 1
                configs.add(Configuration(implementation, rows, inner, cols, 64, count))
        return sorted(configs, key=lambda c: tuple(asdict(c).values()))
    for size in selected_sizes:
        for implementation in (*LOOP_ORDERS, "accelerate"):
            configs.add(Configuration(implementation, size, size, size))
        for block in blocks:
            configs.add(Configuration("blocked", size, size, size, block))
        for implementation in ("parallel",):
            for count in threads:
                configs.add(Configuration(implementation, size, size, size, 64, count))
    rectangles = RECTANGLES if preset == "full" else ((31, 47, 23),)
    for rows, inner, cols in rectangles:
        for implementation in (*LOOP_ORDERS, "blocked", "parallel", "accelerate"):
            count = min(4, cpu_count) if implementation == "parallel" else 1
            configs.add(Configuration(implementation, rows, inner, cols, 64, count))
    return sorted(configs, key=lambda c: tuple(asdict(c).values()))


def iteration_counts(config: Configuration, warmups: int, repetitions: int,
                     adaptive: bool = False, threshold: int = 512 ** 3,
                     large_warmups: int = 1, large_repetitions: int = 3) -> tuple[int, int]:
    """Use a declared workload threshold, never choose iterations from observed winners."""
    if adaptive and config.rows_a * config.cols_a * config.cols_b >= threshold:
        return min(warmups, large_warmups), min(repetitions, large_repetitions)
    return warmups, repetitions


def _probe(command: list[str]) -> dict[str, Any]:
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
        if process.returncode == 0:
            return {"available": True, "output": process.stdout.strip()}
        return {"available": False, "reason": process.stderr.strip() or f"exit {process.returncode}"}
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": str(error)}


def collect_environment() -> dict[str, Any]:
    """Record observations without asserting that an unobserved machine is idle or cool."""
    machine: dict[str, Any] = {"system": platform.system(), "release": platform.release(),
        "architecture": platform.machine(), "processor": platform.processor(),
        "hostname": platform.node(), "logical_cpus": os.cpu_count()}
    controls: dict[str, Any] = {"observed_utc": datetime.now(timezone.utc).isoformat()}
    try:
        controls["load_average_1_5_15_minutes"] = list(os.getloadavg())
    except (AttributeError, OSError):
        controls["load_average_1_5_15_minutes"] = None
    if sys.platform == "darwin":
        for name in ("hw.model", "hw.memsize", "machdep.cpu.brand_string"):
            machine[name] = _probe(["sysctl", "-n", name])
        controls["power"] = _probe(["pmset", "-g", "batt"])
        controls["thermal"] = _probe(["pmset", "-g", "therm"])
    else:
        controls["power"] = {"available": False, "reason": "power probe implemented for macOS only"}
        controls["thermal"] = {"available": False, "reason": "thermal probe implemented for macOS only"}
    power_output = controls["power"].get("output", "")
    power_source = ("AC Power" if "AC Power" in power_output else
                    "Battery Power" if "Battery Power" in power_output else "unknown")
    environment = {"power_source": power_source, "thread_environment": {
        name: os.environ.get(name) for name in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")}}
    def digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {"machine_id": digest(machine), "environment_id": digest(environment),
            "machine": machine, "environment": environment, "controls": controls}


def key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def read_results(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields or "implementation" not in fields:
            raise ValueError(f"{path}: not a MatMul results CSV")
        return fields, list(reader)


def append_result(path: Path, row: dict[str, Any]) -> None:
    """Preserve existing rows if a newer benchmark adds result columns."""
    fields, rows = read_results(path)
    new_fields = [field for field in row if field not in fields]
    if not fields or new_fields:
        fields.extend(new_fields)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(row)
        temporary.replace(path)
    else:
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=Path("build/release/matmul"))
    parser.add_argument("--preset", choices=("quick", "full", "comparison"), default="quick")
    parser.add_argument("--output", type=Path, default=Path("results/raw/results.csv"))
    parser.add_argument("--build-label", default="local")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=positive, default=2)
    parser.add_argument("--repetitions", type=positive, default=9)
    parser.add_argument("--adaptive-iterations", action="store_true", help="reduce counts for declared large workloads")
    parser.add_argument("--large-threshold", type=positive, default=512 ** 3, help="M*K*N threshold for reduced counts")
    parser.add_argument("--large-warmups", type=positive, default=1)
    parser.add_argument("--large-repetitions", type=positive, default=3)
    parser.add_argument("--sweeps", type=positive, default=1, help="independent benchmark processes per configuration")
    parser.add_argument("--sweep-index", type=int, default=0, help="starting index for independent repeat/resume identity")
    parser.add_argument("--cooldown-seconds", type=float, default=0, help="optional pause between independent sweeps")
    parser.add_argument("--control-notes", default="", help="operator observations; recorded verbatim, not inferred")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per configuration")
    parser.add_argument("--max-threads", type=positive, help="cap the detected logical CPU count")
    parser.add_argument("--sizes", nargs="+", type=positive, help="override the preset square sizes")
    parser.add_argument("--skip-rectangles", action="store_true", help="measure only selected square sizes")
    parser.add_argument("--implementations", nargs="+", help="restrict implementation names")
    parser.add_argument("--fresh", action="store_true", help="replace the chosen result CSV and skip log")
    parser.add_argument("--dry-run", action="store_true", help="print planned configurations without executing")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("--seed must fit an unsigned 32-bit integer")
    if args.sweep_index < 0:
        parser.error("--sweep-index must be nonnegative")
    if not math.isfinite(args.cooldown_seconds) or args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be finite and nonnegative")
    if args.adaptive_iterations and (args.repetitions < 3 or args.large_repetitions < 3):
        parser.error("adaptive measurements require at least three repetitions")
    cpu_count = min(os.cpu_count() or 1, args.max_threads or (os.cpu_count() or 1))
    planned = configurations(args.preset, cpu_count, args.sizes)
    if args.skip_rectangles:
        planned = [config for config in planned if config.rows_a == config.cols_a == config.cols_b]
    if args.implementations:
        planned = [config for config in planned if config.implementation in args.implementations]
        if not planned:
            parser.error("no configurations match --implementations")
    indexed_plans = []
    for sweep_index in range(args.sweep_index, args.sweep_index + args.sweeps):
        shuffled = list(planned)
        random.Random(args.seed + sweep_index).shuffle(shuffled)
        indexed_plans.extend((sweep_index, config) for config in shuffled)
    if args.dry_run:
        print(json.dumps([{**asdict(config), **({"sweep_index": sweep_index} if args.sweeps > 1 else {})}
                          for sweep_index, config in indexed_plans], indent=2))
        return 0
    if not args.binary.is_file():
        parser.error(f"benchmark binary does not exist: {args.binary}")
    binary = str(args.binary.resolve())
    with args.binary.open("rb") as handle:
        binary_sha256 = hashlib.sha256(handle.read()).hexdigest()
    try:
        listed = subprocess.run([binary, "list", "--json"], check=True, capture_output=True,
                                text=True, timeout=args.timeout)
        available = set(json.loads(listed.stdout))
    except (subprocess.SubprocessError, ValueError, TypeError) as error:
        parser.error(f"cannot query available implementations: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    skip_path = args.output.with_suffix(args.output.suffix + ".skipped.jsonl")
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.jsonl")
    if args.fresh:
        for path in (args.output, skip_path, metadata_path):
            if path.exists():
                path.unlink()
    try:
        _, existing = read_results(args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    completed = {key(row) for row in existing}
    succeeded = resumed = skipped = failed = 0
    environment: dict[str, Any] = {}
    current_sweep: int | None = None

    def snapshot(phase: str, sweep_index: int) -> dict[str, Any]:
        observed = collect_environment()
        # Notes are part of the declared environment, while instantaneous load is not.
        observed["environment_id"] = hashlib.sha256(json.dumps({
            "detected_environment_id": observed["environment_id"],
            "control_notes": args.control_notes}, sort_keys=True).encode()).hexdigest()
        with metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"phase": phase, "sweep_index": sweep_index,
                "build_label": args.build_label, "binary_sha256": binary_sha256,
                "seed": args.seed, "order_seed": args.seed + sweep_index,
                "control_notes": args.control_notes, "timeout_seconds": args.timeout,
                "adaptive_iterations": args.adaptive_iterations,
                "large_threshold": args.large_threshold, **observed}) + "\n")
        return observed

    for index, (sweep_index, config) in enumerate(indexed_plans, 1):
        if sweep_index != current_sweep:
            if current_sweep is not None:
                snapshot("after", current_sweep)
                remaining = args.cooldown_seconds
                while remaining > 0:
                    pause = min(remaining, 30)
                    time.sleep(pause)
                    remaining -= pause
            current_sweep = sweep_index
            environment = snapshot("before", sweep_index)
        warmups, repetitions = iteration_counts(config, args.warmups, args.repetitions,
            args.adaptive_iterations, args.large_threshold, args.large_warmups, args.large_repetitions)
        metadata = {**asdict(config), "build_label": args.build_label,
                    "binary_sha256": binary_sha256,
                    "warmups": warmups, "repetitions": repetitions, "seed": args.seed,
                    "sweep_index": sweep_index, "order_seed": args.seed + sweep_index,
                    "machine_id": environment["machine_id"], "environment_id": environment["environment_id"]}
        if key(metadata) in completed:
            resumed += 1
            continue
        command = [binary, "benchmark", "--implementation", config.implementation,
                   "--rows", str(config.rows_a), "--inner", str(config.cols_a),
                   "--cols", str(config.cols_b), "--block-size", str(config.block_size),
                   "--threads", str(config.requested_threads), "--warmups", str(warmups),
                   "--repetitions", str(repetitions), "--seed", str(args.seed), "--json"]
        reason: str | None = None
        if config.implementation not in available:
            reason = "implementation unavailable on this platform/build"
        else:
            try:
                process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=args.timeout)
                result = json.loads(process.stdout)
                if not isinstance(result, dict) or not {"elapsed_time_ms", "gflops"} <= result.keys():
                    raise ValueError("benchmark returned an invalid JSON result")
                if not all(math.isfinite(float(result[field])) and float(result[field]) >= 0
                           for field in ("elapsed_time_ms", "gflops")) or float(result["elapsed_time_ms"]) == 0:
                    raise ValueError("benchmark returned nonfinite or invalid timing/performance")
                if result.get("correct") is not True:
                    raise ValueError("benchmark failed numerical verification")
                for field, expected in (("warmups", warmups), ("repetitions", repetitions)):
                    if field in result and result[field] != expected:
                        raise ValueError(f"benchmark returned inconsistent {field}")
                row = {**result, **metadata}
                for field, value in list(row.items()):
                    if isinstance(value, (dict, list)):
                        row[field] = json.dumps(value, separators=(",", ":"))
                append_result(args.output, row)
                completed.add(key(metadata))
                succeeded += 1
                print(f"[{index}/{len(indexed_plans)}] sweep={sweep_index} {config.implementation} "
                      f"{config.rows_a}x{config.cols_a}x{config.cols_b} "
                      f"block={config.block_size} threads={config.requested_threads}: "
                      f"{float(result['gflops']):.3f} GFLOP/s", flush=True)
            except subprocess.TimeoutExpired:
                reason = f"timeout after {args.timeout:g} seconds"
            except subprocess.CalledProcessError as error:
                reason = f"exit {error.returncode}: {error.stderr.strip()[-2000:]}"
                failed += 1
            except (ValueError, TypeError) as error:
                reason = str(error)
                failed += 1
        if reason is not None:
            skipped += 1
            entry = {**metadata, "reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()}
            with skip_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(f"[{index}/{len(indexed_plans)}] skip {config.implementation}: {reason}", file=sys.stderr, flush=True)
    if current_sweep is not None:
        snapshot("after", current_sweep)
    print(f"Sweep: {succeeded} measured, {resumed} resumed, {skipped} skipped; output: {args.output}")
    return 1 if failed or (succeeded == 0 and resumed == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
