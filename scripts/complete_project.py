#!/usr/bin/env python3
"""Reproduce validation, the full study, independent repeats, charts, and profiles.

  uv run --extra dev --extra analysis python scripts/complete_project.py --dry-run
  uv run --extra dev --extra analysis python scripts/complete_project.py

All measurements and profiling run sequentially. Existing sweep measurements
resume by exact identity; no result files are deleted. A full run can take hours.
The default notes do not assert AC power or a quiet desktop: inspect the retained
power, thermal, and load metadata and provide operator observations explicitly.
--skip-build-validation resumes work after separately verified builds;
--profile-backend skip explicitly leaves profiling pending.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from run_experiments import ROOT, run
from run_sweep import positive


DEFAULT_NOTES = ("Serial benchmark processes; power, thermal and load probes retained; "
                 "desktop applications may be active")
STUDY_MODES = ("O3", "O2", "fast", "O0")


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    log: Path | None = None


def build_plan(args: argparse.Namespace, cpu_count: int) -> list[Step]:
    """Construct a reviewable execution order without changing files or probing tools."""
    build_root = args.build_root.resolve()
    results = args.results_root.resolve()
    steps: list[Step] = []
    python = sys.executable
    scripts = ROOT / "scripts"
    if not args.skip_build_validation:
        for name, kind, address, thread in (
            ("debug", "Debug", "OFF", "OFF"),
            ("asan", "Debug", "ON", "OFF"),
            ("tsan", "Debug", "OFF", "ON"),
            ("profile", "RelWithDebInfo", "OFF", "OFF"),
        ):
            build = build_root / name
            steps.extend([
                Step(f"configure-{name}", ["cmake", "-S", str(ROOT), "-B", str(build),
                     f"-DCMAKE_BUILD_TYPE={kind}", "-DBUILD_TESTING=ON", "-DMATMUL_OPT_LEVEL=3",
                     "-DMATMUL_FAST_MATH=OFF", "-DMATMUL_DISABLE_VECTORIZATION=OFF",
                     "-DMATMUL_VECTORIZATION_REPORT=OFF", f"-DMATMUL_ASAN_UBSAN={address}",
                     f"-DMATMUL_TSAN={thread}", f"-DPython3_EXECUTABLE={python}"],
                     build / "configure.log"),
                Step(f"build-{name}", ["cmake", "--build", str(build), "--parallel", str(args.jobs)],
                     build / "build.log"),
                Step(f"test-{name}", ["ctest", "--test-dir", str(build), "--output-on-failure"],
                     build / "ctest.log"),
            ])
        steps.append(Step("python-tests", ["cmake", "-E", "env",
            f"MATMUL_BINARY={build_root / 'profile/matmul'}", python,
            "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
            build_root / "profile/python-tests.log"))

    experiments = build_root / "experiments"
    shared = ["--timeout", f"{args.timeout:g}", "--adaptive-iterations",
              "--max-threads", str(cpu_count), "--control-notes", args.control_notes]
    steps.append(Step("study", [python, str(scripts / "run_experiments.py"),
        "--preset", "study", "--modes", *STUDY_MODES,
        "--build-root", str(experiments), "--results-root", str(results),
        "--jobs", str(args.jobs), "--vectorization-report", "--skip-plots", *shared]))

    repeat_inputs: list[Path] = []
    for filename, mode, preset, sizes, implementations in (
        ("O3_small", "O3", "full", (64, 128, 256, 512),
         ("naive", "reordered", "blocked", "parallel", "accelerate")),
        ("O3_large", "O3", "full", (1024, 2048),
         ("reordered", "blocked", "parallel", "accelerate")),
        ("O0", "O0", "comparison", (128, 512), ()),
        ("O2", "O2", "comparison", (128, 512), ()),
        ("fast", "fast", "comparison", (128, 512), ()),
    ):
        output = results / "repeat_raw" / f"{filename}.csv"
        repeat_inputs.append(output)
        command = [python, str(scripts / "run_sweep.py"),
            "--binary", str(experiments / mode / "matmul"), "--build-label", mode,
            "--preset", preset, "--sizes", *map(str, sizes), "--skip-rectangles",
            "--sweeps", str(args.repeats), "--sweep-index", "1", "--cooldown-seconds", "2",
            "--output", str(output), *shared]
        if implementations:
            command.extend(["--implementations", *implementations])
        steps.append(Step(f"repeats-{filename}", command))

    raw = results / "raw"
    steps.append(Step("audit", [python, str(scripts / "validate_study.py"),
        "--input-dir", str(raw), "--output-dir", str(results / "audit"),
        "--adaptive-iterations", "--cpu-count", str(cpu_count)]))
    inputs = [raw / f"{mode}.csv" for mode in STUDY_MODES] + repeat_inputs
    steps.extend([
        Step("analyze-repeats", [python, str(scripts / "analyze_repeats.py"), *map(str, inputs),
             "--output-dir", str(results / "repeats")]),
        Step("plot-results", [python, str(scripts / "plot_results.py"), *map(str, inputs),
             "--output-dir", str(results / "figures")]),
    ])
    if args.profile_backend != "skip":
        steps.append(Step("profiles", [python, str(scripts / "profile.py"),
            "--binary", str(build_root / "profile/matmul"), "--backend", args.profile_backend,
            "--implementations", "naive", "reordered", "blocked", "parallel",
            "--size", "512", "--duration", "5s", "--output-dir", str(args.profile_output_dir.resolve())]))
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/final")
    parser.add_argument("--build-root", type=Path, default=ROOT / "build")
    parser.add_argument("--timeout", type=float, default=600, help="seconds per benchmark configuration")
    parser.add_argument("--repeats", type=positive, default=3, help="independent repeat sweeps, at least three")
    parser.add_argument("--jobs", type=positive, default=2, help="parallel compiler processes only")
    parser.add_argument("--max-threads", type=positive, help="cap the detected logical CPU count")
    parser.add_argument("--profile-backend", choices=("auto", "instruments", "sample", "skip"), default="auto")
    parser.add_argument("--profile-output-dir", type=Path, default=ROOT / "traces/profiles")
    parser.add_argument("--skip-build-validation", action="store_true", help="reuse separately validated builds")
    parser.add_argument("--control-notes", default=DEFAULT_NOTES)
    parser.add_argument("--dry-run", action="store_true", help="print all commands without executing or writing files")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.repeats < 3:
        parser.error("--repeats must be at least three independent sweeps")
    detected = os.cpu_count() or 1
    cpu_count = min(detected, args.max_threads or detected)
    steps = build_plan(args, cpu_count)
    if args.dry_run:
        for step in steps:
            print(f"[{step.name}] {shlex.join(step.command)}")
            if step.log:
                print(f"  log: {step.log}")
        if args.profile_backend == "skip":
            print("Profiling pending: --profile-backend skip was requested.")
        if args.skip_build_validation:
            print("Build validation omitted: existing validation must be supplied separately.")
        return 0

    status = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "complete": False, "cpu_count": cpu_count, "control_notes": args.control_notes,
        "build_validation": "not_run" if args.skip_build_validation else "pending",
        "profiling": "pending" if args.profile_backend == "skip" else "scheduled",
        "requested_profile_backend": args.profile_backend, "completed_steps": [],
    }
    status_path = args.results_root.resolve() / "completion_status.json"
    failed = False
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        for step in steps:
            status["active_step"] = step.name
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            if step.log:
                step.log.parent.mkdir(parents=True, exist_ok=True)
            run(step.command, step.log)
            status["completed_steps"].append(step.name)
            if step.name == "python-tests":
                status["build_validation"] = "passed"
            if step.name == "profiles":
                status["profiling"] = "captured"
        status["status"] = "finished"
        status["complete"] = status["build_validation"] == "passed" and status["profiling"] == "captured"
        status.pop("active_step", None)
    except (OSError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        failed = True
        print(f"Completion workflow stopped at {status.get('active_step', 'startup')}: {error}", file=sys.stderr)
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            print(f"Could not save workflow status: {error}", file=sys.stderr)
            failed = True
    if failed:
        return 1
    print(f"Workflow finished: {args.results_root.resolve()}. "
          f"Build validation: {status['build_validation']}; profiling: {status['profiling']}.")
    if not status["complete"]:
        print("Full reproduction remains pending for the explicitly omitted validation or profiling steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
