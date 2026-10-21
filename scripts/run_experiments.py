#!/usr/bin/env python3
"""Build, verify, benchmark, and plot a reproducible MatMul experiment.

  uv run --extra dev --extra analysis python scripts/run_experiments.py --preset quick
  uv run --extra dev --extra analysis python scripts/run_experiments.py --preset full
  python scripts/run_experiments.py --modes O0 O2 O3 fast --skip-plots

Quick defaults to O3. Full defaults to O0/O2/O3/fast and covers the roadmap's
complete size, block, thread, and rectangular sweeps. Each build has separate
raw results and flags. Every configuration has a timeout; reruns resume existing
measurements. Large full sweeps can take hours. Profiling is a separate exercise;
see profile.py. Add --vectorization-report to save compiler diagnostics in each
build directory's build.log; use a Clang compiler for the roadmap's remarks.

Study defaults to all four policies: the full loop/block/thread study runs on O3,
while O0/O2/fast compare five implementations at every required square size and
rectangle. This varies one dimension at a time rather than multiplying all axes.
--sweeps repeats independent processes; --sizes/--implementations select smaller
controlled repeat studies. --adaptive-iterations retains 2+9 by default below
512^3 scalar products and uses 1+3 at or above that declared workload threshold.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys

from run_sweep import positive


ROOT = Path(__file__).resolve().parents[1]
MODES = {"O0": ("0", "OFF"), "O2": ("2", "OFF"), "O3": ("3", "OFF"), "fast": ("3", "ON")}


def executable(name: str) -> str:
    """Find environment-installed build tools even when its bin directory is not on PATH."""
    if shutil.which(name):
        return name
    sibling = Path(sys.executable).parent / name
    if sibling.is_file():
        return str(sibling)
    raise FileNotFoundError(f"required tool {name!r} is absent from PATH and {sibling.parent}")


def run(command: list[str], log: Path | None = None) -> None:
    if command[0] in ("cmake", "ctest"):
        command = [executable(command[0]), *command[1:]]
    print("+ " + " ".join(command), flush=True)
    if log is None:
        subprocess.run(command, check=True, cwd=ROOT)
    else:
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
        if process.returncode:
            print(log.read_text(encoding="utf-8")[-12000:], file=sys.stderr)
            raise subprocess.CalledProcessError(process.returncode, command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=("quick", "full", "comparison", "study"), default="quick")
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES))
    parser.add_argument("--build-root", type=Path, default=ROOT / "build/experiments")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--jobs", type=positive, default=2, help="parallel compiler processes")
    parser.add_argument("--max-threads", type=positive)
    parser.add_argument("--warmups", type=positive, default=2)
    parser.add_argument("--repetitions", type=positive, default=9)
    parser.add_argument("--adaptive-iterations", action="store_true")
    parser.add_argument("--large-threshold", type=positive, default=512 ** 3)
    parser.add_argument("--large-warmups", type=positive, default=1)
    parser.add_argument("--large-repetitions", type=positive, default=3)
    parser.add_argument("--sweeps", type=positive, default=1)
    parser.add_argument("--sweep-index", type=int, default=0)
    parser.add_argument("--cooldown-seconds", type=float, default=0)
    parser.add_argument("--control-notes", default="")
    parser.add_argument("--sizes", nargs="+", type=positive)
    parser.add_argument("--skip-rectangles", action="store_true")
    parser.add_argument("--implementations", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--vectorization-report", action="store_true")
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
    if not args.skip_plots:
        try:
            import matplotlib  # noqa: F401
            import pandas  # noqa: F401
        except ImportError:
            parser.error("plotting requires: uv sync --extra analysis (or pass --skip-plots)")
    modes = args.modes or (list(MODES) if args.preset in ("full", "study") else ["O3"])
    outputs = []
    try:
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_tools.py"])
        for mode in dict.fromkeys(modes):
            opt, fast = MODES[mode]
            build = args.build_root.resolve() / mode
            build.mkdir(parents=True, exist_ok=True)
            run(["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                 "-DBUILD_TESTING=ON",
                 f"-DMATMUL_OPT_LEVEL={opt}", f"-DMATMUL_FAST_MATH={fast}",
                 "-DMATMUL_DISABLE_VECTORIZATION=OFF", "-DMATMUL_ASAN_UBSAN=OFF", "-DMATMUL_TSAN=OFF",
                 f"-DMATMUL_VECTORIZATION_REPORT={'ON' if args.vectorization_report else 'OFF'}"],
                build / "configure.log")
            run(["cmake", "--build", str(build), "--parallel", str(args.jobs)], build / "build.log")
            run(["ctest", "--test-dir", str(build), "--output-on-failure"])
            binary = build / "matmul"
            if not binary.exists():
                binary = build / "Release/matmul.exe"
            output = args.results_root.resolve() / "raw" / f"{mode}.csv"
            preset = ("full" if mode == "O3" else "comparison") if args.preset == "study" else args.preset
            command = [sys.executable, str(ROOT / "scripts/run_sweep.py"), "--binary", str(binary),
                       "--preset", preset, "--output", str(output), "--build-label", mode,
                       "--timeout", str(args.timeout), "--warmups", str(args.warmups),
                       "--repetitions", str(args.repetitions), "--seed", str(args.seed),
                       "--sweeps", str(args.sweeps), "--sweep-index", str(args.sweep_index),
                       "--cooldown-seconds", str(args.cooldown_seconds), "--control-notes", args.control_notes]
            if args.adaptive_iterations:
                command.extend(["--adaptive-iterations", "--large-threshold", str(args.large_threshold),
                    "--large-warmups", str(args.large_warmups), "--large-repetitions", str(args.large_repetitions)])
            if args.sizes:
                command.extend(["--sizes", *map(str, args.sizes)])
            if args.skip_rectangles:
                command.append("--skip-rectangles")
            if args.implementations:
                command.extend(["--implementations", *args.implementations])
            if args.max_threads:
                command.extend(["--max-threads", str(args.max_threads)])
            if args.fresh:
                command.append("--fresh")
            run(command)
            outputs.append(output)
        if not args.skip_plots:
            run([sys.executable, str(ROOT / "scripts/plot_results.py"), *map(str, outputs),
                 "--output-dir", str(args.results_root.resolve() / "figures")])
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Experiment stopped: {error}", file=sys.stderr)
        return 1
    print(f"Experiment complete: {args.results_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
