#!/usr/bin/env python3
"""Compare independent sweep medians without pooling within-process timings.

Example: python scripts/analyze_repeats.py results/raw/repeat*.csv \
    --output-dir results/repeats

Explicit sweep_index values must identify the same collection across input files.
Legacy rows without sweep_index use their resolved input filename as collection
identity. Missing binary hashes are isolated per file and cannot establish matched
binary repeat evidence across files. Duplicate configuration/collection rows are
errors, including overlapping inputs; remove the overlap before analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


CONFIG_FIELDS = (
    "build_label", "binary_sha256", "implementation", "rows_a", "cols_a", "cols_b",
    "block_size", "requested_threads", "thread_count", "warmups", "repetitions", "seed",
    "absolute_tolerance", "relative_tolerance", "compiler", "compiler_flags",
    "machine_architecture", "build_type", "logical_cpu_count", "verification_reference",
    "machine_id", "environment_id",
)
REQUIRED_FIELDS = (
    "implementation", "rows_a", "cols_a", "cols_b", "block_size", "warmups",
    "repetitions", "seed", "elapsed_time_ms", "correct",
)
SUMMARY_FIELDS = (
    *CONFIG_FIELDS, "identity_scope", "independent_sweeps", "median_of_medians_ms",
    "min_ms", "max_ms", "population_std_ms", "cv_percent", "bootstrap95_low_ms",
    "bootstrap95_high_ms", "ci_status", "sweep_ids", "source_files",
)
BOOTSTRAP_SAMPLES = 10_000


def percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation between adjacent ordered bootstrap observations."""
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap_interval(values: list[float], identity: tuple[str, ...]) -> tuple[float, float]:
    # A stable digest avoids randomized Python hash() and input-order dependence.
    seed = int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest()[:8], "big")
    generator = random.Random(seed)
    values = sorted(values)
    estimates = sorted(statistics.median(generator.choices(values, k=len(values)))
                       for _ in range(BOOTSTRAP_SAMPLES))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def analyze(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[str, tuple[float, str]]] = defaultdict(dict)
    sources: list[dict[str, Any]] = []
    for path in paths:
        resolved = str(path.resolve())
        row_count = legacy_count = unhashed_count = 0
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
            if not {"requested_threads", "thread_count"}.intersection(reader.fieldnames or []):
                raise ValueError(f"{path}: missing requested_threads or thread_count")
            for line, raw in enumerate(reader, 2):
                row = {key: (value or "").strip() for key, value in raw.items() if key is not None}
                if any(not row.get(field) for field in REQUIRED_FIELDS):
                    raise ValueError(f"{path}:{line}: required value is empty")
                if row["correct"].lower() not in {"true", "1"}:
                    raise ValueError(f"{path}:{line}: measurement did not pass numerical verification")
                try:
                    elapsed = float(row["elapsed_time_ms"])
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise ValueError
                    for field in ("rows_a", "cols_a", "cols_b", "block_size", "repetitions"):
                        if int(row[field]) <= 0:
                            raise ValueError
                    if int(row["warmups"]) < 0 or int(row["seed"]) < 0:
                        raise ValueError
                    if row.get("requested_threads") and int(row["requested_threads"]) <= 0:
                        raise ValueError
                    # Accelerate reports zero for its externally managed worker
                    # count; this is metadata, not a request for zero workers.
                    if row.get("thread_count") and int(row["thread_count"]) < 0:
                        raise ValueError
                except ValueError as error:
                    raise ValueError(f"{path}:{line}: invalid timing or configuration") from error
                sweep = row.get("sweep_index", "")
                if sweep:
                    collection = f"sweep:{sweep}"
                else:
                    legacy_count += 1
                    collection = f"legacy-file:{resolved}"
                if row.get("binary_sha256"):
                    scope = "matched_binary_hash"
                else:
                    # Unknown binaries in different files must never be merged.
                    unhashed_count += 1
                    scope = f"unknown_binary_file:{resolved}"
                    collection = f"legacy-file:{resolved}"
                identity = (*[row.get(field, "") for field in CONFIG_FIELDS], scope)
                if collection in groups[identity]:
                    previous = groups[identity][collection][1]
                    raise ValueError(f"{path}:{line}: duplicate configuration in {collection}; "
                                     f"first observed in {previous}. Independent sweeps require distinct IDs.")
                groups[identity][collection] = (elapsed, f"{resolved}:{line}")
                row_count += 1
        if not row_count:
            raise ValueError(f"{path}: no measurement rows")
        sources.append({"path": resolved, "rows": row_count, "legacy_rows": legacy_count,
                        "missing_binary_hash_rows": unhashed_count})

    summary: list[dict[str, Any]] = []
    for identity, observations in sorted(groups.items()):
        values = [value for value, _ in observations.values()]
        count = len(values)
        low: float | str = ""
        high: float | str = ""
        status = "insufficient_independent_sweeps"
        if count >= 3:
            low, high = bootstrap_interval(values, identity)
            status = "exploratory_small_sample" if count < 10 else "percentile_bootstrap"
        average = statistics.mean(values)
        deviation = statistics.pstdev(values)
        row = dict(zip(CONFIG_FIELDS, identity[:-1]))
        row.update(identity_scope=identity[-1], independent_sweeps=count,
                   median_of_medians_ms=statistics.median(values), min_ms=min(values),
                   max_ms=max(values), population_std_ms=deviation,
                   cv_percent=100 * deviation / average, bootstrap95_low_ms=low,
                   bootstrap95_high_ms=high, ci_status=status,
                   sweep_ids=json.dumps(sorted(observations)),
                   source_files=json.dumps(sorted({source.rsplit(":", 1)[0]
                                                   for _, source in observations.values()})))
        summary.append(row)
    counts = Counter(row["independent_sweeps"] for row in summary)
    coverage = {
        "input_files": sources, "measurement_rows": sum(source["rows"] for source in sources),
        "exact_configuration_groups": len(summary),
        "groups_by_independent_sweep_count": {str(key): value for key, value in sorted(counts.items())},
        "groups_with_at_least_three_independent_sweeps": sum(
            row["independent_sweeps"] >= 3 for row in summary),
        "matched_binary_groups_with_at_least_three_independent_sweeps": sum(
            row["independent_sweeps"] >= 3 and row["identity_scope"] == "matched_binary_hash"
            for row in summary),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "method": "Percentile bootstrap of independent sweep medians; no pooling of iteration durations.",
        "limitations": [
            "Collection IDs declare independence; CSV metadata cannot prove separate physical executions.",
            "Legacy CSVs use one collection per resolved input filename; copied files are not new experiments.",
            "Missing binary hashes prevent establishing matched-binary evidence across input files.",
            "Intervals from fewer than ten independent sweeps are exploratory and may understate uncertainty.",
            "Bootstrap intervals do not account for systematic thermal, load, power, or measurement bias.",
            "Coverage describes supplied observations, not completion against a planned experiment matrix.",
        ],
    }
    return summary, coverage


def write_outputs(output: Path, summary: list[dict[str, Any]], coverage: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "repeat_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    count = coverage["groups_with_at_least_three_independent_sweeps"]
    lines = ["# Independent repeat analysis", "",
             f"Analyzed {coverage['measurement_rows']} verified observations across "
             f"{len(coverage['input_files'])} input CSVs and {len(summary)} exact configurations. "
             f"{count} configurations have at least three independent sweeps.", "",
             "Each observation is one sweep's benchmark median. Configuration groups retain binary "
             "hash, build, shape, implementation, block size, requested/actual threads, input seed, "
             "warmups, repetitions, verification settings, and available compiler/machine metadata. "
             "Different configurations and iteration durations are never pooled.", "",
             "`repeat_summary.csv` reports the median of sweep medians, range, population standard "
             "deviation, and coefficient of variation (population standard deviation divided by the "
             "arithmetic mean). Deterministic 95% percentile bootstrap intervals resample the sweep "
             f"medians {BOOTSTRAP_SAMPLES:,} times and are omitted below three sweeps. "
             "Intervals with fewer than ten sweeps are explicitly exploratory; three repeats provide "
             "only a coarse view of variability and cannot establish a reliable confidence level.", "",
             "| Independent sweeps | Exact configurations |", "| --- | ---: |"]
    lines.extend(f"| {key} | {value} |" for key, value in
                 coverage["groups_by_independent_sweep_count"].items())
    lines.extend(["", "Limitations:", ""])
    lines.extend(f"- {limitation}" for limitation in coverage["limitations"])
    lines.extend(["", "The complete measurements and identities are in `repeat_summary.csv`; "
                  "source and coverage details are in `coverage.json`.", ""])
    (output / "repeat_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/repeats"))
    args = parser.parse_args(argv)
    try:
        summary, coverage = analyze(args.inputs)
        write_outputs(args.output_dir, summary, coverage)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Analyzed {len(summary)} exact configurations; outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
