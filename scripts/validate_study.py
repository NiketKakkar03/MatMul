#!/usr/bin/env python3
"""Audit study coverage and recompute metrics from saved benchmark samples.

python scripts/validate_study.py --input-dir results/final/raw --cpu-count 8 \
    --adaptive-iterations --output-dir results/final/audit

Each binary/machine/environment identity must independently cover its mode's
plan. Repeated sweeps are allowed; partial identities are never joined to claim
completion. Historical skip events are classified against verified saved data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from run_sweep import Configuration, configurations, iteration_counts, positive


MODES = ("O3", "O2", "fast", "O0")
WORKLOAD_FIELDS = tuple(Configuration.__dataclass_fields__)
IDENTITY_FIELDS = ("binary_sha256", "machine_id", "environment_id")
REQUIRED_METADATA = (*IDENTITY_FIELDS, "timestamp", "compiler", "compiler_flags",
                     "machine_architecture", "build_type", "verification_reference")


def planned_study(cpu_count: int) -> dict[str, set[Configuration]]:
    return {mode: set(configurations("full" if mode == "O3" else "comparison", cpu_count))
            for mode in MODES}


def workload(row: dict[str, Any]) -> Configuration:
    return Configuration(str(row["implementation"]),
                         *(int(row[field]) for field in WORKLOAD_FIELDS[1:]))


def validate_row(row: dict[str, str], cpu_count: int, adaptive: bool,
                 warmups: int = 2, repetitions: int = 9, seed: int = 42,
                 large_threshold: int = 512 ** 3, large_warmups: int = 1,
                 large_repetitions: int = 3) -> Configuration:
    """Raise ValueError for unverifiable data; all recomputation uses milliseconds."""
    missing = [field for field in REQUIRED_METADATA if not row.get(field, "").strip()]
    if missing:
        raise ValueError("missing metadata: " + ", ".join(missing))
    for field in IDENTITY_FIELDS:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", row[field]):
            raise ValueError(f"invalid {field}: expected SHA-256 identity")
    timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    if str(row.get("correct", "")).lower() not in ("true", "1"):
        raise ValueError("numerical verification did not pass")
    config = workload(row)
    if any(getattr(config, field) <= 0 for field in WORKLOAD_FIELDS[1:]):
        raise ValueError("nonpositive matrix dimension, block size or requested thread count")
    if config.requested_threads > cpu_count:
        raise ValueError("requested thread count exceeds declared CPU cap")
    if int(row["seed"]) != seed or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("input seed differs from declared study seed")
    if int(row["sweep_index"]) < 0:
        raise ValueError("negative sweep index")
    counts = iteration_counts(config, warmups, repetitions, adaptive,
                              large_threshold, large_warmups, large_repetitions)
    if (int(row["warmups"]), int(row["repetitions"])) != counts:
        raise ValueError("warmup/repetition counts differ from declared iteration policy")
    if int(row["warmups"]) < 1 or int(row["repetitions"]) < 3:
        raise ValueError("study requires at least one warmup and three measured samples")
    actual_threads = int(row["thread_count"])
    if (config.implementation == "accelerate" and actual_threads != 0) or (
            config.implementation != "accelerate" and not 1 <= actual_threads <= config.requested_threads):
        raise ValueError("actual thread count inconsistent with kernel metadata")
    if int(row["logical_cpu_count"]) < cpu_count:
        raise ValueError("declared CPU cap exceeds recorded machine logical CPU count")
    raw = row["durations_ms"].strip()
    durations = ([float(value) for value in json.loads(raw)] if raw.startswith("[") else
                 [float(value) for value in raw.split(";")])
    if len(durations) != int(row["repetitions"]):
        raise ValueError("saved sample count differs from repetitions")
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("saved durations must be finite and positive")
    median = statistics.median(durations)
    expected = {"elapsed_time_ms": median, "minimum_time_ms": min(durations),
                "standard_deviation_ms": statistics.pstdev(durations),
                "gflops": 2 * config.rows_a * config.cols_a * config.cols_b / median / 1e6}
    for field, reference in expected.items():
        value = float(row[field])
        if not math.isfinite(value) or not math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{field} inconsistent with saved samples or matrix shape")
    for field in ("maximum_absolute_error", "maximum_relative_error", "absolute_tolerance",
                  "relative_tolerance", "initialization_time_ms", "verification_time_ms", "estimated_bandwidth"):
        value = float(row[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    if not math.isfinite(float(row["checksum"])):
        raise ValueError("nonfinite checksum")
    return config


def audit(input_dir: Path, cpu_count: int, adaptive: bool = False, **policy: Any) -> dict[str, Any]:
    plans = planned_study(cpu_count)
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    files = sorted(input_dir.glob("*.csv"))
    total_rows = 0
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            for line, row in enumerate(csv.DictReader(handle), 2):
                total_rows += 1
                location = {"file": str(path), "line": line}
                try:
                    mode = row.get("build_label", "")
                    if mode not in plans:
                        raise ValueError(f"unknown study build label {mode!r}")
                    config = validate_row(row, cpu_count, adaptive, **policy)
                    identity = (mode, *(row[field] for field in IDENTITY_FIELDS))
                    group = groups.setdefault(identity, {"configs": set(), "observations": set(),
                        "recovery_keys": set(), "source_files": set(), "sweep_indices": set()})
                    observation = (config, int(row["sweep_index"]))
                    if observation in group["observations"]:
                        raise ValueError("duplicate workload and sweep index in the same identity")
                    group["observations"].add(observation)
                    group["recovery_keys"].add((*observation, int(row["warmups"]),
                                                int(row["repetitions"]), int(row["seed"])))
                    group["configs"].add(config)
                    group["source_files"].add(str(path))
                    group["sweep_indices"].add(int(row["sweep_index"]))
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    invalid.append({**location, "reason": str(error)})

    modes: dict[str, Any] = {}
    for mode, planned in plans.items():
        identities = []
        for identity, group in sorted(groups.items()):
            if identity[0] != mode:
                continue
            missing = planned - group["configs"]
            unexpected = group["configs"] - planned
            serialize = lambda values: [asdict(config) for config in sorted(
                values, key=lambda config: tuple(asdict(config).values()))]
            identities.append({**dict(zip(IDENTITY_FIELDS, identity[1:])),
                "planned": len(planned), "completed": len(planned & group["configs"]),
                "observations": len(group["observations"]), "missing": serialize(missing),
                "unexpected": serialize(unexpected), "source_files": sorted(group["source_files"]),
                "sweep_indices": sorted(group["sweep_indices"]), "complete": not missing and not unexpected})
        modes[mode] = {"planned": len(planned), "identity_groups": identities,
                       "complete": bool(identities) and all(item["complete"] for item in identities)}

    skip_events: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.skipped.jsonl")):
        for line, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(raw)
                config = workload(event)
                identity = (event["build_label"], *(event.get(field, "") for field in IDENTITY_FIELDS))
                recovery_key = (config, int(event.get("sweep_index", 0)), int(event["warmups"]),
                                int(event["repetitions"]), int(event["seed"]))
                recovered = recovery_key in groups.get(identity, {}).get("recovery_keys", set())
                skip_events.append({"file": str(path), "line": line, "reason": event.get("reason", ""),
                                    "status": "recovered" if recovered else "unresolved", **asdict(config)})
            except (ValueError, TypeError, KeyError) as error:
                invalid.append({"file": str(path), "line": line, "reason": f"invalid skip event: {error}"})
    return {"complete": all(item["complete"] for item in modes.values()) and not invalid,
        "cpu_count": cpu_count, "adaptive_iterations": adaptive, "iteration_policy": policy,
        "planned_total": sum(map(len, plans.values())), "measurement_rows": total_rows,
        "valid_observations": sum(len(group["observations"]) for group in groups.values()),
        "modes": modes, "invalid_rows": invalid, "historical_skip_events": skip_events,
        "method": "Coverage is computed separately per build/binary/machine/environment identity. "
                  "Distinct sweep indices identify repeated observations, not extra workload coverage. "
                  "Historical skips do not remove later valid results."}


def write_outputs(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "coverage.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Study coverage audit", "", f"Status: {'complete' if result['complete'] else 'incomplete'}.", "",
        f"{result['valid_observations']} valid observations from {result['measurement_rows']} saved rows; "
        f"{result['planned_total']} planned workloads at a {result['cpu_count']}-CPU cap. "
        f"Invalid rows: {len(result['invalid_rows'])}.", "", result["method"], "",
        "| Mode | Binary / machine / environment | Planned | Completed | Missing | Unexpected | Observations |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for mode, details in result["modes"].items():
        if not details["identity_groups"]:
            lines.append(f"| {mode} | No valid measurements | {details['planned']} | 0 | {details['planned']} | 0 | 0 |")
        for group in details["identity_groups"]:
            identity = " / ".join(group[field][:12] for field in IDENTITY_FIELDS)
            lines.append(f"| {mode} | {identity} | {group['planned']} | {group['completed']} | "
                f"{len(group['missing'])} | {len(group['unexpected'])} | {group['observations']} |")
    recovered = sum(event["status"] == "recovered" for event in result["historical_skip_events"])
    lines.extend(["", f"Historical skip events: {len(result['historical_skip_events'])}; recovered by "
        f"valid data: {recovered}. Remaining historical events do not determine coverage; the planned "
        "workload comparison does.", "", "Median, minimum, population standard deviation and GFLOP/s "
        "were recomputed from every saved sample. Full missing/unexpected workload lists, invalid-row "
        "reasons and historical-event classifications are in `coverage.json`.", ""])
    (output / "coverage.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("results/final/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final/audit"))
    parser.add_argument("--cpu-count", type=positive, default=8)
    parser.add_argument("--adaptive-iterations", action="store_true")
    parser.add_argument("--warmups", type=positive, default=2)
    parser.add_argument("--repetitions", type=positive, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--large-threshold", type=positive, default=512 ** 3)
    parser.add_argument("--large-warmups", type=positive, default=1)
    parser.add_argument("--large-repetitions", type=positive, default=3)
    args = parser.parse_args(argv)
    if args.repetitions < 3 or args.large_repetitions < 3 or not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("study repetitions must be at least three and seed must fit unsigned 32 bits")
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    policy = {name: getattr(args, name) for name in ("warmups", "repetitions", "seed", "large_threshold",
                                                    "large_warmups", "large_repetitions")}
    try:
        result = audit(args.input_dir, args.cpu_count, args.adaptive_iterations, **policy)
        write_outputs(args.output_dir, result)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Study {'complete' if result['complete'] else 'incomplete'}: "
          f"{result['valid_observations']} valid observations; outputs: {args.output_dir}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
