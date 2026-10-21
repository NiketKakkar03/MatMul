"""Synthetic study audit tests; no benchmark subprocesses are launched."""

from contextlib import redirect_stdout
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_study as study  # noqa: E402


def measurement(mode, config, **overrides):
    warmups, repetitions = study.iteration_counts(config, 2, 9, True)
    durations = [1 + index / 10 for index in range(repetitions)]
    median = statistics.median(durations)
    return {**asdict(config), "build_label": mode, "binary_sha256": "a" * 64,
        "machine_id": "b" * 64, "environment_id": "c" * 64,
        "timestamp": "2026-10-20T19:00:00Z", "compiler": "Clang", "compiler_flags": "-O3",
        "machine_architecture": "arm64", "build_type": "Release", "logical_cpu_count": 8,
        "verification_reference": "FP64 reference", "correct": True, "seed": 42, "sweep_index": 0,
        "thread_count": 0 if config.implementation == "accelerate" else config.requested_threads,
        "warmups": warmups, "repetitions": repetitions,
        "durations_ms": ";".join(map(str, durations)), "elapsed_time_ms": median,
        "minimum_time_ms": min(durations), "standard_deviation_ms": statistics.pstdev(durations),
        "gflops": 2 * config.rows_a * config.cols_a * config.cols_b / median / 1e6,
        "maximum_absolute_error": 0, "maximum_relative_error": 0, "absolute_tolerance": 0.0001,
        "relative_tolerance": 0.0001, "initialization_time_ms": 0.25,
        "verification_time_ms": 0.5, "estimated_bandwidth": 0.01, "checksum": 1,
        **overrides}


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="matmul-study-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.plan = study.planned_study(8)
        self.config = study.Configuration("naive", 32, 32, 32)

    def write(self, name, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_complete_study_and_repeated_sweep_cover_432_workloads(self):
        self.assertEqual({mode: len(plan) for mode, plan in self.plan.items()},
                         {"O3": 252, "O2": 60, "fast": 60, "O0": 60})
        for mode, plan in self.plan.items():
            rows = [measurement(mode, config) for config in plan]
            rows.append(measurement(mode, next(iter(plan)), sweep_index=1))
            self.write(mode + ".csv", rows)
        result = study.audit(self.root, 8, True)
        self.assertTrue(result["complete"])
        self.assertEqual(result["valid_observations"], 436)
        self.assertEqual(result["planned_total"], 432)
        study.write_outputs(self.root / "audit", result)
        self.assertTrue(json.loads((self.root / "audit/coverage.json").read_text())["complete"])
        self.assertIn("Status: complete", (self.root / "audit/coverage.md").read_text())

    def test_partial_mode_and_missing_modes_fail_without_guessing_coverage(self):
        self.write("O3.csv", [measurement("O3", self.config)])
        result = study.audit(self.root, 8, True)
        self.assertFalse(result["complete"])
        group = result["modes"]["O3"]["identity_groups"][0]
        self.assertEqual((group["planned"], group["completed"], len(group["missing"])), (252, 1, 251))
        self.assertEqual(result["modes"]["O0"]["identity_groups"], [])

    def test_partial_different_environments_never_merge(self):
        rows = [measurement("O3", config, environment_id=("c" if index % 2 else "d") * 64)
                for index, config in enumerate(self.plan["O3"])]
        self.write("O3.csv", rows)
        groups = study.audit(self.root, 8, True)["modes"]["O3"]["identity_groups"]
        self.assertEqual(len(groups), 2)
        self.assertEqual(sum(group["completed"] for group in groups), 252)
        self.assertTrue(all(not group["complete"] for group in groups))

    def test_metric_sample_and_metadata_corruption_are_invalid(self):
        baseline = measurement("O3", self.config)
        modifications = [{"elapsed_time_ms": 99}, {"minimum_time_ms": 99},
            {"standard_deviation_ms": 99}, {"gflops": 99}, {"durations_ms": "1;2;3"},
            {"durations_ms": ";".join(["nan"] * 9)}, {"compiler_flags": ""},
            {"environment_id": "unknown"}, {"timestamp": "2026-10-20T19:00:00"},
            {"correct": "false"}, {"warmups": 1}, {"requested_threads": 9}, {"seed": 43}]
        for changes in modifications:
            with self.subTest(changes=changes):
                raw = {key: str(value) for key, value in {**baseline, **changes}.items()}
                with self.assertRaises(ValueError):
                    study.validate_row(raw, 8, True)

    def test_duplicate_workload_sweep_is_invalid_but_another_sweep_is_valid(self):
        row = measurement("O3", self.config)
        self.write("O3.csv", [row, row, measurement("O3", self.config, sweep_index=1)])
        result = study.audit(self.root, 8, True)
        self.assertEqual(result["valid_observations"], 2)
        self.assertEqual(len(result["invalid_rows"]), 1)
        self.assertIn("duplicate", result["invalid_rows"][0]["reason"])

    def test_unexpected_configuration_is_reported(self):
        self.write("O3.csv", [measurement("O3", study.Configuration("naive", 7, 7, 7))])
        result = study.audit(self.root, 8, True)
        group = result["modes"]["O3"]["identity_groups"][0]
        self.assertEqual(group["completed"], 0)
        self.assertEqual(len(group["unexpected"]), 1)
        self.assertFalse(group["complete"])

    def test_historical_timeout_recovers_only_from_matching_valid_observation(self):
        row = measurement("O3", self.config)
        self.write("O3.csv", [row])
        events = [{**row, "reason": "timeout after 10 seconds"},
                  {**row, "sweep_index": 1, "reason": "timeout after 10 seconds"}]
        (self.root / "O3.csv.skipped.jsonl").write_text(
            "\n".join(map(json.dumps, events)) + "\n", encoding="utf-8")
        result = study.audit(self.root, 8, True)
        self.assertEqual([event["status"] for event in result["historical_skip_events"]],
                         ["recovered", "unresolved"])
        self.assertEqual(result["modes"]["O3"]["identity_groups"][0]["completed"], 1)

    def test_empty_input_cli_writes_incomplete_report_and_returns_nonzero(self):
        with redirect_stdout(io.StringIO()):
            code = study.main(["--input-dir", str(self.root), "--output-dir", str(self.root / "audit"),
                               "--cpu-count", "8", "--adaptive-iterations"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads((self.root / "audit/coverage.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
