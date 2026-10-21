"""Dependency-free regression tests for the bounded experiment runner.

All benchmark subprocesses are mocked; these tests never collect measurements.
Run with: python -m unittest discover -s tests -p test_tools.py
"""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_sweep as sweep  # noqa: E402
import run_experiments as experiments  # noqa: E402


class PlanningTests(unittest.TestCase):
    def test_full_plan_covers_roadmap_dimensions(self):
        self.assertEqual(sweep.SIZES, (32, 64, 128, 256, 512, 768, 1024, 1536, 2048))
        self.assertEqual(sweep.BLOCKS, (8, 16, 24, 32, 48, 64, 96, 128, 192, 256))
        self.assertEqual(sweep.RECTANGLES, ((128, 1024, 64), (1024, 128, 1024), (64, 4096, 64)))
        plan = sweep.configurations("full", cpu_count=7)
        self.assertEqual(len(plan), len(set(plan)))
        squares = [c for c in plan if c.rows_a == c.cols_a == c.cols_b]
        self.assertEqual({c.rows_a for c in squares}, set(sweep.SIZES))
        for size in sweep.SIZES:
            current = [c for c in squares if c.rows_a == size]
            self.assertEqual({c.implementation for c in current},
                             {*sweep.LOOP_ORDERS, "accelerate", "blocked", "parallel"})
            self.assertEqual({c.block_size for c in current if c.implementation == "blocked"},
                             set(sweep.BLOCKS))
            self.assertEqual({c.requested_threads for c in current if c.implementation == "parallel"},
                             set(range(1, 8)))
        rectangles = [c for c in plan if c.rows_a != c.cols_a or c.cols_a != c.cols_b]
        self.assertEqual({(c.rows_a, c.cols_a, c.cols_b) for c in rectangles},
                         set(sweep.RECTANGLES))
        for shape in sweep.RECTANGLES:
            self.assertEqual({c.implementation for c in rectangles
                              if (c.rows_a, c.cols_a, c.cols_b) == shape},
                             {*sweep.LOOP_ORDERS, "blocked", "parallel", "accelerate"})
        self.assertTrue(all(1 <= c.requested_threads <= 7 for c in plan))

    def test_plan_order_is_stable_and_seeded_shuffle_is_reproducible(self):
        plan = sweep.configurations("quick", cpu_count=4, sizes=[8, 16])
        self.assertEqual(plan, sweep.configurations("quick", cpu_count=4, sizes=[8, 16]))
        expected = list(plan)
        random.Random(123).shuffle(expected)
        arguments = ["--dry-run", "--sizes", "8", "16", "--seed", "123"]
        outputs = []
        with mock.patch.object(sweep.os, "cpu_count", return_value=4):
            for _ in range(2):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(sweep.main(arguments), 0)
                outputs.append(json.loads(output.getvalue()))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], [sweep.asdict(c) for c in expected])

    def test_main_caps_detected_threads_and_filters_implementations(self):
        output = io.StringIO()
        with mock.patch.object(sweep.os, "cpu_count", return_value=32), redirect_stdout(output):
            code = sweep.main(["--dry-run", "--preset", "full", "--sizes", "8",
                               "--max-threads", "3", "--implementations", "parallel"])
        self.assertEqual(code, 0)
        plan = json.loads(output.getvalue())
        self.assertEqual({c["implementation"] for c in plan}, {"parallel"})
        self.assertEqual({c["requested_threads"] for c in plan}, {1, 2, 3})
        squares = [c for c in plan if c["rows_a"] == c["cols_a"] == c["cols_b"]]
        self.assertEqual({c["rows_a"] for c in squares}, {8})

    def test_single_cpu_quick_plan_has_no_duplicate_thread_counts(self):
        parallel = [c for c in sweep.configurations("quick", 1) if c.implementation == "parallel"]
        self.assertEqual(len(parallel), 4)
        self.assertEqual({c.requested_threads for c in parallel}, {1})

    def test_resume_key_distinguishes_every_experiment_input(self):
        row = {"build_label": "O3", "binary_sha256": "abc123", "implementation": "blocked",
               "rows_a": 8, "cols_a": 9, "cols_b": 10, "block_size": 16,
               "requested_threads": 2, "warmups": 1, "repetitions": 3, "seed": 42,
               "sweep_index": 0, "machine_id": "machine-a", "environment_id": "environment-a"}
        baseline = sweep.key(row)
        self.assertEqual(baseline, sweep.key({field: str(value) for field, value in row.items()}))
        for field in row:
            with self.subTest(field=field):
                changed = {**row, field: f"different-{row[field]}"}
                self.assertNotEqual(baseline, sweep.key(changed))

    def test_comparison_plan_covers_common_kernels_shapes_and_sizes(self):
        plan = sweep.configurations("comparison", cpu_count=4)
        self.assertEqual(len(plan), len(set(plan)))
        expected_shapes = {(size, size, size) for size in sweep.SIZES} | set(sweep.RECTANGLES)
        self.assertEqual({(c.rows_a, c.cols_a, c.cols_b) for c in plan}, expected_shapes)
        for shape in expected_shapes:
            with self.subTest(shape=shape):
                self.assertEqual({c.implementation for c in plan
                                  if (c.rows_a, c.cols_a, c.cols_b) == shape},
                                 {"naive", "reordered", "blocked", "parallel", "accelerate"})

    def test_adaptive_iterations_use_operation_volume_and_are_opt_in(self):
        below = sweep.Configuration("naive", 511, 512, 512)
        boundary = sweep.Configuration("naive", 512, 512, 512)
        rectangular = sweep.Configuration("naive", 128, 4096, 256)
        for config in (below, boundary, rectangular):
            self.assertEqual(sweep.iteration_counts(config, 2, 9), (2, 9))
        self.assertEqual(sweep.iteration_counts(below, 2, 9, adaptive=True), (2, 9))
        for config in (boundary, rectangular):
            self.assertEqual(sweep.iteration_counts(config, 2, 9, adaptive=True), (1, 3))
        self.assertEqual(sweep.iteration_counts(below, 4, 7, adaptive=True, threshold=1,
                                               large_warmups=2, large_repetitions=5), (2, 5))

    def test_multi_sweep_dry_run_has_reproducible_distinct_trial_identity(self):
        arguments = ["--dry-run", "--sizes", "8", "--seed", "19",
                     "--sweeps", "2", "--sweep-index", "3"]
        outputs = []
        with mock.patch.object(sweep.os, "cpu_count", return_value=4):
            for _ in range(2):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(sweep.main(arguments), 0)
                outputs.append(json.loads(output.getvalue()))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual({row["sweep_index"] for row in outputs[0]}, {3, 4})
        first = [{k: v for k, v in row.items() if k != "sweep_index"}
                 for row in outputs[0] if row["sweep_index"] == 3]
        second = [{k: v for k, v in row.items() if k != "sweep_index"}
                  for row in outputs[0] if row["sweep_index"] == 4]
        self.assertEqual({json.dumps(row, sort_keys=True) for row in first},
                         {json.dumps(row, sort_keys=True) for row in second})

    def test_square_only_filter_applies_to_full_and_comparison_plans(self):
        for preset in ("full", "comparison"):
            output = io.StringIO()
            with self.subTest(preset=preset), redirect_stdout(output):
                self.assertEqual(sweep.main(["--dry-run", "--preset", preset,
                                             "--sizes", "8", "16", "--skip-rectangles"]), 0)
                plan = json.loads(output.getvalue())
                self.assertEqual({row["rows_a"] for row in plan}, {8, 16})
                self.assertTrue(all(row["rows_a"] == row["cols_a"] == row["cols_b"] for row in plan))


class ExperimentTests(unittest.TestCase):
    def test_build_tool_resolution_uses_path_then_interpreter_sibling(self):
        with mock.patch.object(experiments.shutil, "which", return_value="/usr/bin/cmake"):
            self.assertEqual(experiments.executable("cmake"), "cmake")
        with tempfile.TemporaryDirectory(prefix="matmul-tool-resolution-") as directory:
            root = Path(directory)
            sibling = root / "cmake"
            sibling.touch()
            with mock.patch.object(experiments.shutil, "which", return_value=None), \
                    mock.patch.object(experiments.sys, "executable", str(root / "python")):
                self.assertEqual(experiments.executable("cmake"), str(sibling))
                with self.assertRaisesRegex(FileNotFoundError, "required tool 'ctest' is absent"):
                    experiments.executable("ctest")

    def test_study_routes_modes_and_forwards_repeat_controls(self):
        with tempfile.TemporaryDirectory(prefix="matmul-study-") as directory:
            root = Path(directory)
            with mock.patch.object(experiments, "run") as run, redirect_stdout(io.StringIO()):
                code = experiments.main(["--preset", "study", "--skip-plots",
                    "--build-root", str(root / "build"), "--results-root", str(root / "results"),
                    "--sweeps", "2", "--sweep-index", "5", "--adaptive-iterations",
                    "--large-threshold", "100", "--large-warmups", "1", "--large-repetitions", "3",
                    "--sizes", "8", "16", "--skip-rectangles", "--implementations", "naive", "blocked",
                    "--control-notes", "connected to power", "--cooldown-seconds", "2"])
        self.assertEqual(code, 0)
        commands = [call.args[0] for call in run.call_args_list]
        sweeps = [command for command in commands if any(str(item).endswith("run_sweep.py") for item in command)]
        self.assertEqual(len(sweeps), 4)
        mapping = {command[command.index("--build-label") + 1]: command[command.index("--preset") + 1]
                   for command in sweeps}
        self.assertEqual(mapping, {"O0": "comparison", "O2": "comparison", "O3": "full", "fast": "comparison"})
        for command in sweeps:
            with self.subTest(command=command):
                for field, value in (("--sweeps", "2"), ("--sweep-index", "5"),
                                     ("--large-threshold", "100"), ("--large-repetitions", "3"),
                                     ("--control-notes", "connected to power"), ("--cooldown-seconds", "2.0")):
                    self.assertEqual(command[command.index(field) + 1], value)
                self.assertIn("--adaptive-iterations", command)
                self.assertIn("--skip-rectangles", command)
                self.assertEqual(command[command.index("--sizes") + 1:command.index("--sizes") + 3], ["8", "16"])
                self.assertEqual(command[command.index("--implementations") + 1:command.index("--implementations") + 3],
                                 ["naive", "blocked"])

    def test_failed_build_stops_before_collecting_measurements(self):
        with tempfile.TemporaryDirectory(prefix="matmul-study-failure-") as directory:
            def build_failure(command, log=None):
                if command[:2] == ["cmake", "--build"]:
                    raise subprocess.CalledProcessError(1, command)
            with mock.patch.object(experiments, "run", side_effect=build_failure) as run, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = experiments.main(["--skip-plots", "--build-root", directory])
        self.assertEqual(code, 1)
        self.assertFalse(any(any(str(item).endswith("run_sweep.py") for item in call.args[0])
                             for call in run.call_args_list))


class SweepFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="matmul-tools-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "results.csv"

    def test_read_missing_empty_and_malformed_csv(self):
        self.assertEqual(sweep.read_results(self.path), ([], []))
        self.path.touch()
        self.assertEqual(sweep.read_results(self.path), ([], []))
        self.path.write_text("unrelated,value\nx,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a MatMul results CSV"):
            sweep.read_results(self.path)

    def test_append_expands_schema_without_losing_old_rows(self):
        sweep.append_result(self.path, {"implementation": "naive", "elapsed_time_ms": 1.25,
                                       "notes": 'comma, quote " and café'})
        sweep.append_result(self.path, {"implementation": "blocked", "elapsed_time_ms": 0.5,
                                       "notes": "new\nline", "new_metric": 99})
        sweep.append_result(self.path, {"implementation": "parallel", "elapsed_time_ms": 0.4})
        fields, rows = sweep.read_results(self.path)
        self.assertEqual(fields, ["implementation", "elapsed_time_ms", "notes", "new_metric"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["notes"], 'comma, quote " and café')
        self.assertEqual(rows[0]["new_metric"], "")
        self.assertEqual(rows[1]["notes"], "new\nline")
        self.assertEqual(rows[1]["new_metric"], "99")
        self.assertEqual(rows[2]["notes"], "")
        self.assertFalse(self.path.with_suffix(".csv.tmp").exists())


class MainTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="matmul-sweep-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.binary = self.root / "mock-binary"
        self.binary.write_bytes(b"test binary version one")
        self.output = self.root / "nested" / "results.csv"
        self.skip_log = self.output.with_suffix(".csv.skipped.jsonl")
        self.config = sweep.Configuration("naive", 2, 3, 4)
        self.plan_patch = mock.patch.object(sweep, "configurations", return_value=[self.config])
        self.plan_patch.start()
        self.addCleanup(self.plan_patch.stop)
        self.environment = {"machine_id": "test-machine", "environment_id": "test-environment",
                            "machine": {"model": "mock machine"},
                            "environment": {"power_source": "mock"}}
        self.environment_patch = mock.patch.object(sweep, "collect_environment",
                                                   side_effect=lambda: self.environment.copy())
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        self.args = ["--binary", str(self.binary), "--output", str(self.output),
                     "--build-label", "test", "--warmups", "1", "--repetitions", "2",
                     "--timeout", "0.25"]

    @staticmethod
    def response(value):
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(value), stderr="")

    @staticmethod
    def measurement(**overrides):
        return {"elapsed_time_ms": 1.25, "gflops": 0.125, "correct": True,
                "durations_ms": [1.0, 1.5], **overrides}

    def invoke(self, extra=()):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = sweep.main([*self.args, *extra])
        return result, stdout.getvalue(), stderr.getvalue()

    def logged_skips(self):
        return [json.loads(line) for line in self.skip_log.read_text(encoding="utf-8").splitlines()]

    def test_success_and_resume_use_binary_hash_and_exact_configuration(self):
        with mock.patch.object(sweep.subprocess, "run", side_effect=[
                self.response(["naive"]), self.response(self.measurement())]) as run:
            code, output, _ = self.invoke()
        self.assertEqual(code, 0)
        self.assertIn("1 measured", output)
        self.assertEqual(run.call_count, 2)
        command = run.call_args_list[1].args[0]
        self.assertEqual(command[:4], [str(self.binary.resolve()), "benchmark", "--implementation", "naive"])
        self.assertEqual(command[command.index("--rows") + 1], "2")
        self.assertEqual(command[command.index("--inner") + 1], "3")
        self.assertEqual(command[command.index("--cols") + 1], "4")
        self.assertEqual(run.call_args.kwargs["timeout"], 0.25)
        _, rows = sweep.read_results(self.output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["binary_sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())
        self.assertEqual(json.loads(rows[0]["durations_ms"]), [1.0, 1.5])
        self.assertEqual(rows[0]["build_label"], "test")
        with mock.patch.object(sweep.subprocess, "run", return_value=self.response(["naive"])) as run:
            code, output, _ = self.invoke()
        self.assertEqual(code, 0)
        self.assertIn("1 resumed", output)
        self.assertEqual(run.call_count, 1)
        self.binary.write_bytes(b"test binary version two")
        with mock.patch.object(sweep.subprocess, "run", side_effect=[
                self.response(["naive"]), self.response(self.measurement())]):
            self.assertEqual(self.invoke()[0], 0)
        _, rows = sweep.read_results(self.output)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["binary_sha256"], rows[1]["binary_sha256"])

    def test_timeout_is_logged_without_fabricated_measurement(self):
        with mock.patch.object(sweep.subprocess, "run", side_effect=[
                self.response(["naive"]), subprocess.TimeoutExpired("mock", 0.25)]):
            code, output, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("0 measured", output)
        self.assertIn("timeout after 0.25 seconds", stderr)
        self.assertFalse(self.output.exists())
        skips = self.logged_skips()
        self.assertEqual(len(skips), 1)
        self.assertIn("timeout", skips[0]["reason"])
        self.assertNotIn("elapsed_time_ms", skips[0])
        self.assertNotIn("gflops", skips[0])

    def test_unavailable_implementation_does_not_launch_benchmark(self):
        with mock.patch.object(sweep.subprocess, "run", return_value=self.response([])) as run:
            self.assertEqual(self.invoke()[0], 1)
        self.assertEqual(run.call_count, 1)
        self.assertIn("unavailable", self.logged_skips()[0]["reason"])
        self.assertFalse(self.output.exists())

    def test_failed_subprocess_returns_error_and_records_stderr(self):
        failure = subprocess.CalledProcessError(7, "mock", stderr="specific failure")
        with mock.patch.object(sweep.subprocess, "run", side_effect=[self.response(["naive"]), failure]):
            self.assertEqual(self.invoke()[0], 1)
        self.assertEqual(self.logged_skips()[0]["reason"], "exit 7: specific failure")
        self.assertFalse(self.output.exists())

    def test_invalid_results_are_rejected_without_writing_csv(self):
        missing_verification = self.measurement()
        del missing_verification["correct"]
        invalid_results = [[], {}, self.measurement(elapsed_time_ms=0),
                           self.measurement(elapsed_time_ms=float("nan")),
                           self.measurement(gflops=-1), self.measurement(correct=False),
                           self.measurement(elapsed_time_ms="not a number"),
                           missing_verification, self.measurement(correct="false"),
                           self.measurement(correct=0), self.measurement(warmups=3),
                           self.measurement(repetitions=9)]
        for result in invalid_results:
            with self.subTest(result=result):
                with mock.patch.object(sweep.subprocess, "run", side_effect=[
                        self.response(["naive"]), self.response(result)]):
                    self.assertEqual(self.invoke()[0], 1)
                self.assertFalse(self.output.exists())
        self.assertEqual(len(self.logged_skips()), len(invalid_results))

    def test_fresh_replaces_previous_csv_and_skip_log(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_text("invalid prior results\n", encoding="utf-8")
        self.skip_log.write_text('{"reason":"old failure"}\n', encoding="utf-8")
        with mock.patch.object(sweep.subprocess, "run", side_effect=[
                self.response(["naive"]), self.response(self.measurement())]):
            self.assertEqual(self.invoke(["--fresh"])[0], 0)
        self.assertEqual(len(sweep.read_results(self.output)[1]), 1)
        self.assertFalse(self.skip_log.exists())

    def test_implementation_query_failure_is_a_cli_error(self):
        with mock.patch.object(sweep.subprocess, "run", side_effect=subprocess.TimeoutExpired("list", 0.25)):
            with self.assertRaises(SystemExit) as failure:
                self.invoke()
        self.assertEqual(failure.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_independent_sweeps_resume_separately_and_preserve_input_seed(self):
        with mock.patch.object(sweep.subprocess, "run", side_effect=[
                self.response(["naive"]), self.response(self.measurement()),
                self.response(self.measurement())]) as run:
            self.assertEqual(self.invoke(["--sweeps", "2"])[0], 0)
        _, rows = sweep.read_results(self.output)
        self.assertEqual({row["sweep_index"] for row in rows}, {"0", "1"})
        self.assertEqual({row["seed"] for row in rows}, {"42"})
        self.assertEqual({row["machine_id"] for row in rows}, {"test-machine"})
        self.assertEqual(len({row["environment_id"] for row in rows}), 1)
        sidecar = self.output.with_suffix(".csv.metadata.jsonl")
        snapshots = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(row["phase"], row["sweep_index"]) for row in snapshots],
                         [("before", 0), ("after", 0), ("before", 1), ("after", 1)])
        self.assertEqual({row["environment_id"] for row in rows},
                         {row["environment_id"] for row in snapshots})
        self.assertEqual(run.call_count, 3)
        with mock.patch.object(sweep.subprocess, "run", return_value=self.response(["naive"])) as run:
            code, output, _ = self.invoke(["--sweeps", "2"])
        self.assertEqual(code, 0)
        self.assertIn("2 resumed", output)
        self.assertEqual(run.call_count, 1)

    def test_environment_change_does_not_resume_incompatible_measurement(self):
        for environment_id in ("test-environment", "new-environment"):
            self.environment["environment_id"] = environment_id
            with mock.patch.object(sweep.subprocess, "run", side_effect=[
                    self.response(["naive"]), self.response(self.measurement())]):
                self.assertEqual(self.invoke()[0], 0)
        _, rows = sweep.read_results(self.output)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row["environment_id"] for row in rows}), 2)

    def test_adaptive_cli_records_and_runs_actual_iteration_counts(self):
        with mock.patch.object(sweep.subprocess, "run", side_effect=[
                self.response(["naive"]), self.response(self.measurement())]) as run:
            self.assertEqual(self.invoke(["--warmups", "4", "--repetitions", "9",
                                         "--adaptive-iterations", "--large-threshold", "1",
                                         "--large-warmups", "2", "--large-repetitions", "3"])[0], 0)
        command = run.call_args_list[1].args[0]
        self.assertEqual(command[command.index("--warmups") + 1], "2")
        self.assertEqual(command[command.index("--repetitions") + 1], "3")
        _, rows = sweep.read_results(self.output)
        self.assertEqual((rows[0]["warmups"], rows[0]["repetitions"]), ("2", "3"))


if __name__ == "__main__":
    unittest.main()
