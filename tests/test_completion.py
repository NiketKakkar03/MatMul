"""Check reproduction planning and failure handling without running benchmarks."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import complete_project as completion  # noqa: E402


class CompletionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="matmul-completion-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.results = self.root / "results"
        self.build = self.root / "build"
        self.arguments = ["--results-root", str(self.results), "--build-root", str(self.build)]

    def invoke(self, *extra):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = completion.main([*self.arguments, *extra])
        return result, output.getvalue()

    def status(self):
        return json.loads((self.results / "completion_status.json").read_text(encoding="utf-8"))

    @staticmethod
    def script_calls(run):
        return [call.args[0] for call in run.call_args_list if call.args[0][0] == sys.executable]

    def test_dry_run_is_reviewable_and_has_no_execution_or_file_writes(self):
        with mock.patch.object(completion, "run") as run, mock.patch.object(completion.os, "cpu_count", return_value=8):
            code, output = self.invoke("--dry-run", "--control-notes", "operator's observation; literal $data")
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertFalse(self.results.exists())
        self.assertFalse(self.build.exists())
        for step in ("configure-debug", "test-asan", "test-tsan", "python-tests", "study",
                     "repeats-O3_small", "repeats-O3_large", "audit", "analyze-repeats", "plot-results", "profiles"):
            self.assertIn(f"[{step}]", output)
        self.assertIn("literal $data", output)
        self.assertIn("--cpu-count 8", output)

    def test_full_run_orders_validation_measurements_analysis_and_profiles_serially(self):
        with mock.patch.object(completion, "run") as run, mock.patch.object(completion.os, "cpu_count", return_value=8):
            self.assertEqual(self.invoke()[0], 0)
        calls = [call.args[0] for call in run.call_args_list]
        study_index = next(index for index, command in enumerate(calls) if "study" in command)
        self.assertEqual(study_index, 13)  # Four build/configure/test triples and Python discovery.
        scripts = self.script_calls(run)
        self.assertEqual([Path(command[1]).name for command in scripts],
                         ["run_experiments.py"] + ["run_sweep.py"] * 5 +
                         ["validate_study.py", "analyze_repeats.py", "plot_results.py", "profile.py"])
        for name in ("debug", "asan", "tsan", "profile"):
            self.assertTrue((self.build / name).is_dir())
        python_command = calls[12]
        self.assertIn(f"MATMUL_BINARY={self.build / 'profile/matmul'}", python_command)
        self.assertEqual(python_command[-3:], ["tests", "-p", "test_*.py"])
        status = self.status()
        self.assertTrue(status["complete"])
        self.assertEqual(status["build_validation"], "passed")
        self.assertEqual(status["profiling"], "captured")
        self.assertEqual(status["completed_steps"][-1], "profiles")

    def test_configurations_and_binary_paths_preserve_the_declared_study(self):
        notes = "Observed AC supply; applications active"
        with mock.patch.object(completion, "run") as run, mock.patch.object(completion.os, "cpu_count", return_value=16):
            self.assertEqual(self.invoke("--max-threads", "6", "--repeats", "4", "--timeout", "700",
                                         "--control-notes", notes, "--profile-backend", "sample")[0], 0)
        scripts = self.script_calls(run)
        study = scripts[0]
        self.assertEqual(study[study.index("--modes") + 1:study.index("--build-root")],
                         ["O3", "O2", "fast", "O0"])
        self.assertIn("--vectorization-report", study)
        self.assertIn("--skip-plots", study)
        repeats = scripts[1:6]
        self.assertEqual([Path(command[command.index("--output") + 1]).stem for command in repeats],
                         ["O3_small", "O3_large", "O0", "O2", "fast"])
        for command in [study, *repeats]:
            self.assertEqual(command[command.index("--max-threads") + 1], "6")
            self.assertEqual(command[command.index("--control-notes") + 1], notes)
            self.assertEqual(command[command.index("--timeout") + 1], "700")
            self.assertIn("--adaptive-iterations", command)
        for command in repeats:
            mode = command[command.index("--build-label") + 1]
            self.assertEqual(command[command.index("--binary") + 1], str(self.build / "experiments" / mode / "matmul"))
            self.assertEqual(command[command.index("--sweeps") + 1], "4")
            self.assertEqual(command[command.index("--sweep-index") + 1], "1")
            self.assertEqual(command[command.index("--cooldown-seconds") + 1], "2")
            self.assertIn("--skip-rectangles", command)
        self.assertEqual(repeats[0][repeats[0].index("--sizes") + 1:repeats[0].index("--skip-rectangles")],
                         ["64", "128", "256", "512"])
        self.assertEqual(repeats[1][repeats[1].index("--sizes") + 1:repeats[1].index("--skip-rectangles")],
                         ["1024", "2048"])
        audit = scripts[6]
        self.assertEqual(audit[audit.index("--cpu-count") + 1], "6")
        for command in scripts[7:9]:
            self.assertEqual(sum(str(item).endswith(".csv") for item in command), 9)
        profile = scripts[-1]
        self.assertEqual(profile[profile.index("--backend") + 1], "sample")
        self.assertEqual(profile[profile.index("--duration") + 1], "5s")
        self.assertEqual(profile[profile.index("--size") + 1], "512")

    def test_skipped_profiling_is_explicitly_pending(self):
        with mock.patch.object(completion, "run") as run:
            code, output = self.invoke("--profile-backend", "skip")
        self.assertEqual(code, 0)
        self.assertFalse(any(Path(command[1]).name == "profile.py" for command in self.script_calls(run)))
        self.assertEqual(self.status()["profiling"], "pending")
        self.assertFalse(self.status()["complete"])
        self.assertIn("remains pending", output)

    def test_skipped_build_validation_never_claims_fresh_checks(self):
        with mock.patch.object(completion, "run") as run:
            self.assertEqual(self.invoke("--skip-build-validation")[0], 0)
        self.assertEqual(Path(run.call_args_list[0].args[0][1]).name, "run_experiments.py")
        self.assertFalse(self.build.exists())
        self.assertEqual(self.status()["build_validation"], "not_run")
        self.assertFalse(self.status()["complete"])

    def test_failure_stops_before_later_measurements_and_is_recorded(self):
        failure = subprocess.CalledProcessError(1, ["ctest"])
        with mock.patch.object(completion, "run", side_effect=[None, None, failure]) as run:
            code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 3)
        status = self.status()
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["active_step"], "test-debug")
        self.assertFalse(status["complete"])
        self.assertIn("test-debug", output)

    def test_unavailable_cpu_detection_falls_back_to_one(self):
        with mock.patch.object(completion, "run") as run, mock.patch.object(completion.os, "cpu_count", return_value=None):
            self.assertEqual(self.invoke("--skip-build-validation", "--profile-backend", "skip")[0], 0)
        study = self.script_calls(run)[0]
        self.assertEqual(study[study.index("--max-threads") + 1], "1")
        self.assertNotIn("AC power observed", self.status()["control_notes"])

    def test_invalid_settings_fail_before_any_execution(self):
        for arguments in (("--timeout", "nan"), ("--timeout", "0"), ("--timeout", "inf"),
                          ("--repeats", "2"), ("--jobs", "0"), ("--max-threads", "0")):
            with self.subTest(arguments=arguments), mock.patch.object(completion, "run") as run:
                with self.assertRaises(SystemExit) as error:
                    self.invoke(*arguments)
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
