"""Mocked profiling lifecycle tests; no native profiler or kernel is launched."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("matmul_profile_tool", SCRIPTS / "profile.py")
profiling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiling)


class ProfilePlanningTests(unittest.TestCase):
    def test_duration_conversion_and_invalid_values(self):
        for value, expected in (("2", 2), ("1.5s", 1.5), ("250ms", .25), ("2m", 120)):
            with self.subTest(value=value):
                self.assertEqual(profiling.duration_seconds(value), expected)
        for value in ("0", "0ms", "-1", "nan", "inf", "1h", "1 s", "", "1e3", "9" * 400):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                profiling.duration_seconds(value)

    def test_recording_commands_attach_to_existing_target(self):
        output = Path("profiles/example")
        sample = profiling.recording_command("sample", "123", output, 1.25, 2, "Time Profiler")
        self.assertEqual(sample, ["sample", "123", "2", "2", "-file", str(output.resolve())])
        instruments = profiling.recording_command("instruments", "123", output, 1.25, 2, "Time Profiler")
        self.assertEqual(instruments[-2:], ["--attach", "123"])
        self.assertEqual(instruments[instruments.index("--time-limit") + 1], "1.25s")
        self.assertNotIn("--launch", instruments)
        self.assertNotIn("benchmark", instruments)

    def test_auto_backend_uses_instruments_when_present_and_sample_otherwise(self):
        with mock.patch.object(profiling.subprocess, "run") as run:
            for code, expected in ((0, "instruments"), (1, "sample")):
                run.return_value = subprocess.CompletedProcess([], code, stdout="", stderr="")
                self.assertEqual(profiling.resolve_backend("auto"), expected)
            run.assert_called_with(["xcrun", "--find", "xctrace"], capture_output=True,
                                   text=True, timeout=10)
            for error in (FileNotFoundError("xcrun"), subprocess.TimeoutExpired("xcrun", 10)):
                run.side_effect = error
                self.assertEqual(profiling.resolve_backend("auto"), "sample")
            run.reset_mock()
            self.assertEqual(profiling.resolve_backend("sample"), "sample")
            run.assert_not_called()

    def test_dry_run_spawns_no_process_and_creates_no_files(self):
        with tempfile.TemporaryDirectory(prefix="matmul-profile-plan-") as directory:
            root = Path(directory)
            for backend in ("sample", "instruments"):
                output = io.StringIO()
                with self.subTest(backend=backend), mock.patch.object(profiling.subprocess, "Popen") as popen, \
                        mock.patch.object(profiling.subprocess, "run") as run, redirect_stdout(output):
                    self.assertEqual(profiling.main(["--dry-run", "--backend", backend,
                        "--binary", str(root / "missing-binary"), "--output-dir", str(root / "nested"),
                        "--duration", "1.25s", "--block-size", "32", "--threads", "3"]), 0)
                    popen.assert_not_called()
                    run.assert_not_called()
                    commands = [shlex.split(line) for line in output.getvalue().splitlines()]
                    self.assertEqual(len(commands), 2)
                    target, recorder = commands
                    self.assertEqual(target[1], "profile")
                    self.assertIn("--wait-for-profiler", target)
                    self.assertNotIn("benchmark", target)
                    self.assertNotIn("--launch", recorder)
                    self.assertIn("--block-size", target)
                    self.assertEqual(target[target.index("--block-size") + 1], "32")
                    self.assertEqual(target[target.index("--threads") + 1], "3")
                    expected_duration = 4 if backend == "sample" else 16.25
                    self.assertEqual(float(target[target.index("--duration") + 1]), expected_duration)
                self.assertEqual(list(root.iterdir()), [])

    def test_multiple_implementations_reject_single_output_and_unsafe_names(self):
        bad_arguments = [
            ["--implementations", "naive", "blocked", "--output", "one.trace"],
            ["--implementation", "../escape"],
            ["--implementation", "/absolute"],
            ["--threads", "0"],
            ["--threads", "invalid"],
        ]
        for arguments in bad_arguments:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), \
                    mock.patch.object(profiling.subprocess, "Popen") as popen, \
                    self.assertRaises(SystemExit) as error:
                profiling.main(["--dry-run", "--backend", "sample", *arguments])
            self.assertEqual(error.exception.code, 2)
            popen.assert_not_called()

    def test_multiple_implementations_plan_distinct_capture_paths(self):
        output = io.StringIO()
        with mock.patch.object(profiling.subprocess, "Popen") as popen, redirect_stdout(output):
            self.assertEqual(profiling.main(["--dry-run", "--backend", "sample",
                                            "--implementations", "naive", "blocked"]), 0)
        commands = [shlex.split(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(commands), 4)
        targets, recorders = commands[::2], commands[1::2]
        self.assertEqual([command[command.index("--implementation") + 1] for command in targets],
                         ["naive", "blocked"])
        self.assertEqual(len({command[-1] for command in recorders}), 2)
        popen.assert_not_called()


class ProfileLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="matmul-profile-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.binary = self.root / "matmul"
        self.binary.write_bytes(b"mock executable")
        self.output = self.root / "nested" / "profile.sample.txt"
        self.args = argparse.Namespace(binary=self.binary, size=8, block_size=4, threads="2",
            duration=.5, sampling_interval_ms=1, template="Time Profiler", startup_timeout=2)
        self.result = {"implementation": "blocked", "iterations": 10,
                       "elapsed_time_ms": 3000, "checksum": 12.5}
        self.target = mock.Mock(pid=123, returncode=None)
        self.target.stdin = io.StringIO()
        self.target.poll.side_effect = lambda: self.target.returncode

        def target_done(*args, **kwargs):
            self.target.returncode = 0
            return json.dumps(self.result), ""
        self.target.communicate.side_effect = target_done
        self.recorder = mock.Mock(returncode=None)
        self.recorder.poll.side_effect = lambda: self.recorder.returncode

        def recorder_done(*args, **kwargs):
            self.output.write_text("recorded stacks\n", encoding="utf-8")
            self.recorder.returncode = 0
            return 0
        self.recorder.wait.side_effect = recorder_done
        for name, replacement in (("collect_environment", mock.Mock(return_value={"machine_id": "mock"})),
                                  ("wait_ready", mock.Mock()),
                                  ("subprocess.Popen", mock.Mock(side_effect=[self.target, self.recorder])),
                                  ("time.sleep", mock.Mock())):
            # The module is loaded without installing its name in sys.modules.
            target, _, attribute = name.rpartition(".")
            owner = profiling
            for component in target.split(".") if target else ():
                owner = getattr(owner, component)
            patch = mock.patch.object(owner, attribute or name, replacement)
            patch.start()
            self.addCleanup(patch.stop)

    def metadata(self):
        return json.loads(self.output.with_name(self.output.name + ".metadata.json").read_text(encoding="utf-8"))

    def record(self):
        with redirect_stdout(io.StringIO()):
            return profiling.record(self.args, "sample", "blocked", self.output)

    def test_success_records_artifact_commands_and_environment_metadata(self):
        returned = self.record()
        metadata = self.metadata()
        self.assertEqual(returned, metadata)
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["kernel_workload"], self.result)
        self.assertEqual(metadata["environment_before"], {"machine_id": "mock"})
        self.assertEqual(metadata["environment_after"], {"machine_id": "mock"})
        self.assertEqual(metadata["target_command"][1], "profile")
        self.assertNotIn("benchmark", metadata["target_command"])
        self.assertEqual(metadata["recording_command"][1], "123")
        self.assertEqual(self.target.stdin.getvalue(), "start\n")
        self.assertTrue(self.output.with_name(self.output.name + ".recorder.log").exists())
        self.target.terminate.assert_not_called()
        self.recorder.terminate.assert_not_called()

    def test_artifact_is_required_before_claiming_success(self):
        for empty_file in (False, True):
            with self.subTest(empty_file=empty_file):
                def no_artifact(*args, **kwargs):
                    if empty_file:
                        self.output.touch()
                    self.recorder.returncode = 0
                    return 0
                self.recorder.wait.side_effect = no_artifact
                with self.assertRaisesRegex(RuntimeError, "profile artifact"):
                    self.record()
                self.assertEqual(self.metadata()["status"], "failed")
                for path in self.output.parent.iterdir():
                    path.unlink()
                self.target.returncode = self.recorder.returncode = None
                profiling.subprocess.Popen.side_effect = [self.target, self.recorder]

    def test_failed_recorder_stops_owned_processes_and_writes_failure_metadata(self):
        self.recorder.wait.side_effect = None
        self.recorder.wait.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "recorder failed"):
            self.record()
        self.target.terminate.assert_called_once()
        self.recorder.terminate.assert_called_once()
        self.assertEqual(self.metadata()["status"], "failed")
        self.assertIn("recorder failed", self.metadata()["error"])

    def test_invalid_target_statistics_never_produce_complete_metadata(self):
        invalid_results = [[], {}]
        invalid_results.extend({**self.result, "iterations": value} for value in (0, "1", True, None))
        invalid_results.extend({**self.result, "elapsed_time_ms": value}
                               for value in (0, -1, float("nan"), "10"))
        invalid_results.extend({**self.result, "checksum": value}
                               for value in (None, float("inf"), "10"))
        for result in invalid_results:
            with self.subTest(result=result):
                self.result = result
                profiling.subprocess.Popen.side_effect = [self.target, self.recorder]
                self.target.returncode = self.recorder.returncode = None
                with self.assertRaises(RuntimeError):
                    self.record()
                self.assertEqual(self.metadata()["status"], "failed")
                self.assertNotIn("kernel_workload", self.metadata())
                for path in self.output.parent.iterdir():
                    path.unlink()

    def test_readiness_failure_cleans_target_without_spawning_recorder(self):
        profiling.wait_ready.side_effect = RuntimeError("startup timeout")
        with self.assertRaisesRegex(RuntimeError, "startup timeout"):
            self.record()
        self.assertEqual(profiling.subprocess.Popen.call_count, 1)
        self.target.terminate.assert_called_once()
        self.assertEqual(self.metadata()["status"], "failed")

    def test_early_recorder_exit_never_releases_kernel_workload(self):
        self.recorder.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "before kernel execution"):
            self.record()
        self.assertEqual(self.target.stdin.getvalue(), "")
        self.target.terminate.assert_called_once()
        self.recorder.terminate.assert_not_called()
        self.assertEqual(self.metadata()["status"], "failed")

    def test_existing_artifacts_are_preserved_without_starting_processes(self):
        self.output.parent.mkdir()
        for path in (self.output, self.output.with_name(self.output.name + ".recorder.log"),
                     self.output.with_name(self.output.name + ".metadata.json")):
            with self.subTest(path=path):
                path.write_text("keep this", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    self.record()
                self.assertEqual(path.read_text(encoding="utf-8"), "keep this")
                path.unlink()
        profiling.subprocess.Popen.assert_not_called()


class ReadinessAndCleanupTests(unittest.TestCase):
    def test_readiness_eof_invalid_signal_and_timeout(self):
        process = mock.Mock()
        process.stdout.fileno.return_value = 123
        with mock.patch.object(profiling.select, "select", return_value=([process.stdout], [], [])), \
                mock.patch.object(profiling.os, "read", return_value=b"PROFILE_READY\n") as read:
            profiling.wait_ready(process, 2)
            for signal in (b"", b"unexpected\n"):
                read.return_value = signal
                with self.subTest(signal=signal), self.assertRaisesRegex(RuntimeError, "readiness signal"):
                    profiling.wait_ready(process, 2)
        with mock.patch.object(profiling.select, "select", return_value=([], [], [])), \
                self.assertRaisesRegex(RuntimeError, "startup timeout"):
            profiling.wait_ready(process, 2)

    def test_partial_readiness_signal_remains_bounded_by_startup_timeout(self):
        process = mock.Mock()
        process.stdout.fileno.return_value = 123
        with mock.patch.object(profiling.select, "select", side_effect=[([process.stdout], [], []), ([], [], [])]), \
                mock.patch.object(profiling.os, "read", return_value=b"PROFILE_") as read, \
                self.assertRaisesRegex(RuntimeError, "startup timeout"):
            profiling.wait_ready(process, 2)
        read.assert_called_once_with(123, 4096)

    def test_cleanup_escalates_to_kill_when_owned_process_does_not_terminate(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.communicate.side_effect = [subprocess.TimeoutExpired("owned", 5), ("", "")]
        profiling.stop_owned(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.call_count, 2)
        profiling.stop_owned(None)


if __name__ == "__main__":
    unittest.main()
