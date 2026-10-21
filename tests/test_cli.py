"""Small end-to-end CLI contracts; no performance measurements are retained.

Set MATMUL_BINARY to the built executable. CTest sets this automatically.
Run manually with: MATMUL_BINARY=build/release/matmul python -m unittest discover -s tests -p test_cli.py
"""

import csv
import io
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import unittest


@unittest.skipUnless(os.environ.get("MATMUL_BINARY"), "set MATMUL_BINARY to run CLI tests")
class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = str(Path(os.environ["MATMUL_BINARY"]).resolve())
        if not Path(cls.binary).is_file():
            raise RuntimeError(f"MATMUL_BINARY does not exist: {cls.binary}")

    def invoke(self, *arguments):
        return subprocess.run([self.binary, *arguments], capture_output=True, text=True, timeout=20)

    def test_rejects_invalid_options_with_diagnostic(self):
        cases = [
            ("unknown-command",),
            ("benchmark", "--unknown"),
            ("benchmark", "--size"),
            ("benchmark", "--size", "2", "--size", "3"),
            ("benchmark", "--size", "-1"),
            ("benchmark", "--size", "2.5"),
            ("benchmark", "--size", "184467440737095516160"),
            ("benchmark", "--size", "2", "--rows", "2"),
            ("benchmark", "--rows", "2", "--inner", "2"),
            ("benchmark", "--block-size", "0"),
            ("benchmark", "--threads", "0"),
            ("benchmark", "--threads", "many"),
            ("benchmark", "--warmups", "0"),
            ("benchmark", "--repetitions", "0"),
            ("benchmark", "--seed", "4294967296"),
            ("benchmark", "--atol", "nan"),
            ("benchmark", "--rtol", "inf"),
            ("benchmark", "--atol", "-0.1"),
            ("benchmark", "--sizes", "2,3"),
            ("benchmark", "--append"),
            ("sweep", "--sizes", "2,"),
            ("sweep", "--sizes", "2,,3"),
            ("sweep", "--size", "2"),
            ("list", "--size", "2"),
            ("verify", "--json"),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("matmul:", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_listing_and_every_implementation_rectangular_json(self):
        listing = self.invoke("list", "--json")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        names = json.loads(listing.stdout)
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue({"naive", "reordered", "blocked", "parallel"} <= set(names))
        for name in names:
            with self.subTest(implementation=name):
                process = self.invoke("benchmark", "--implementation", name,
                                      "--rows", "3", "--inner", "5", "--cols", "2",
                                      "--threads", "2", "--block-size", "4",
                                      "--warmups", "1", "--repetitions", "2",
                                      "--seed", "4294967295", "--json")
                self.assertEqual(process.returncode, 0, process.stderr)
                result = json.loads(process.stdout)
                self.assertTrue(result["correct"])
                self.assertEqual((result["rows_a"], result["cols_a"], result["cols_b"]), (3, 5, 2))
                self.assertEqual(result["seed"], 4294967295)
                self.assertEqual(result["repetitions"], 2)
                self.assertTrue(math.isfinite(result["checksum"]))

    def test_csv_append_and_schema_mismatch_preserve_file(self):
        with tempfile.TemporaryDirectory(prefix="matmul-cli-") as directory:
            output = Path(directory) / "nested" / "results.csv"
            arguments = ["benchmark", "--size", "2", "--repetitions", "1", "--warmups", "1",
                         "--output", str(output), "--json"]
            self.assertEqual(self.invoke(*arguments).returncode, 0)
            self.assertEqual(self.invoke(*arguments, "--append").returncode, 0)
            rows = list(csv.DictReader(io.StringIO(output.read_text(encoding="utf-8"))))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["correct"] == "true" for row in rows))
            original = "incompatible,column\nkeep,this\n"
            output.write_text(original, encoding="utf-8")
            rejected = self.invoke(*arguments, "--append")
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("schema mismatch", rejected.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), original)

    def test_zero_budget_sweep_creates_no_results(self):
        with tempfile.TemporaryDirectory(prefix="matmul-budget-") as directory:
            output = Path(directory) / "results.csv"
            process = self.invoke("sweep", "--sizes", "2", "--implementations", "naive",
                                  "--max-seconds", "0", "--output", str(output), "--json")
            self.assertEqual(process.returncode, 1)
            self.assertIn("Wrote 0 configurations", process.stderr)
            self.assertEqual(process.stdout, "")
            self.assertFalse(output.exists())

    def test_profile_runs_only_after_readiness_handshake(self):
        command = [self.binary, "profile", "--implementation", "blocked", "--size", "3",
                   "--block-size", "2", "--duration", "0.01", "--wait-for-profiler"]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            ready = queue.Queue()
            threading.Thread(target=lambda: ready.put(process.stdout.readline()), daemon=True).start()
            self.assertEqual(ready.get(timeout=20).strip(), "PROFILE_READY")
            self.assertIsNone(process.poll())
            stdout, stderr = process.communicate(input="start\n", timeout=20)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        self.assertEqual(process.returncode, 0, stderr)
        result = json.loads(stdout)
        self.assertEqual(result["implementation"], "blocked")
        self.assertGreater(result["iterations"], 0)
        self.assertGreaterEqual(result["elapsed_time_ms"], 10)
        self.assertTrue(math.isfinite(result["checksum"]))
        self.assertNotIn("verification_reference", result)
        cancelled = subprocess.run(command, input="", capture_output=True, text=True, timeout=20)
        self.assertEqual(cancelled.returncode, 2)
        self.assertEqual(cancelled.stdout.strip(), "PROFILE_READY")
        self.assertTrue(cancelled.stderr.strip())

    def test_profile_options_are_validated_and_scoped_to_command(self):
        for duration in ("0", "-1", "nan", "inf"):
            with self.subTest(duration=duration):
                process = self.invoke("profile", "--size", "2", "--duration", duration)
                self.assertEqual(process.returncode, 2, process.stderr)
        for arguments in [
            ("benchmark", "--duration", "0.01"),
            ("benchmark", "--wait-for-profiler"),
            ("profile", "--json"),
            ("profile", "--output", "unused.csv"),
            ("profile", "--append"),
            ("profile", "--max-seconds", "1"),
        ]:
            with self.subTest(arguments=arguments):
                process = self.invoke(*arguments)
                self.assertEqual(process.returncode, 2, process.stderr)


if __name__ == "__main__":
    unittest.main()
