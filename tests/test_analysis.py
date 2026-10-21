"""Synthetic regression checks for independent-sweep analysis (no benchmarks)."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_repeats as analysis  # noqa: E402
import plot_results  # noqa: E402

try:
    import pandas as pd
except ImportError:
    pd = None


class RepeatAnalysisTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="matmul-analysis-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    @staticmethod
    def row(**overrides):
        return {"build_label": "O3", "binary_sha256": "binary-A", "implementation": "blocked",
                "rows_a": "8", "cols_a": "9", "cols_b": "10", "block_size": "16",
                "requested_threads": "2", "thread_count": "2", "warmups": "1",
                "repetitions": "9", "seed": "42", "correct": "True", "sweep_index": "1",
                "elapsed_time_ms": "2", "durations_ms": "9999;9999;9999", **overrides}

    def write(self, name, rows):
        path = self.root / name
        fields = list(dict.fromkeys(field for row in rows for field in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_uses_sweep_medians_and_population_variability_without_pooling(self):
        path = self.write("repeat.csv", [self.row(sweep_index=str(index), elapsed_time_ms=str(value))
                                         for index, value in enumerate((1, 2, 3), 1)])
        rows, coverage = analysis.analyze([path])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["independent_sweeps"], 3)
        self.assertEqual(row["median_of_medians_ms"], 2)
        self.assertEqual((row["min_ms"], row["max_ms"]), (1, 3))
        self.assertAlmostEqual(row["population_std_ms"], math.sqrt(2 / 3))
        self.assertAlmostEqual(row["cv_percent"], 100 * math.sqrt(2 / 3) / 2)
        self.assertEqual(row["ci_status"], "exploratory_small_sample")
        self.assertGreaterEqual(row["bootstrap95_low_ms"], 1)
        self.assertLessEqual(row["bootstrap95_high_ms"], 3)
        self.assertEqual(coverage["groups_by_independent_sweep_count"], {"3": 1})

    def test_bootstrap_is_deterministic_and_order_independent(self):
        values = [2, 4, 9, 11, 30]
        self.assertEqual(analysis.bootstrap_interval(values, ("A",)),
                         analysis.bootstrap_interval(list(reversed(values)), ("A",)))

    def test_different_inputs_builds_and_runtime_settings_stay_separate(self):
        changes = {"binary_sha256": "binary-B", "build_label": "fast", "rows_a": "10",
                   "cols_a": "10", "cols_b": "11", "block_size": "32", "seed": "43",
                   "warmups": "2", "repetitions": "7", "thread_count": "1",
                   "requested_threads": "3", "compiler_flags": "-ffast-math",
                   "absolute_tolerance": "0.01", "verification_reference": "other",
                   "machine_id": "other-machine", "environment_id": "battery-mode"}
        path = self.write("configs.csv", [self.row()] +
                          [self.row(**{field: value}) for field, value in changes.items()])
        rows, _ = analysis.analyze([path])
        self.assertEqual(len(rows), len(changes) + 1)
        self.assertTrue(all(row["independent_sweeps"] == 1 for row in rows))
        self.assertTrue(all(row["bootstrap95_low_ms"] == "" for row in rows))

    def test_two_sweeps_have_no_confidence_interval(self):
        path = self.write("two.csv", [self.row(), self.row(sweep_index="2")])
        row = analysis.analyze([path])[0][0]
        self.assertEqual(row["independent_sweeps"], 2)
        self.assertEqual(row["ci_status"], "insufficient_independent_sweeps")
        self.assertEqual(row["bootstrap95_high_ms"], "")

    def test_accelerate_managed_threads_zero_is_valid_metadata(self):
        path = self.write("accelerate.csv", [self.row(implementation="accelerate", thread_count="0")])
        self.assertEqual(analysis.analyze([path])[0][0]["thread_count"], "0")

    def test_duplicate_collection_rejected_within_and_across_inputs(self):
        first = self.write("a.csv", [self.row()])
        second = self.write("b.csv", [self.row(elapsed_time_ms="9")])
        duplicate = self.write("duplicate.csv", [self.row(), self.row()])
        for paths in ([first, second], [duplicate], [first, first]):
            with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "duplicate configuration"):
                analysis.analyze(paths)

    def test_legacy_files_supply_one_collection_each_and_match_binary(self):
        paths = [self.write(f"legacy{index}.csv", [self.row(sweep_index="")]) for index in range(3)]
        rows, coverage = analysis.analyze(paths)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["independent_sweeps"], 3)
        self.assertEqual(coverage["matched_binary_groups_with_at_least_three_independent_sweeps"], 1)
        self.assertTrue(all(source["legacy_rows"] == 1 for source in coverage["input_files"]))
        different = self.write("newbinary.csv", [self.row(sweep_index="", binary_sha256="binary-B")])
        self.assertEqual(len(analysis.analyze([*paths, different])[0]), 2)

    def test_unknown_binary_cannot_establish_repeats_across_files_or_ids(self):
        paths = [self.write(f"unknown{index}.csv", [self.row(binary_sha256="", sweep_index=str(index))])
                 for index in range(3)]
        rows, coverage = analysis.analyze(paths)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["independent_sweeps"] == 1 for row in rows))
        self.assertEqual(coverage["matched_binary_groups_with_at_least_three_independent_sweeps"], 0)
        duplicate = self.write("unknown-ids.csv", [self.row(binary_sha256=""),
                                                     self.row(binary_sha256="", sweep_index="2")])
        with self.assertRaisesRegex(ValueError, "duplicate configuration"):
            analysis.analyze([duplicate])

    def test_invalid_or_unverified_observations_fail_explicitly(self):
        for index, changes in enumerate(({"elapsed_time_ms": "nan"}, {"elapsed_time_ms": "inf"},
                                         {"elapsed_time_ms": "0"}, {"elapsed_time_ms": "bad"},
                                         {"rows_a": "-1"}, {"thread_count": "-1"},
                                         {"correct": "False"}, {"seed": ""})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                analysis.analyze([self.write(f"invalid{index}.csv", [self.row(**changes)])])

    def test_cli_outputs_report_summary_and_coverage(self):
        path = self.write("input.csv", [self.row(sweep_index=str(index)) for index in range(3)])
        output = self.root / "output"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(analysis.main([str(path), "--output-dir", str(output)]), 0)
        with (output / "repeat_summary.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["independent_sweeps"], "3")
        self.assertEqual(json.loads((output / "coverage.json").read_text())["measurement_rows"], 3)
        self.assertIn("exploratory", (output / "repeat_report.md").read_text())
        self.assertIn("no pooling", json.loads((output / "coverage.json").read_text())["method"])

    def test_cli_reports_error_without_creating_misleading_outputs(self):
        path = self.write("bad.csv", [self.row(), self.row()])
        output = self.root / "output"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            analysis.main([str(path), "--output-dir", str(output)])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())


@unittest.skipUnless(pd is not None, "optional pandas analysis dependency not installed")
class PlotAggregationTests(unittest.TestCase):
    @staticmethod
    def row(**overrides):
        return {"build_label": "O3", "binary_sha256": "hash-A", "implementation": "blocked",
                "rows_a": 8, "cols_a": 8, "cols_b": 8, "block_size": 16,
                "requested_threads": 1, "thread_count": 1, "warmups": 1, "repetitions": 9,
                "seed": 42, "elapsed_time_ms": 2.0, "gflops": 0.1, "sweep_index": "1",
                "_source_file": "file.csv", **overrides}

    def test_aggregate_latency_before_selecting_block_winner(self):
        # One unusually fast sample cannot win against a consistently faster block.
        rows = [self.row(block_size=16, elapsed_time_ms=value, sweep_index=str(index))
                for index, value in enumerate((0.01, 10, 10), 1)]
        rows += [self.row(block_size=32, elapsed_time_ms=value, sweep_index=str(index))
                 for index, value in enumerate((2, 3, 4), 1)]
        result = plot_results.aggregate_measurements(pd.DataFrame(rows))
        self.assertEqual(len(result), 2)
        winner = result.loc[result.gflops.idxmax()]
        self.assertEqual(winner.block_size, 32)
        self.assertEqual(winner.elapsed_time_ms, 3)
        self.assertEqual(winner.independent_sweeps, 3)
        self.assertAlmostEqual(winner.gflops, 2 * 8**3 / (3e6))

    def test_even_count_throughput_uses_median_latency_not_median_reciprocal(self):
        result = plot_results.aggregate_measurements(pd.DataFrame([
            self.row(elapsed_time_ms=1), self.row(elapsed_time_ms=9, sweep_index="2")]))
        self.assertEqual(result.iloc[0].elapsed_time_ms, 5)
        self.assertAlmostEqual(result.iloc[0].gflops, 2 * 8**3 / (5e6))

    def test_contexts_preserve_machine_environment_binary_and_input_identity(self):
        rows = [self.row()]
        for field, value in {"binary_sha256": "hash-B", "machine_id": "other", "environment_id": "other",
                             "repetitions": 3, "seed": 43, "warmups": 2}.items():
            rows.append(self.row(**{field: value}))
        result = plot_results.aggregate_measurements(pd.DataFrame(rows))
        self.assertEqual(len(result), len(rows))
        self.assertEqual(result.comparison_context.nunique(), len(rows))
        self.assertEqual(result.context_label.nunique(), len(rows))

    def test_duplicate_collection_and_unknown_binary_inputs(self):
        with self.assertRaisesRegex(ValueError, "duplicate configuration"):
            plot_results.aggregate_measurements(pd.DataFrame([self.row(), self.row()]))
        result = plot_results.aggregate_measurements(pd.DataFrame([
            self.row(binary_sha256="", _source_file="one.csv"),
            self.row(binary_sha256="", _source_file="two.csv")]))
        self.assertEqual(len(result), 2)
        self.assertEqual(result.comparison_context.nunique(), 2)
        self.assertTrue((result.independent_sweeps == 1).all())

    def test_adaptive_policies_connect_sizes_without_combining_independent_counts(self):
        rows = []
        for size, warmups, repetitions, reference in (
            (128, 2, 9, "independent FP64 dot products"),
            (512, 1, 3, "Apple Accelerate FP32"),
        ):
            rows.extend(self.row(rows_a=size, cols_a=size, cols_b=size, warmups=warmups,
                                 repetitions=repetitions, verification_reference=reference,
                                 sweep_index=str(index)) for index in range(3))
        exact = plot_results.aggregate_measurements(pd.DataFrame(rows))
        original = exact.copy(deep=True)
        display = plot_results.display_series(exact, "rows_a")
        self.assertEqual(len(display), 2)
        self.assertEqual(display.comparison_context.nunique(), 2)
        self.assertEqual(display.display_series.nunique(), 1)
        self.assertEqual(display.display_series.iloc[0], "blocked")
        self.assertTrue((display.independent_sweeps == 3).all())
        pd.testing.assert_frame_equal(exact, original)

    def test_conflicting_policies_at_same_x_get_qualified_series_without_selection(self):
        raw = [self.row(rows_a=128, repetitions=9, sweep_index=str(index)) for index in range(2)]
        raw += [self.row(rows_a=128, repetitions=3, sweep_index=str(index)) for index in range(2)]
        raw += [self.row(rows_a=512, repetitions=3)]
        exact = plot_results.aggregate_measurements(pd.DataFrame(raw))
        display = plot_results.display_series(exact, "rows_a")
        self.assertEqual(len(display), 3)
        self.assertEqual(display.display_series.nunique(), 2)
        self.assertEqual(set(display.display_series), {"blocked / r=9", "blocked / r=3"})
        self.assertEqual(sorted(display[display.rows_a == 128].independent_sweeps), [2, 2])
        self.assertEqual(len(display[display.rows_a == 128]), 2)

    def test_display_preserves_binary_machine_environment_and_seed_boundaries(self):
        raw = [self.row()]
        raw.extend(self.row(**{field: value}) for field, value in {
            "binary_sha256": "hash-B", "machine_id": "machine-B",
            "environment_id": "battery", "seed": 43,
        }.items())
        exact = plot_results.aggregate_measurements(pd.DataFrame(raw))
        display = plot_results.display_series(exact, "rows_a")
        self.assertEqual(display.display_context.nunique(), 5)
        self.assertEqual(display.display_series.nunique(), 5)

    def test_display_qualifies_builds_when_no_facet_separates_them(self):
        exact = plot_results.aggregate_measurements(pd.DataFrame([
            self.row(build_label="O0"), self.row(build_label="O2")]))
        display = plot_results.display_series(exact, "rows_a")
        self.assertEqual(set(display.display_series), {"O0 / blocked", "O2 / blocked"})

    def test_four_builds_render_in_separate_panels_with_compact_adaptive_legends(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.figure import Figure
        except ImportError:
            self.skipTest("optional matplotlib analysis dependency not installed")
        raw = []
        for build in ("O3", "O2", "fast", "O0"):
            for implementation in ("naive", "blocked"):
                for size, count in ((128, 9), (512, 3)):
                    raw.append(self.row(build_label=build, implementation=implementation,
                                        rows_a=size, cols_a=size, cols_b=size, repetitions=count,
                                        correct=True, maximum_absolute_error=0.00001,
                                        maximum_relative_error=0.0001))
        figures = {}

        def inspect_figure(fig, path):
            figures[Path(path).name] = [(axis.get_title(), axis.get_visible(),
                                        axis.get_legend_handles_labels()[1]) for axis in fig.axes]

        with tempfile.TemporaryDirectory(prefix="matmul-chart-test-") as directory:
            source = Path(directory) / "synthetic.csv"
            output = Path(directory) / "figures"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
                writer.writeheader()
                writer.writerows(raw)
            with mock.patch.object(Figure, "savefig", new=inspect_figure), redirect_stdout(io.StringIO()):
                self.assertEqual(plot_results.main([str(source), "--output-dir", str(output)]), 0)
            with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 16)
            self.assertEqual({row["build_label"] for row in summary}, {"O3", "O2", "fast", "O0"})
        self.assertEqual(len(figures), 9)
        for filename in ("gflops_vs_size.png", "loop_ordering.png", "versus_accelerate.png"):
            panels = figures[filename]
            self.assertEqual({title for title, visible, _ in panels if visible}, {"O3", "O2", "fast", "O0"})
            self.assertTrue(all("repetitions=" not in label for _, _, labels in panels for label in labels))
        self.assertTrue(figures["gflops_vs_block_size.png"][0][0].startswith("O3:"))


if __name__ == "__main__":
    unittest.main()
