#!/usr/bin/env python3
"""Create the roadmap's seven figures and measured summary tables.

  python scripts/plot_results.py results/raw/O3.csv --output-dir results/figures
  python scripts/plot_results.py results/raw/*.csv --output-dir results/figures

Install analysis dependencies with `uv sync --extra analysis` first. Results from
different builds retain separate labels. The implementation comparison selects
the best measured block/thread configuration for each size; it does not imply a
single configuration is optimal everywhere. Estimated bandwidth is not plotted
as measured hardware traffic. Empty chart panels explicitly report missing data.
Performance comparisons use one panel per build. Block/thread/stage charts focus
on O3 when supplied; CSV tables retain every build and exact sampling policy.
Display lines can connect sizes with different iteration/reference policies, but
conflicting policies at the same x coordinate always retain separate series.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from analyze_repeats import CONFIG_FIELDS


CONTEXT_FIELDS = tuple(field for field in CONFIG_FIELDS if field not in {
    "implementation", "rows_a", "cols_a", "cols_b", "block_size", "requested_threads", "thread_count",
}) + ("identity_scope",)
DISPLAY_POLICY_FIELDS = ("warmups", "repetitions", "verification_reference")
DISPLAY_CONTEXT_FIELDS = tuple(field for field in CONTEXT_FIELDS if field not in DISPLAY_POLICY_FIELDS)


def aggregate_measurements(data):
    """Aggregate exact matched configurations before selecting observed winners.

    Each source row represents one independent sweep median; duplicate collection
    IDs are errors. Unknown binaries are confined to their source files. Error
    columns retain maxima, while stage times use medians across collections.
    """
    import pandas as pd

    data = data.copy()
    if "_source_file" not in data:
        data["_source_file"] = "unspecified-input"
    for field in (*CONFIG_FIELDS, "sweep_index"):
        if field not in data:
            data[field] = ""
    # Canonical textual metadata prevents pandas numeric inference from merging
    # missing values with literal zero or producing inconsistent group labels.
    for field in CONFIG_FIELDS:
        if field not in {"rows_a", "cols_a", "cols_b", "block_size", "requested_threads", "thread_count",
                         "warmups", "repetitions", "seed", "logical_cpu_count"}:
            data[field] = data[field].fillna("").astype(str)
    data["identity_scope"] = data["binary_sha256"].map(
        lambda value: "matched_binary_hash" if value else "unknown_binary")
    unknown = data["binary_sha256"] == ""
    data.loc[unknown, "identity_scope"] += ":" + data.loc[unknown, "_source_file"].astype(str)
    data["_collection"] = data["sweep_index"].fillna("").astype(str).map(
        lambda value: f"sweep:{value}" if value else "")
    legacy = (data["_collection"] == "") | unknown
    data.loc[legacy, "_collection"] = "legacy-file:" + data.loc[legacy, "_source_file"].astype(str)
    keys = [*CONFIG_FIELDS, "identity_scope"]
    if data.duplicated([*keys, "_collection"]).any():
        raise ValueError("duplicate configuration/collection rows; overlapping inputs are not independent sweeps")
    observations = []
    for _, group in data.groupby(keys, dropna=False, sort=False):
        row = group.iloc[0].to_dict()
        row["elapsed_time_ms"] = group.elapsed_time_ms.median()
        # Throughput must derive from the aggregate latency, not the fastest row
        # or the median of reciprocals (different for even observation counts).
        row["gflops"] = 2 * row["rows_a"] * row["cols_a"] * row["cols_b"] / (row["elapsed_time_ms"] * 1e6)
        row["independent_sweeps"] = len(group)
        row["source_files"] = json.dumps(sorted(set(group._source_file.astype(str))))
        row["sweep_ids"] = json.dumps(sorted(set(group._collection.astype(str))))
        # Per-process samples and timestamps do not describe the aggregate.
        for field in ("_source_file", "_collection", "sweep_index", "timestamp", "durations_ms",
                      "minimum_time_ms", "standard_deviation_ms", "estimated_bandwidth"):
            row.pop(field, None)
        for field in ("initialization_time_ms", "verification_time_ms"):
            if field in group:
                row[field] = group[field].median()
        for field in ("maximum_absolute_error", "maximum_relative_error"):
            if field in group:
                row[field] = group[field].max()
        observations.append(row)
    result = pd.DataFrame(observations)
    # Build names already identify their own compiler policies. Add qualifiers
    # only for metadata that would otherwise make one build's series ambiguous.
    varying = [field for field in CONTEXT_FIELDS if field != "build_label"
               and result.groupby("build_label", dropna=False)[field].nunique(dropna=False).max() > 1]

    def context(row):
        return json.dumps([str(row[field]) for field in CONTEXT_FIELDS], separators=(",", ":"))

    def label(row):
        parts = [row["build_label"] or "local"]
        for field in varying:
            value = str(row[field])
            if field in {"binary_sha256", "machine_id", "environment_id"}:
                value = value[:8]
            elif field == "compiler_flags":
                value = "sha256:" + hashlib.sha256(value.encode()).hexdigest()[:8]
            parts.append(f"{field}={value or 'unknown'}")
        return " / ".join(parts)

    result["comparison_context"] = result.apply(context, axis=1)
    result["context_label"] = result.apply(label, axis=1)
    result["series"] = result.implementation.astype(str) + " / " + result.context_label
    return result


def display_series(frame, x, series_fields=("implementation",)):
    """Label plotting lines without changing exact observations or selecting data.

    Sampling/reference policies may change with x (the declared adaptive study).
    A policy conflict at any shared x splits that entire line into qualified
    policy series; no rows are pooled or chosen between conflicting policies.
    Binary, environment, machine, input and all other contexts remain distinct.
    """
    result = frame.copy()
    if result.empty:
        for field in ("display_context", "display_context_label", "display_series"):
            result[field] = ""
        return result
    varying = [field for field in DISPLAY_CONTEXT_FIELDS if field != "build_label"
               and result.groupby("build_label", dropna=False)[field].nunique(dropna=False).max() > 1]

    def context_label(row):
        parts = [str(row.build_label) or "local"]
        aliases = {"binary_sha256": "binary", "machine_id": "machine", "environment_id": "env"}
        for field in varying:
            value = str(row[field])
            if field in aliases:
                value = value[:8]
            elif field in {"compiler_flags", "identity_scope"}:
                value = "#" + hashlib.sha256(value.encode()).hexdigest()[:6]
            parts.append(f"{aliases.get(field, field)}={value or '?'}")
        return " / ".join(parts)

    result["display_context"] = result.apply(
        lambda row: json.dumps([str(row[field]) for field in DISPLAY_CONTEXT_FIELDS]), axis=1)
    result["display_context_label"] = result.apply(context_label, axis=1)
    result["_policy"] = result.apply(
        lambda row: json.dumps([str(row[field]) for field in DISPLAY_POLICY_FIELDS]), axis=1)
    result["display_series"] = ""
    multiple_builds = result.build_label.nunique(dropna=False) > 1
    for _, group in result.groupby(["display_context", *series_fields], dropna=False, sort=False):
        conflicting = group.groupby(x, dropna=False)._policy.nunique().max() > 1
        changing_policy = [field for field in DISPLAY_POLICY_FIELDS
                           if conflicting and group[field].nunique(dropna=False) > 1]
        for index, row in group.iterrows():
            parts = [str(row.build_label)] if multiple_builds else []
            for field in series_fields:
                value = row[field]
                if field == "implementation":
                    parts.append(str(value))
                elif field in {"requested_threads", "thread_count"} and result[field].nunique(dropna=False) == 1:
                    continue
                else:
                    alias = {"rows_a": "n", "block_size": "block", "requested_threads": "requested",
                             "thread_count": "workers"}.get(field, field)
                    parts.append(f"{alias}={value}")
            if varying:
                # The build already names the facet; append only its qualifiers.
                parts.extend(row.display_context_label.split(" / ")[1:])
            for field in changing_policy:
                value = str(row[field])
                if field == "verification_reference":
                    value = ("FP64 ref" if value == "independent FP64 dot products" else
                             "Accelerate ref" if value == "Apple Accelerate FP32" else
                             "ref#" + hashlib.sha256(value.encode()).hexdigest()[:6])
                    parts.append(value)
                else:
                    parts.append(f"{'w' if field == 'warmups' else 'r'}={value}")
            result.at[index, "display_series"] = " / ".join(parts)
    return result.drop(columns="_policy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="one or more benchmark CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as error:
        parser.error(f"analysis dependency missing ({error.name}); run: uv sync --extra analysis")
    try:
        frames = []
        for path in args.inputs:
            frame = pd.read_csv(path, dtype={"sweep_index": str, "binary_sha256": str,
                                             "machine_id": str, "environment_id": str})
            frame["_source_file"] = str(path.resolve())
            if "build_label" not in frame:
                frame["build_label"] = frame.get("compiler_flags", path.stem)
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    required = {"implementation", "rows_a", "cols_a", "cols_b", "elapsed_time_ms", "gflops"}
    if not required <= set(data.columns):
        parser.error("missing CSV columns: " + ", ".join(sorted(required - set(data.columns))))
    if data.empty:
        parser.error("no measured benchmark results")
    for column in ("rows_a", "cols_a", "cols_b", "elapsed_time_ms", "gflops", "block_size", "thread_count",
                   "initialization_time_ms", "verification_time_ms", "maximum_absolute_error", "maximum_relative_error"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "correct" in data:
        data = data[data["correct"].astype(str).str.lower().isin(("true", "1", "yes"))]
    data = data.dropna(subset=["gflops", "elapsed_time_ms"])
    data = data[data.elapsed_time_ms.map(math.isfinite) & data.gflops.map(math.isfinite)]
    data = data[(data["elapsed_time_ms"] > 0) & (data["gflops"] >= 0)].copy()
    if data.empty:
        parser.error("no valid, verified benchmark measurements")
    data["build_label"] = data["build_label"].fillna("local").astype(str)
    observation_count = len(data)
    try:
        data = aggregate_measurements(data)
    except ValueError as error:
        parser.error(str(error))
    square = data[(data.rows_a == data.cols_a) & (data.cols_a == data.cols_b)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})

    def finish(fig, ax, filename, title, xlabel, ylabel):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if len(handles) > 10:
                fig.set_size_inches(12, max(5, len(handles) * 0.18))
                ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
            else:
                ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "No matching measurements", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(args.output_dir / filename)
        plt.close(fig)

    def lines(frame, x, y, group_columns):
        fig, ax = plt.subplots(figsize=(9, 5))
        if not frame.empty:
            for names, group in frame.groupby(group_columns, dropna=False):
                names = names if isinstance(names, tuple) else (names,)
                group = group.sort_values(x)
                ax.plot(group[x], group[y], marker="o", markersize=3, label=" / ".join(map(str, names)))
        return fig, ax

    def build_panels(frame, filename, title):
        """Keep each compiler policy readable with compact implementation legends."""
        displayed = frame
        builds = sorted(displayed.build_label.unique()) if not displayed.empty else ["No measurements"]
        columns = min(2, len(builds))
        rows = math.ceil(len(builds) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(6.4 * columns, 4.5 * rows), squeeze=False)
        colors = {implementation: plt.get_cmap("tab10")(index % 10)
                  for index, implementation in enumerate(sorted(data.implementation.unique()))}
        for ax, build in zip(axes.flat, builds):
            subset = displayed[displayed.build_label == build] if not displayed.empty else displayed
            subset = display_series(subset, "rows_a")
            for label, group in subset.groupby("display_series", dropna=False):
                group = group.sort_values("rows_a")
                ax.plot(group.rows_a, group.gflops, marker="o", markersize=3, label=label,
                        color=colors[group.implementation.iloc[0]])
            ax.set(title=str(build), xlabel="Square matrix size", ylabel="GFLOP/s (log scale)")
            if not subset.empty:
                # Logarithmic size spacing accommodates the full 32..2048 study
                # without compressing the smaller sizes into the left margin.
                sizes = sorted(subset.rows_a.unique())
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_xticks(sizes, [str(int(size)) for size in sizes], rotation=30)
                ax.legend(fontsize=7, ncol=2 if subset.display_series.nunique() > 5 else 1)
            else:
                ax.text(0.5, 0.5, "No matching measurements", ha="center", transform=ax.transAxes)
        for ax in list(axes.flat)[len(builds):]:
            ax.set_visible(False)
        fig.suptitle(title, fontsize=12)
        fig.text(0.5, 0.01, "Sampling/reference policies may vary across sizes; conflicting policies at the same size remain separate.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(args.output_dir / filename)
        plt.close(fig)

    # Preserve independent build labels; select only observed configurations.
    best = square.loc[square.groupby(["comparison_context", "implementation", "rows_a"])["gflops"].idxmax()] if not square.empty else square
    best.drop(columns=["series"], errors="ignore").to_csv(args.output_dir / "summary.csv", index=False)
    build_panels(best, "gflops_vs_size.png", "Best configuration after aggregating independent sweep medians")
    loop_orders = best[best.implementation.isin(("naive", "reordered", "ijk", "ikj", "jik", "jki", "kij", "kji"))]
    build_panels(loop_orders, "loop_ordering.png", "Loop orders (single-thread, unblocked)")

    blocked = square[square.implementation == "blocked"].copy()
    if "block_size" in blocked:
        block_curves = blocked
        best_blocks = blocked.loc[blocked.groupby(["comparison_context", "rows_a"])["gflops"].idxmax()]
        best_blocks.drop(columns=["series"], errors="ignore").to_csv(args.output_dir / "best_blocks.csv", index=False)
    else:
        block_curves = pd.DataFrame()
    engineering_build = "O3" if "O3" in square.build_label.values else None

    def engineering(frame):
        return frame[frame.build_label == engineering_build].copy() if engineering_build and not frame.empty else frame.copy()

    engineering_title = f"{engineering_build}: " if engineering_build else ""
    block_display = display_series(engineering(block_curves), "block_size",
                                   ("rows_a", "requested_threads", "thread_count"))
    fig, ax = lines(block_display, "block_size", "gflops", ["display_series"])
    finish(fig, ax, "gflops_vs_block_size.png", engineering_title + "Cache block sweep (all builds retained in CSV)", "Block size", "GFLOP/s")

    scaling_rows = []
    threaded = square[square.implementation.isin(("parallel", "blocked_parallel"))].copy()
    if "thread_count" in threaded and "block_size" in threaded:
        # Do not mix block sizes when computing speedup against one thread.
        for names, group in threaded.groupby(["comparison_context", "implementation", "rows_a", "block_size"]):
            # If requests clamp to the same actual worker count, select the best
            # aggregate exact configuration, never pool their observations.
            selected = group.loc[group.groupby("thread_count").gflops.idxmax()]
            times = selected.set_index("thread_count").elapsed_time_ms
            if 1 not in times or times[1] <= 0:
                continue
            for count, elapsed in times.items():
                source = selected[selected.thread_count == count].iloc[0]
                scaling_rows.append(source.to_dict() |
                                    {"context_label": group.iloc[0].context_label,
                                     "thread_count": count, "elapsed_time_ms": elapsed,
                                     "speedup": times[1] / elapsed, "efficiency": times[1] / elapsed / count})
    scaling = pd.DataFrame(scaling_rows)
    if not scaling.empty:
        scaling.to_csv(args.output_dir / "thread_scaling.csv", index=False)
    for filename, y, title, ylabel in (
        ("speedup_vs_threads.png", "speedup", "Speedup relative to the same kernel at one thread", "Speedup"),
        ("efficiency_vs_threads.png", "efficiency", "Parallel efficiency (speedup / thread count)", "Parallel efficiency"),
        ("time_vs_threads.png", "elapsed_time_ms", "Multiplication time versus thread count", "Median time (ms)"),
    ):
        thread_display = display_series(engineering(scaling), "thread_count", ("implementation", "rows_a", "block_size"))
        fig, ax = lines(thread_display, "thread_count", y, ["display_series"])
        finish(fig, ax, filename, engineering_title + title, "Threads", ylabel)

    fig, ax = plt.subplots(figsize=(10, 6))
    breakdown_columns = ["initialization_time_ms", "elapsed_time_ms", "verification_time_ms"]
    if not best.empty and set(breakdown_columns) <= set(best.columns):
        stage_data = engineering(best)
        largest = display_series(stage_data[stage_data.rows_a == stage_data.rows_a.max()], "rows_a").set_index("display_series")
        largest[breakdown_columns].rename(columns={"initialization_time_ms": "Initialization (sweep median)",
            "elapsed_time_ms": "Multiplication (median of sweep medians)", "verification_time_ms": "Verification (sweep median)"}).plot.bar(
                stacked=True, ax=ax, rot=35)
    finish(fig, ax, "execution_time_breakdown.png", engineering_title + "Stage costs at the largest measured square size", "Implementation", "Time (ms)")

    comparison = best[best.implementation.isin(("naive", "reordered", "blocked", "parallel", "blocked_parallel", "accelerate"))]
    build_panels(comparison, "versus_accelerate.png", "Custom kernels and Apple Accelerate (where available)")

    fig, ax = plt.subplots(figsize=(10, 5))
    error_columns = [name for name in ("maximum_absolute_error", "maximum_relative_error") if name in data]
    if error_columns:
        # Keep every exact policy in the table; the chart shows maxima across
        # sizes/sampling policies within each protected build/machine identity.
        numerical_metadata = data[["comparison_context", *CONTEXT_FIELDS]].drop_duplicates(
            "comparison_context").set_index("comparison_context")
        numerical_table = numerical_metadata.join(data.groupby("comparison_context")[error_columns].max())
        numerical_table.to_csv(args.output_dir / "numerical_error.csv")
        numerical_display = display_series(data, "rows_a")
        numerical = numerical_display.groupby("display_context_label")[error_columns].max()
        numerical.plot.bar(ax=ax, rot=15)
        ax.set_yscale("symlog", linthresh=1e-8)
    finish(fig, ax, "numerical_error_vs_mode.png", "Maximum observed error by compiler build (mixed workloads)", "Build / floating-point policy", "Maximum error (symlog scale)")
    print(f"Aggregated {observation_count} observations into {len(data)} exact configurations; "
          f"wrote nine charts and summary tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
