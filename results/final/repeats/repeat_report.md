# Independent repeat analysis

Analyzed 894 verified observations across 9 input CSVs and 432 exact configurations. 154 configurations have at least three independent sweeps.

Each observation is one sweep's benchmark median. Configuration groups retain binary hash, build, shape, implementation, block size, requested/actual threads, input seed, warmups, repetitions, verification settings, and available compiler/machine metadata. Different configurations and iteration durations are never pooled.

`repeat_summary.csv` reports the median of sweep medians, range, population standard deviation, and coefficient of variation (population standard deviation divided by the arithmetic mean). Deterministic 95% percentile bootstrap intervals resample the sweep medians 10,000 times and are omitted below three sweeps. Intervals with fewer than ten sweeps are explicitly exploratory; three repeats provide only a coarse view of variability and cannot establish a reliable confidence level.

| Independent sweeps | Exact configurations |
| --- | ---: |
| 1 | 278 |
| 4 | 154 |

Limitations:

- Collection IDs declare independence; CSV metadata cannot prove separate physical executions.
- Legacy CSVs use one collection per resolved input filename; copied files are not new experiments.
- Missing binary hashes prevent establishing matched-binary evidence across input files.
- Intervals from fewer than ten independent sweeps are exploratory and may understate uncertainty.
- Bootstrap intervals do not account for systematic thermal, load, power, or measurement bias.
- Coverage describes supplied observations, not completion against a planned experiment matrix.

The complete measurements and identities are in `repeat_summary.csv`; source and coverage details are in `coverage.json`.
