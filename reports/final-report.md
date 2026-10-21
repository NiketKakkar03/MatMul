# MatMul technical report

Recorded on 2026-10-20. The implementation, validation, audited study, independent
repeats, visualizations, and native stack profiling are complete. These results
describe this Apple M1 Pro and do not establish performance across all machines.

## Final completion update

The final study contains 432/432 valid planned observations: 252 O3 workloads and
60 each for O2, O0, and O3 fast-math. The audit recomputed statistics and
throughput from every saved sample and found no invalid, missing, or unexpected
rows. A previously timed-out O0 naive 2048 case was recovered with a longer
process bound. The final evidence adds 462 repeat observations, for 894 total;
154 exact configurations have four independent process-level sweeps. Bootstrap
intervals based on four sweeps are retained as exploratory rather than presented
as publication-grade confidence claims.

Representative O3 square results below use the median of independent sweep
medians where repeats exist and select the best measured block or thread count
for the implementation and size.

| Size | Naive | Reordered | Best blocked | Best parallel | Accelerate |
|---:|---:|---:|---:|---:|---:|
| 128 | 2.665 | 32.023 | 31.926 | 34.077 | 890.794 |
| 512 | 1.802 | 25.639 | 23.733 | 86.892 | 1989.944 |
| 1024 | 1.598* | 25.575 | 21.255 | 99.820 | 1928.733 |
| 2048 | 0.572* | 22.976 | 18.147 | 82.069 | 1413.772 |

Values are GFLOP/s. An asterisk marks sizes with one base observation rather than
four sweeps. At 2048, the best custom result is the seven-thread parallel kernel:
143.5× the naive baseline, while Accelerate remains 17.2× faster than that custom
result. The gap is consistent with missing packing, register blocking, tuned SIMD
microkernels, and persistent worker infrastructure.

Four five-second native `sample` captures cover naive, reordered, blocked, and
parallel kernels at 512. The single-thread captures place nearly all active
samples in their intended multiply function. The parallel capture also exposes
substantial `__ulock_wait` time and visible thread creation, supporting the
benchmark observation that per-call worker creation and synchronization are real
costs. These are stack samples, not hardware-counter measurements; full Xcode was
not installed, so this report does not claim Instruments CPU Counters, cache-miss
rates, or stall-cycle attribution.

The authoritative completion artifacts are
[`coverage.md`](../results/final/audit/coverage.md),
[`repeat_report.md`](../results/final/repeats/repeat_report.md), the tables and
charts under `results/final/figures/`, and the profile index in
[`traces/README.md`](../traces/README.md).

## Historical initial study

The remaining sections preserve the earlier 430-observation study and its
bounded-timeout limitations for experiment provenance. Statements there about
pending measurements are superseded by the final completion update above.

## Problem

MatMul computes dense FP32 matrix products, `C = A × B`, in portable C++20.
The arithmetic count is approximately `2MNK` for an `M×K` input multiplied by a
`K×N` input. The project holds that arithmetic workload constant while changing
memory traversal, blocking, compiler policy, and output-row parallelism.

## Machine

| Item | Recorded configuration |
|---|---|
| Processor | Apple M1 Pro |
| Mac model identifier | MacBookPro18,3 |
| Architecture | arm64 |
| Logical CPUs | 8 |
| Memory | 16 GiB |
| Operating system | macOS 26.7 (25G229) |
| Compiler | Apple Clang 21.0.0.21000334 |
| Build policies | Release, explicit `-O0`, `-O2`, `-O3`, and `-O3 -ffast-math` kernels |

Each raw row records compiler flags, architecture, CPU count, input configuration,
timestamp, and executable SHA256. The final optimization flag selects the
requested level even when CMake's inherited Release flags contain an earlier
`-O3`. Fast-math applies only to arithmetic kernel sources. The input generator,
FP64 oracle, numerical comparison, and timing/statistics code retain strict
floating-point behavior. Apple's precompiled Accelerate library is unchanged by
these compiler switches.

## Implementations

`Matrix` owns one contiguous row-major allocation, validates storage sizes, and
provides deterministic random inputs and debug bounds checking. The independent
oracle accumulates dot products in double precision and converts the answer to
FP32. It is separate from the baseline FP32 multiplication implementation.

The six traversal experiments are `naive` (`ijk`), `reordered` (`ikj`), `jik`,
`jki`, `kij`, and `kji`. `blocked` uses clamped tiles and an inner `ikj` pointer
kernel. `parallel` applies that tiled kernel to disjoint output-row ranges; each
call creates and joins its workers. `blocked_parallel` is an accepted alias for
`parallel`, not a separate algorithm. The one-thread parallel path creates no
workers. All implementations overwrite their output and reject invalid shapes
or unsupported input/output aliasing.

The Apple baseline calls `cblas_sgemm` with row-major, non-transposed inputs,
leading dimensions `K` and `N`, `alpha=1`, and `beta=0`. Its actual worker count
is library-managed; CSV `thread_count=0` denotes unknown library-managed
parallelism, not zero work or a controlled single-thread baseline.

## Methodology

The completed quick study contains 192 measured configurations: 48 each in
[O0](../results/raw/O0.csv), [O2](../results/raw/O2.csv),
[O3](../results/raw/O3.csv), and [fast](../results/raw/fast.csv). Each build covers
square sizes 32, 64, and 128; block edges 16, 32, and 64; worker counts 1, 2,
and 4; and a `31×47` by `47×23` rectangular workload. Configuration order is
randomized with seed 42. Inputs use seeds 42 and 43.

Every configuration performs two warmups and nine measured multiplications using
`std::chrono::steady_clock`. The CSV retains every duration, the median, minimum,
and population standard deviation. Input/output allocation, random generation,
checksums, and verification are outside the measured interval. Multiplication
timing includes output preparation and, for parallel execution, worker creation
and joining. Initialization and verification are separately measured once;
their stage-chart bars are not nine-run medians.

Verification uses `|actual−expected| <= 1e-4 + 1e-4×max(|actual|,|expected|)`
per element and rejects nonfinite values. Every quick result uses the independent
FP64 oracle. For larger non-Accelerate workloads above 16,777,216 multiply-add
positions, the harness uses Accelerate FP32 as the verification reference when
available; Accelerate itself remains checked against the independent oracle.
Each row identifies the reference actually used.

The larger O3 sweep retained 234 verified measurements from 252 planned
configurations in [O3_full.csv](../results/raw/O3_full.csv), across
sizes 32, 64, 128, 256, 512, 768, 1024, 1536, and 2048; block edges 8 through
256; all worker counts 1 through 8; and the roadmap's three rectangular shapes.
A 15-second timeout initially bounded each subprocess, including setup and
verification: 233 cases passed and 19 timed out. Repeating just Accelerate at
2048 with a 120-second allowance recovered one result; the other three
Accelerate rectangular cases resumed from existing results. Eighteen planned
configurations remain unmeasured. The adjacent JSONL log retains all 19 original
timeout events, including the recovered case. All 27 rectangular configurations
passed. Full O0, O2, and fast-math sweeps are not claimed here.

A further four verified observations in
[O3_no_vectorize.csv](../results/raw/O3_no_vectorize.csv) isolate auto-vectorization
for the reordered kernel at 128, 512, 1024, and the small rectangular shape.
The total retained dataset has 430 observations, including repeated
configurations across studies rather than 430 unique inputs.

System load was not controlled, and power state and thermal state were not
recorded. These results support an initial study rather than publication-grade
claims about stable hardware limits.

## Results

The following table reports GFLOP/s at `128×128` from the completed quick study.
For blocked and parallel kernels, each cell selects the fastest observed
configuration within that build, rather than imposing one universally optimal
configuration.

| Implementation | O0 | O2 | O3 | O3 fast-math |
|---|---:|---:|---:|---:|
| Naive / ijk | 0.370 | 2.729 | 2.629 | 2.734 |
| Reordered / ikj | 1.705 | 32.368 | 30.728 | 31.825 |
| jik | 0.602 | 2.694 | 2.678 | 2.755 |
| jki | 1.235 | 3.184 | 3.094 | 3.065 |
| kij | 1.710 | 31.428 | 30.040 | 32.008 |
| kji | 1.238 | 4.380 | 4.252 | 4.240 |
| Blocked, best measured block | 1.719 | 20.902 | 21.404 | 21.528 |
| Parallel, best measured workers | 5.295 | 34.486 | 33.059 | 36.779 |
| Apple Accelerate | 883.011 | 923.449 | 915.187 | 923.449 |

The selected block edge is 32 at O0 and 64 for the other builds. The selected
parallel worker count is four at O0 and fast-math, and two at O2 and O3. These
choices only describe the tested sizes and candidates. Accelerate's differing
numbers across builds reflect run variation and wrapper/build effects; they
cannot be attributed to recompiling its precompiled kernel with these flags.

For O3 at 128, reordering improves throughput by 11.69× relative to the O3 naive
kernel. The best measured O3 parallel result improves by 12.58× relative to that
same baseline. It reaches 3.61% of the measured Accelerate throughput, leaving
an approximately 27.68× gap. These are within-study observations for this size,
not general speedup guarantees. Comparing O0 and O3 reordered kernels at this
size gives an 18.02× improvement, illustrating that compiler policy itself is
a major experimental variable.

Correct speedup normalization compares a threaded kernel against its own
one-thread configuration:

| O3 parallel at 128, block 64 | Median ms | Speedup | Efficiency |
|---|---:|---:|---:|
| 1 worker | 0.189458 | 1.000× | 100.0% |
| 2 workers | 0.126875 | 1.493× | 74.7% |
| 4 workers | 0.128625 | 1.473× | 36.8% |

All 192 quick configurations pass verification. All four modes have the same
largest observed absolute error, `5.7220458984375e-6`, and relative error,
`0.0074474439415699`. The per-configuration maximum error metrics also match
between O3 and fast-math in this study. A relatively large relative error near
zero can coexist with a small absolute error and pass the combined tolerance.
These observations do not establish that fast-math preserves all floating-point
semantics or all possible inputs.

Nine generated charts and measured summary tables are in
[results/figures](../results/figures/): the seven required views plus dedicated
loop-order and execution-time-versus-worker plots. The tables include
[best blocks](../results/figures/best_blocks.csv),
[thread scaling](../results/figures/thread_scaling.csv), and
[numerical errors](../results/figures/numerical_error.csv). Chart contents reflect
the input CSVs supplied to the plotting command; the completed larger study has
its own chart directory below.

The larger O3 study provides the following GFLOP/s comparison. Parentheses give
the best measured block edge or worker count; these values are selected only
from configurations that completed and passed verification.

| Size | Reordered | Blocked (edge) | Parallel (workers) | Accelerate |
|---|---:|---:|---:|---:|
| 512 | 26.040 | 23.396 (192) | 76.535 (8) | 1936.414 |
| 768 | 25.109 | 31.055 (96) | 115.007 (6) | 2458.534 |
| 1024 | 25.860 | 21.243 (256) | 101.413 (6) | 2132.909 |
| 1536 | 23.785 | 19.817 (96) | 110.223 (8) | 2180.898 |
| 2048 | 23.343 | 17.908 (256) | 86.946 (8) | 1749.648 |

At 1024, six workers outperform eight (101.413 versus 81.032 GFLOP/s). At 768,
blocking improves on reordered traversal, while it loses at several other sizes.
At 2048 the best custom result reaches about 4.97% of Accelerate throughput.
The largest observed absolute error in this larger study is
`0.00022125244140625`; every retained observation passes the combined tolerance.

The recovered Accelerate 2048 result illustrates the timeout distinction:
multiplication's median is 9.819 ms, but independent FP64 verification takes
23,668.6 ms. Its initial timeout cannot be attributed to slow BLAS execution.
Larger-study charts and tables are kept separately in
[results/full_figures](../results/full_figures/), preserving the original quick
compiler comparison charts.

## Analysis

In row-major storage, `ikj` and `kij` stream across rows of B and C while reusing
an A scalar. Their approximately 30 GFLOP/s O3 results at 128 substantially exceed
the strided alternatives. All orders remain cubic; arithmetic complexity alone
does not account for cache-line use, address generation, and vectorizable access.
For the small rectangular case, `kij` reaches 14.756 GFLOP/s at O3 versus
6.933 for `ikj`, reinforcing that the winning traversal can depend on shape.

Compiler remarks in `build/experiments/O3/build.log` report vectorized loops
with width four and interleave count four in `src/reordered.cpp` and
`src/kernel_internal.hpp:31`. The naive source also receives a vectorization
remark, so SIMD being present is not sufficient evidence of good locality.
Random generation in `src/matrix.cpp:38` misses vectorization because of its
dependencies, and numerical comparison in `src/verification.cpp:14` has a
reduction/control-flow constraint. These reports establish compiler decisions,
not hardware utilization or cache-miss rates.

An isolated Clang ablation keeps O3 and strict floating point while disabling
loop and SLP vectorization using `MATMUL_DISABLE_VECTORIZATION=ON`:

| Reordered square size | Normal O3 GFLOP/s | Vectorization disabled | Ratio |
|---|---:|---:|---:|
| 128 | 30.606 | 5.622 | 5.44× |
| 512 | 26.040 | 5.913 | 4.40× |
| 1024 | 25.860 | 6.050 | 4.27× |

Assembly inspection finds `fmla.4s` in the normal reordered object and no such
vector FMA instructions in the disabled object; scalar `fmadd` remains. These
ratios support a substantial SIMD contribution, with the same uncontrolled-load
and separate-run limitations as the rest of this study.

Blocking trades loop overhead for reuse. A rough working-set model for three
square FP32 tiles is `12×block_size²` bytes: 48 KiB for an edge of 64 and
768 KiB for an edge of 256. Actual cache behavior also depends on associativity,
other resident data, and the matrix shape; no cache capacity or miss rate was
measured here. At 128, O3 block 64 gives 21.404 GFLOP/s, versus 9.051 for block
16, but both trail unblocked `ikj` at 30.728. A tiny problem need not benefit
from additional tiling.

An ideal compulsory-traffic model reads A and B once and writes C once, giving
arithmetic intensity `2MNK / (4×(MK+KN+MN))` FLOP/byte, or `N/6` for squares.
Real traffic can exceed this lower bound when reuse is lost. The roofline view
would compare this intensity against independently measured bandwidth and peak
compute throughput. Neither ceiling was measured, so the project does not claim
a numerical roofline or prove that a kernel is bandwidth-bound. The CSV's
`estimated_bandwidth` is that compulsory byte count divided by runtime, not a
hardware DRAM counter.

Thread creation and scheduling must be amortized. At size 32, four O3 workers
take 0.039625 ms compared with 0.004916 ms for one, about 8.06× slower. At size
128, two workers help, while four add little and reduce efficiency. Possible
causes include startup costs, shared resources, and scheduling across different
cores. Their individual contributions require profiling. Disjoint row ownership
avoids output-element races; it does not prove that adjacent row boundaries
never share a cache line.

Accelerate remains much faster in these measurements. Packing, tuned register
blocking, instruction selection, and processor-specific scheduling are plausible
sources of a production library's advantage, but the internal selected kernel
was not inspected. No specific instruction set or private implementation
mechanism is asserted from throughput alone.

## Failed experiments

Additional workers lose at 32 and 64 in the O3 quick sweep. A block edge of 16
also loses to 64 and to the unblocked reordered kernel at 128. O3 is slightly
slower than O2 for several measured kernels; fast-math produces modest changes
rather than a universal improvement. One uncontrolled sweep cannot distinguish
all small compiler-policy differences from measurement noise. These outcomes
are retained in the raw data instead of being discarded to present only wins.

The first vectorization control supplied disabling flags in `CMAKE_CXX_FLAGS`.
A later target `-O3` re-enabled vectorization, producing an apparent null result
(about 25.8 GFLOP/s at 1024). Assembly exposed the invalid control. The dedicated
CMake option places disabling flags last; the corrected four measurements
replace those initial control values. See [vectorization.md](vectorization.md).

## Limitations

The bounded larger-size study has 18 remaining timeout gaps. The complete
four-policy, large-size experiment matrix has not
been completed. One machine, one input seed, uncontrolled background load, and
unrecorded thermal/power state limit generalization. Tiny Accelerate timings
are particularly sensitive to timer and system noise. No confidence intervals
across independent repeated sweeps are reported.

Seventeen unit cases and four CTest entries passed in Debug, ASan/UBSan, separate
TSan, and each of the four optimized Release variants. Those checks cover the
project's instrumented code, not the internals of the precompiled Accelerate
library. Large inputs may use the FP32 BLAS verification reference, as recorded
per row. The engine supports FP32 only and has no persistent worker pool,
handwritten SIMD microkernel, or packed-matrix implementation.

Fourteen additional Python tests pass for experiment planning, thread caps,
CSV schema preservation, executable-aware resume, and subprocess failure/timeout
handling. Those tests use mocked child processes and do not add timing samples.

Full Xcode/Instruments is unavailable in the selected command-line-tools
environment: `xcrun --find xctrace` failed. No trace has been collected. The
[experiment log](experiment-log.md) records predictions and this failed attempt;
`scripts/profile.py` provides a reproducible recording helper. Profiling at least
three implementations, observing their hotspots, and replacing predictions with
trace-backed evidence remain necessary to satisfy the roadmap's empirical
definition of done. Compiler diagnostics alone do not complete that requirement.

## Conclusion

The completed studies show large effects from compiler optimization,
auto-vectorization, and row-major traversal, and document cases where tiling and
more workers lose.
The O3 reordered kernel is 11.69× faster than its same-build naive baseline at
128; the best measured O3 custom kernel still trails Accelerate by 27.68× at
that size. The larger O3 study reaches 2048 and retains all timeout gaps; its best
custom implementation still trails Accelerate substantially. Broader claims
require controlled repetitions and the pending Instruments exercise.

To reproduce the quick study with all four compiler policies:

```bash
uv run --extra dev --extra analysis python scripts/run_experiments.py --preset quick --modes O0 O2 O3 fast
```

Use `--preset full` for the complete automated sweep; its default timeout is
120 seconds per configuration. The current bounded O3 run uses 15 seconds
instead. Raw logs and skip records should accompany any claim based on either
policy.
