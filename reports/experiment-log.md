# MatMul experiment log

Date: 2026-10-20. Machine: Apple M1 Pro, 8 logical CPUs, 16 GiB RAM,
arm64, macOS 26.7 (25G229), Apple Clang 21.0.0.21000334.

## E1 — Establish correctness before performance

**Prediction:** Contiguous ownership and explicit shape validation should make
small hand-computed, rectangular, zero, identity, and partial-tile products agree
with a double-accumulation reference.

**Change:** Implemented the matrix container, independent oracle, frozen FP32
naive kernel, all traversal orders, blocking, disjoint-row threading, and BLAS.

**Observation:** 17 unit cases and four CTest entries pass in Debug, ASan/UBSan,
and separate TSan builds. No sanitizer findings. A review caught an unnecessary
row loop for huge zero-storage shapes in the reference; early return and
regression tests now cover this, along with zero-output bandwidth accounting.

**Explanation:** Empty extents have a defined shape without allocating elements.
Storage safety alone does not guarantee prompt completion; loops must handle
zero output dimensions too. Independent double arithmetic avoids trusting the
optimized implementation to validate itself.

**Remaining question:** External Accelerate binaries are not sanitizer-instrumented
by this project. Larger measured workloads must also pass numerical verification.

## E2 — Loop traversal and compiler diagnostics

**Prediction:** Unit-stride `ikj`/`kij` inner loops should use SIMD effectively;
strided reads/writes in other orders should become expensive with larger shapes.

**Change:** Added five alternative loop orders and a raw-pointer tile kernel.
Kept element accessors inline so the scalar baseline is not penalized by an
unnecessary cross-translation-unit accessor call. The naive loop body is unchanged.

**Observation:** Saved Clang build logs show vectorized hot loops in
`reordered.cpp` and `kernel_internal.hpp`, with width 4 and interleave count 4.
The naive source can also receive a vectorization remark; a vectorized loop is
not sufficient evidence of good locality. Matrix random generation and numerical
comparison have missed-vectorization remarks due to dependencies/control flow.
Raw measurements and dedicated loop-order charts are in `results/`.

**Explanation:** Every traversal has the same cubic arithmetic count, but
different unit-stride streams, reused scalar values, and repeated cache-line
loads. Compiler reports describe transformations; they do not measure cache
misses or CPU stalls.

**Remaining question:** Instruments is needed to establish actual cache/stall
behavior. Strided access being costly is a hypothesis supported by timing and
layout, not a measured hardware-counter result.

## E3 — Blocking and parallel overhead

**Prediction:** Small tiles can add overhead when the working set already fits
in cache. Worker creation can dominate tiny products. Larger matrices may benefit
from reuse and multiple cores, with a non-universal best tile/worker count.

**Change:** Sweeps vary tile edges and worker counts independently, using the
same seed and overwrite semantics. Parallel timing includes thread creation and
joining. One thread calls the same tiled kernel without starting workers.

**Observation:** Quick measurements show that increasing workers on a 32×32
matrix can make it substantially slower, and small tile edges can lose to the
unblocked reordered kernel. See `thread_scaling.csv`, `best_blocks.csv`, and
`final-report.md` for the retained measurements.

**Explanation:** More parallelism and more cache tiling are not automatically
improvements. Setup/scheduling and loop overhead must be amortized. Row ownership
avoids shared output elements; adjacent boundaries may still share cache lines,
so claiming all false sharing is eliminated would be too strong.

**Remaining question:** A persistent worker pool and packed/register-blocked
kernels are optional follow-ups after actual profiling identifies the bottleneck.

## E4 — Compiler policy and numerical error

**Prediction:** Optimized builds should beat O0 substantially. Fast-math may
change dot-product rounding and is not guaranteed to improve already-vectorized
unit-stride loops.

**Change:** Built O0, O2, O3, and O3 with source-scoped `-ffast-math`; kept input
generation, validation, comparison, and the FP64 oracle strict. Every configuration
runs two warmups and nine measured repetitions by default.

**Observation:** All four quick sweeps pass correctness and produce separate
CSV files and numerical-error curves. Fast-math is an explicit separate policy;
its performance must be read from measurements rather than presumed.

**Explanation:** A policy permitting reassociation need not produce a different
instruction sequence for every loop. Accelerate is precompiled and unchanged
between these compiler builds; variation in its timings is measurement noise.

**Remaining question:** The bounded full sweep provides larger-size evidence;
repeat under controlled load and thermal conditions before publishing broad claims.

## E5 — Instruments exercise (pending)

**Prediction before profiling:** Naive multiplication should spend most time
inside its dot-product loop. Reordered/blocked execution should expose different
memory behavior; small parallel products may spend a substantial fraction in
thread startup and scheduling.

**Attempt:** `xcrun --find xctrace` failed because full Xcode/Instruments is not
installed or selected (`xcode-select -p` selects command-line tools).

**Observation:** No trace collected. No sampled hotspot, stall, or hardware-counter
claim is made. `scripts/profile.py` is implemented and dry-run reviewable.

**Next action:** Install/select full Xcode, list templates, then profile at least
three implementations and replace these predictions with trace-backed findings.

## E6 — Isolate SIMD rather than all O3 transformations

**Prediction:** Removing automatic loop/SLP vectorization while retaining O3
should reduce throughput of the contiguous reordered inner loop.

**Failed control:** Disabling flags supplied before the target O3 flag were
overridden. The first 1024 measurement stayed near 25.8 GFLOP/s, and disassembly
still contained vector `fmla.4s` instructions. This did not test the hypothesis.

**Change:** Added `MATMUL_DISABLE_VECTORIZATION` with compiler-specific
flags placed after O3. Rebuilt and passed the complete CTest suite.

**Observation:** Corrected reordered results are 5.622, 5.913, and 6.050 GFLOP/s
at sizes 128, 512, and 1024. Normal O3 measurements are 5.44×, 4.40×, and 4.27×
higher. Disabled-object disassembly has scalar FMA but no vector FMA at the
checked locations. Four corrected measurements replace the initial control CSV.

**Explanation:** Flag ordering is part of experimental validity. Comparing only
O0 and O3 would mix SIMD with many other transformations.

**Remaining question:** Load was uncontrolled and runs occurred separately;
repeat under controlled conditions to establish stable ratios.

## E7 — Complete bounded larger O3 study

**Observation:** Of 252 planned configurations, 233 passed with the 15-second
process budget and 19 timed out. A targeted 120-second retry recovered the
2048 Accelerate result, leaving 234 verified rows and 18 missing configurations.
All 27 roadmap rectangular configurations passed. The original 19 timeout events
remain in the JSONL log.

At 2048, Accelerate's median multiplication is 9.819 ms while FP64 reference
verification takes 23.669 seconds. A process timeout cannot be treated as a
measurement of the multiplication kernel. The best custom 2048 result is
86.946 GFLOP/s with eight workers, versus Accelerate's 1749.648 GFLOP/s.

**Interpretation:** Larger problems amortize worker startup, but more workers
still do not always help: at 1024, six outperform eight. Tiling helps at 768
for the best tested edge but loses to unblocked reordered traversal at several
other sizes. See the final report and `results/full_figures/` for complete
measured comparisons.

## E8 — Complete audited study, independent repeats, and native profiles

**Observation:** The final adaptive study completed all 432 planned workloads:
252 at O3 and 60 each at O2, O0, and O3 fast-math. The coverage validator found
432 valid rows with no missing, unexpected, or invalid observations. The O0 naive
2048 workload exceeded the original 600-second process bound and was recovered
under a 1,800-second bound while on AC power.

An additional 462 observations provide three new process-level collections for
selected workloads. Together with the base collection, 154 exact configurations
have four independent sweeps. The analysis keeps builds, binaries, machines,
environments, verification methods, and iteration policies separate.

Four five-second macOS `sample` captures completed for naive, reordered, blocked,
and parallel kernels at size 512. The three serial captures are dominated by the
intended multiplication function. The parallel profile additionally shows lock
waiting and thread-creation paths, consistent with the implementation's choice to
create and join workers on every multiplication.

**Interpretation:** The strongest custom O3 result at 2048 is the seven-thread
parallel kernel at 82.069 GFLOP/s, 143.5× the naive baseline but still 17.2×
behind Accelerate. Reordering is decisive, blocking is workload-dependent, and
multithreading pays off only once useful work amortizes lifecycle overhead.

**Limitation:** The repeat bootstrap intervals have only four independent sweeps
and are exploratory. Desktop load was not isolated. Full Xcode was unavailable,
so stack sampling was completed but Instruments CPU Counters and hardware cache
or stall metrics are not claimed.
