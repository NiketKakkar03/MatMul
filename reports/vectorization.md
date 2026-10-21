# Retained compiler vectorization evidence

Collected on 2026-10-20 with Apple Clang 21.0.0.21000334, arm64, Release
`-O3 -DNDEBUG`, fast-math OFF. The orchestrator enabled:

```text
-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
```

Selected remarks from `build/experiments/O3/build.log` (local absolute source
prefixes omitted):

```text
src/naive.cpp:12:13: remark: vectorized loop (vectorization width: 4, interleaved count: 4)
src/naive.cpp:15:21: remark: loop not vectorized: call instruction cannot be vectorized
src/naive.cpp:9:5: remark: loop not vectorized
src/reordered.cpp:40:17: remark: vectorized loop (vectorization width: 4, interleaved count: 4)
src/reordered.cpp:47:17: remark: vectorized loop (vectorization width: 4, interleaved count: 4)
src/reordered.cpp:57:17: remark: vectorized loop (vectorization width: 4, interleaved count: 4)
src/reordered.cpp:68:17: remark: vectorized loop (vectorization width: 4, interleaved count: 4)
src/reordered.cpp:75:17: remark: vectorized loop (vectorization width: 4, interleaved count: 4)
src/kernel_internal.hpp:31:25: remark: vectorized loop (vectorization width: 4, interleaved count: 4)
src/matrix.cpp:38:5: remark: loop not vectorized: value that could not be identified as reduction is used outside the loop
src/verification.cpp:14:5: remark: loop not vectorized: value that could not be identified as reduction is used outside the loop
```

The five reordered source locations correspond to ikj, jik, jki, kij, and kji.
The blocked and parallel translation units share the same tiled inner loop.
The naive hot dot-product loop also vectorizes in this build, while the outer
loop is not vectorized. A missed outer-loop transformation does not mean the
whole function is scalar. Diagnostics can mention compiler-generated paths and
calls even when source accessors are inline.

These observations show why predicting performance from a vectorization remark
alone is unreliable. Strided gather/scatter patterns, reductions, and cache-line
reuse can still dominate. A missed-vectorization example also appears in seeded
random generation: the generator's state introduces dependencies. Moving this
work outside timing avoids confusing setup cost with multiplication throughput.

The quick experiment compares O0/O2/O3/fast. Its speed differences include other
compiler transformations and must not be attributed entirely to SIMD.

An additional isolated experiment disables loop and SLP vectorization after the
O3 flag using `MATMUL_DISABLE_VECTORIZATION=ON`. At square sizes 128, 512,
and 1024, reordered throughput falls from 30.606/26.040/25.860 GFLOP/s to
5.622/5.913/6.050 GFLOP/s respectively. Four verified observations, including
the small rectangular case, are retained in `results/raw/O3_no_vectorize.csv`.
The experiment uses the same two warmups, nine samples, and seed as the normal
build. System-load controls remain limited, so the ratios are observations on
this machine rather than guarantees.

`otool -tvV` inspection finds `fmla.4s` in the normal O3 reordered object. The
disabled object's disassembly contains scalar `fmadd` and no `fmla.4s` or
`fmla.2d` instructions. This confirms that the control actually changed SIMD
generation for these loops.

The first attempt placed `-fno-vectorize -fno-slp-vectorize` in CMAKE_CXX_FLAGS,
before the target's O3 flag. It showed nearly unchanged performance and still
contained vector FMA instructions: a later O3 re-enabled vectorization. That
invalid control was replaced after adding the dedicated option, which puts the
disabling flags after optimization. This is why inspecting effective code
generation matters even when the requested flags look correct.

Regenerate complete logs with:

```sh
uv run --extra dev --extra analysis python scripts/run_experiments.py \
  --preset quick --modes O0 O2 O3 fast --vectorization-report

uv run --extra dev cmake -S . -B build/experiments/no-vectorize \
  -DCMAKE_BUILD_TYPE=Release -DMATMUL_DISABLE_VECTORIZATION=ON
uv run --extra dev cmake --build build/experiments/no-vectorize --parallel 4
uv run python scripts/run_sweep.py --binary build/experiments/no-vectorize/matmul \
  --preset quick --sizes 128 512 1024 --implementations reordered \
  --build-label O3_no_vectorize --output results/raw/O3_no_vectorize.csv
```

This is compiler evidence, not an Instruments trace or measurement of stalls,
cache misses, memory bandwidth, or Apple core scheduling.
