# Profiling evidence

Native macOS stack profiles were captured on 2026-10-20 for four 512×512
workloads. Each profile contains the raw `sample` output; its adjacent metadata
records the exact target and recorder commands, executable hash, power/load
observations, workload result, and limitations. Recorder logs are retained too.

| Implementation | Active-stack observation |
|---|---|
| naive | 4,115 collapsed top-of-stack samples in `multiply_naive` |
| reordered | 4,069 samples in `multiply_loop_order` |
| blocked | 4,060 samples in `multiply_blocked`; five in output fill |
| parallel | Active samples include `multiply_parallel`, 1,608 `__ulock_wait` samples, and visible worker-creation paths |

The profiles are under `traces/profiles/`, with filenames beginning `naive-`,
`reordered-`, `blocked-`, and `parallel-`. The readiness handshake appears as a
short pre-capture read in some call graphs and is not part of multiplication.
Correctness verification was intentionally excluded from the profiled target.

These captures were recorded before the project was renamed to MatMul. Their raw
process names and metadata paths retain the old executable name intentionally so
the evidence continues to describe the command that was actually profiled. New
builds and profiling runs produce the `matmul` executable.

These observations demonstrate that the intended kernels dominate the serial
workloads and that the current parallel design pays synchronization and worker
lifecycle costs. Stack samples are neither function-call counts nor hardware
counters, so they cannot establish cache-miss rates or stall-cycle causes.

Full Xcode was unavailable: `xcrun --find xctrace` could not locate Instruments.
Consequently no `.trace` bundle or Instruments CPU Counters capture is claimed.
After installing/selecting full Xcode, `scripts/profile.py --list-templates` and
`--backend instruments` can add that optional hardware-specific evidence without
changing the completed `sample` artifacts.
