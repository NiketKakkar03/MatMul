# Study coverage audit

Status: complete.

432 valid observations from 432 saved rows; 432 planned workloads at a 8-CPU cap. Invalid rows: 0.

Coverage is computed separately per build/binary/machine/environment identity. Distinct sweep indices identify repeated observations, not extra workload coverage. Historical skips do not remove later valid results.

| Mode | Binary / machine / environment | Planned | Completed | Missing | Unexpected | Observations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| O3 | 2118f62f9ecf / 09986280057c / daa73a13e5a7 | 252 | 252 | 0 | 0 | 252 |
| O2 | 383bc130e3ff / 09986280057c / daa73a13e5a7 | 60 | 60 | 0 | 0 | 60 |
| fast | 29bd086fda79 / 09986280057c / daa73a13e5a7 | 60 | 60 | 0 | 0 | 60 |
| O0 | 918ea76e2881 / 09986280057c / daa73a13e5a7 | 60 | 60 | 0 | 0 | 60 |

Historical skip events: 1; recovered by valid data: 1. Remaining historical events do not determine coverage; the planned workload comparison does.

Median, minimum, population standard deviation and GFLOP/s were recomputed from every saved sample. Full missing/unexpected workload lists, invalid-row reasons and historical-event classifications are in `coverage.json`.
