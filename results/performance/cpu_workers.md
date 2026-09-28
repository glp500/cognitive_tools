# CPU worker benchmark — 2026-09-28

Implementation: `2ac3a887f271874d9ff71726a3d0ed9e83601efc`.
Hardware: i7-12700H, 14 physical cores / 20 logical CPUs, approximately 64 GB RAM.

Identical workload: 40 independent adaptive local-rewiring conditions, uniform
high ecology, N=64, 100 training / 20 evaluation steps, seed 20260928. Timing
includes process startup, per-condition JSON checkpoint writes, and ordered
checkpoint reads. Numerical-library threads were capped at one. The original
single-worker B0 campaign remained active during measurement.

| Workers | First timing (seconds) | Repeat (seconds) |
|---|---:|---:|
| 1 | — | 79.80 |
| 8 | 11.06 | — |
| 14 | 8.45 | 8.21 |
| 20 | 8.80 | 9.44 |

Keep process-level parallelism: 14 workers achieved about 9.7× speedup on this
short workload. Recommend 14 as the measured starting point; `WORKERS=0` selects
all available logical CPUs (20 here). These short runs do not establish the
full-campaign ETA or prove that 14 remains optimal under prolonged thermal load.

Correctness: serial/parallel full condition outputs match for fixed, adaptive,
and matched-random treatments. Completed checkpoints are reused without
simulation. The suite contains 158 passing tests. Scientific model equations,
seeds, treatment grid, and final CSV ordering are unchanged.
