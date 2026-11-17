# Serial American option pricer — Longstaff-Schwartz Monte Carlo

Serial C++23 baseline of the parallel-architectures project. It prices one
American option with the Longstaff-Schwartz (2001) least-squares Monte Carlo
method. The OpenMP / MPI / CUDA
versions will be built from this code: every loop they need to change is marked
with a `// [PARALLEL: ...]` comment.

No dividends are modelled, so early exercise of a call is never optimal.

## Build (Linux)

Requirements: GCC ≥ 14 (`-std=c++23`), GNU make, Python 3 with numpy and
matplotlib for the plots. Verified on Arch Linux with GCC 16.2.

```bash
make            # builds bin/lsm
make test       # builds and runs bin/lsm_tests (25 checks, exit != 0 on failure)
make run        # prices the example below and draws the plots
make debug      # -O0 -g with ASan/UBSan
make clean
```

## Run

```
./bin/lsm put|call S0 K r sigma T [paths=100000] [steps_per_year=50] [seed=42]
python3 scripts/plot.py            # out/paths.png, out/boundary.png
```

The number of steps is `M = round(steps_per_year · T)` (at least 1) and
`dt = T/M`. The regression basis (quadratic) and the output folder (`out/`) are
constants at the top of [src/main.cpp](src/main.cpp).

Example (first row of the Longstaff-Schwartz table):

```
$ ./bin/lsm put 36 40 0.06 0.2 1
put S0=36 K=40 r=0.06 sigma=0.2 T=1   paths=100000 steps=50 seed=42
LSM       4.4655 ± 0.0092
wrote out/summary.csv boundary.csv paths.csv
time      generate 0.094s accumulate_sums 0.019s solve 0.000s apply_exercise 0.026s final_price 0.000s
total     0.139s  (phases 0.138s + other 0.001s)
```

The price is slightly low by design (about 0.01–0.03 for this option): the
fitted exercise policy is slightly sub-optimal, and LSM only exercises on M
dates.

Exit code: 0 success, 2 bad arguments.

Other useful cases:

| Command | Expected |
|---|---|
| `./bin/lsm put 80 100 0.05 0.2 1` | deep ITM: exactly 20.0000 (exercise now) |
| `./bin/lsm call 100 100 0.05 0.2 1` | no-dividend call: early exercise is worthless, ≈ 10.45 |
| `./bin/lsm put 50 50 0.10 0.40 0.416667` | Hull's 5-month put, M = 21 |
| `./bin/lsm put 36 40 0.06 0.2 1 1000000 365` | heavy run for timing (≈ 1.5 GB of paths) |

## Output (`out/`)

| File | Content |
|---|---|
| `summary.csv` | one row: `type,S0,K,r,sigma,T,paths,steps,seed,lsm,se` |
| `boundary.csv` | `step,time,boundary,exercised,itm` for steps 1..M. Boundary = highest S that exercised (put) or lowest (call), `nan` if none. The last row (step M) is expiry, not a decision. |
| `paths.csv` | `time,p0..p49`: the first 50 simulated paths |
| `paths.png` | the 50 paths coloured by whether they finish in the money, the strike and the exercise boundary |
| `boundary.png` | the exercise boundary over time (top), paths in the money and paths exercising per step (bottom) |

## Code layout

```
include/lsm/  option.hpp (OptionSpec, payoff, num_steps)   types.hpp (PathMatrix)
              rng.hpp (Philox4x32-10)   paths.hpp   regression.hpp   lsm.hpp
              timer.hpp
src/          main.cpp (glue, ~110 lines)  paths.cpp  regression.cpp  lsm.cpp
tests/        test_main.cpp  (toy example 10.215, Philox KATs, determinism, ...)
scripts/      plot.py
```

Design points that matter for the parallel versions:

- **Counter-based RNG (Philox4x32-10).** Each normal is a pure function of
  (seed, option index, path, step). The layout is documented in `rng.hpp`:
  `key = seed`, `counter = {path lo, path hi, block, option}`, and one call gives
  4 consecutive steps. Any worker can generate any path, and every version
  produces bit-identical paths.
- **Time-major path storage `S[t][p]`.** Each backward step sweeps one contiguous
  row, so reads are cache-friendly on the CPU and coalesced on the GPU.
- **`RegressionSums` (8 doubles, standard layout, mergeable with `+=`).**
  Merging these sums is the reduction step: `MPI_Allreduce` sends them as
  `double[8]`.
- **Discount table** `disc[k] = exp(-r·dt·k)`, built once per option, so the hot
  loops never call `exp`. The toy test uses the same mechanism to inject a
  per-step discount of 0.9.
- **Regression on `X = S/K`.** All matrix entries are then of order 1, which keeps
  the normal equations well conditioned.
- **t = 0 rule.** Step t = 0 is not regressed. The final price is
  `max(mean discounted cash flow, payoff(S0))`, which gives the deep-ITM put
  exactly 20.00.

## Parallelism map

Run `grep -rn "\[PARALLEL" src/ include/` to list all locations.

| Location | Pattern | OpenMP | MPI | CUDA | Hazard |
|---|---|---|---|---|---|
| `generate_paths` outer loop | map | `parallel for` | each rank owns path range | 1 thread/path | none (counter RNG) |
| `init_at_expiry` | map | `parallel for` | local | 1 thread/path | none |
| `accumulate_sums` | **map + reduce** | user-declared reduction on `RegressionSums` | local sums + `MPI_Allreduce` of 8 doubles | block shared-mem / warp-shuffle reduce, then atomics or 2nd pass | floating-point sum order changes last digits; compare with tolerance |
| `solve_normal_equations` | serial | `single` | every rank solves redundantly (cheaper than broadcast) | 1 thread or on host | none |
| `apply_exercise` | map | `parallel for` | local | 1 thread/path | none; ITM branch → warp divergence on GPU |
| time loop `t = M-1..1` | **sequential** | implicit barrier per step | 1 Allreduce per step = M-1 syncs | 1+ kernel launch per step (CUDA Graphs later) | the scaling bottleneck |
| `final_price` | reduce | `reduction(+:sum,sum2)` | `MPI_Reduce` | reduce | — |

## Serial baseline

`put 36 40 0.06 0.2 1`, 100 000 paths, 50 steps, `-O3 -march=native`, float
paths, seed 42: about 0.3 s, with path generation about two thirds of it. Path
generation dominates (`exp`, `log`, `sin`/`cos` per step, and strided
time-major writes), followed by `apply_exercise` and `accumulate_sums`.
