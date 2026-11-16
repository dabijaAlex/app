# Serial American option pricer — Longstaff-Schwartz Monte Carlo

Serial C++23 baseline of the parallel-architectures project. It prices American
options with the Longstaff-Schwartz (2001) least-squares Monte Carlo method and
validates every price against a CRR binomial tree (American). The OpenMP / MPI / CUDA versions will be built from this code: every
loop they need to change is marked with a `// [PARALLEL: ...]` comment.

No dividends are modelled. For this reason early exercise of a call is never
optimal (the `Call_noDiv_check` row).

## Build

Requirements: GCC ≥ 14 with `-std=c++23`, GNU make, and only the standard library.

```bash
make            # builds bin/lsm
make test       # builds and runs bin/lsm_tests (exit code != 0 on failure)
make clean
make debug      # -O0 -g (+ ASan/UBSan on Linux)
```

**Windows (MSYS2 UCRT64):** use `mingw32-make` instead of `make`. Run it from a
shell that has `C:\msys64\ucrt64\bin` **first** on `PATH`. From Git Bash, run
`export PATH=/c/msys64/ucrt64/bin:$PATH`, because Git's own `/mingw64/bin` makes
the linker fail without printing a message. The Makefile adds two
Windows-specific link flags:
- `-lstdc++exp`, where MinGW keeps the console part of `std::print`;
- `-static`, so `bin/lsm.exe` runs without MSYS2 DLLs.

ASan is not available on MinGW, so `make debug` there builds without sanitizers.

Developed and verified with GCC 16.1 (MSYS2). GCC 14 has not been tested, but no
library feature newer than GCC 14 is used.

## Run

```
./bin/lsm [--input FILE] [--paths N] [--steps-per-year M] [--degree 1|2]
          [--seed S] [--binomial-steps B] [--out DIR] [--dump-paths]
          [--no-boundary] [--quiet] [--preset dev|bench|heavy]
          [--convergence] [--repeat R]
```

| Preset | paths | steps/year | Purpose |
|---|---|---|---|
| `dev`   | 20 000    | 50  | seconds; development |
| `bench` | 100 000   | 50  | the paper's precision regime |
| `heavy` | 1 000 000 | 365 | minutes; performance measurements |

The preset is applied first and explicit flags then override it, whatever their
order on the command line. The number of steps for an option is
`M = round(steps_per_year · T)` (at least 1), and the time step is `dt = T/M`.

Before allocating, the program prints the path-matrix size `(M+1)·N·sizeof(path_t)`.
It refuses to run if that size exceeds the `LSM_MAX_GB` environment variable
(default 8). One matrix is allocated and reused for every option.

Exit code: 0 if every option is `OK`, 1 if any is `CHECK`, 2 for CLI/input
errors, 3 if the memory limit is exceeded.

Examples:

```bash
./bin/lsm --preset bench --input data/options_benchmark.csv
./bin/lsm --preset bench --input data/options_realistic.csv --dump-paths
./bin/lsm --preset dev --convergence --out out/conv      # convergence study
./bin/lsm --preset heavy --repeat 3 --out out/heavy      # benchmarking
python scripts/plot.py out                                # PNGs in out/plots/
```

`Status` is `OK` when `|LSM − Binomial| ≤ 3·SE + 0.02`. The `+0.02` absorbs two
small downward biases: the regression's slightly sub-optimal exercise policy, and
the fact that LSM only exercises on M dates (a Bermudan option), while the
binomial reference is effectively American.

## Input data

- `data/options_benchmark.csv`: the Longstaff & Schwartz (2001) Table 1 grid
  (20 puts, K = 40, r = 0.06).
- `data/options_realistic.csv`: 7 market-style cases. These include a deep-ITM
  put whose correct price is immediate exercise (20.00), and a no-dividend call
  that should never be exercised early.

Format: `name,type,S0,K,r,sigma,T`. Lines starting with `#` are comments.

## Output files (`--out DIR`, default `out/`)

| File | Content |
|---|---|
| `results.csv` | name,type,S0,K,r,sigma,T,paths,steps,lsm_price,lsm_se,binomial,diff,status |
| `timings.csv` | name,generate_s,accumulate_s,solve_s,exercise_s,final_s,reference_s,total_s: median over `--repeat`. Appended columns: `repeats` and the `min_*` per phase |
| `boundary_<name>.csv` | step,time,boundary_price,exercised_paths,itm_paths. Boundary = max S among exercising paths (put) or min S (call), `nan` if none. The last row (step M) counts the paths in the money at expiry. |
| `sample_paths.csv` | step,time,p0..p49 for the first option (only with `--dump-paths`) |
| `convergence.csv` | paths,price,se,abs_error_vs_binomial,time_s (only with `--convergence`) |
| `run_info.txt` | command line, configuration, compiler version, date, hostname |

All CSVs have a header row, use `.` as the decimal separator and 6 decimal places.

## Code layout

```
include/lsm/  types.hpp (PathMatrix, path_t)   config.hpp (Config, OptionSpec, CLI)
              rng.hpp (Philox4x32-10)          paths.hpp   regression.hpp
              lsm.hpp   reference.hpp   timer.hpp   report.hpp
src/          main.cpp config.cpp paths.cpp regression.cpp lsm.cpp reference.cpp report.cpp
tests/        test_main.cpp  (28 checks: toy example 10.215, Philox KATs, ...)
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
- **Discount table** `disc[k] = exp(-r·dt·k)`. It is built once per option, so
  the hot loops never call `exp`. `accumulate_sums` takes this table instead of
  `dt`: a small, deliberate deviation from the spec that lets the toy test inject
  a per-step discount of 0.9.
- **Regression on `X = S/K`.** All matrix entries are then of order 1, which keeps
  the normal equations well conditioned.
- **t = 0 rule.** The step t = 0 is not regressed. Instead, the final price is
  `max(mean discounted cash flow, payoff(S0))`, which gives DeepITM exactly 20.00.

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
| loop over options | independent tasks | `parallel for` over options (alt. strategy) | split options across ranks | streams | memory: one PathMatrix per concurrent option |

## Serial baseline results

Machine: Intel Core i7-13650HX, 16 GB RAM, Windows 11, GCC 16.1, `-O3 -march=native`,
float paths, seed 42.

| Run | Wall time | Status |
|---|---|---|
| `--preset bench`, benchmark grid (20 options) | ~4.8 s (generate 67 %, accumulate 15 %, exercise 17 %) | 20/20 OK |
| `--preset bench`, realistic (7 options) | ~0.9 s | 7/7 OK |
| `--preset heavy`, benchmark grid (20 options) | **415 s (≈ 6.9 min)**: generate 279 s (67 %), accumulate_sums 65 s (16 %), apply_exercise 71 s (17 %), solve 0.01 s | 20/20 OK, SE ≈ 0.002–0.007 |

Observations:

- Path generation dominates the serial time (`exp`, `log`, `sin`/`cos` per step,
  and strided time-major writes).
- In the convergence study the absolute error stops falling below about 0.02.
  That residual is the bias described in the Run section (exercise on discrete
  dates plus the regression policy), not Monte Carlo noise.
- For `Call_noDiv_check`, LSM exercises about 11 % of paths early, all in the last
  two steps before expiry. There the theoretical value of holding over exercising
  (about `K·r·dt ≈ 0.1`) is smaller than the regression error, so the effect on
  the price is small.
