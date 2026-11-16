# Navigation guide

This guide is a map of the project. It explains the idea first, then where each
part lives, how to read the code, and what every output means. It assumes you
know C++ but not option pricing.

Contents:

1. [The idea in five minutes](#1-the-idea-in-five-minutes)
2. [The algorithm, step by step](#2-the-algorithm-step-by-step)
3. [Folder map](#3-folder-map)
4. [Reading the code, in order](#4-reading-the-code-in-order)
5. [How data flows through one run](#5-how-data-flows-through-one-run)
6. [Design decisions worth knowing](#6-design-decisions-worth-knowing)
7. [Running things](#7-running-things)
8. [Outputs and how to read them](#8-outputs-and-how-to-read-them)
9. [Tests](#9-tests)
10. [Where parallelism goes](#10-where-parallelism-goes)
11. [Known quirks](#11-known-quirks)
12. [Glossary](#12-glossary)

---

## 1. The idea in five minutes

### What is being priced

A **put option** is a contract that gives you the right, but not the obligation,
to **sell** one share at a fixed price `K` (the **strike**). A **call** is the
same, but it gives the right to **buy**.

- The value of using the right right now (the **payoff**) is:
  - put: `max(K − S, 0)`
  - call: `max(S − K, 0)`

  Here `S` is the share price at that moment.
- An option is **in the money (ITM)** when its payoff is above 0. For a put,
  that means `S < K`.
- **American** means you may use the right on **any day** up to expiry `T`.

So the question is: what is that right worth **today**, when the share is at
`S0`?

### Why it's hard

For an American option, the price depends on **when** you exercise. Every day
you face a choice: take the payoff now, or wait in case it gets bigger. The
right exercise rule depends on the unknown future, so there is no closed-form
formula.

### How we model the future

We assume the share follows **geometric Brownian motion (GBM)**: each step, the
price is multiplied by a random factor:

```
S[t+1] = S[t] · exp( (r − σ²/2)·dt + σ·√dt·Z ),   Z ~ Normal(0,1)
```

- `r` is the risk-free interest rate.
- `σ` (sigma) is the volatility, i.e. how wild the share price is.
- `dt = T / M` is one time step.

We draw `N` such random futures, called **paths**, each with `M` steps.

### Pricing rule

The fair price is the **average over all paths of the discounted cash flow**.
"Discounted" means money received at time `t` is worth `exp(−r·t)` today. The
hard part is deciding, on each path, **when** the cash flow is received.

### The Longstaff-Schwartz trick (LSM)

At each date you need to know: "if I *don't* exercise, what is holding on worth
on average?" That is the **continuation value**. LSM estimates it with a simple
**least-squares regression** across all paths:

- **X:** today's price on the path (scaled as `S/K`).
- **Y:** the cash flow that path actually received later, discounted back to
  today.
- **Fit:** `Y ≈ a + b·X + c·X²`. That parabola is the estimated continuation
  value as a function of the price.

Then the rule is: **exercise if payoff now > fitted continuation value**. Work
**backwards** from expiry to today, updating each path's cash flow when it
exercises. At the end, average everything.

### How we know it's right

The program also computes a **binomial tree** price. This is a completely
different, deterministic method (a lattice of up/down moves) that gives an
accurate American price. If LSM and the binomial tree agree within the
statistical error, the row is marked `OK`.

---

## 2. The algorithm, step by step

```
generate N paths × (M+1) time points               ← paths.cpp
cashflow[p] = payoff(S[M][p]), received at step M  ← init_at_expiry
for t = M−1 down to 1:                             ← price_lsm_table (sequential!)
    sums = Σ over ITM paths of (X, Y) moments      ← accumulate_sums   (hot spot)
    (a,b,c) = solve 3×3 normal equations           ← solve_normal_equations
    for each ITM path:                             ← apply_exercise
        if payoff(S[t][p]) > a + b·X + c·X²:
            cashflow[p] = payoff;  exercise_step[p] = t
price = mean_p( cashflow[p] · disc[exercise_step[p]] )   ← final_price
price = max(price, payoff(S0))                    ← "t = 0 rule"
```

Four rules the code follows deliberately:

1. **Only ITM paths enter the regression and the exercise decision.** An OTM path
   would never exercise, and including it distorts the fit.
2. **Decide using the fitted value, never the path's own future.** Using the
   path's own future would be cheating with foresight and would bias the price
   upward.
3. **Step `t = 0` is not regressed.** Every path has the same `S0` there, so the
   regression is singular. Instead, the final price is compared with exercising
   immediately, and the larger one is kept.
4. **The time loop is inherently sequential.** Step `t` needs the decisions of
   step `t+1`. Only the work *inside* one step can run in parallel.

---

## 3. Folder map

```
lsm/
├── GUIDE.md            ← you are here
├── README.md           build/run reference, parallelism table, baseline timings
├── Makefile            make / make test / make clean (mingw32-make on Windows)
├── data/               input option lists (CSV)
│   ├── options_benchmark.csv   20 puts from the Longstaff-Schwartz 2001 paper
│   └── options_realistic.csv   7 market-style cases incl. edge cases
├── include/lsm/        all headers; declarations + doc comments live HERE
├── src/                implementations
├── tests/test_main.cpp unit tests (plain checks, no framework)
├── scripts/plot.py     CSV → PNG graphs (matplotlib)
├── bin/                built executables (lsm.exe, lsm_tests.exe)
├── build/              object files
└── out/                run outputs (one sub-folder per --out DIR)
```

The original build spec is `C:\Users\alexd\Downloads\LSM_SERIAL_BUILD_INSTRUCTIONS.md`.
Code comments cite its section numbers (e.g. "section 5.3").

### One module per concern

| Module | Header | Source | Job |
|---|---|---|---|
| types | [types.hpp](include/lsm/types.hpp) | — | `path_t` (float), `PathMatrix` (time-major storage) |
| config | [config.hpp](include/lsm/config.hpp) | [config.cpp](src/config.cpp) | `OptionSpec`, `Config`, `payoff()`, CLI + CSV parsing |
| rng | [rng.hpp](include/lsm/rng.hpp) | — (header-only) | Philox4x32-10 RNG, uniforms, Box-Muller normals |
| paths | [paths.hpp](include/lsm/paths.hpp) | [paths.cpp](src/paths.cpp) | GBM path generation |
| regression | [regression.hpp](include/lsm/regression.hpp) | [regression.cpp](src/regression.cpp) | `RegressionSums`, `Coeffs`, 3×3 solver |
| lsm | [lsm.hpp](include/lsm/lsm.hpp) | [lsm.cpp](src/lsm.cpp) | backward induction, pricing |
| reference | [reference.hpp](include/lsm/reference.hpp) | [reference.cpp](src/reference.cpp) | binomial tree |
| timer | [timer.hpp](include/lsm/timer.hpp) | — (header-only) | per-phase wall-clock timing |
| report | [report.hpp](include/lsm/report.hpp) | [report.cpp](src/report.cpp) | console table, CSV writers |
| driver | — | [main.cpp](src/main.cpp) | glue: parse → loop options → write outputs |

Dependencies only point downward: `main` → `report` / `lsm` / `paths` /
`reference` → `regression` / `rng` / `config` / `types`. No module has global
state; everything is passed explicitly.

---

## 4. Reading the code, in order

Read the **header first** in each pair: the doc comments explaining the math
live there. The `.cpp` holds the mechanics.

### Step 1: the data shapes (10 min)

- [types.hpp:37](include/lsm/types.hpp:37), `PathMatrix`. Element `(t, p)` is at
  `data[t·N + p]` (**time-major**). The ASCII diagram above it explains why.
- [config.hpp:20](include/lsm/config.hpp:20): `OptionSpec` describes one option,
  and [config.hpp:30](include/lsm/config.hpp:30): `Config` holds the run settings.
- [config.hpp:47](include/lsm/config.hpp:47), `payoff()`. This one-liner defines
  put vs call everywhere.
- [lsm.hpp:21](include/lsm/lsm.hpp:21), `LsmState`. These two per-path arrays
  hold the whole algorithm's state: `cashflow` and `exercise_step`.

### Step 2: the core loop (20 min), the heart of the project

- [lsm.cpp:141](src/lsm.cpp:141), `price_lsm_table`. Read this function top to
  bottom; it matches the pseudocode in section 2. The backward loop is at
  [lsm.cpp:179](src/lsm.cpp:179), and the t = 0 rule at
  [lsm.cpp:208](src/lsm.cpp:208).
- Then the three kernels it calls:
  - [lsm.cpp:41](src/lsm.cpp:41) `accumulate_sums`: the ITM filter, `X = S/K`,
    and `Y` = cash flow × discount.
  - [regression.cpp:23](src/regression.cpp:23) `solve_normal_equations`:
    Gaussian elimination with pivoting. The comment at the top of the file
    explains why the regression uses `S/K` rather than `S`.
  - [lsm.cpp:64](src/lsm.cpp:64) `apply_exercise`: the exercise decision and the
    boundary statistic.
- [lsm.cpp:119](src/lsm.cpp:119) `final_price`: the mean and standard error
  (`MeanAcc` just above it uses a shifted sum of squares to reduce cancellation).
- [regression.hpp:17](include/lsm/regression.hpp:17) `RegressionSums`: 8 doubles
  with `add()` and `operator+=`. The `+=` is what the parallel reduction will be.

### Step 3: where the paths come from (15 min)

- [paths.cpp:31](src/paths.cpp:31) `generate_paths`: outer loop over paths, inner
  loop over steps. The running price stays in `double` and is stored as `float`.
- [rng.hpp:21](include/lsm/rng.hpp:21) `Philox4x32`: 10 rounds of
  multiply/xor. [rng.hpp:69](include/lsm/rng.hpp:69) `uniform4` and
  [rng.hpp:86](include/lsm/rng.hpp:86) `normal4` (Box-Muller). The
  **counter-layout comment** above `uniform4` is the contract that the CUDA/MPI
  versions must match.

### Step 4: validation (5 min)

- [reference.cpp:14](src/reference.cpp:14) `binomial_american_single`: a CRR tree
  in one vector, updated in place. `binomial_american` averages the B-step and
  (B+1)-step trees.

### Step 5: plumbing (skim)

- [main.cpp:90](src/main.cpp:90) `main`: parse, memory check, the repeat × option
  loops ([main.cpp:144](src/main.cpp:144)), median/min timings, then the writers.
- [main.cpp:45](src/main.cpp:45) `price_option`: one option end to end.
- [config.cpp:59](src/config.cpp:59) `parse_cli`: two passes, with presets first
  ([config.cpp:62](src/config.cpp:62)) so explicit flags always win.
- [report.cpp](src/report.cpp): formatting only. `OptionResult::ok()` at
  [report.cpp:17](src/report.cpp:17) is the `OK`/`CHECK` rule.
- [timer.hpp:30](include/lsm/timer.hpp:30) `PhaseTimer`: an RAII scope per
  phase. `time_phase(nullptr, …)` does no timing, which lets the tests call the
  pricer without a timer.

### Step 6: the toy example (10 min), the best way to *verify* your understanding

[test_main.cpp:53](tests/test_main.cpp:53) builds 5 hand-written paths and walks
the algorithm one step at a time. It asserts the regression line and the
exercise decisions at each step, then the final price, 10.215. Work it out on
paper once and the whole algorithm will feel concrete.

---

## 5. How data flows through one run

```
argv ──parse_cli──► Config
CSV  ──read_options_csv──► vector<OptionSpec>
                                │
            for each option i (and each repeat):
                                │
   generate_paths(opt, cfg, i) ─► PathMatrix S  (reused buffer, (M+1)×N floats)
                                │
   price_lsm(S, opt, degree, dt)
       ├─ make_discount_table_rate  → disc[k] = exp(−r·dt·k)
       ├─ init_at_expiry            → LsmState
       ├─ for t = M−1..1: accumulate_sums → solve → apply_exercise
       └─ final_price (+ t=0 rule)  → LsmResult {estimate, boundary[], counts[]}
                                │
   binomial_american(opt, B)    → reference price
                                │
            OptionResult {spec, lsm, binomial, timing}
                                │
   print_row / write_boundary_csv     (per option)
   print_summary / write_results_csv / write_timings_csv / write_run_info
```

---

## 6. Design decisions worth knowing

| Decision | Why |
|---|---|
| **Counter-based RNG (Philox)**, not `mt19937` | Each random number is a pure function of `(seed, option, path, step)`, with no state. Any thread, rank or GPU thread can generate any path, so the serial and parallel versions produce **bit-identical** paths. That is how the parallel versions will be proven correct. |
| **Time-major `S[t][p]`** | Each backward step reads one whole row (all paths at one `t`) contiguously: cache-friendly on the CPU, coalesced on the GPU. The price is strided writes during generation, which is accepted for now. |
| **`float` storage, `double` math** | This halves memory (the heavy preset needs about 2.9 GB per option). Every sum and price is computed in `double`. Compile with `-DLSM_PATHS_DOUBLE` to switch. |
| **Regress on `S/K`** | This keeps X ≈ 1, so `ΣX⁴` stays of order `n` and the 3×3 system is well conditioned. The decisions are mathematically the same as regressing on `S`. |
| **Discount table `disc[k]`** | It is built once per option, so the hot loops never call `exp`. The same table mechanism lets the toy test inject a per-step discount of 0.9. |
| **`RegressionSums` = 8 doubles, standard layout** | It can be sent with MPI as `double[8]`. Merging with `+=` is the reduction step. |
| **One `PathMatrix` reused** | Allocated once for the largest option, so there is no reallocation churn. |
| **No `-ffast-math`** | Fast-math reorders floating-point operations, which would break the bit-identical reproducibility. |

Where the code differs from the build spec's sketches:

- `accumulate_sums` and `final_price` take the discount table instead of `dt` or
  `(r, dt)`.
- `apply_exercise` returns `ExerciseStats` instead of a plain count.
- `LsmResult` also holds `itm_count` and `exercise_at_t0`.
- The `price_lsm*` functions take an optional `PhaseTimer*`.

---

## 7. Running things

Build from PowerShell (or MSYS2 UCRT64) in `lsm/`:

```bash
mingw32-make
```

```bash
mingw32-make test
```

From Git Bash, run `export PATH=/c/msys64/ucrt64/bin:$PATH` first. Without it,
the linker fails without printing a message.

Typical runs:

```bash
./bin/lsm.exe --preset bench --input data/options_benchmark.csv --out out/bench
```

```bash
./bin/lsm.exe --preset bench --input data/options_realistic.csv --out out/realistic --dump-paths
```

```bash
./bin/lsm.exe --preset dev --convergence --out out/conv
```

```bash
py scripts/plot.py out/realistic
```

| Preset | paths | steps/year | Time (this laptop) |
|---|---|---|---|
| `dev` | 20 000 | 50 | ~1 s for the 20-option grid |
| `bench` | 100 000 | 50 | ~5 s |
| `heavy` | 1 000 000 | 365 | ~7 min (415 s measured) |

The other flags (`--paths`, `--seed`, `--degree 1|2`, `--repeat R`, …) are
listed by `./bin/lsm.exe --help`. Exit code: 0 if all rows are `OK`, 1 if any is
`CHECK`, 2 for bad input, 3 if over the `LSM_MAX_GB` memory limit.

---

## 8. Outputs and how to read them

### Console table

```
Option          Type   S0     K     r     σ    T  │ LSM(Amer) ±SE      Binomial     Diff  Status
LS_S36_v20_T1   put  36.00 40.00 0.060 0.20 1.00  │    4.4655 ±0.0092    4.4867  -0.0212  OK
```

- **LSM(Amer) ±SE:** the Monte Carlo price and its standard error. The true
  Monte Carlo mean is within about ±2·SE of the estimate 95 % of the time. SE
  shrinks like `1/√N`: 10× more paths gives about 3.2× smaller SE.
- **Binomial:** the independent reference price.
- **Diff:** LSM − Binomial. It is usually **slightly negative**. That is expected
  (see quirks), not a bug.
- **Status:** `OK` when `|Diff| ≤ 3·SE + 0.02`.
- **Timing block:** seconds per phase. `generate` is about 67 %,
  `accumulate_sums` about 15 % and `apply_exercise` about 17 %. These are the
  numbers the parallel versions will speed up.

### Files (in `out/<dir>/`)

| File | What to look at |
|---|---|
| `results.csv` | One row per option; the same data as the table. Diff this against the parallel versions' output. |
| `timings.csv` | Per-option seconds per phase. With `--repeat R` the base columns are medians, and the `min_*` columns are minima. This is the baseline for speedup plots. |
| `boundary_<name>.csv` | Per step: `boundary_price` (the highest S that exercised, for a put), `exercised_paths` and `itm_paths`. The last row (step M) is expiry, not a decision. |
| `sample_paths.csv` | 50 paths of the first option (only with `--dump-paths`). |
| `convergence.csv` | Price, SE and \|error\| vs number of paths (only with `--convergence`). |
| `run_info.txt` | The exact command, config, compiler and date, for reproducing a run. |

### Plots (`py scripts/plot.py out/<dir>` → `out/<dir>/plots/`)

- **`paths.png`:** spaghetti of possible futures. Orange paths end below the
  strike, so the put pays off there.
- **`boundary_<name>.png`:** the **early-exercise boundary**. For a put,
  exercising is optimal when the price drops *below* this curve. It rises
  toward `K` as expiry approaches, because less time remains for the price to
  recover. A noisy, jagged curve near `t = 0` is normal: few paths are ITM that
  early.
- **`lsm_vs_binomial.png`:** LSM with error bars against the binomial ×. The ×
  should sit at or just above the dots.
- **`timing_breakdown.png`:** stacked phase times, showing where parallelism
  pays off.
- **`convergence.png`:** the error falls like `1/√N` at first, then **plateaus
  around 0.02**. The plateau is LSM's bias, not noise.

---

## 9. Tests

`mingw32-make test` runs [tests/test_main.cpp](tests/test_main.cpp) and should
report 28 checks passing:

| Test | Function | What it proves |
|---|---|---|
| Toy example | [`test_toy_example`](tests/test_main.cpp:53) | The algorithm logic, exactly: hand-computed regression lines, decisions, price 10.215 |
| Philox KAT | [`test_philox_kat`](tests/test_main.cpp:105) | The RNG matches the published Random123 vectors bit for bit |
| Normals | [`test_normals`](tests/test_main.cpp:119) | 10⁶ normals have mean ≈ 0 and variance ≈ 1 |
| Determinism | [`test_determinism`](tests/test_main.cpp:132) | Same seed gives bit-identical prices; different options get different streams |
| Merge | [`test_regression_merge`](tests/test_main.cpp:158) | 1 chunk vs 7 chunks give the same fit (simulates the parallel reduction) |
| Solver | [`test_regression_exact`](tests/test_main.cpp:185) | Recovers `1 + 2x + 3x²` exactly and rejects singular or too-small inputs |
| Binomial | [`test_binomial`](tests/test_main.cpp:206) | Tree matches the reference values (put 4.4867, call 10.4506) |
| LSM + t=0 | [`test_lsm_and_t0`](tests/test_main.cpp:214) | Paper's first row in [4.45, 4.50]; the deep-ITM put prices at exactly 20.00 |
| CLI | [`test_cli`](tests/test_main.cpp:232) | Explicit flags override presets; bad input is rejected; M = 21 for T = 5/12 |

When you change the algorithm, the toy test is the one that catches logic
errors. The statistical tests only catch large ones.

---

## 10. Where parallelism goes

Every loop a parallel version will change has a marker comment:

```bash
grep -rn "\[PARALLEL" src/ include/
```

In short:

- **Maps** (fully independent per path): `generate_paths`, `init_at_expiry` and
  `apply_exercise`.
- **Map + reduce:** `accumulate_sums`, which is the hot spot. `final_price` is a
  reduce.
- **Serial:** `solve_normal_equations` (tiny).
- **Sequential:** the time loop, which is the scaling bottleneck (one sync per
  step).
- **Independent tasks:** the loop over options (an alternative strategy).

The README has the full OpenMP / MPI / CUDA table.

---

## 11. Known quirks

- **LSM reads about 0.01–0.03 below the binomial price.** Its exercise rule
  comes from a quadratic fit, so it is slightly sub-optimal, and a sub-optimal
  rule can only lose value. The binomial tree is also effectively continuous
  exercise, while LSM only exercises on M dates. The `+0.02` in the `OK` rule
  absorbs this. More paths shrink the SE but **not** this bias, which is why the
  convergence plot plateaus.
- **`DeepITM_1y_put` = 20.0000 with SE 0.** Exercising immediately
  (`K − S0 = 20`) beats the continuation value, so the t = 0 rule takes it, and
  there is no randomness left in the answer.
- **The call exercises some paths early.** Without dividends it never should. In
  `boundary_Call_noDiv_check.csv`, about 11 % of paths exercise, all in the last
  two steps. There the true advantage of waiting is tiny (about `K·r·dt ≈ 0.1`)
  and the regression error is larger than that. The impact on the price is
  small.
- **`Growth_ATM_1y_put` has a big Diff (≈ −0.26).** It is still `OK`: with
  σ = 0.6 and S = 250, its SE is 0.16, so the tolerance is about 0.51.
- **Same seed, same output, bit for bit.** If two runs differ, something
  non-deterministic crept in.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| S, S0 | share price; price today |
| K | strike price |
| r | risk-free interest rate (continuous, per year) |
| σ (sigma) | volatility: the annual standard deviation of log-returns |
| T | time to expiry in years |
| M, dt | number of time steps; `dt = T/M` |
| N | number of simulated paths |
| payoff | value of exercising now: put `max(K−S,0)`, call `max(S−K,0)` |
| ITM / OTM | in the money (payoff > 0) / out of the money |
| American / Bermudan | exercise any time / only on fixed dates (LSM is really Bermudan with M dates) |
| continuation value | expected discounted value of *not* exercising now |
| exercise boundary | the price threshold at which exercising becomes optimal, per date |
| discounting | money at time t is worth `exp(−r·t)` today |
| SE | standard error of the Monte Carlo mean = `std / √N` |
| GBM | geometric Brownian motion, the random-walk model for the share price |
| CRR / binomial tree | Cox-Ross-Rubinstein lattice; the deterministic reference price |
| Philox | counter-based random number generator (stateless, parallel-friendly) |
| Box-Muller | turns 2 uniform randoms into 2 normal randoms |
| normal equations | the small linear system whose solution is the least-squares fit |
