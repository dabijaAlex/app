# Guide

This guide follows one run of the program from start to finish: how to run it,
what goes in, what happens to the input at each stage (with the real numbers
from one example), and what comes out. It assumes you know C++ but not option
pricing.

Contents:

1. [Quick start](#1-quick-start)
2. [The idea in five minutes](#2-the-idea-in-five-minutes)
3. [What goes in](#3-what-goes-in)
4. [How it goes: one example, stage by stage](#4-how-it-goes-one-example-stage-by-stage)
5. [What comes out](#5-what-comes-out)
6. [How to run: more examples](#6-how-to-run-more-examples)
7. [Code map and reading order](#7-code-map-and-reading-order)
8. [Tests](#8-tests)
9. [Design decisions worth knowing](#9-design-decisions-worth-knowing)
10. [Where parallelism goes](#10-where-parallelism-goes)
11. [Known quirks](#11-known-quirks)
12. [Glossary](#12-glossary)

---

## 1. Quick start

Requirements: Linux, GCC ≥ 14, GNU make, Python 3 with numpy and matplotlib.

```bash
cd ~/app
make                                 # build bin/lsm
./bin/lsm put 36 40 0.06 0.2 1       # price one option
python3 scripts/plot.py              # draw out/paths.png and out/boundary.png
```

`make run` does the last two lines in one go. `make test` runs the unit tests.

The rest of this guide explains exactly what that second command does.

---

## 2. The idea in five minutes

### What is being priced

A **put option** is a contract that gives you the right, but not the obligation,
to **sell** one share at a fixed price `K` (the **strike**). A **call** gives
the right to **buy** at `K`.

- The value of using the right right now (the **payoff**) is
  put `max(K − S, 0)`, call `max(S − K, 0)`, where `S` is the share price at
  that moment.
- An option is **in the money (ITM)** when its payoff is above 0. For a put,
  that means `S < K`.
- **American** means you may use the right on **any day** up to expiry `T`.

The question: what is that right worth **today**, when the share is at `S0`?

### Why it's hard

The price depends on **when** you exercise. Every day you choose: take the
payoff now, or wait in case it gets bigger. The best rule depends on the
unknown future, so there is no closed-form formula.

### How the future is modelled

The share follows **geometric Brownian motion (GBM)**: each step, the price is
multiplied by a random factor:

```
S[t+1] = S[t] · exp( (r − σ²/2)·dt + σ·√dt·Z ),   Z ~ Normal(0,1)
```

`r` is the risk-free interest rate, `σ` the volatility (how wild the price is),
and `dt` one time step. The program draws `N` such random futures (**paths**),
each with `M` steps.

### The pricing rule

The fair price is the **average over all paths of the discounted cash flow**.
Money received at time `t` is worth `exp(−r·t)` today. The hard part is
deciding, on each path, **when** the cash flow is received.

### The Longstaff-Schwartz trick (LSM)

At each date you need to know: "if I *don't* exercise, what is holding on
worth on average?" That is the **continuation value**. LSM estimates it with a
**least-squares regression** across all paths:

- **X:** today's price on the path (scaled as `S/K`).
- **Y:** the cash flow that path actually received later, discounted to today.
- **Fit:** `Y ≈ a + b·X + c·X²`. That parabola is the estimated continuation
  value as a function of price.

Rule: **exercise if payoff now > fitted continuation value**. Work **backwards**
from expiry to today, updating a path's cash flow whenever it exercises. At the
end, average.

### How we know it's right

The tests do the checking (section 8). The toy test walks 5 hand-written paths
through every step and compares each number with a hand calculation. Another
test checks that the example below lands in [4.45, 4.50], around the accepted
value of about 4.49 for this option.

---

## 3. What goes in

Everything comes from the command line. Nothing is read from files.

```
./bin/lsm put|call S0 K r sigma T [paths] [steps_per_year] [seed]
```

| # | Argument | Meaning | Example | Rule |
|---|---|---|---|---|
| 1 | `put` / `call` | option type | `put` | exactly one of the two |
| 2 | `S0` | share price today | `36` | > 0 |
| 3 | `K` | strike price | `40` | > 0 |
| 4 | `r` | risk-free rate, annual, continuous (6 % = `0.06`) | `0.06` | any number |
| 5 | `sigma` | volatility, annual (20 % = `0.2`) | `0.2` | > 0 |
| 6 | `T` | time to expiry in years (6 months = `0.5`) | `1` | > 0 |
| 7 | `paths` | number of simulated paths `N` | default `100000` | > 0 |
| 8 | `steps_per_year` | exercise dates per year | default `50` | > 0 |
| 9 | `seed` | random seed | default `42` | any unsigned integer |

The optional arguments are positional: to set `steps_per_year` you must also
give `paths`.

Fixed settings, changed by editing the constants at the top of
[src/main.cpp](src/main.cpp):

| Constant | Value | Meaning |
|---|---|---|
| `kDegree` | 2 | regression basis `1, X, X²` (1 = straight line) |
| `kSamplePaths` | 50 | paths written to `paths.csv` |
| `kOutDir` | `out` | output folder (relative to where you run) |

Bad input prints the usage line and exits with code 2:

```
$ ./bin/lsm put -1 40 0.06 0.2 1
usage: lsm put|call S0 K r sigma T [paths=100000] [steps_per_year=50] [seed=42]
example: lsm put 36 40 0.06 0.2 1
```

**Memory:** the program stores every path at every step as a `float`:
`(M+1) × N × 4` bytes. The defaults need 20 MB; 1 000 000 paths at 365
steps/year need about 1.5 GB. There is no limit check, so mind the size.

---

## 4. How it goes: one example, stage by stage

The example is the first row of the Longstaff-Schwartz (2001) paper:

```bash
./bin/lsm put 36 40 0.06 0.2 1
```

A put with strike 40 on a share now at 36, 6 % interest, 20 % volatility, one
year to expiry, 100 000 paths, 50 steps per year, seed 42. Exercising today
would pay `40 − 36 = 4`. The question is how much the right to wait adds.

The whole pipeline:

```
argv ─► OptionSpec + N, M, dt                       (main.cpp)
          │
          ▼
   4.2 generate_paths ─► S[t][p]: 51 × 100 000 floats  (paths.cpp, rng.hpp)
          │
          ▼
   price_lsm                                          (lsm.cpp)
     4.3 discount table   disc[k] = exp(−r·dt·k)
     4.4 init_at_expiry   cashflow[p] = payoff(S[50][p]), received at step 50
     4.5 for t = 49 … 1:  accumulate_sums ─► solve ─► apply_exercise
     4.6 final_price      mean and SE of discounted cash flows
     4.7 t = 0 rule       max(price, payoff(S0))
          │
          ▼
   print 4 lines + write out/summary.csv, boundary.csv, paths.csv
          │
          ▼
   scripts/plot.py ─► out/paths.png, out/boundary.png
```

### 4.1 Parse

[main.cpp](src/main.cpp#L43) turns the arguments into an `OptionSpec`
(`type=Put, S0=36, K=40, r=0.06, sigma=0.2, T=1`) and computes:

```
M  = round(steps_per_year · T) = round(50 · 1) = 50 steps
dt = T / M                     = 0.02 years (about 1 week)
```

### 4.2 Generate the paths

[generate_paths](src/paths.cpp#L25) fills a 51 × 100 000 matrix (step 0 to 50,
20.4 MB). Two constants are computed once:

```
drift       = (r − σ²/2)·dt = (0.06 − 0.02)·0.02 = 0.0008
vol_sqrt_dt = σ·√dt         = 0.2·0.141421       = 0.028284
```

The random numbers come from the Philox generator. Each call gives 4 normals
for 4 consecutive steps of one path. For path 0, steps 1–4:

```
Z = −0.970190, 0.194019, −1.235621, 1.923927
```

Step 1 of path 0:

```
S[1] = 36 · exp(0.0008 + 0.028284 · (−0.970190)) = 36 · exp(−0.026641) = 35.0536
```

The first five paths as stored (`paths.csv` holds the first 50):

| step | time | p0 | p1 | p2 | p3 | p4 |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 36.0000 | 36.0000 | 36.0000 | 36.0000 | 36.0000 |
| 1 | 0.02 | 35.0536 | 35.9543 | 35.6871 | 35.6439 | 36.8473 |
| 2 | 0.04 | 35.2747 | 36.1250 | 35.1954 | 34.9757 | 35.5774 |
| … | | | | | | |
| 50 | 1.00 | 30.6181 | 40.4270 | 42.8554 | 37.5630 | 40.6451 |

Storage is **time-major**: all 100 000 prices at step `t` sit next to each
other, because the backward pass reads one whole step at a time.

### 4.3 Discount table

`disc[k] = exp(−r·dt·k)` for k = 0…50, built once so the hot loops never call
`exp`. `disc[1] = 0.998801` (one week), `disc[50] = 0.941765` (one year).

### 4.4 Start at expiry

[init_at_expiry](src/lsm.cpp#L26): at step 50 there is no choice left, so each
path's cash flow is its payoff, received at step 50.

| | p0 | p1 | p2 | p3 | p4 |
|---|---|---|---|---|---|
| S[50] | 30.6181 | 40.4270 | 42.8554 | 37.5630 | 40.6451 |
| cashflow | **9.3819** | 0 | 0 | **2.4370** | 0 |
| exercise_step | 50 | 50 | 50 | 50 | 50 |

62 906 of the 100 000 paths finish in the money. If you could only exercise
at expiry (a **European** put), the price would be the mean discounted
payoff: about **3.855**. The rest of the algorithm finds how much more early
exercise is worth.

### 4.5 Walk backwards: t = 49 down to 1

[price_lsm_table](src/lsm.cpp#L140) repeats three kernels at every step. Here is
what they do at three of the 49 steps.

**Step t = 49** (one week before expiry)

1. [accumulate_sums](src/lsm.cpp#L40): take the 63 112 paths in the money at
   step 49. For each, `X = S[49]/40` and `Y` = its current cash flow discounted
   back to step 49. Add `X, X², X³, X⁴, Y, XY, X²Y` into 8 running sums.
2. [solve_normal_equations](src/regression.cpp#L22): solve the 3 × 3 system for
   the best parabola:
   ```
   continuation(X) = 41.7621 − 44.8746·X + 3.2228·X²
   ```
3. [apply_exercise](src/lsm.cpp#L63): for each in-the-money path, compare the
   payoff now with the parabola.

| | S[49] | payoff now | continuation | decision |
|---|---|---|---|---|
| p0 | 30.6738 | 9.3262 | 9.2454 | **exercise** (cash flow ← 9.3262, step ← 49) |
| p1 | 40.5284 | 0 | — | out of the money, skip |
| p2 | 42.7590 | 0 | — | skip |
| p3 | 37.4691 | 2.5309 | 2.5547 | wait |
| p4 | 40.7667 | 0 | — | skip |

For p0: `X = 30.6738/40 = 0.766845`, so the continuation is
`41.7621 − 44.8746·0.766845 + 3.2228·0.766845² = 9.2454`. The payoff 9.3262 is
higher, so p0 exercises.

44 315 paths exercise at this step. The highest price at which any path
exercised, 36.5969, is the **exercise boundary** for step 49.

**Step t = 25** (half way)

```
continuation(X) = 56.0895 − 83.0652·X + 28.3953·X²       (72 805 ITM paths)
```

| | S[25] | payoff now | continuation | decision |
|---|---|---|---|---|
| p0 | 33.1857 | 6.8143 | 6.7199 | **exercise** (overwrites the step-49 decision) |
| p1 | 39.4312 | 0.5688 | 1.7991 | wait: barely ITM, a lot of time left |
| p3 | 34.2976 | 5.7024 | 5.7425 | wait |

28 243 paths exercise; boundary 34.0017.

**Step t = 1** (one week after today)

```
continuation(X) = 90.7957 − 163.7187·X + 75.3123·X²      (99 994 ITM paths)
```

All five sample paths wait: with a year left, holding is worth more. Only 342
paths exercise (those that fell to 33.40 or below).

Because the loop runs backwards, a path's final cash flow is its **earliest**
exercise. Each exercise overwrites the later one.

### 4.6 Average

[final_price](src/lsm.cpp#L118) discounts each path's cash flow from its
exercise step back to today and averages over all 100 000 paths:

| | cash flow | received at step | × disc[step] |
|---|---|---|---|
| p0 | 7.0237 | 13 | 6.9150 |
| p1 | 0 | 50 | 0 |
| p2 | 0 | 50 | 0 |
| p3 | 7.5635 | 6 | 7.5092 |
| p4 | 0 | 50 | 0 |

```
price = mean = 4.465527
SE    = std / √N = 0.009228
```

### 4.7 The t = 0 rule

All paths share `S0 = 36` at step 0, so no regression is possible there.
Instead: exercising today pays `40 − 36 = 4.00`, and holding is worth
4.4655, so the program keeps 4.4655. (For a deep in-the-money put such as
`put 80 100 0.05 0.2 1`, exercising today wins and the price is exactly
20.0000.)

So: the European put is worth about 3.86, and the right to exercise early adds
about 0.61, for an American price of about 4.47.

---

## 5. What comes out

### Console

```
put S0=36 K=40 r=0.06 sigma=0.2 T=1   paths=100000 steps=50 seed=42
LSM       4.4655 ± 0.0092
wrote out/summary.csv boundary.csv paths.csv
time      generate 0.094s accumulate_sums 0.019s solve 0.000s apply_exercise 0.026s final_price 0.000s
total     0.139s  (phases 0.138s + other 0.001s)
```

| Line | Meaning |
|---|---|
| 1 | the input as understood, plus the computed `M` (`steps`) |
| `LSM` | Monte Carlo price ± standard error. The true Monte Carlo mean is within about ±2·SE 95 % of the time. 10× more paths gives about 3.2× smaller SE. `(exercise now)` appears when the t = 0 rule won. |
| `time` | seconds spent in each phase of the algorithm, the baseline for the parallel versions |
| `total` | wall-clock time of the whole run, from the start of `main` to after the CSVs are written. `phases` is the sum of the `time` line; `other` is everything outside it (parsing, writing files), normally about 1 ms |

Exit code: **0** success, **2** bad arguments.

### Files in `out/`

**`summary.csv`**: one row, the input and the result.

```
type,S0,K,r,sigma,T,paths,steps,seed,lsm,se
put,36,40,0.06,0.2,1,100000,50,42,4.465527,0.009228
```

**`boundary.csv`**: one row per step, 1 to M.

```
step,time,boundary,exercised,itm
1,0.020000,33.396297,342,99994
2,0.040000,33.600796,3760,99508
...
25,0.500000,34.001713,28243,72805
...
49,0.980000,36.596905,44315,63112
50,1.000000,39.999649,62906,62906
```

| Column | Meaning |
|---|---|
| `boundary` | put: highest price at which a path exercised at that step (call: lowest). `nan` if none did. |
| `exercised` | paths whose decision at that step, during the backward pass, was "exercise". A later (earlier-in-time) exercise can override it, so these counts do not add up to distinct paths. |
| `itm` | paths in the money at that step, the regression sample |

The last row (step M) is expiry, not a decision: every in-the-money path is
counted, and the boundary is just below K.

**`paths.csv`**: the first 50 paths, one row per time point (`time,p0..p49`),
the values shown in the table in [4.2](#42-generate-the-paths).

### Plots (`python3 scripts/plot.py`)

The script reads the three CSVs and writes two PNGs into `out/`. Pass another
folder as the first argument if needed.

- **`paths.png`**: the 50 sample paths. Orange ones finish in the money, blue
  ones don't. The dashed line is the strike; the black line is the exercise
  boundary. For a put, a path that drops below the black line is exercised.
- **`boundary.png`**: top, the boundary over time. For this put it sits around
  34 for most of the year and climbs toward 40 in the last weeks: as expiry
  nears, there is less time for the price to recover, so you exercise sooner.
  Bottom, per step, the in-the-money paths (light) and the paths exercising
  (dark).

---

## 6. How to run: more examples

```bash
make                     # build
make test                # unit tests (25 checks)
make run                 # the example above + plots
make debug               # -O0 -g with AddressSanitizer / UBSan
make clean               # remove bin/ and build/
```

| Command | What it shows |
|---|---|
| `./bin/lsm put 36 40 0.06 0.2 1` | the example above: 4.4655 ± 0.0092 |
| `./bin/lsm put 36 40 0.06 0.2 1 1000` | only 1 000 paths: 4.5653 ± 0.0980, ten times the SE |
| `./bin/lsm put 36 40 0.06 0.2 1 100000 50 7` | same option, different seed: a slightly different price |
| `./bin/lsm put 80 100 0.05 0.2 1` | deep in the money: exactly 20.0000, `(exercise now)` |
| `./bin/lsm call 100 100 0.05 0.2 1` | call, no dividends: early exercise is worthless, 10.4462 |
| `./bin/lsm put 50 50 0.10 0.40 0.416667` | 5-month put: `M = round(50 · 5/12) = 21` |
| `./bin/lsm put 36 40 0.06 0.2 1 1000000 365` | heavy run for timing, about 1.5 GB of paths |

Each run overwrites `out/`. To keep a run, copy the folder, e.g.
`cp -r out out_seed7`, then `python3 scripts/plot.py out_seed7`.

---

## 7. Code map and reading order

```
app/
├── GUIDE.md            ← you are here
├── README.md           short build/run reference, parallelism table
├── Makefile
├── include/lsm/        all headers; doc comments explaining the math live HERE
├── src/                implementations (main.cpp is the only glue)
├── tests/test_main.cpp unit tests (plain checks, no framework)
├── scripts/plot.py     CSV → PNG
├── bin/ build/ out/    generated, not in git
```

| Module | Header | Source | Job |
|---|---|---|---|
| option | [option.hpp](include/lsm/option.hpp) | — | `OptionSpec`, `payoff()`, `num_steps()` |
| types | [types.hpp](include/lsm/types.hpp) | — | `path_t` (float), `PathMatrix` (time-major) |
| rng | [rng.hpp](include/lsm/rng.hpp) | — | Philox4x32-10, uniforms, Box-Muller normals |
| paths | [paths.hpp](include/lsm/paths.hpp) | [paths.cpp](src/paths.cpp) | GBM path generation |
| regression | [regression.hpp](include/lsm/regression.hpp) | [regression.cpp](src/regression.cpp) | `RegressionSums`, `Coeffs`, 3×3 solver |
| lsm | [lsm.hpp](include/lsm/lsm.hpp) | [lsm.cpp](src/lsm.cpp) | backward induction, pricing |
| timer | [timer.hpp](include/lsm/timer.hpp) | — | per-phase wall-clock timing |
| driver | — | [main.cpp](src/main.cpp) | parse → price → print → write CSVs |

Dependencies only point downward (`main` → `lsm` / `paths` →
`regression` / `rng` / `option` / `types`). No global state.

Suggested reading order (about an hour). Read each header before its `.cpp`.

1. **Data shapes:** [PathMatrix](include/lsm/types.hpp#L36),
   [OptionSpec](include/lsm/option.hpp#L12), [payoff](include/lsm/option.hpp#L22),
   [LsmState](include/lsm/lsm.hpp#L20) (the two per-path arrays that are the
   algorithm's whole state).
2. **The core loop:** [price_lsm_table](src/lsm.cpp#L140), with the backward loop
   at [lsm.cpp:178](src/lsm.cpp#L178) and the t = 0 rule at
   [lsm.cpp:211](src/lsm.cpp#L211). Then its kernels:
   [accumulate_sums](src/lsm.cpp#L40),
   [solve_normal_equations](src/regression.cpp#L22),
   [apply_exercise](src/lsm.cpp#L63), [final_price](src/lsm.cpp#L118), and
   [RegressionSums](include/lsm/regression.hpp#L16).
3. **Paths:** [generate_paths](src/paths.cpp#L25), then
   [Philox4x32](include/lsm/rng.hpp#L20), [uniform4](include/lsm/rng.hpp#L68)
   (its counter-layout comment is the contract the CUDA/MPI versions must match)
   and [normal4](include/lsm/rng.hpp#L85).
4. **Glue:** [main](src/main.cpp#L43) and [PhaseTimer](include/lsm/timer.hpp#L27).
5. **Verify your understanding:** [test_toy_example](tests/test_main.cpp#L51)
   walks 5 hand-written paths through the algorithm and checks every
   regression line and decision, ending at the price 10.215. Work it out on
   paper once.

---

## 8. Tests

`make test` runs [tests/test_main.cpp](tests/test_main.cpp); 25 checks should pass.

| Test | What it proves |
|---|---|
| [`test_toy_example`](tests/test_main.cpp#L51) | The algorithm logic, exactly: hand-computed regression lines, decisions, price 10.215 |
| [`test_philox_kat`](tests/test_main.cpp#L103) | The RNG matches the published Random123 vectors bit for bit |
| [`test_normals`](tests/test_main.cpp#L117) | 10⁶ normals have mean ≈ 0 and variance ≈ 1 |
| [`test_determinism`](tests/test_main.cpp#L130) | Same seed gives bit-identical prices; different option indices get different streams |
| [`test_regression_merge`](tests/test_main.cpp#L155) | 1 chunk vs 7 chunks give the same fit (simulates the parallel reduction) |
| [`test_regression_exact`](tests/test_main.cpp#L182) | Recovers `1 + 2x + 3x²` exactly; rejects singular or too-small inputs |
| [`test_lsm_and_t0`](tests/test_main.cpp#L203) | The example above lands in [4.45, 4.50]; the deep-ITM put is exactly 20.00 |
| [`test_num_steps`](tests/test_main.cpp#L221) | M = 21 for T = 5/12; M is never 0 |

When you change the algorithm, the toy test catches logic errors. The
statistical tests only catch large ones.

---

## 9. Design decisions worth knowing

| Decision | Why |
|---|---|
| **Counter-based RNG (Philox)**, not `mt19937` | Each random number is a pure function of `(seed, option, path, step)`, with no state. Any thread, rank or GPU thread can generate any path, so the serial and parallel versions produce **bit-identical** paths. That is how the parallel versions will be proven correct. |
| **Time-major `S[t][p]`** | Each backward step reads one whole row contiguously: cache-friendly on the CPU, coalesced on the GPU. The cost is strided writes during generation. |
| **`float` storage, `double` math** | Halves memory (1 M paths × 366 steps is 1.5 GB as float, 2.9 GB as double). Every sum and price is in `double`. Compile with `-DLSM_PATHS_DOUBLE` to switch. |
| **Regress on `S/K`** | Keeps X ≈ 1, so the 3×3 system is well conditioned. The decisions are mathematically the same as regressing on `S`. |
| **Discount table `disc[k]`** | Built once, so hot loops never call `exp`. The toy test uses the same mechanism to inject a per-step discount of 0.9. |
| **Only ITM paths in the regression** | An out-of-the-money path never exercises; including it distorts the fit. |
| **Decide with the fitted value, never the path's own future** | Using the path's own future would be foresight and bias the price upward. |
| **`RegressionSums` = 8 doubles, standard layout** | Sent with MPI as `double[8]`; merging with `+=` is the reduction. |
| **No `-ffast-math`** | It reorders floating-point operations and would break bit-identical reproducibility. |

---

## 10. Where parallelism goes

Every loop a parallel version will change has a marker comment:

```bash
grep -rn "\[PARALLEL" src/ include/
```

- **Maps** (independent per path): `generate_paths`, `init_at_expiry`,
  `apply_exercise`.
- **Map + reduce:** `accumulate_sums`, the hot spot. `final_price` is a reduce.
- **Serial:** `solve_normal_equations` (tiny).
- **Sequential:** the time loop `t = M−1 … 1`. Step `t` needs the decisions of
  step `t+1`, so only the work *inside* a step runs in parallel. This is the
  scaling bottleneck (one sync per step).

In the example, path generation takes about two thirds of the time,
`apply_exercise` and `accumulate_sums` most of the rest. The README has the full
OpenMP / MPI / CUDA table.

---

## 11. Known quirks

- **LSM reads about 0.01–0.03 below the true American price.** Its exercise rule
  comes from a quadratic fit, so it is slightly sub-optimal, and a sub-optimal
  rule can only lose value. LSM also only exercises on M dates (it is really a
  Bermudan option), while the true American option can be exercised at any
  moment. More paths shrink the SE but **not** this bias.
- **Deep in-the-money put = exactly 20.0000 with SE 0.** In
  `put 80 100 0.05 0.2 1`, exercising today (`K − S0 = 20`) beats holding, so
  the t = 0 rule takes it and no randomness is left.
- **The call sometimes exercises paths early.** Without dividends it never
  should. For `call 100 100 0.05 0.2 1`, seeds 1, 2, 4 and 5 show exercises
  in `boundary.csv` before expiry, while 3 and 6–8 show none. Where it
  happens, the true advantage of waiting is tiny (about `K·r·dt ≈ 0.1`) and the
  regression error is larger. The effect on the price is small.
- **The boundary is noisy near t = 0.** Far from expiry few paths exercise, so
  the highest exercising price is set by a handful of paths.
- **Same seed, same output, bit for bit.** If two runs differ, something
  non-deterministic crept in.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| S, S0 | share price; price today |
| K | strike price |
| r | risk-free interest rate (continuous, per year) |
| σ (sigma) | volatility: annual standard deviation of log-returns |
| T | time to expiry in years |
| M, dt | number of time steps; `dt = T/M` |
| N | number of simulated paths |
| payoff | value of exercising now: put `max(K−S,0)`, call `max(S−K,0)` |
| ITM / OTM | in the money (payoff > 0) / out of the money |
| American / European / Bermudan | exercise any time / only at expiry / only on fixed dates (LSM is really Bermudan with M dates) |
| continuation value | expected discounted value of *not* exercising now |
| exercise boundary | the price at which exercising becomes optimal, per date |
| discounting | money at time t is worth `exp(−r·t)` today |
| SE | standard error of the Monte Carlo mean = `std / √N` |
| GBM | geometric Brownian motion, the random-walk model for the share price |
| Philox | counter-based random number generator (stateless, parallel-friendly) |
| Box-Muller | turns 2 uniform random numbers into 2 normal ones |
| normal equations | the small linear system whose solution is the least-squares fit |
