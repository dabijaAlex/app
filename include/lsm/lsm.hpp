// lsm.hpp - Longstaff-Schwartz backward induction and pricing.
// Implements section 5.3 of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// Parallelism: each backward step = map+reduce (accumulate_sums), a tiny
// serial solve, then a map (apply_exercise). The time loop itself is
// SEQUENTIAL (step t needs the decisions of step t+1).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lsm/config.hpp"
#include "lsm/regression.hpp"
#include "lsm/timer.hpp"
#include "lsm/types.hpp"

namespace lsm {

// Per-path state of the backward induction.
struct LsmState {
    std::vector<double> cashflow;              // per path, value at its exercise step (undiscounted)
    std::vector<std::uint32_t> exercise_step;  // per path, step index where cashflow is received
};

// Discount table: disc[k] = (one-step discount factor)^k, k = 0..M.
// Built ONCE per option so the hot loops never call exp().
//   - real pricing:  disc[k] = exp(-r * dt * k)   (computed directly, not by
//                    repeated multiplication, to avoid error build-up)
//   - test hook:     disc[k] = std::pow(step_discount, k)
// [PARALLEL-NOTE] Parallel versions must build this table the same way
// (on the host, then broadcast / copy to the GPU) to stay bit-identical.
std::vector<double> make_discount_table_rate(double r, double dt, std::size_t M);
std::vector<double> make_discount_table_step(double step_discount, std::size_t M);

// Step M: cashflow = payoff(S[M][p]), exercise_step = M for every path
// (zero-payoff paths too: they simply carry a 0 cash flow).
void init_at_expiry(const PathMatrix& S, const OptionSpec& opt, LsmState& st);

// Loops over in-the-money paths at step t and adds (X, Y) to the sums:
//   X = S[t][p] / K
//   Y = cashflow[p] * disc[exercise_step[p] - t]   (future cash flow discounted to t)
// Note: takes the discount table instead of dt (spec deviation, documented):
// that is what lets the toy test inject a custom per-step discount.
RegressionSums accumulate_sums(const PathMatrix& S, const OptionSpec& opt, const LsmState& st,
                               std::size_t t, std::span<const double> disc);

// Result of apply_exercise at one step (for the boundary output).
struct ExerciseStats {
    std::size_t exercised = 0;  // number of paths that exercised at t
    double boundary = 0;        // put: max S among exercised; call: min S. NaN if none.
};

// For in-the-money paths at t: if payoff(S) > eval(coeffs, S/K), exercise now:
// cashflow[p] = payoff, exercise_step[p] = t. Out-of-the-money paths and
// paths whose continuation estimate is higher keep their future cash flow.
ExerciseStats apply_exercise(const PathMatrix& S, const OptionSpec& opt, const Coeffs& k,
                             std::size_t t, LsmState& st);

// Mean and standard error of a Monte Carlo estimate.
struct PriceEstimate {
    double price = 0;
    double std_error = 0;
};

// Discount every path's cashflow to t = 0 (disc[exercise_step[p]]) and return
// the sample mean and its standard error  sqrt(var / N).
PriceEstimate final_price(const LsmState& st, std::span<const double> disc);

struct LsmResult {
    PriceEstimate estimate;                    // American (LSM) price
    bool exercise_at_t0 = false;               // true if immediate exercise beat continuation
    std::vector<double> boundary;              // per step: put max S / call min S among exercised, NaN if none
    std::vector<std::size_t> exercised_count;  // per step
    std::vector<std::size_t> itm_count;        // per step (paths entering the regression)
};

// Full pricer with dt = T/M and discount exp(-r dt) per step.
LsmResult price_lsm(const PathMatrix& S, const OptionSpec& opt, int degree, double dt,
                    PhaseTimer* timer = nullptr);

// Same, with a custom per-step discount factor (needed by the hand-worked
// toy example, section 8.1). dt is unused by the math; kept for symmetry.
LsmResult price_lsm_with_discount(const PathMatrix& S, const OptionSpec& opt, int degree,
                                  double dt, double step_discount, PhaseTimer* timer = nullptr);

// Core implementation shared by both entry points.
LsmResult price_lsm_table(const PathMatrix& S, const OptionSpec& opt, int degree,
                          std::span<const double> disc, PhaseTimer* timer = nullptr);

}  // namespace lsm
