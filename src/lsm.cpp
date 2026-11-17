// lsm.cpp - Longstaff-Schwartz backward induction and final pricing.
// Parallelism: per time step, accumulate_sums is map+REDUCE (the hot spot),
// solve is serial, apply_exercise is a conflict-free map. The time loop in
// price_lsm_table is SEQUENTIAL and is the scaling bottleneck (M-1 syncs).

#include "lsm/lsm.hpp"

#include <cmath>
#include <limits>

namespace lsm {

std::vector<double> make_discount_table_rate(double r, double dt, std::size_t M) {
    std::vector<double> d(M + 1);
    for (std::size_t k = 0; k <= M; ++k) d[k] = std::exp(-r * dt * static_cast<double>(k));
    return d;
}

std::vector<double> make_discount_table_step(double step_discount, std::size_t M) {
    std::vector<double> d(M + 1);
    for (std::size_t k = 0; k <= M; ++k) d[k] = std::pow(step_discount, static_cast<double>(k));
    return d;
}

// Every path starts with the payoff at expiry, received at step M.
void init_at_expiry(const PathMatrix& S, const OptionSpec& opt, LsmState& st) {
    const std::size_t N = S.num_paths;
    const std::size_t M = S.num_steps - 1;
    st.cashflow.resize(N);
    st.exercise_step.resize(N);
    const auto last = S.row(M);
    // [PARALLEL: map] OpenMP: parallel for over p. MPI: local paths only.
    // CUDA: 1 thread/path. Hazard: none.
    for (std::size_t p = 0; p < N; ++p) {
        st.cashflow[p] = payoff(opt, static_cast<double>(last[p]));
        st.exercise_step[p] = static_cast<std::uint32_t>(M);
    }
}

RegressionSums accumulate_sums(const PathMatrix& S, const OptionSpec& opt, const LsmState& st,
                               std::size_t t, std::span<const double> disc) {
    RegressionSums sums;
    const auto row = S.row(t);
    const double inv_K = 1.0 / opt.K;
    // [PARALLEL: map + REDUCE] the hot spot. OpenMP: user-declared reduction
    // on RegressionSums (#pragma omp declare reduction(+:RegressionSums:...))
    // or per-thread partials merged with operator+=. MPI: local sums then
    // MPI_Allreduce of 8 doubles. CUDA: block shared-mem / warp-shuffle
    // reduce, then atomics or a 2nd pass. Hazard: floating-point summation
    // order changes the last digits -> compare with a tolerance.
    for (std::size_t p = 0; p < row.size(); ++p) {
        const double s = static_cast<double>(row[p]);
        // Only IN-THE-MONEY paths enter the regression (out-of-the-money paths
        // would never exercise, and including them distorts the fit).
        if (payoff(opt, s) <= 0.0) continue;
        const double x = s * inv_K;  // regress on S/K, see regression.cpp
        const double y = st.cashflow[p] * disc[st.exercise_step[p] - t];
        sums.add(x, y);
    }
    return sums;
}

ExerciseStats apply_exercise(const PathMatrix& S, const OptionSpec& opt, const Coeffs& k,
                             std::size_t t, LsmState& st) {
    ExerciseStats es;
    const bool is_put = opt.type == OptionType::Put;
    double boundary = is_put ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity();
    const auto row = S.row(t);
    const double inv_K = 1.0 / opt.K;
    // [PARALLEL: map] OpenMP: parallel for over p (exercised count / boundary
    // via reduction(+:count) reduction(max/min:boundary)). MPI: local paths.
    // CUDA: 1 thread/path. Hazard: none - each path writes only its own
    // entries; the ITM branch causes warp divergence on GPU.
    for (std::size_t p = 0; p < row.size(); ++p) {
        const double s = static_cast<double>(row[p]);
        const double immediate = payoff(opt, s);
        if (immediate <= 0.0) continue;  // only ITM paths can exercise
        // The decision uses the REGRESSION estimate of continuation, never the
        // path's own realised future cash flow (that would be foresight and
        // bias the price upward).
        if (immediate > eval(k, s * inv_K)) {
            st.cashflow[p] = immediate;
            st.exercise_step[p] = static_cast<std::uint32_t>(t);
            ++es.exercised;
            boundary = is_put ? std::max(boundary, s) : std::min(boundary, s);
        }
    }
    es.boundary = es.exercised ? boundary : std::numeric_limits<double>::quiet_NaN();
    return es;
}

// Sample mean and standard error of values v_p (computed in one pass).
// Uses a shifted sum of squares around the first value to reduce
// cancellation in the variance.
namespace {
struct MeanAcc {
    double shift = 0, sum = 0, sum2 = 0;
    std::size_t n = 0;
    void add(double v) {
        if (n == 0) shift = v;
        const double d = v - shift;
        sum += d;
        sum2 += d * d;
        ++n;
    }
    PriceEstimate result() const {
        if (n == 0) return {};
        const double nn = static_cast<double>(n);
        const double mean_d = sum / nn;
        double var = (n > 1) ? (sum2 - nn * mean_d * mean_d) / (nn - 1.0) : 0.0;
        if (var < 0) var = 0;
        return {shift + mean_d, std::sqrt(var / nn)};
    }
};
}  // namespace

PriceEstimate final_price(const LsmState& st, std::span<const double> disc) {
    MeanAcc acc;
    // [PARALLEL: reduce] OpenMP: reduction(+:sum,sum2). MPI: MPI_Reduce of
    // (n, sum, sum2). CUDA: block reduce. Hazard: summation order -> last digits.
    // (Parallel versions: use shift = 0 or a common broadcast shift.)
    for (std::size_t p = 0; p < st.cashflow.size(); ++p)
        acc.add(st.cashflow[p] * disc[st.exercise_step[p]]);
    return acc.result();
}

LsmResult price_lsm(const PathMatrix& S, const OptionSpec& opt, int degree, double dt,
                    PhaseTimer* timer) {
    const auto disc = make_discount_table_rate(opt.r, dt, S.num_steps - 1);
    return price_lsm_table(S, opt, degree, disc, timer);
}

LsmResult price_lsm_with_discount(const PathMatrix& S, const OptionSpec& opt, int degree,
                                  double /*dt*/, double step_discount, PhaseTimer* timer) {
    const auto disc = make_discount_table_step(step_discount, S.num_steps - 1);
    return price_lsm_table(S, opt, degree, disc, timer);
}

LsmResult price_lsm_table(const PathMatrix& S, const OptionSpec& opt, int degree,
                          std::span<const double> disc, PhaseTimer* timer) {
    const std::size_t M = S.num_steps - 1;
    LsmResult res;
    res.boundary.assign(M + 1, std::numeric_limits<double>::quiet_NaN());
    res.exercised_count.assign(M + 1, 0);
    res.itm_count.assign(M + 1, 0);

    LsmState st;
    {
        auto _ = time_phase(timer, Phase::ApplyExercise);
        init_at_expiry(S, opt, st);
    }
    // Expiry row of the boundary output: every ITM path "exercises" at M.
    {
        const auto last = S.row(M);
        double b = (opt.type == OptionType::Put) ? -std::numeric_limits<double>::infinity()
                                                 : std::numeric_limits<double>::infinity();
        std::size_t cnt = 0;
        for (std::size_t p = 0; p < last.size(); ++p) {
            const double s = static_cast<double>(last[p]);
            if (payoff(opt, s) > 0.0) {
                ++cnt;
                b = (opt.type == OptionType::Put) ? std::max(b, s) : std::min(b, s);
            }
        }
        res.exercised_count[M] = cnt;
        res.itm_count[M] = cnt;
        if (cnt) res.boundary[M] = b;
    }

    // Backward induction. Step t = 0 is NOT regressed: all paths share S0, so
    // the regression matrix is singular there (handled after the loop).
    // [PARALLEL: SEQUENTIAL time loop] step t depends on the decisions of step
    // t+1, so this loop cannot be parallelised. OpenMP: one parallel region
    // around it with an implicit barrier per step. MPI: 1 Allreduce per step =
    // M-1 syncs. CUDA: 1+ kernel launches per step (CUDA Graphs later).
    // This is the scaling bottleneck.
    for (std::size_t t = M - 1; t >= 1; --t) {
        RegressionSums sums;
        {
            auto _ = time_phase(timer, Phase::AccumulateSums);
            sums = accumulate_sums(S, opt, st, t, disc);
        }
        res.itm_count[t] = static_cast<std::size_t>(sums.n);

        Coeffs k;
        {
            // [PARALLEL: serial] OpenMP: `single`. MPI: every rank solves
            // redundantly (cheaper than a broadcast). CUDA: 1 thread or host.
            auto _ = time_phase(timer, Phase::Solve);
            k = solve_normal_equations(sums, degree);
        }

        if (k.valid) {  // invalid fit (too few ITM paths / singular) -> no exercise at t
            auto _ = time_phase(timer, Phase::ApplyExercise);
            const ExerciseStats es = apply_exercise(S, opt, k, t, st);
            res.exercised_count[t] = es.exercised;
            res.boundary[t] = es.boundary;
        }
        if (t == 1) break;  // size_t loop: avoid wrap-around below 1
    }

    {
        auto _ = time_phase(timer, Phase::FinalPrice);
        res.estimate = final_price(st, disc);

        // t = 0: every path has the same S0, so compare the discounted
        // continuation value (the MC mean) with exercising immediately and
        // take the larger. This matters for deep-ITM puts (DeepITM_1y_put),
        // where the correct answer is immediate exercise = K - S0 exactly.
        const double now = payoff(opt, S.num_paths ? static_cast<double>(S.at(0, 0)) : opt.S0);
        if (now > res.estimate.price) {
            res.estimate = {now, 0.0};
            res.exercise_at_t0 = true;
        }
    }
    return res;
}

}  // namespace lsm
