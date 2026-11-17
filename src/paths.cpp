// paths.cpp - Monte Carlo path generation (geometric Brownian motion).
// Parallelism: the outer loop over paths is an embarrassingly parallel map,
// made possible by the counter-based Philox RNG (rng.hpp): path p needs no
// state from any other path, so any worker can generate any range of paths.

#include "lsm/paths.hpp"

#include <cmath>

#include "lsm/rng.hpp"

namespace lsm {

// GBM, exact discretization (risk-neutral):
//   S[t+1] = S[t] * exp( (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
// drift and vol_sqrt_dt are precomputed once.
//
// Normals: step t (1..M) uses normal number (t-1) of its path. Normals come in
// blocks of 4 per Philox call: block = (t-1)/4, lane = (t-1)%4, with
//   counter_hi = (option_index << 32) | block,  counter_lo = p   (see rng.hpp).
//
// The running price S is kept in DOUBLE and only the stored copy is rounded
// to path_t; we never read the float back to continue the path, so float
// storage does not accumulate rounding error along the path.
void generate_paths(const OptionSpec& opt, std::size_t M, std::size_t N, std::uint64_t seed,
                    std::size_t option_index, PathMatrix& out) {
    out.resize(M + 1, N);
    const double dt = opt.T / static_cast<double>(M);
    const double drift = (opt.r - 0.5 * opt.sigma * opt.sigma) * dt;
    const double vol_sqrt_dt = opt.sigma * std::sqrt(dt);
    const std::uint64_t opt_hi = static_cast<std::uint64_t>(option_index) << 32;

    // [PARALLEL: map over paths] OpenMP: #pragma omp parallel for schedule(static)
    // over p. MPI: each rank generates ONLY its own range [p_begin, p_end) - no
    // data distribution needed thanks to the counter-based RNG. CUDA: one thread
    // per path; writes S[t][p] are coalesced across a warp.
    // Hazard: none (no shared state). NOTE: with outer-p / inner-t loops the
    // writes into time-major storage are STRIDED on the CPU (stride N). That is
    // acceptable for the serial baseline; a known later experiment is to
    // generate step-by-step across paths instead (contiguous writes).
    for (std::size_t p = 0; p < N; ++p) {
        double S = opt.S0;
        out.at(0, p) = static_cast<path_t>(S);
        std::array<double, 4> z{};
        for (std::size_t t = 1; t <= M; ++t) {
            const std::size_t k = t - 1;  // index of the normal used for this step
            if ((k & 3u) == 0) z = normal4(seed, opt_hi | static_cast<std::uint64_t>(k >> 2), p);
            S *= std::exp(drift + vol_sqrt_dt * z[k & 3u]);
            out.at(t, p) = static_cast<path_t>(S);
        }
    }
}

}  // namespace lsm
