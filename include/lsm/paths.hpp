// paths.hpp - Monte Carlo path generation (geometric Brownian motion).
// Parallelism: every path is independent (counter-based RNG), so the outer
// loop over paths is a pure "map" for OpenMP / MPI / CUDA.
#pragma once

#include "lsm/option.hpp"
#include "lsm/types.hpp"

namespace lsm {

// Fill `out` (resized to (M+1) x N) with N risk-neutral GBM paths of M steps:
//   S[0]   = S0
//   S[t+1] = S[t] * exp( (r - sigma^2/2) dt + sigma sqrt(dt) Z[t] ),  dt = T/M
// Z[t] comes from Philox keyed by (seed, option_index, path, step) - see rng.hpp.
// The running price is kept in double; only the stored copy is path_t.
void generate_paths(const OptionSpec& opt, std::size_t num_steps_M, std::size_t num_paths,
                    std::uint64_t seed, std::size_t option_index, PathMatrix& out);

}  // namespace lsm
