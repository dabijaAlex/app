// reference.hpp - independent reference price used to validate LSM.
// Implements section 5.4 of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// Parallelism: none needed (cheap, deterministic). In the parallel versions
// this can simply run on rank 0 / the host.
#pragma once

#include <cstddef>

#include "lsm/config.hpp"

namespace lsm {

// Cox-Ross-Rubinstein binomial tree for the AMERICAN option.
//   u = exp(sigma sqrt(dt)), d = 1/u, p = (exp(r dt) - d) / (u - d)
//   backward: V = max(payoff, disc * (p V_up + (1-p) V_down))
// Single std::vector<double> of size B+1 updated in place (O(B) memory).
// Returns the average of the B-step and (B+1)-step trees, which cancels most
// of the odd/even oscillation of CRR prices.
double binomial_american(const OptionSpec& opt, std::size_t B);

// One tree with exactly B steps (exposed for tests).
double binomial_american_single(const OptionSpec& opt, std::size_t B);

}  // namespace lsm
