// reference.cpp - CRR binomial tree (American) reference price.
// Implements section 5.4 of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// Parallelism: none needed; this is a cheap deterministic reference. (The
// binomial inner loop over nodes is a map per level if ever needed.)

#include "lsm/reference.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace lsm {

double binomial_american_single(const OptionSpec& o, std::size_t B) {
    const double dt = o.T / static_cast<double>(B);
    const double u = std::exp(o.sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double disc = std::exp(-o.r * dt);
    const double p = (std::exp(o.r * dt) - d) / (u - d);
    const double q = 1.0 - p;

    // Node (i, j) after i steps with j up-moves has price S0 * u^(2j - i).
    // Precompute u^k for k = -B..B once (index k + B) to avoid pow() per node.
    std::vector<double> upow(2 * B + 1);
    const double logu = std::log(u);
    for (std::size_t k = 0; k <= 2 * B; ++k)
        upow[k] = std::exp(logu * (static_cast<double>(k) - static_cast<double>(B)));

    // Values at expiry (i = B), then roll back in place: V[j] uses V[j], V[j+1].
    std::vector<double> V(B + 1);
    for (std::size_t j = 0; j <= B; ++j) V[j] = payoff(o, o.S0 * upow[2 * j]);  // 2j - B + B
    for (std::size_t i = B; i-- > 0;) {
        for (std::size_t j = 0; j <= i; ++j) {
            const double cont = disc * (p * V[j + 1] + q * V[j]);
            const double S = o.S0 * upow[2 * j + B - i];  // exponent 2j - i
            V[j] = std::max(payoff(o, S), cont);
        }
    }
    return V[0];
}

double binomial_american(const OptionSpec& o, std::size_t B) {
    return 0.5 * (binomial_american_single(o, B) + binomial_american_single(o, B + 1));
}

}  // namespace lsm
