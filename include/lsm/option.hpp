// option.hpp - the option being priced and its payoff.
// Plain values, no globals, so they can be copied to every thread / rank / GPU kernel.
#pragma once

#include <cmath>
#include <cstddef>

namespace lsm {

enum class OptionType { Put, Call };

struct OptionSpec {
    OptionType type = OptionType::Put;
    double S0 = 0;     // spot price today
    double K = 0;      // strike
    double r = 0;      // risk-free rate (continuous, annual), e.g. 0.06
    double sigma = 0;  // volatility (annual), e.g. 0.20
    double T = 0;      // maturity in years
};

// Payoff of the option at spot S: put max(K-S,0), call max(S-K,0).
inline double payoff(const OptionSpec& o, double S) {
    const double v = (o.type == OptionType::Put) ? (o.K - S) : (S - o.K);
    return v > 0.0 ? v : 0.0;
}

// Number of time steps M = round(steps_per_year * T), at least 1.
// dt is then T / M (NOT 1/steps_per_year: T = 5/12 gives M = 21, dt = T/21).
inline std::size_t num_steps(double T, std::size_t steps_per_year) {
    const double m = std::round(static_cast<double>(steps_per_year) * T);
    return m < 1.0 ? std::size_t{1} : static_cast<std::size_t>(m);
}

}  // namespace lsm
