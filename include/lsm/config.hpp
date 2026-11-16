// config.hpp - run configuration, option specification, CLI and CSV parsing.
// Implements section 3 (Config, OptionSpec, CLI, presets) and the input format
// of section 7 of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// No parallelism here; Config / OptionSpec are plain values passed explicitly
// (no globals), so they can be copied to every thread / rank / GPU kernel.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lsm {

enum class OptionType { Put, Call };

struct OptionSpec {
    std::string name;
    OptionType type = OptionType::Put;
    double S0 = 0;     // spot price today
    double K = 0;      // strike
    double r = 0;      // risk-free rate (continuous, annual), e.g. 0.06
    double sigma = 0;  // volatility (annual), e.g. 0.20
    double T = 0;      // maturity in years
};

struct Config {
    std::size_t num_paths = 100'000;   // N
    std::size_t steps_per_year = 50;   // M = round(steps_per_year * T), min 1
    int basis_degree = 2;              // 1 = line, 2 = parabola (default)
    std::uint64_t seed = 42;
    std::size_t binomial_steps = 2000;
    std::string input_csv = "data/options_benchmark.csv";
    std::string output_dir = "out";
    bool dump_paths = false;     // write sample_paths.csv (first 50 paths)
    bool dump_boundary = true;   // write boundary_<name>.csv
    bool quiet = false;
    bool convergence = false;    // section 10: convergence study on first option
    std::size_t repeat = 1;      // section 10: run everything R times
    std::string preset = "";     // name of the preset used (for run_info.txt)
};

// Payoff of the option at spot S: put max(K-S,0), call max(S-K,0).
inline double payoff(const OptionSpec& o, double S) {
    const double v = (o.type == OptionType::Put) ? (o.K - S) : (S - o.K);
    return v > 0.0 ? v : 0.0;
}

// Number of time steps M = round(steps_per_year * T), at least 1.
// dt is then T / M (NOT 1/steps_per_year: T = 5/12 gives M = 21, dt = T/21).
std::size_t num_steps_for(const OptionSpec& o, const Config& c);

// Parse argv. Presets are applied first, explicit flags then override them,
// independent of the order on the command line.
// Returns an error message (including the usage text for --help) on failure.
std::expected<Config, std::string> parse_cli(std::span<const std::string_view> args);

std::string usage_text();

// Read "name,type,S0,K,r,sigma,T" rows; '#' lines and the header are skipped.
std::expected<std::vector<OptionSpec>, std::string> read_options_csv(const std::string& path);

std::string_view type_name(OptionType t);

}  // namespace lsm
