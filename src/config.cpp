// config.cpp - command-line parsing, presets and options-CSV reader.
// Implements section 3 (CLI, presets) and section 7 (input format) of
// LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// No parallelism. In MPI versions rank 0 parses and broadcasts (or every
// rank parses the same argv/CSV - both are deterministic).

#include "lsm/config.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <format>
#include <ranges>

namespace lsm {

std::size_t num_steps_for(const OptionSpec& o, const Config& c) {
    const double m = std::round(static_cast<double>(c.steps_per_year) * o.T);
    return m < 1.0 ? std::size_t{1} : static_cast<std::size_t>(m);
}

std::string_view type_name(OptionType t) { return t == OptionType::Put ? "put" : "call"; }

std::string usage_text() {
    return "Usage: lsm [--input FILE] [--paths N] [--steps-per-year M] [--degree 1|2]\n"
           "           [--seed S] [--binomial-steps B] [--out DIR] [--dump-paths]\n"
           "           [--no-boundary] [--quiet] [--preset dev|bench|heavy]\n"
           "           [--convergence] [--repeat R] [--help]\n"
           "\n"
           "Presets set paths / steps-per-year; explicit flags override them:\n"
           "  dev    20000 paths,   50 steps/year  (seconds)\n"
           "  bench  100000 paths,  50 steps/year  (Longstaff-Schwartz paper regime)\n"
           "  heavy  1000000 paths, 365 steps/year (minutes; performance runs)\n"
           "Env LSM_MAX_GB (default 8): refuse to allocate a larger path matrix.\n";
}

namespace {

// Strict integer / float parsing: the whole string must be consumed.
template <class T>
std::expected<T, std::string> parse_num(std::string_view flag, std::string_view s) {
    T v{};
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size())
        return std::unexpected(std::format("invalid value '{}' for {}", s, flag));
    return v;
}

std::string_view trim(std::string_view s) {
    const auto ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string_view::npos) return {};
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

}  // namespace

std::expected<Config, std::string> parse_cli(std::span<const std::string_view> args) {
    Config c;

    // Pass 1: presets, so that explicit flags (pass 2) always win.
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] != "--preset") continue;
        if (i + 1 >= args.size()) return std::unexpected("missing value for --preset");
        const auto p = args[i + 1];
        if (p == "dev") { c.num_paths = 20'000; c.steps_per_year = 50; }
        else if (p == "bench") { c.num_paths = 100'000; c.steps_per_year = 50; }
        else if (p == "heavy") { c.num_paths = 1'000'000; c.steps_per_year = 365; }
        else return std::unexpected(std::format("unknown preset '{}' (dev|bench|heavy)", p));
        c.preset = std::string(p);
    }

    // Pass 2: everything else.
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto a = args[i];
        auto next = [&]() -> std::expected<std::string_view, std::string> {
            if (i + 1 >= args.size()) return std::unexpected(std::format("missing value for {}", a));
            return args[++i];
        };
        if (a == "--help" || a == "-h") return std::unexpected(usage_text());
        else if (a == "--preset") { ++i; }
        else if (a == "--dump-paths") c.dump_paths = true;
        else if (a == "--no-boundary") c.dump_boundary = false;
        else if (a == "--quiet") c.quiet = true;
        else if (a == "--convergence") c.convergence = true;
        else if (a == "--input" || a == "--out") {
            auto v = next();
            if (!v) return std::unexpected(v.error());
            (a == "--input" ? c.input_csv : c.output_dir) = std::string(*v);
        } else if (a == "--paths" || a == "--steps-per-year" || a == "--binomial-steps" ||
                   a == "--repeat") {
            auto v = next();
            if (!v) return std::unexpected(v.error());
            auto n = parse_num<std::size_t>(a, *v);
            if (!n) return std::unexpected(n.error());
            if (*n == 0) return std::unexpected(std::format("{} must be > 0", a));
            if (a == "--paths") c.num_paths = *n;
            else if (a == "--steps-per-year") c.steps_per_year = *n;
            else if (a == "--binomial-steps") c.binomial_steps = *n;
            else c.repeat = *n;
        } else if (a == "--seed") {
            auto v = next();
            if (!v) return std::unexpected(v.error());
            auto n = parse_num<std::uint64_t>(a, *v);
            if (!n) return std::unexpected(n.error());
            c.seed = *n;
        } else if (a == "--degree") {
            auto v = next();
            if (!v) return std::unexpected(v.error());
            auto n = parse_num<int>(a, *v);
            if (!n) return std::unexpected(n.error());
            if (*n != 1 && *n != 2) return std::unexpected("--degree must be 1 or 2");
            c.basis_degree = *n;
        } else {
            return std::unexpected(std::format("unknown argument '{}'\n\n{}", a, usage_text()));
        }
    }
    return c;
}

std::expected<std::vector<OptionSpec>, std::string> read_options_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) return std::unexpected(std::format("cannot open input file '{}'", path));

    std::vector<OptionSpec> out;
    std::string line;
    std::size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        const auto s = trim(line);
        if (s.empty() || s.front() == '#') continue;
        if (s.starts_with("name,")) continue;  // header row

        std::vector<std::string_view> f;
        for (auto part : s | std::views::split(','))
            f.push_back(trim(std::string_view(part.begin(), part.end())));
        if (f.size() != 7)
            return std::unexpected(std::format("{}:{}: expected 7 fields, got {}", path, lineno, f.size()));

        OptionSpec o;
        o.name = std::string(f[0]);
        if (f[1] == "put") o.type = OptionType::Put;
        else if (f[1] == "call") o.type = OptionType::Call;
        else return std::unexpected(std::format("{}:{}: type must be put or call", path, lineno));

        double* dst[5] = {&o.S0, &o.K, &o.r, &o.sigma, &o.T};
        for (int k = 0; k < 5; ++k) {
            auto v = parse_num<double>("CSV field", f[2 + k]);
            if (!v) return std::unexpected(std::format("{}:{}: {}", path, lineno, v.error()));
            *dst[k] = *v;
        }
        if (o.S0 <= 0 || o.K <= 0 || o.sigma <= 0 || o.T <= 0)
            return std::unexpected(std::format("{}:{}: S0, K, sigma, T must be > 0", path, lineno));
        out.push_back(std::move(o));
    }
    if (out.empty()) return std::unexpected(std::format("no options found in '{}'", path));
    return out;
}

}  // namespace lsm
