// main.cpp - driver: parse CLI, read options, price each option, write output.
// Implements the overall pipeline of section 1 and the run modes of
// section 10 (--convergence, --repeat) of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// Parallelism: the loop over options is an alternative parallel strategy
// (independent tasks); inside each option the LSM kernels are parallelised.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <print>
#include <string>
#include <string_view>
#include <vector>

#include "lsm/config.hpp"
#include "lsm/lsm.hpp"
#include "lsm/paths.hpp"
#include "lsm/reference.hpp"
#include "lsm/report.hpp"
#include "lsm/timer.hpp"
#include "lsm/types.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace lsm;

namespace {

// Median of a small vector (copy, then nth_element).
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Price one option end to end: paths -> LSM -> references. Phase times are
// added to `timer`. `S` is a reusable buffer (allocated once for the run).
OptionResult price_option(const OptionSpec& opt, const Config& cfg, std::size_t option_index,
                          PathMatrix& S, PhaseTimer& timer) {
    OptionResult r;
    r.spec = opt;
    r.paths = cfg.num_paths;
    r.steps = num_steps_for(opt, cfg);
    const double dt = opt.T / static_cast<double>(r.steps);

    {
        auto _ = timer.scope(Phase::Generate);
        generate_paths(opt, cfg, option_index, S);
    }
    r.lsm = price_lsm(S, opt, cfg.basis_degree, dt, &timer);
    {
        auto _ = timer.scope(Phase::Reference);
        r.binomial = binomial_american(opt, cfg.binomial_steps);
    }
    return r;
}

// Section 10: LSM on the first option for an increasing number of paths.
std::vector<ConvergenceRow> run_convergence(const OptionSpec& opt, const Config& cfg, PathMatrix& S) {
    static constexpr std::size_t kPaths[] = {1'000, 2'000, 5'000, 10'000, 20'000,
                                            50'000, 100'000, 200'000, 500'000};
    const double bin = binomial_american(opt, cfg.binomial_steps);
    const std::size_t M = num_steps_for(opt, cfg);
    const double dt = opt.T / static_cast<double>(M);
    std::vector<ConvergenceRow> rows;
    if (!cfg.quiet) std::println("\nConvergence study on {} (binomial {:.4f})", opt.name, bin);
    for (std::size_t n : kPaths) {
        const auto t0 = std::chrono::steady_clock::now();
        generate_paths(opt, M, n, cfg.seed, 0, S);
        const auto res = price_lsm(S, opt, cfg.basis_degree, dt);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        rows.push_back({n, res.estimate.price, res.estimate.std_error,
                        std::abs(res.estimate.price - bin), secs});
        if (!cfg.quiet)
            std::println("  paths={:>7}  price={:.4f} ±{:.4f}  |err|={:.4f}  {:.3f} s", n,
                         res.estimate.price, res.estimate.std_error, rows.back().abs_error, secs);
    }
    return rows;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);  // box-drawing characters and σ in the table
#endif
    std::vector<std::string_view> args(argv + 1, argv + argc);
    auto cfg_or = parse_cli(args);
    if (!cfg_or) {
        const bool help = std::ranges::find(args, "--help") != args.end() ||
                          std::ranges::find(args, "-h") != args.end();
        std::println(help ? stdout : stderr, "{}", cfg_or.error());
        return help ? 0 : 2;
    }
    const Config cfg = *cfg_or;

    auto opts_or = read_options_csv(cfg.input_csv);
    if (!opts_or) {
        std::println(stderr, "error: {}", opts_or.error());
        return 2;
    }
    const auto& options = *opts_or;

    // Memory estimate: one PathMatrix, sized for the largest option, reused.
    std::size_t maxM = 0;
    for (const auto& o : options) maxM = std::max(maxM, num_steps_for(o, cfg));
    const std::size_t maxN = cfg.convergence ? std::max<std::size_t>(cfg.num_paths, 500'000) : cfg.num_paths;
    const double bytes = path_memory_bytes(maxM, maxN);
    double max_gb = 8.0;
    if (const char* e = std::getenv("LSM_MAX_GB")) max_gb = std::atof(e);
    if (bytes > max_gb * 1e9) {
        std::println(stderr,
                     "error: path matrix needs {:.2f} GB ((M+1)*N*sizeof(path_t) = {}*{}*{}), "
                     "above the LSM_MAX_GB limit of {:.2f} GB. Reduce --paths / --steps-per-year "
                     "or raise LSM_MAX_GB.",
                     bytes / 1e9, maxM + 1, maxN, sizeof(path_t), max_gb);
        return 3;
    }
    if (!cfg.quiet) {
        print_header(cfg, bytes);
        if (bytes > 1e9) std::println("warning: allocating {:.2f} GB for paths\n", bytes / 1e9);
    }

    std::error_code ec;
    std::filesystem::create_directories(cfg.output_dir, ec);
    if (ec) {
        std::println(stderr, "error: cannot create output dir '{}': {}", cfg.output_dir, ec.message());
        return 2;
    }

    PathMatrix S;  // allocated once (on first use), reused for every option
    std::vector<OptionResult> results(options.size());
    // timings[i][rep] = phase times of option i in repetition rep
    std::vector<std::vector<PhaseTimer>> timings(options.size());

    if (!cfg.quiet) print_table_header();
    for (std::size_t rep = 0; rep < cfg.repeat; ++rep) {
        // [PARALLEL: independent tasks] loop over options - alternative strategy:
        // OpenMP parallel for over options; MPI: split options across ranks;
        // CUDA: one stream per option. Hazard: memory - one PathMatrix per
        // concurrent option (up to ~3 GB each in the heavy preset).
        for (std::size_t i = 0; i < options.size(); ++i) {
            PhaseTimer t;
            results[i] = price_option(options[i], cfg, i, S, t);
            timings[i].push_back(t);
            if (rep + 1 == cfg.repeat) {
                if (!cfg.quiet) print_row(results[i]);
                if (cfg.dump_paths && i == 0)
                    write_sample_paths_csv(cfg.output_dir, S, options[i].T / static_cast<double>(results[i].steps));
                if (cfg.dump_boundary) write_boundary_csv(cfg.output_dir, results[i]);
            }
        }
    }

    // Median / min per phase over repeats.
    PhaseTimer total;
    for (std::size_t i = 0; i < options.size(); ++i) {
        for (std::size_t ph = 0; ph < kPhaseNames.size(); ++ph) {
            std::vector<double> v;
            for (const auto& t : timings[i]) v.push_back(t.values()[ph]);
            results[i].timing.values()[ph] = median(v);
            results[i].timing_min.values()[ph] = *std::ranges::min_element(v);
        }
        total += results[i].timing;
    }

    if (!cfg.quiet) print_summary(results, total);

    std::string cmdline = "lsm";
    for (auto a : args) cmdline += " " + std::string(a);
    bool ok_files = write_results_csv(cfg.output_dir, results);
    ok_files &= write_timings_csv(cfg.output_dir, results, cfg.repeat);
    ok_files &= write_run_info(cfg.output_dir, cfg, cmdline);

    if (cfg.convergence) ok_files &= write_convergence_csv(cfg.output_dir, run_convergence(options[0], cfg, S));

    if (!cfg.quiet) std::println("\nOutput written to {}/", cfg.output_dir);
    if (!ok_files) return 4;
    const bool all_ok = std::ranges::all_of(results, [](const OptionResult& r) { return r.ok(); });
    return all_ok ? 0 : 1;
}
