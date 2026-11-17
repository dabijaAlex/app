// main.cpp - price ONE American option with Longstaff-Schwartz and write three
// small CSVs for scripts/plot.py.
//
//   lsm put|call S0 K r sigma T [paths] [steps_per_year] [seed]
//
// Outputs (in out/):
//   summary.csv   one row: the option, the LSM price and its standard error
//   boundary.csv  step,time,boundary,exercised,itm   (one row per step 1..M)
//   paths.csv     time,p0..p49                       (first 50 simulated paths)

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <print>
#include <string_view>

#include "lsm/lsm.hpp"
#include "lsm/option.hpp"
#include "lsm/paths.hpp"
#include "lsm/timer.hpp"
#include "lsm/types.hpp"

using namespace lsm;

namespace {

constexpr int kDegree = 2;                 // regression basis: 1, X, X^2
constexpr std::size_t kSamplePaths = 50;   // paths written to paths.csv
constexpr const char* kOutDir = "out";

// Strict number parse: the whole string must be consumed.
template <class T>
bool parse(std::string_view s, T& v) {
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    return ec == std::errc{} && ptr == s.data() + s.size();
}

}  // namespace

int main(int argc, char** argv) {
    const auto run_start = std::chrono::steady_clock::now();
    // ---- input: one option from the command line ---------------------------
    OptionSpec opt;
    std::size_t N = 100'000, steps_per_year = 50;
    std::uint64_t seed = 42;

    const std::string_view type = argc > 1 ? argv[1] : "";
    bool ok = (argc >= 7 && argc <= 10) && (type == "put" || type == "call") &&
              parse(argv[2], opt.S0) && parse(argv[3], opt.K) && parse(argv[4], opt.r) &&
              parse(argv[5], opt.sigma) && parse(argv[6], opt.T) &&
              (argc < 8 || parse(argv[7], N)) && (argc < 9 || parse(argv[8], steps_per_year)) &&
              (argc < 10 || parse(argv[9], seed));
    ok = ok && opt.S0 > 0 && opt.K > 0 && opt.sigma > 0 && opt.T > 0 && N > 0 && steps_per_year > 0;
    if (!ok) {
        std::println(stderr, "usage: lsm put|call S0 K r sigma T [paths=100000] [steps_per_year=50] [seed=42]\n"
                             "example: lsm put 36 40 0.06 0.2 1");
        return 2;
    }
    opt.type = (type == "put") ? OptionType::Put : OptionType::Call;

    // ---- the algorithm --------------------------------------------------------
    const std::size_t M = num_steps(opt.T, steps_per_year);
    const double dt = opt.T / static_cast<double>(M);

    PhaseTimer timer;
    PathMatrix S;
    {
        auto _ = timer.scope(Phase::Generate);
        generate_paths(opt, M, N, seed, /*option_index=*/0, S);
    }
    const LsmResult res = price_lsm(S, opt, kDegree, dt, &timer);

    // ---- output ---------------------------------------------------------------
    const double price = res.estimate.price, se = res.estimate.std_error;

    std::println("{} S0={} K={} r={} sigma={} T={}   paths={} steps={} seed={}", type, opt.S0, opt.K,
                 opt.r, opt.sigma, opt.T, N, M, seed);
    std::println("LSM       {:.4f} ± {:.4f}{}", price, se, res.exercise_at_t0 ? "  (exercise now)" : "");

    std::filesystem::create_directories(kOutDir);
    const std::string dir = kOutDir;

    std::ofstream sum(dir + "/summary.csv");
    std::println(sum, "type,S0,K,r,sigma,T,paths,steps,seed,lsm,se");
    std::println(sum, "{},{},{},{},{},{},{},{},{},{:.6f},{:.6f}", type, opt.S0, opt.K, opt.r,
                 opt.sigma, opt.T, N, M, seed, price, se);

    // boundary = put: highest S that exercised at t; call: lowest. nan if none.
    // The last row (t = M) is expiry: every in-the-money path "exercises".
    std::ofstream bnd(dir + "/boundary.csv");
    std::println(bnd, "step,time,boundary,exercised,itm");
    for (std::size_t t = 1; t <= M; ++t)
        std::println(bnd, "{},{:.6f},{:.6f},{},{}", t, dt * static_cast<double>(t), res.boundary[t],
                     res.exercised_count[t], res.itm_count[t]);

    std::ofstream pth(dir + "/paths.csv");
    const std::size_t np = std::min(kSamplePaths, N);
    std::print(pth, "time");
    for (std::size_t p = 0; p < np; ++p) std::print(pth, ",p{}", p);
    std::println(pth, "");
    for (std::size_t t = 0; t <= M; ++t) {
        std::print(pth, "{:.6f}", dt * static_cast<double>(t));
        for (std::size_t p = 0; p < np; ++p) std::print(pth, ",{:.4f}", S.at(t, p));
        std::println(pth, "");
    }

    std::println("wrote {}/summary.csv boundary.csv paths.csv", kOutDir);

    // total = the whole run: parsing, allocation, every phase, writing the CSVs.
    // "other" is the part of it outside the timed phases.
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
    std::print("time     ");
    for (std::size_t i = 0; i < kPhaseNames.size(); ++i)
        std::print(" {} {:.3f}s", kPhaseNames[i], timer.values()[i]);
    std::println("");
    std::println("total     {:.3f}s  (phases {:.3f}s + other {:.3f}s)", total, timer.total(), total - timer.total());
    return 0;
}
