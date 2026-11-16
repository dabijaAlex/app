// report.cpp - console table, timing summary and CSV / text output files.
// Implements section 6 of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// No parallelism: in the parallel versions only rank 0 / the host prints and
// writes files. CSV format: header row, '.' decimal separator, 6 decimals.

#include "lsm/report.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <fstream>
#include <print>

namespace lsm {

bool OptionResult::ok() const {
    return std::abs(diff()) <= 3.0 * lsm.estimate.std_error + 0.02;
}

double path_memory_bytes(std::size_t M, std::size_t N) {
    return static_cast<double>(M + 1) * static_cast<double>(N) * sizeof(path_t);
}

namespace {

std::string fmt_bytes(double b) {
    if (b >= 1e9) return std::format("{:.2f} GB", b / 1e9);
    return std::format("{:.1f} MB", b / 1e6);
}

// Column widths shared by the header and the rows (keep them in sync).
constexpr const char* kSep = "──────────────────────────────────────────────────────────────┼"
                             "─────────────────────────────────────────────────";

std::ofstream open_out(const std::string& dir, const std::string& file) {
    std::ofstream f(dir + "/" + file);
    if (!f) std::println(stderr, "error: cannot write {}/{}", dir, file);
    return f;
}

// CSV number: 6 decimals, "nan" for NaN.
std::string csv_num(double v) {
    if (std::isnan(v)) return "nan";
    return std::format("{:.6f}", v);
}

}  // namespace

void print_header(const Config& cfg, double max_bytes) {
    std::println("American option pricer — Longstaff-Schwartz (serial)");
    std::println("paths={}  steps/year={}  basis={}  seed={}  binomial_steps={}{}", cfg.num_paths,
                 cfg.steps_per_year, cfg.basis_degree == 2 ? "quadratic" : "linear", cfg.seed,
                 cfg.binomial_steps, cfg.repeat > 1 ? std::format("  repeat={}", cfg.repeat) : "");
    std::println("memory per option ≈ {} ({} paths, largest option)", fmt_bytes(max_bytes),
                 sizeof(path_t) == 4 ? "float" : "double");
    std::println("");
}

void print_table_header() {
    std::println("{:<20} {:<5}{:>7}{:>7}{:>7}{:>6}{:>6}   │ {:>9} {:<8} {:>8} {:>8}  {}",
                 "Option", "Type", "S0", "K", "r", "σ", "T", "LSM(Amer)", "±SE", "Binomial",
                 "Diff", "Status");
    std::println("{}", kSep);
}

void print_row(const OptionResult& r) {
    const auto& o = r.spec;
    std::println("{:<20} {:<5}{:>7.2f}{:>7.2f}{:>7.3f}{:>6.2f}{:>6.2f}   │ "
                 "{:>9.4f} ±{:<7.4f} {:>8.4f} {:>+8.4f}  {}",
                 o.name.size() > 20 ? o.name.substr(0, 20) : o.name, type_name(o.type), o.S0, o.K,
                 o.r, o.sigma, o.T, r.lsm.estimate.price,
                 r.lsm.estimate.std_error, r.binomial, r.diff(), r.ok() ? "OK" : "CHECK");
}

void print_summary(const std::vector<OptionResult>& results, const PhaseTimer& total) {
    std::println("");
    std::size_t bad = 0;
    for (const auto& r : results) bad += !r.ok();
    if (bad == 0) {
        std::println("All {} options OK", results.size());
    } else {
        std::print("{} of {} options marked CHECK:", bad, results.size());
        for (const auto& r : results)
            if (!r.ok()) std::print(" {}", r.spec.name);
        std::println("");
    }

    std::println("");
    std::println("Timing (total over all options{})", "");
    const double tot = total.total();
    for (std::size_t i = 0; i < kPhaseNames.size(); ++i) {
        const double s = total.values()[i];
        std::println("  {:<16}{:>8.2f} s {:>6.1f} %", kPhaseNames[i], s, tot > 0 ? 100.0 * s / tot : 0.0);
    }
    std::println("  {:<16}{:>8.2f} s", "TOTAL", tot);
}

bool write_results_csv(const std::string& dir, const std::vector<OptionResult>& results) {
    auto f = open_out(dir, "results.csv");
    if (!f) return false;
    std::println(f, "name,type,S0,K,r,sigma,T,paths,steps,lsm_price,lsm_se,binomial,diff,status");
    for (const auto& r : results) {
        const auto& o = r.spec;
        std::println(f, "{},{},{},{},{},{},{},{},{},{},{},{},{},{}", o.name, type_name(o.type),
                     csv_num(o.S0), csv_num(o.K), csv_num(o.r), csv_num(o.sigma), csv_num(o.T), r.paths,
                     r.steps, csv_num(r.lsm.estimate.price),
                     csv_num(r.lsm.estimate.std_error), csv_num(r.binomial), csv_num(r.diff()),
                     r.ok() ? "OK" : "CHECK");
    }
    return true;
}

// Base columns as in the spec (median over repeats; equals the single run
// when repeat = 1), then appended extras: repeats and the per-phase minima.
bool write_timings_csv(const std::string& dir, const std::vector<OptionResult>& results,
                       std::size_t repeats) {
    auto f = open_out(dir, "timings.csv");
    if (!f) return false;
    std::println(f, "name,generate_s,accumulate_s,solve_s,exercise_s,final_s,reference_s,total_s,"
                    "repeats,min_generate_s,min_accumulate_s,min_solve_s,min_exercise_s,"
                    "min_final_s,min_reference_s,min_total_s");
    for (const auto& r : results) {
        std::string line = r.spec.name;
        for (double v : r.timing.values()) line += "," + csv_num(v);
        line += "," + csv_num(r.timing.total());
        line += std::format(",{}", repeats);
        for (double v : r.timing_min.values()) line += "," + csv_num(v);
        line += "," + csv_num(r.timing_min.total());
        std::println(f, "{}", line);
    }
    return true;
}

bool write_boundary_csv(const std::string& dir, const OptionResult& r) {
    auto f = open_out(dir, "boundary_" + sanitize_filename(r.spec.name) + ".csv");
    if (!f) return false;
    std::println(f, "step,time,boundary_price,exercised_paths,itm_paths");
    const std::size_t M = r.steps;
    const double dt = r.spec.T / static_cast<double>(M);
    for (std::size_t t = 1; t <= M; ++t)
        std::println(f, "{},{},{},{},{}", t, csv_num(dt * static_cast<double>(t)),
                     csv_num(r.lsm.boundary[t]), r.lsm.exercised_count[t], r.lsm.itm_count[t]);
    return true;
}

bool write_sample_paths_csv(const std::string& dir, const PathMatrix& S, double dt,
                            std::size_t max_paths) {
    auto f = open_out(dir, "sample_paths.csv");
    if (!f) return false;
    const std::size_t np = std::min(max_paths, S.num_paths);
    std::string h = "step,time";
    for (std::size_t p = 0; p < np; ++p) h += std::format(",p{}", p);
    std::println(f, "{}", h);
    for (std::size_t t = 0; t < S.num_steps; ++t) {
        std::string line = std::format("{},{}", t, csv_num(dt * static_cast<double>(t)));
        for (std::size_t p = 0; p < np; ++p) line += "," + csv_num(static_cast<double>(S.at(t, p)));
        std::println(f, "{}", line);
    }
    return true;
}

bool write_run_info(const std::string& dir, const Config& cfg, const std::string& cmdline) {
    auto f = open_out(dir, "run_info.txt");
    if (!f) return false;
    const char* host = std::getenv("COMPUTERNAME");
    if (!host) host = std::getenv("HOSTNAME");
    const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println(f, "program        : lsm (serial Longstaff-Schwartz baseline)");
    std::println(f, "command        : {}", cmdline);
    std::println(f, "date (UTC)     : {:%Y-%m-%d %H:%M:%S}", now);
    std::println(f, "hostname       : {}", host ? host : "unknown");
    std::println(f, "compiler       : GCC {}", __VERSION__);
    std::println(f, "C++ standard   : {}", __cplusplus);
    std::println(f, "path_t         : {}", sizeof(path_t) == 4 ? "float" : "double");
    std::println(f, "preset         : {}", cfg.preset.empty() ? "(none)" : cfg.preset);
    std::println(f, "input_csv      : {}", cfg.input_csv);
    std::println(f, "num_paths      : {}", cfg.num_paths);
    std::println(f, "steps_per_year : {}", cfg.steps_per_year);
    std::println(f, "basis_degree   : {}", cfg.basis_degree);
    std::println(f, "seed           : {}", cfg.seed);
    std::println(f, "binomial_steps : {}", cfg.binomial_steps);
    std::println(f, "repeat         : {}", cfg.repeat);
    std::println(f, "convergence    : {}", cfg.convergence);
    return true;
}

bool write_convergence_csv(const std::string& dir, const std::vector<ConvergenceRow>& rows) {
    auto f = open_out(dir, "convergence.csv");
    if (!f) return false;
    std::println(f, "paths,price,se,abs_error_vs_binomial,time_s");
    for (const auto& r : rows)
        std::println(f, "{},{},{},{},{}", r.paths, csv_num(r.price), csv_num(r.se), csv_num(r.abs_error),
                     csv_num(r.time_s));
    return true;
}

std::string sanitize_filename(const std::string& s) {
    std::string out = s;
    for (char& c : out)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.')) c = '_';
    return out;
}

}  // namespace lsm
