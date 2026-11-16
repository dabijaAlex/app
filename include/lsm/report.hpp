// report.hpp - console table and CSV output.
// Implements section 6 of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// No parallelism: in the parallel versions only rank 0 / the host writes.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "lsm/config.hpp"
#include "lsm/lsm.hpp"
#include "lsm/timer.hpp"
#include "lsm/types.hpp"

namespace lsm {

// Everything we know about one priced option.
struct OptionResult {
    OptionSpec spec;
    std::size_t paths = 0;
    std::size_t steps = 0;  // M
    double binomial = 0;
    LsmResult lsm;
    PhaseTimer timing;          // median over repeats (single run if R = 1)
    PhaseTimer timing_min;      // min over repeats
    double diff() const { return lsm.estimate.price - binomial; }
    bool ok() const;            // |LSM - Binomial| <= 3 SE + 0.02
};

// Estimated PathMatrix size in bytes: (M+1) * N * sizeof(path_t).
double path_memory_bytes(std::size_t M, std::size_t N);

void print_header(const Config& cfg, double max_bytes);
void print_table_header();
void print_row(const OptionResult& r);
void print_summary(const std::vector<OptionResult>& results, const PhaseTimer& total);

// CSV writers (section 6.2). Each returns false and prints an error if the
// file cannot be opened.
bool write_results_csv(const std::string& dir, const std::vector<OptionResult>& results);
bool write_timings_csv(const std::string& dir, const std::vector<OptionResult>& results,
                       std::size_t repeats);
bool write_boundary_csv(const std::string& dir, const OptionResult& r);
bool write_sample_paths_csv(const std::string& dir, const PathMatrix& S, double dt,
                            std::size_t max_paths = 50);
bool write_run_info(const std::string& dir, const Config& cfg, const std::string& cmdline);

struct ConvergenceRow {
    std::size_t paths;
    double price, se, abs_error, time_s;
};
bool write_convergence_csv(const std::string& dir, const std::vector<ConvergenceRow>& rows);

// Characters not allowed in file names are replaced by '_'.
std::string sanitize_filename(const std::string& s);

}  // namespace lsm
