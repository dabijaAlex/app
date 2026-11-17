// types.hpp - core numeric aliases and the PathMatrix container.
// Parallelism: PathMatrix is TIME-MAJOR so that every backward-induction step
// reads one contiguous row (all paths at one time t). That row is what an
// OpenMP thread team / MPI rank slice / CUDA grid will sweep in parallel.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lsm {

// Precision of STORED path values. float halves memory (1M paths x 366 steps
// = 1.46 GB as float vs 2.9 GB as double). All ACCUMULATIONS use double.
// [PARALLEL-NOTE] On consumer GPUs float is much faster; keep this switchable
// (compile with -DLSM_PATHS_DOUBLE to store doubles).
#ifdef LSM_PATHS_DOUBLE
using path_t = double;
#else
using path_t = float;
#endif

using real_t = double;  // type of every accumulator / price / coefficient

// Memory layout (why time-major):
//
//   time-major  S[t][p]:  t=5: | p0 | p1 | p2 | p3 | ...   <- one backward step reads one
//                                                            contiguous row: coalesced
//   path-major  S[p][t]:  p0: | t0 | t1 | t2 | ... | p1: | t0 | t1 | ...
//                         reading all p at t=5 jumps by (M+1) elements: strided
//
// Element (t, p) lives at data[t * num_paths + p].
// In every backward step we sweep all paths at ONE time t, so contiguous p
// means cache-friendly on CPU and COALESCED loads on GPU.
struct PathMatrix {
    std::size_t num_steps = 0;  // M + 1 time points (t = 0..M)
    std::size_t num_paths = 0;  // N
    std::vector<path_t> data;

    // (Re)shape the matrix. Reuses the existing allocation when it is large
    // enough, so one buffer can serve every option in a run.
    void resize(std::size_t steps, std::size_t paths) {
        num_steps = steps;
        num_paths = paths;
        data.resize(steps * paths);
    }

    path_t& at(std::size_t t, std::size_t p) { return data[t * num_paths + p]; }
    path_t at(std::size_t t, std::size_t p) const { return data[t * num_paths + p]; }
    std::span<const path_t> row(std::size_t t) const {
        return {data.data() + t * num_paths, num_paths};
    }
};

}  // namespace lsm
