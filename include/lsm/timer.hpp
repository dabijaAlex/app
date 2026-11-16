// timer.hpp - per-phase wall-clock accumulation.
// Implements section 5.5 of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// These phases are exactly the ones whose speedup the team will plot after
// parallelising, so keep them separate (even `solve`, which is tiny).
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <string_view>

namespace lsm {

enum class Phase : std::size_t {
    Generate = 0,
    AccumulateSums,
    Solve,
    ApplyExercise,
    FinalPrice,
    Reference,
    Count
};

inline constexpr std::array<std::string_view, static_cast<std::size_t>(Phase::Count)>
    kPhaseNames = {"generate", "accumulate_sums", "solve", "apply_exercise", "final_price",
                   "reference"};

// Accumulates seconds per phase. Usage:
//   { auto s = timer.scope(Phase::Generate); generate_paths(...); }
class PhaseTimer {
public:
    using clock = std::chrono::steady_clock;

    class Scope {
    public:
        Scope(PhaseTimer* t, Phase p) : timer_(t), phase_(p), start_(clock::now()) {}
        ~Scope() {
            if (timer_) timer_->add(phase_, std::chrono::duration<double>(clock::now() - start_).count());
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        PhaseTimer* timer_;
        Phase phase_;
        clock::time_point start_;
    };

    // Returns an RAII guard; passing a null timer (via the free helper below)
    // makes timing a no-op, so library code can be called without a timer.
    Scope scope(Phase p) { return Scope(this, p); }

    void add(Phase p, double seconds) { s_[static_cast<std::size_t>(p)] += seconds; }
    double get(Phase p) const { return s_[static_cast<std::size_t>(p)]; }
    double total() const {
        double t = 0;
        for (double v : s_) t += v;
        return t;
    }
    PhaseTimer& operator+=(const PhaseTimer& o) {
        for (std::size_t i = 0; i < s_.size(); ++i) s_[i] += o.s_[i];
        return *this;
    }
    const std::array<double, static_cast<std::size_t>(Phase::Count)>& values() const { return s_; }
    std::array<double, static_cast<std::size_t>(Phase::Count)>& values() { return s_; }

private:
    std::array<double, static_cast<std::size_t>(Phase::Count)> s_{};
};

// Null-safe scope: timing is skipped when `t` is nullptr.
inline PhaseTimer::Scope time_phase(PhaseTimer* t, Phase p) { return PhaseTimer::Scope(t, p); }

}  // namespace lsm
