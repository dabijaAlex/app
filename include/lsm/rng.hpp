// rng.hpp - Philox4x32-10 counter-based random number generator + normals.
// Implements section 4 of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// A random number is a PURE FUNCTION of (seed, counter). There is no hidden
// state, so any thread / MPI rank / CUDA thread can generate any path on its
// own, and the serial, OpenMP, MPI and CUDA versions produce BIT-IDENTICAL
// paths. This is how the parallel versions will be proven correct.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace lsm {

// ---------------------------------------------------------------------------
// Philox4x32-10 (Salmon et al., SC'11). Matches Random123's philox4x32_R(10)
// and cuRAND's curand_Philox4x32_10 bit for bit (verified by the known-answer
// test in tests/test_main.cpp).
// ---------------------------------------------------------------------------
struct Philox4x32 {
    using ctr_t = std::array<std::uint32_t, 4>;
    using key_t = std::array<std::uint32_t, 2>;

    static constexpr std::uint32_t M0 = 0xD2511F53u;  // round multipliers
    static constexpr std::uint32_t M1 = 0xCD9E8D57u;
    static constexpr std::uint32_t W0 = 0x9E3779B9u;  // Weyl key increments
    static constexpr std::uint32_t W1 = 0xBB67AE85u;  // (golden ratio, sqrt(3)-1)

    // One Philox round: two 32x32->64 multiplies, swap and XOR with the key.
    static constexpr ctr_t round(const ctr_t& c, const key_t& k) {
        const std::uint64_t p0 = std::uint64_t{M0} * c[0];
        const std::uint64_t p1 = std::uint64_t{M1} * c[2];
        const auto hi0 = static_cast<std::uint32_t>(p0 >> 32), lo0 = static_cast<std::uint32_t>(p0);
        const auto hi1 = static_cast<std::uint32_t>(p1 >> 32), lo1 = static_cast<std::uint32_t>(p1);
        return {hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0};
    }

    // 10 rounds; the key is bumped by the Weyl constants between rounds.
    static constexpr ctr_t generate(ctr_t c, key_t k) {
        for (int i = 0; i < 10; ++i) {
            if (i > 0) { k[0] += W0; k[1] += W1; }
            c = round(c, k);
        }
        return c;
    }
};

// Map a 32-bit integer to a double strictly inside (0,1):
//   u = (x + 0.5) / 2^32   ->   min 1.16e-10, max 1 - 1.16e-10.
// Never exactly 0, so log(u) in Box-Muller is always finite.
constexpr double u32_to_open01(std::uint32_t x) {
    return (static_cast<double>(x) + 0.5) * 0x1.0p-32;
}

// ---------------------------------------------------------------------------
// COUNTER LAYOUT (the CUDA / MPI teammates MUST replicate this exactly):
//
//   key     = { seed bits 0..31,  seed bits 32..63 }
//   counter = { counter_lo bits 0..31, counter_lo bits 32..63,
//               counter_hi bits 0..31, counter_hi bits 32..63 }
//
// For path generation (paths.cpp) we use
//   counter_lo = path_id
//   counter_hi = (option_index << 32) | block,   block = (t-1) / 4
// so one Philox call gives the 4 normals for steps t = 4*block+1 .. 4*block+4
// of one path of one option.
// ---------------------------------------------------------------------------
inline std::array<double, 4> uniform4(std::uint64_t seed, std::uint64_t counter_hi,
                                      std::uint64_t counter_lo) {
    const Philox4x32::key_t key = {static_cast<std::uint32_t>(seed),
                                   static_cast<std::uint32_t>(seed >> 32)};
    const Philox4x32::ctr_t ctr = {
        static_cast<std::uint32_t>(counter_lo), static_cast<std::uint32_t>(counter_lo >> 32),
        static_cast<std::uint32_t>(counter_hi), static_cast<std::uint32_t>(counter_hi >> 32)};
    const auto r = Philox4x32::generate(ctr, key);
    return {u32_to_open01(r[0]), u32_to_open01(r[1]), u32_to_open01(r[2]), u32_to_open01(r[3])};
}

// Standard normals via Box-Muller: 4 uniforms -> 4 independent N(0,1).
//   (u0,u1) -> R = sqrt(-2 ln u0), z0 = R cos(2 pi u1), z1 = R sin(2 pi u1)
//   (u2,u3) -> same for z2, z3
// Uses std::cos/std::sin separately (not sincos) so results are the same
// on every platform that has a correctly rounded-ish libm; GPU results may
// differ in the last ulp - compare with a tolerance there.
inline std::array<double, 4> normal4(std::uint64_t seed, std::uint64_t counter_hi,
                                     std::uint64_t counter_lo) {
    const auto u = uniform4(seed, counter_hi, counter_lo);
    constexpr double two_pi = 2.0 * std::numbers::pi;
    const double r0 = std::sqrt(-2.0 * std::log(u[0]));
    const double r1 = std::sqrt(-2.0 * std::log(u[2]));
    return {r0 * std::cos(two_pi * u[1]), r0 * std::sin(two_pi * u[1]),
            r1 * std::cos(two_pi * u[3]), r1 * std::sin(two_pi * u[3])};
}

}  // namespace lsm
