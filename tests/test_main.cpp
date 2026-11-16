// test_main.cpp - unit tests, plain asserts with readable output, no framework.
// Implements section 8 of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// Parallelism: test 4 (chunked RegressionSums merge) simulates the future
// parallel reduction; test 3 (determinism) is the property the parallel
// versions will be checked against.
// Exit code is non-zero if any check fails.

#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <print>
#include <string>
#include <vector>

#include "lsm/config.hpp"
#include "lsm/lsm.hpp"
#include "lsm/paths.hpp"
#include "lsm/reference.hpp"
#include "lsm/regression.hpp"
#include "lsm/rng.hpp"
#include "lsm/types.hpp"

using namespace lsm;

namespace {

int g_failed = 0;
int g_passed = 0;

void check(bool cond, const std::string& name, const std::string& detail = "") {
    if (cond) {
        ++g_passed;
        std::println("[PASS] {}", name);
    } else {
        ++g_failed;
        std::println("[FAIL] {}{}{}", name, detail.empty() ? "" : ": ", detail);
    }
}

void check_near(double got, double expected, double tol, const std::string& name) {
    check(std::abs(got - expected) <= tol, name,
          std::format("expected {:.10g} got {:.10g} (|diff| {:.3g} > tol {:.3g})", expected, got,
                      std::abs(got - expected), tol));
}

OptionSpec ls_s36() { return {"LS_S36_v20_T1", OptionType::Put, 36, 40, 0.06, 0.20, 1.0}; }

// ---------------------------------------------------------------------------
// 8.1 Hand-worked toy example (Longstaff-Schwartz style), K=100, degree 1,
// step discount 0.9. Paths are injected directly - no RNG.
// ---------------------------------------------------------------------------
void test_toy_example() {
    const double P[5][4] = {{100, 90, 85, 80},
                            {100, 105, 80, 105},
                            {100, 95, 102, 110},
                            {100, 88, 90, 75},
                            {100, 110, 115, 108}};
    PathMatrix S;
    S.resize(4, 5);
    for (std::size_t p = 0; p < 5; ++p)
        for (std::size_t t = 0; t < 4; ++t) S.at(t, p) = static_cast<path_t>(P[p][t]);
    const OptionSpec opt{"toy", OptionType::Put, 100, 100, 0.0, 0.2, 3.0};
    const double K = opt.K;
    const auto disc = make_discount_table_step(0.9, 3);

    // Step by step, to assert the intermediate decisions.
    LsmState st;
    init_at_expiry(S, opt, st);

    // t = 2
    auto s2 = accumulate_sums(S, opt, st, 2, disc);
    check(s2.n == 3, "toy t=2: ITM = {P1,P2,P4} (3 paths)", std::format("n = {}", s2.n));
    auto k2 = solve_normal_equations(s2, 1);
    // Fit is in X = S/K; the raw-S line is Y = a + (b/K) S.
    check_near(k2.a, -177.75, 1e-9, "toy t=2: raw-S intercept -177.75");
    check_near(k2.b / K, 2.25, 1e-9, "toy t=2: raw-S slope 2.25");
    apply_exercise(S, opt, k2, 2, st);
    check(st.exercise_step[0] == 2 && st.exercise_step[1] == 2 && st.exercise_step[3] == 3,
          "toy t=2: P1, P2 exercise; P4 waits");

    // t = 1
    auto s1 = accumulate_sums(S, opt, st, 1, disc);
    check(s1.n == 3, "toy t=1: ITM = {P1,P3,P4} (3 paths)", std::format("n = {}", s1.n));
    auto k1 = solve_normal_equations(s1, 1);
    check_near(k1.a, 271.13, 0.01, "toy t=1: raw-S intercept ~271.13");
    check_near(k1.b / K, -2.856, 0.001, "toy t=1: raw-S slope ~-2.856");
    apply_exercise(S, opt, k1, 1, st);
    check(st.exercise_step[0] == 2 && st.exercise_step[2] == 1 && st.exercise_step[3] == 3,
          "toy t=1: only P3 exercises");

    const double cf_exp[5] = {15, 20, 5, 25, 0};
    const std::uint32_t at_exp[5] = {2, 2, 1, 3, 3};
    bool cf_ok = true;
    for (int p = 0; p < 5; ++p)
        cf_ok &= (st.cashflow[p] == cf_exp[p]) && (p == 4 || st.exercise_step[p] == at_exp[p]);
    check(cf_ok, "toy final cash flows: P1=15@2 P2=20@2 P3=5@1 P4=25@3 P5=0");

    // Full pricer through the test hook.
    const auto res = price_lsm_with_discount(S, opt, 1, 1.0, 0.9);
    check_near(res.estimate.price, 10.215, 1e-9, "toy LSM price = 10.215");
}

// 8.2.1 Philox known-answer vectors (Random123 kat_vectors).
void test_philox_kat() {
    const auto a = Philox4x32::generate({0, 0, 0, 0}, {0, 0});
    check(a == Philox4x32::ctr_t{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u},
          "Philox4x32-10 KAT: counter 0, key 0",
          std::format("got {:08x} {:08x} {:08x} {:08x}", a[0], a[1], a[2], a[3]));
    const auto b = Philox4x32::generate({0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
                                        {0xffffffffu, 0xffffffffu});
    check(b == Philox4x32::ctr_t{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu},
          "Philox4x32-10 KAT: all 0xffffffff",
          std::format("got {:08x} {:08x} {:08x} {:08x}", b[0], b[1], b[2], b[3]));
    check(u32_to_open01(0) > 0.0 && u32_to_open01(0xffffffffu) < 1.0, "uniforms strictly inside (0,1)");
}

// 8.2.2 Normals: mean ~ 0, variance ~ 1 over 1e6 samples.
void test_normals() {
    const std::size_t n = 1'000'000;
    double s = 0, s2 = 0;
    for (std::uint64_t i = 0; i < n / 4; ++i) {
        const auto z = normal4(12345, 7, i);
        for (double v : z) { s += v; s2 += v * v; }
    }
    const double mean = s / n, var = s2 / n - mean * mean;
    check_near(mean, 0.0, 0.005, "normals: mean of 1e6 within 0.005 of 0");
    check_near(var, 1.0, 0.005, "normals: variance of 1e6 within 0.005 of 1");
}

// 8.2.3 Determinism: same seed -> bit-identical prices.
void test_determinism() {
    const auto opt = ls_s36();
    Config cfg;
    cfg.num_paths = 20'000;
    auto run = [&] {
        PathMatrix S;
        generate_paths(opt, cfg, 0, S);
        return price_lsm(S, opt, 2, opt.T / static_cast<double>(num_steps_for(opt, cfg)));
    };
    const auto r1 = run(), r2 = run();
    check(std::bit_cast<std::uint64_t>(r1.estimate.price) == std::bit_cast<std::uint64_t>(r2.estimate.price) &&
              std::bit_cast<std::uint64_t>(r1.estimate.std_error) == std::bit_cast<std::uint64_t>(r2.estimate.std_error),
          "determinism: two runs with the same seed are bit-identical");

    // A different option index must give different paths (RNG streams separate).
    PathMatrix A, B;
    generate_paths(opt, cfg, 0, A);
    generate_paths(opt, cfg, 1, B);
    check(A.at(10, 0) != B.at(10, 0), "different option_index -> different random stream");
}

// 8.2.4 RegressionSums merge: 1 chunk vs 7 chunks -> same coefficients.
// Data are dyadic rationals (x = k/64, y = j/64), so every sum is exact in
// double and the result must not depend on the summation order at all -
// this isolates the merge logic from rounding. (Real data: order changes
// only the last digits; parallel versions compare with a tolerance.)
void test_regression_merge() {
    std::vector<std::pair<double, double>> pts;
    for (int i = 0; i < 1000; ++i) {
        const double x = (48 + (i * 37) % 33) / 64.0;  // 0.75 .. 1.25
        const double y = ((i * 53) % 640) / 64.0;      // 0 .. 10
        pts.emplace_back(x, y);
    }
    RegressionSums one;
    for (auto [x, y] : pts) one.add(x, y);

    // 7 unequal chunks, merged with operator+= (what the reduction does).
    const std::size_t cuts[8] = {0, 13, 150, 151, 400, 777, 900, 1000};
    RegressionSums merged;
    for (int c = 0; c < 7; ++c) {
        RegressionSums part;
        for (std::size_t i = cuts[c]; i < cuts[c + 1]; ++i) part.add(pts[i].first, pts[i].second);
        merged += part;
    }
    const auto k1 = solve_normal_equations(one, 2), k7 = solve_normal_equations(merged, 2);
    auto rel = [](double a, double b) { return std::abs(a - b) / std::max(1.0, std::abs(a)); };
    check(k1.valid && k7.valid && rel(k1.a, k7.a) <= 1e-12 && rel(k1.b, k7.b) <= 1e-12 &&
              rel(k1.c, k7.c) <= 1e-12,
          "RegressionSums: 1 chunk vs 7 chunks give the same coefficients",
          std::format("({}, {}, {}) vs ({}, {}, {})", k1.a, k1.b, k1.c, k7.a, k7.b, k7.c));
}

// 8.2.5 Exact quadratic data is recovered.
void test_regression_exact() {
    RegressionSums s;
    for (int i = 0; i <= 20; ++i) {
        const double x = 0.5 + 0.05 * i;
        s.add(x, 1 + 2 * x + 3 * x * x);
    }
    const auto k = solve_normal_equations(s, 2);
    check(k.valid && std::abs(k.a - 1) < 1e-9 && std::abs(k.b - 2) < 1e-9 && std::abs(k.c - 3) < 1e-9,
          "solve_normal_equations recovers y = 1 + 2x + 3x^2",
          std::format("got ({}, {}, {})", k.a, k.b, k.c));

    RegressionSums few;
    few.add(1.0, 1.0);
    few.add(1.1, 2.0);
    check(!solve_normal_equations(few, 2).valid, "fewer than degree+1 points -> invalid");
    RegressionSums same;
    for (int i = 0; i < 10; ++i) same.add(1.0, i);
    check(!solve_normal_equations(same, 2).valid, "all X equal (singular) -> invalid");
}

// 8.2.6 Binomial American put and call.
void test_binomial() {
    check_near(binomial_american(ls_s36(), 2000), 4.4867, 0.003, "binomial LS_S36_v20_T1 (2000 steps)");
    const OptionSpec call{"call", OptionType::Call, 100, 100, 0.05, 0.20, 1.0};
    check_near(binomial_american(call, 2000), 10.4506, 0.003,
               "binomial American call (no dividends) = 10.4506");
}

// Extra: LSM on the paper's first row; plus deep-ITM t=0 rule.
void test_lsm_and_t0() {
    const auto opt = ls_s36();
    Config cfg;  // 100k paths, 50 steps/year
    PathMatrix S;
    generate_paths(opt, cfg, 0, S);
    const auto r = price_lsm(S, opt, 2, opt.T / static_cast<double>(num_steps_for(opt, cfg)));
    check(r.estimate.price > 4.45 && r.estimate.price < 4.50, "LSM LS_S36_v20_T1 in [4.45, 4.50]",
          std::format("got {:.5f}", r.estimate.price));

    const OptionSpec deep{"DeepITM_1y_put", OptionType::Put, 80, 100, 0.05, 0.20, 1.0};
    cfg.num_paths = 20'000;
    generate_paths(deep, cfg, 0, S);
    const auto d = price_lsm(S, deep, 2, deep.T / static_cast<double>(num_steps_for(deep, cfg)));
    check(d.exercise_at_t0 && d.estimate.price == 20.0, "deep ITM put: immediate exercise = 20.00",
          std::format("got {:.6f}", d.estimate.price));
}

// CLI and CSV smoke tests.
void test_cli() {
    const std::vector<std::string_view> a = {"--paths", "123", "--preset", "heavy"};
    auto c = parse_cli(a);
    check(c && c->num_paths == 123 && c->steps_per_year == 365, "CLI: explicit flag overrides preset");
    const std::vector<std::string_view> bad = {"--degree", "3"};
    check(!parse_cli(bad), "CLI: invalid --degree rejected");
    const OptionSpec hull{"Hull", OptionType::Put, 50, 50, 0.1, 0.4, 0.416667};
    check(num_steps_for(hull, Config{}) == 21, "M = round(50 * 5/12) = 21");
}

}  // namespace

int main() {
    test_toy_example();
    test_philox_kat();
    test_normals();
    test_determinism();
    test_regression_merge();
    test_regression_exact();
    test_binomial();
    test_lsm_and_t0();
    test_cli();
    std::println("\n{} passed, {} failed", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
