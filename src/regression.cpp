// regression.cpp - solve the least-squares normal equations (2x2 or 3x3).
// Parallelism: this solve is O(1) work per time step and stays serial in all
// versions; only the sums feeding it (lsm.cpp: accumulate_sums) are reduced
// in parallel.
//
// WHY we regress on X = S/K and not on S:
//   with S ~ 40, S^4 ~ 2.6e6 and the matrix entries span ~6 orders of
//   magnitude (n vs Sum S^4), which makes the normal equations badly
//   conditioned and wastes precision. With X = S/K ~ 1 every entry is of the
//   same order as n, so the 3x3 system is well scaled. The fitted function
//   (and hence every exercise decision) is mathematically the same.

#include "lsm/regression.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace lsm {

Coeffs solve_normal_equations(const RegressionSums& s, int degree) {
    Coeffs out;
    const int n = (degree >= 2) ? 3 : 2;  // number of unknowns
    if (s.n < static_cast<double>(n)) return out;  // too few points -> invalid

    // Augmented matrix [A | rhs] of the normal equations.
    std::array<std::array<double, 4>, 3> A{};
    if (n == 3) {
        A[0] = {s.n, s.sx, s.sx2, s.sy};
        A[1] = {s.sx, s.sx2, s.sx3, s.sxy};
        A[2] = {s.sx2, s.sx3, s.sx4, s.sx2y};
    } else {
        A[0] = {s.n, s.sx, s.sy, 0};
        A[1] = {s.sx, s.sx2, s.sxy, 0};
    }

    // Scale for the singularity test: largest |entry| of the coefficient matrix.
    double scale = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) scale = std::max(scale, std::abs(A[i][j]));
    const double tiny = 1e-12 * scale;

    // Gaussian elimination with partial pivoting.
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int i = col + 1; i < n; ++i)
            if (std::abs(A[i][col]) > std::abs(A[piv][col])) piv = i;
        if (!(std::abs(A[piv][col]) >= tiny) || A[piv][col] == 0.0) return out;  // singular
        if (piv != col) std::swap(A[piv], A[col]);
        for (int i = col + 1; i < n; ++i) {
            const double f = A[i][col] / A[col][col];
            for (int j = col; j <= n; ++j) A[i][j] -= f * A[col][j];
        }
    }

    // Back substitution.
    std::array<double, 3> x{};
    for (int i = n - 1; i >= 0; --i) {
        double v = A[i][n];
        for (int j = i + 1; j < n; ++j) v -= A[i][j] * x[j];
        x[i] = v / A[i][i];
    }
    out.a = x[0];
    out.b = x[1];
    out.c = (n == 3) ? x[2] : 0.0;
    out.valid = std::isfinite(out.a) && std::isfinite(out.b) && std::isfinite(out.c);
    return out;
}

}  // namespace lsm
