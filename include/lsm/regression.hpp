// regression.hpp - least-squares fit Y ~ a + bX + cX^2 via the normal equations.
// Implements section 5.2 of LSM_SERIAL_BUILD_INSTRUCTIONS.md.
// Parallelism: RegressionSums is a mergeable value (operator+=). Merging
// per-thread / per-rank / per-block partial sums IS the parallel reduction.
// solve_normal_equations is tiny (3x3) and stays serial everywhere.
#pragma once

#include <type_traits>

namespace lsm {

// The 8 sums needed by the normal equations for Y ~ a + bX + cX^2.
// Designed as a mergeable value type: partial sums from different
// threads/ranks/blocks combine with operator+=. This IS the reduction.
// [PARALLEL-NOTE] MPI: send as double[8] with MPI_Allreduce(MPI_SUM).
// Keep the struct standard-layout and exactly 8 doubles (static_asserts below).
struct RegressionSums {
    double n = 0, sx = 0, sx2 = 0, sx3 = 0, sx4 = 0, sy = 0, sxy = 0, sx2y = 0;

    // Elementwise add of another partial sum.
    RegressionSums& operator+=(const RegressionSums& o) {
        n += o.n; sx += o.sx; sx2 += o.sx2; sx3 += o.sx3; sx4 += o.sx4;
        sy += o.sy; sxy += o.sxy; sx2y += o.sx2y;
        return *this;
    }

    // Add one data point (x, y).
    void add(double x, double y) {
        const double x2 = x * x;
        n += 1.0;
        sx += x;
        sx2 += x2;
        sx3 += x2 * x;
        sx4 += x2 * x2;
        sy += y;
        sxy += x * y;
        sx2y += x2 * y;
    }
};
static_assert(std::is_standard_layout_v<RegressionSums>);
static_assert(sizeof(RegressionSums) == 8 * sizeof(double));

// Regression coefficients; valid=false means "do not use this step".
struct Coeffs {
    double a = 0, b = 0, c = 0;
    bool valid = false;
};

// Builds the 3x3 (or 2x2 if degree==1) normal-equation system
//   | n    Sx   Sx2 |   |a|   | Sy   |
//   | Sx   Sx2  Sx3 | . |b| = | Sxy  |
//   | Sx2  Sx3  Sx4 |   |c|   | Sx2y |
// and solves it with Gaussian elimination + partial pivoting.
// Returns valid=false if there are fewer than (degree+1) points or a pivot
// is ~0 (|pivot| < 1e-12 * largest |matrix entry|). The caller then skips
// early exercise at this step.
Coeffs solve_normal_equations(const RegressionSums& s, int degree);

// Evaluate the fitted polynomial at x (c == 0 for degree 1).
inline double eval(const Coeffs& k, double x) { return k.a + k.b * x + k.c * x * x; }

}  // namespace lsm
