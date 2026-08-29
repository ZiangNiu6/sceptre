#ifndef SCEPTRE_CRT_SPA_FAST_H
#define SCEPTRE_CRT_SPA_FAST_H

#include <Rcpp.h>

namespace sceptre {

// Information-studentized CRT saddlepoint approximation with a fastSPA-style
// partial-normal CGF.  A fixed high-leverage block is evaluated exactly and
// the complementary low-leverage block is replaced by its joint Gaussian
// mean and covariance.  The partition never depends on observed treatment.
Rcpp::List crt_spa_full_fast(const Rcpp::NumericVector& a,
                             const Rcpp::NumericVector& w,
                             const Rcpp::NumericMatrix& Z,
                             const Rcpp::NumericVector& propensity,
                             double target,
                             int score_sign = 1,
                             double tolerance = 1e-6,
                             int max_iterations = 50);

// Outward-tail wrapper matching crt_spa_full_outward, but using the
// partial-normal CGF for the one selected root solve.
Rcpp::List crt_spa_full_outward_fast(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::NumericVector& propensity,
    const Rcpp::IntegerVector& treated_indices,
    double tolerance = 1e-6,
    int max_iterations = 50);

}  // namespace sceptre

#endif  // SCEPTRE_CRT_SPA_FAST_H
