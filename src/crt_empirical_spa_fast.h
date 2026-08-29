#ifndef SCEPTRE_CRT_EMPIRICAL_SPA_FAST_H
#define SCEPTRE_CRT_EMPIRICAL_SPA_FAST_H

#include <Rcpp.h>

namespace sceptre {

// Partial-normal fastSPA analogue for the empirically studentized CRT
// statistic. Positive-outcome rows remain exact and zero-outcome rows are
// replaced by their moment-matched joint Gaussian CGF unless promoted.
Rcpp::List crt_empirical_spa_full_fast(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& propensity,
    const Rcpp::NumericVector& y,
    double target,
    int score_sign = 1,
    double tolerance = 1e-5,
    int max_iterations = 60,
    int max_backtracks = 24);

}  // namespace sceptre

#endif  // SCEPTRE_CRT_EMPIRICAL_SPA_FAST_H
