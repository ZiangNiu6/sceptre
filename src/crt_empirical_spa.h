#ifndef SCEPTRE_CRT_EMPIRICAL_SPA_H
#define SCEPTRE_CRT_EMPIRICAL_SPA_H

#include <Rcpp.h>

namespace sceptre {

// Exact two-moment Bernoulli saddlepoint solver for the empirically
// studentized CRT statistic.  This entry point is shared by package
// translation units and does not cross R's .Call boundary.
Rcpp::List crt_empirical_spa_full(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& propensity,
    double target,
    int score_sign = 1,
    double tolerance = 1e-9,
    int max_iterations = 60,
    int max_backtracks = 24);

}  // namespace sceptre

#endif  // SCEPTRE_CRT_EMPIRICAL_SPA_H
