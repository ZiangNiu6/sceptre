#ifndef SCEPTRE_CRT_SPA_H
#define SCEPTRE_CRT_SPA_H

#include <Rcpp.h>

namespace sceptre {

// Exact Full Newton CRT saddlepoint approximation. This is the C++ entry
// point used by other package translation units; it does not cross R's .Call
// boundary.
Rcpp::List crt_spa_full(const Rcpp::NumericVector& a,
                        const Rcpp::NumericVector& w,
                        const Rcpp::NumericMatrix& Z,
                        const Rcpp::NumericVector& propensity,
                        double target,
                        int score_sign = 1,
                        double tolerance = 1e-5,
                        int max_iterations = 50);

// Full-Newton CRT-SPA with automatic outward-tail selection relative to the
// information statistic's untilted null center.  This evaluates the center
// directly and performs only the one root solve that is actually needed.
Rcpp::List crt_spa_full_outward(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::NumericVector& propensity,
    const Rcpp::IntegerVector& treated_indices,
    double tolerance = 1e-5,
    int max_iterations = 50);

}  // namespace sceptre

#endif  // SCEPTRE_CRT_SPA_H
