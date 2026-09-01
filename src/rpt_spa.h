#ifndef SCEPTRE_RPT_SPA_H
#define SCEPTRE_RPT_SPA_H

#include <Rcpp.h>

namespace sceptre {

// Information-studentized saddlepoint approximation for a fixed-count
// randomization (permutation) test. The Bernoulli intercept tilt is profiled
// so that every evaluated saddlepoint has expected treatment count m.
Rcpp::List rpt_spa_full(const Rcpp::NumericVector& a,
                        const Rcpp::NumericVector& w,
                        const Rcpp::NumericMatrix& Z,
                        int m,
                        double target,
                        int score_sign = 1,
                        double tolerance = 1e-5,
                        int max_iterations = 50);

// Evaluate the observed information statistic, reflect a left-tail target,
// and solve only the outward fixed-count saddlepoint root.
Rcpp::List rpt_spa_full_outward(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::IntegerVector& treated_indices,
    double tolerance = 1e-5,
    int max_iterations = 50);

}  // namespace sceptre

#endif  // SCEPTRE_RPT_SPA_H
