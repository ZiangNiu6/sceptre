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
                        double tolerance = 1e-9,
                        int max_iterations = 50);

}  // namespace sceptre

#endif  // SCEPTRE_CRT_SPA_H
