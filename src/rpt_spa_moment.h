#ifndef SCEPTRE_RPT_SPA_MOMENT_H
#define SCEPTRE_RPT_SPA_MOMENT_H

#include <Rcpp.h>

namespace sceptre {

// Native contexts are worker-local and must not be serialized between workers.
SEXP prepare_rpt_spa_response(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::LogicalVector& zero_mask,
    double minimum_zero_fraction = 0.8,
    int maximum_moment_degree = 2);

void release_rpt_spa_response(SEXP prepared_context);

Rcpp::List rpt_spa_moment_prepared(
    SEXP prepared_context, int m, double target, int score_sign = 1,
    double tolerance = 1e-4, double compressed_tolerance = 1e-4,
    int max_iterations = 50, int maximum_polish_updates = 3,
    double delta_limit = 0.06, double exact_audit_tolerance = 1e-4);

Rcpp::List rpt_spa_moment_outward_prepared(
    SEXP prepared_context, const Rcpp::IntegerVector& treated_indices,
    double tolerance = 1e-4, double compressed_tolerance = 1e-4,
    int max_iterations = 50, int maximum_polish_updates = 3,
    double delta_limit = 0.06, double exact_audit_tolerance = 1e-4);

}  // namespace sceptre

#endif  // SCEPTRE_RPT_SPA_MOMENT_H
