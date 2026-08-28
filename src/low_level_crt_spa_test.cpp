#include "crt_spa.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>
#include <vector>

#include "shared_low_level_functions.h"

using namespace Rcpp;

// Implemented in low_level_full_test.cpp. Keeping the declarations here lets
// the CRT-SPA path reuse the exact native SCEPTRE statistic and resampling
// implementation without changing the legacy entry point.
double compute_observed_full_statistic_v2(NumericVector a,
                                          NumericVector w,
                                          NumericMatrix D,
                                          IntegerVector trt_idxs);

std::vector<double> compute_null_full_statistics(
    const NumericVector& a,
    const NumericVector& w,
    const NumericMatrix& D,
    int start_pos,
    int B,
    int n_trt,
    bool use_all_cells,
    SEXP synthetic_idxs);

namespace {

constexpr double kTailTrigger = 0.02;
constexpr double kMinimumPValue = 1.0e-250;

double list_double_or_na(const List& values, const char* name) {
  if (!values.containsElementNamed(name)) return NA_REAL;
  SEXP value = values[name];
  if (Rf_isNull(value) || Rf_length(value) != 1) return NA_REAL;
  return Rcpp::as<double>(value);
}

int list_int_or_na(const List& values, const char* name) {
  if (!values.containsElementNamed(name)) return NA_INTEGER;
  SEXP value = values[name];
  if (Rf_isNull(value) || Rf_length(value) != 1) return NA_INTEGER;
  return Rcpp::as<int>(value);
}

std::string list_string_or(const List& values,
                           const char* name,
                           const std::string& fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  SEXP value = values[name];
  if (Rf_isNull(value) || Rf_length(value) != 1) return fallback;
  return Rcpp::as<std::string>(value);
}

bool list_bool_or_false(const List& values, const char* name) {
  if (!values.containsElementNamed(name)) return false;
  SEXP value = values[name];
  if (Rf_isNull(value) || Rf_length(value) != 1) return false;
  return Rcpp::as<bool>(value);
}

}  // namespace

// Full Newton CRT-SPA version of SCEPTRE's low-level adaptive test.
//
// Stage 1 and the empirical fallback deliberately reuse SCEPTRE's existing
// resampling code. The only changed component is the opt-in stage-2 tail
// calculation.
//
// [[Rcpp::export]]
SEXP run_low_level_test_full_crt_spa_v1(
    NumericVector y,
    NumericVector mu,
    NumericVector a,
    NumericVector w,
    NumericMatrix D,
    NumericMatrix Z,
    NumericVector fitted_probabilities,
    IntegerVector trt_idxs,
    int n_trt,
    bool use_all_cells,
    SEXP synthetic_idxs,
    int B1,
    int B2,
    bool return_resampling_dist,
    int side_code) {
  if (B1 <= 0 || B2 <= 0) stop("B1 and B2 must be positive");
  if (side_code < -1 || side_code > 1) {
    stop("side_code must be -1, 0, or 1");
  }

  const double lfc = estimate_log_fold_change_v2(y, mu, trt_idxs, n_trt);
  const double z_orig =
      compute_observed_full_statistic_v2(a, w, D, trt_idxs);

  std::vector<double> null_statistics = compute_null_full_statistics(
      a, w, D, 0, B1, n_trt, use_all_cells, synthetic_idxs);
  double p = compute_empirical_p_value(null_statistics, z_orig, side_code);
  int stage = 1;
  std::string p_value_source = "B1_empirical";

  LogicalVector spa_converged = LogicalVector::create(NA_LOGICAL);
  std::string spa_reason = "not_attempted";
  int spa_iterations = NA_INTEGER;
  double spa_max_residual = NA_REAL;
  double spa_rate = NA_REAL;
  double spa_r_lr = NA_REAL;
  double spa_q_lr = NA_REAL;

  if (p <= kTailTrigger) {
    int score_sign = side_code == -1 ? -1 : 1;
    double target = score_sign * z_orig;
    double sidedness_multiplier = 1.0;

    if (side_code == 0) {
      const double p_left =
          compute_empirical_p_value(null_statistics, z_orig, -1);
      const double p_right =
          compute_empirical_p_value(null_statistics, z_orig, 1);
      score_sign = p_right <= p_left ? 1 : -1;
      target = score_sign * z_orig;
      sidedness_multiplier = 2.0;
    }

    List spa_result;
    bool converged = false;
    try {
      spa_result = sceptre::crt_spa_full(
          a, w, Z, fitted_probabilities, target, score_sign, 1.0e-9, 50);
      converged = list_bool_or_false(spa_result, "converged");
      spa_converged[0] = converged;
      spa_reason = list_string_or(spa_result, "reason", "unknown_failure");
      spa_iterations = list_int_or_na(spa_result, "iterations");
      spa_max_residual = list_double_or_na(spa_result, "max_residual");
      spa_rate = list_double_or_na(spa_result, "rate");
      spa_r_lr = list_double_or_na(spa_result, "r_lr");
      spa_q_lr = list_double_or_na(spa_result, "q_lr");
    } catch (const std::exception& error) {
      spa_converged[0] = false;
      spa_reason = std::string("input_or_solver_error: ") + error.what();
    } catch (...) {
      spa_converged[0] = false;
      spa_reason = "input_or_solver_error: unknown exception";
    }

    const double spa_p = list_double_or_na(spa_result, "p_value");
    if (converged && std::isfinite(spa_p) && spa_p >= 0.0 && spa_p <= 1.0) {
      p = std::max(kMinimumPValue,
                   std::min(1.0, sidedness_multiplier * spa_p));
      stage = 2;
      p_value_source = "crt_spa";
    } else {
      null_statistics = compute_null_full_statistics(
          a, w, D, B1, B2, n_trt, use_all_cells, synthetic_idxs);
      p = compute_empirical_p_value(null_statistics, z_orig, side_code);
      stage = 3;
      p_value_source = "B2_empirical";
    }
  }

  const NumericVector sn_params =
      NumericVector::create(NA_REAL, NA_REAL, NA_REAL);
  List out = List::create(
      Named("p") = p,
      Named("z_orig") = z_orig,
      Named("lfc") = lfc,
      Named("stage") = stage,
      Named("sn_params") = sn_params,
      Named("p_value_source") = p_value_source,
      Named("spa_converged") = spa_converged,
      Named("spa_reason") = spa_reason,
      Named("spa_iterations") = spa_iterations,
      Named("spa_max_residual") = spa_max_residual,
      Named("spa_rate") = spa_rate,
      Named("spa_r_lr") = spa_r_lr,
      Named("spa_q_lr") = spa_q_lr);
  if (return_resampling_dist) out["resampling_dist"] = null_statistics;
  return out;
}
