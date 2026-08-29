// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>

#include "crt_spa.h"
#include "crt_spa_fast.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
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
constexpr double kSpaRootTolerance = 1.0e-5;
constexpr const char* kStatisticId = "information_studentized_crt_v1";
constexpr const char* kEquationId = "crt_information_full_kkt_v1";
constexpr const char* kTailGeometry =
    "information_studentized_lugannani_rice";

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

NumericVector list_numeric_vector_or_empty(const List& values,
                                           const char* name) {
  if (!values.containsElementNamed(name)) return NumericVector(0);
  SEXP value = values[name];
  if (Rf_isNull(value) || !Rf_isNumeric(value)) return NumericVector(0);
  return Rcpp::as<NumericVector>(value);
}

struct SafeInformationStatisticCache {
  int n = 0;
  int p = 0;
  bool information_invertible = false;
  std::vector<double> a;
  std::vector<double> w;
  Eigen::MatrixXd weighted_design;
  Eigen::MatrixXd information_inverse;
};

SafeInformationStatisticCache build_safe_information_statistic_cache(
    const NumericVector& a,
    const NumericVector& w,
    const NumericMatrix& Z) {
  const int n = a.size();
  const int p = Z.ncol();
  if (n < 2 || p < 1 || w.size() != n || Z.nrow() != n) {
    stop("a, w, and Z have incompatible dimensions");
  }
  SafeInformationStatisticCache cache;
  cache.n = n;
  cache.p = p;
  cache.a.resize(n);
  cache.w.resize(n);
  cache.weighted_design.resize(n, p);
  Eigen::MatrixXd information = Eigen::MatrixXd::Zero(p, p);
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(a[i])) stop("a must be finite");
    if (!std::isfinite(w[i]) || w[i] <= 0.0) {
      stop("w must be positive and finite");
    }
    cache.a[i] = a[i];
    cache.w[i] = w[i];
    for (int j = 0; j < p; ++j) {
      if (!std::isfinite(Z(i, j))) stop("Z must be finite");
      const double weighted_value = w[i] * Z(i, j);
      if (!std::isfinite(weighted_value)) {
        stop("w multiplied by Z must be finite");
      }
      cache.weighted_design(i, j) = weighted_value;
    }
    for (int j = 0; j < p; ++j) {
      for (int k = 0; k <= j; ++k) {
        information(j, k) +=
            cache.weighted_design(i, j) * Z(i, k);
      }
    }
  }
  for (int j = 0; j < p; ++j) {
    for (int k = 0; k < j; ++k) information(k, j) = information(j, k);
  }
  bool invertible = true;
  try {
    Eigen::LLT<Eigen::MatrixXd> decomposition(information);
    if (decomposition.info() != Eigen::Success) {
      invertible = false;
    } else {
      cache.information_inverse = decomposition.solve(
          Eigen::MatrixXd::Identity(p, p));
      invertible = decomposition.info() == Eigen::Success &&
                   cache.information_inverse.allFinite();
    }
  } catch (...) {
    invertible = false;
  }
  if (!invertible) {
    cache.information_inverse = Eigen::MatrixXd::Constant(
        p, p, std::numeric_limits<double>::quiet_NaN());
  } else {
    cache.information_invertible = true;
  }
  return cache;
}

double safe_information_statistic(
    const SafeInformationStatisticCache& cache,
    const std::vector<int>& treated_indices) {
  if (treated_indices.empty()) return NA_REAL;
  if (!cache.information_invertible) return NA_REAL;
  double numerator = 0.0;
  double lower_left = 0.0;
  Eigen::VectorXd weighted_design_sum = Eigen::VectorXd::Zero(cache.p);
  for (std::size_t j = 0; j < treated_indices.size(); ++j) {
    const int index = treated_indices[j];
    if (index < 0 || index >= cache.n) return NA_REAL;
    numerator += cache.a[index];
    lower_left += cache.w[index];
    weighted_design_sum += cache.weighted_design.row(index).transpose();
  }
  const double projection = weighted_design_sum.dot(
      cache.information_inverse * weighted_design_sum);
  const double variance = lower_left - projection;
  const double variance_scale =
      std::max(std::abs(lower_left), std::abs(projection));
  if (!std::isfinite(numerator) || !std::isfinite(variance) ||
      !std::isfinite(variance_scale) || variance_scale <= 0.0 ||
      variance <= 1.0e-14 * variance_scale) {
    return NA_REAL;
  }
  const double statistic = numerator / std::sqrt(variance);
  return std::isfinite(statistic) ? statistic : NA_REAL;
}

std::vector<double> safe_null_information_statistics(
    const SafeInformationStatisticCache& cache,
    const int B,
    SEXP synthetic_idxs) {
  if (B < 0) stop("B must be nonnegative");
  if (TYPEOF(synthetic_idxs) != EXTPTRSXP) {
    stop("synthetic_idxs must be an external-pointer resample bank");
  }
  Rcpp::XPtr<std::vector<std::vector<int>>> bank(synthetic_idxs);
  if (bank.get() == NULL) stop("synthetic_idxs is a null pointer");
  if (static_cast<std::size_t>(B) > bank->size()) {
    stop("the requested B2 range exceeds synthetic_idxs");
  }
  std::vector<double> out(B);
  if (!cache.information_invertible) {
    std::fill(out.begin(), out.end(), NA_REAL);
    return out;
  }
  for (int j = 0; j < B; ++j) {
    out[j] = safe_information_statistic(cache, (*bank)[j]);
    if ((j & 1023) == 1023) Rcpp::checkUserInterrupt();
  }
  return out;
}

bool all_finite(const std::vector<double>& values) {
  for (std::size_t j = 0; j < values.size(); ++j) {
    if (!std::isfinite(values[j])) return false;
  }
  return true;
}

double empirical_p_value_clamped(const std::vector<double>& values,
                                 const double observed,
                                 const int side_code) {
  if (!std::isfinite(observed) || !all_finite(values)) return NA_REAL;
  const double p = compute_empirical_p_value(values, observed, side_code);
  if (!std::isfinite(p)) return NA_REAL;
  return std::max(0.0, std::min(1.0, p));
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
    int side_code,
    bool use_fast = false) {
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
  List spa_diagnostics = List::create(
      Named("converged") = LogicalVector::create(NA_LOGICAL),
      Named("reason") = spa_reason);

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

    bool converged = false;
    try {
      spa_diagnostics = use_fast
                       ? sceptre::crt_spa_full_fast(
                             a, w, y, Z, fitted_probabilities, target,
                             score_sign, kSpaRootTolerance, 50)
                       : sceptre::crt_spa_full(
                             a, w, Z, fitted_probabilities, target,
                             score_sign, kSpaRootTolerance, 50);
      converged = list_bool_or_false(spa_diagnostics, "converged");
      spa_converged[0] = converged;
      spa_reason = list_string_or(
          spa_diagnostics, "reason", "unknown_failure");
      spa_iterations = list_int_or_na(spa_diagnostics, "iterations");
      spa_max_residual = list_double_or_na(
          spa_diagnostics, "max_residual");
      spa_rate = list_double_or_na(spa_diagnostics, "rate");
      spa_r_lr = list_double_or_na(spa_diagnostics, "r_lr");
      spa_q_lr = list_double_or_na(spa_diagnostics, "q_lr");
    } catch (const std::exception& error) {
      spa_converged[0] = false;
      spa_reason = std::string("input_or_solver_error: ") + error.what();
      spa_diagnostics = List::create(
          Named("converged") = false,
          Named("reason") = spa_reason);
    } catch (...) {
      spa_converged[0] = false;
      spa_reason = "input_or_solver_error: unknown exception";
      spa_diagnostics = List::create(
          Named("converged") = false,
          Named("reason") = spa_reason);
    }

    const double spa_p = list_double_or_na(spa_diagnostics, "p_value");
    if (converged && std::isfinite(spa_p) && spa_p >= 0.0 && spa_p <= 1.0) {
      p = std::max(kMinimumPValue,
                   std::min(1.0, sidedness_multiplier * spa_p));
      stage = 2;
      p_value_source = use_fast ? "crt_spa_fast" : "crt_spa";
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
      Named("spa_fast") = use_fast,
      Named("spa_converged") = spa_converged,
      Named("spa_reason") = spa_reason,
      Named("spa_iterations") = spa_iterations,
      Named("spa_max_residual") = spa_max_residual,
      Named("spa_rate") = spa_rate,
      Named("spa_r_lr") = spa_r_lr,
      Named("spa_q_lr") = spa_q_lr,
      Named("spa_diagnostics") = spa_diagnostics);
  if (return_resampling_dist) out["resampling_dist"] = null_statistics;
  return out;
}

// SPA-first information-studentized CRT.  The exact observed statistic and
// the null center are computed from the same information geometry used by
// the Full Newton solver, so this path needs neither D nor a resample bank.
// The core selects the outward direction relative to the untilted center and
// performs exactly one root solve.  An unsuccessful finite attempt requests
// a lazily shared B2 bank from the R orchestration layer.
// [[Rcpp::export]]
SEXP run_low_level_test_full_crt_spa_always_v1(
    NumericVector y,
    NumericVector mu,
    NumericVector a,
    NumericVector w,
    NumericMatrix Z,
    NumericVector fitted_probabilities,
    IntegerVector trt_idxs,
    int n_trt,
    int side_code,
    int max_iterations = 50,
    bool use_fast = false) {
  if (side_code < -1 || side_code > 1) {
    stop("side_code must be -1, 0, or 1");
  }
  if (max_iterations < 0) {
    stop("max_iterations must be nonnegative");
  }
  if (y.size() != a.size() || mu.size() != a.size() ||
      w.size() != a.size() || Z.nrow() != a.size() ||
      fitted_probabilities.size() != a.size()) {
    stop("y, mu, a, w, Z, and fitted_probabilities have incompatible dimensions");
  }
  if (n_trt < 0 || n_trt != trt_idxs.size()) {
    stop("n_trt must equal length(trt_idxs) and be nonnegative");
  }
  std::vector<int> validated_indices(trt_idxs.size());
  for (R_xlen_t j = 0; j < trt_idxs.size(); ++j) {
    if (trt_idxs[j] == NA_INTEGER || trt_idxs[j] < 1 ||
        trt_idxs[j] > a.size()) {
      stop("trt_idxs must contain valid one-based indices");
    }
    validated_indices[j] = trt_idxs[j];
  }
  std::sort(validated_indices.begin(), validated_indices.end());
  if (std::adjacent_find(
          validated_indices.begin(), validated_indices.end()) !=
      validated_indices.end()) {
    stop("trt_idxs cannot contain duplicates");
  }

  const double lfc = n_trt == 0
                         ? NA_REAL
                         : estimate_log_fold_change_v2(
                               y, mu, trt_idxs, n_trt);
  List spa_diagnostics;
  try {
    spa_diagnostics = use_fast
                          ? sceptre::crt_spa_full_outward_fast(
                                a, w, y, Z, fitted_probabilities, trt_idxs,
                                kSpaRootTolerance, max_iterations)
                          : sceptre::crt_spa_full_outward(
                                a, w, Z, fitted_probabilities, trt_idxs,
                                kSpaRootTolerance, max_iterations);
  } catch (const std::exception& error) {
    // Dimension/design errors are programming/input errors and should retain
    // their precise native message.  Regular solver failures are already
    // returned structurally by crt_spa_full_outward().
    stop(std::string("input_or_solver_error: ") + error.what());
  }

  const double z_orig =
      list_double_or_na(spa_diagnostics, "observed_target");
  const bool observed_valid = std::isfinite(z_orig);
  const bool converged = list_bool_or_false(spa_diagnostics, "converged");
  const std::string spa_reason = list_string_or(
      spa_diagnostics, "reason", "unknown_failure");
  const int score_sign = list_int_or_na(
      spa_diagnostics, "outward_score_sign");
  const double directional_tail = list_double_or_na(
      spa_diagnostics, "p_value");

  double p = NA_REAL;
  const bool recoverable_geometry_failure =
      spa_reason == "singular_information_geometry";
  bool needs_empirical_fallback =
      observed_valid || recoverable_geometry_failure;
  std::string p_value_source = needs_empirical_fallback
                                   ? "B2_empirical_pending"
                                   : "invalid_observed_studentizer";
  bool tail_was_complemented = false;
  if (observed_valid && converged &&
      (score_sign == -1 || score_sign == 1) &&
      std::isfinite(directional_tail) && directional_tail >= 0.0 &&
      directional_tail <= 1.0) {
    const double outward_tail = std::max(
        0.0, std::min(1.0, directional_tail));
    const double inward_tail = 1.0 - outward_tail;
    double requested_p = NA_REAL;
    if (side_code == 0) {
      requested_p = 2.0 * std::min(outward_tail, inward_tail);
      tail_was_complemented = inward_tail < outward_tail;
    } else {
      const bool requested_is_outward = side_code == score_sign;
      requested_p = requested_is_outward ? outward_tail : inward_tail;
      tail_was_complemented = !requested_is_outward;
    }
    if (std::isfinite(requested_p)) {
      p = std::max(kMinimumPValue, std::min(1.0, requested_p));
      p_value_source = use_fast ? "crt_spa_always_fast"
                                : "crt_spa_always";
      needs_empirical_fallback = false;
    }
  }

  const NumericVector sn_params =
      NumericVector::create(NA_REAL, NA_REAL, NA_REAL);
  return List::create(
      Named("p") = p,
      Named("z_orig") = z_orig,
      Named("lfc") = lfc,
      Named("stage") = 2,
      Named("sn_params") = sn_params,
      Named("p_value_source") = p_value_source,
      Named("spa_fast") = use_fast,
      Named("needs_empirical_fallback") = needs_empirical_fallback,
      Named("statistic_id") = kStatisticId,
      Named("equation_id") = kEquationId,
      Named("outer_dimension") = Z.ncol() + 2,
      Named("spa_tail_geometry") = kTailGeometry,
      Named("spa_attempted") = true,
      Named("spa_root_attempted") =
          spa_reason != "central_target_requires_empirical_fallback" &&
          spa_reason != "invalid_observed_studentizer" &&
          spa_reason != "non_interior_propensity" &&
          spa_reason != "unsupported_spa_geometry_no_intercept" &&
          spa_reason != "singular_information_geometry",
      Named("spa_converged") = LogicalVector::create(converged),
      Named("spa_reason") = spa_reason,
      Named("spa_iterations") =
          list_int_or_na(spa_diagnostics, "iterations"),
      Named("spa_max_residual") =
          list_double_or_na(spa_diagnostics, "max_residual"),
      Named("spa_rate") = list_double_or_na(spa_diagnostics, "rate"),
      Named("spa_r_lr") = list_double_or_na(spa_diagnostics, "r_lr"),
      Named("spa_q_lr") = list_double_or_na(spa_diagnostics, "q_lr"),
      Named("spa_center") =
          list_double_or_na(spa_diagnostics, "base_center"),
      Named("spa_target") = list_double_or_na(spa_diagnostics, "target"),
      Named("spa_score_sign") = score_sign,
      Named("spa_directional_upper_tail") = directional_tail,
      Named("spa_tail_was_complemented") = tail_was_complemented,
      Named("spa_state") =
          list_numeric_vector_or_empty(spa_diagnostics, "state"),
      Named("spa_diagnostics") = spa_diagnostics,
      Named("resampling_dist") = NumericVector(0));
}

// Complete a failed information-SPA attempt against a lazily generated B2
// bank.  This scorer is independent of legacy compute_null_full_statistics:
// it checks pointer/range/index/variance validity and safely handles empty
// Bernoulli assignments without taking &v[0].
// [[Rcpp::export]]
SEXP finalize_low_level_test_crt_spa_fallback_v1(
    NumericVector a,
    NumericVector w,
    NumericMatrix Z,
    SEXP synthetic_idxs,
    int B2,
    bool return_resampling_dist,
    int side_code,
    List spa_attempt_result) {
  if (B2 <= 0) stop("B2 must be positive");
  if (side_code < -1 || side_code > 1) {
    stop("side_code must be -1, 0, or 1");
  }
  if (!spa_attempt_result.containsElementNamed("z_orig")) {
    stop("spa_attempt_result must contain z_orig");
  }
  const double z_orig =
      list_double_or_na(spa_attempt_result, "z_orig");
  const SafeInformationStatisticCache cache =
      build_safe_information_statistic_cache(a, w, Z);
  const std::vector<double> null_statistics =
      safe_null_information_statistics(cache, B2, synthetic_idxs);
  const bool B2_studentizers_valid = all_finite(null_statistics);
  const double p = std::isfinite(z_orig) && B2_studentizers_valid
                       ? empirical_p_value_clamped(
                             null_statistics, z_orig, side_code)
                       : NA_REAL;

  List out = Rcpp::clone(spa_attempt_result);
  out["p"] = p;
  out["stage"] = 3;
  out["p_value_source"] =
      std::isfinite(p) ? "B2_empirical" : "invalid_B2_studentizer";
  out["needs_empirical_fallback"] = false;
  out["empirical_fallback_reason"] =
      std::isfinite(p) ? "spa_failed_or_nonregular"
                       : (!std::isfinite(z_orig)
                              ? "invalid_observed_studentizer"
                              : "invalid_B2_studentizer");
  if (return_resampling_dist) {
    out["resampling_dist"] = null_statistics;
  }
  return out;
}
