// Adaptive low-level SCEPTRE test using an empirically studentized CRT score
// in every observed and resampled branch.
//
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>

#include "crt_empirical_spa.h"
#include "shared_low_level_functions.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <vector>

using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::LogicalVector;
using Rcpp::Named;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

namespace {

constexpr double kTailTrigger = 0.02;
constexpr double kMinimumPValue = 1.0e-250;
constexpr const char* kStatisticId = "empirical_studentized_crt_v1";
constexpr const char* kEquationId = "crt_studentized_reduced_root_v1";
constexpr const char* kTailGeometry =
    "directional_tangent_halfspace_lugannani_rice";

struct SparseEmpiricalStatisticCache {
  int n = 0;
  std::vector<double> a;
  std::vector<double> d;
  double sc = 0.0;
  double qc = 0.0;
};

SparseEmpiricalStatisticCache build_statistic_cache(
    const NumericVector& a,
    const NumericVector& fitted_probabilities) {
  const int n = a.size();
  if (n < 2 || fitted_probabilities.size() != n) {
    Rcpp::stop(
        "a and fitted_probabilities must have the same length of at least two");
  }
  SparseEmpiricalStatisticCache cache;
  cache.n = n;
  cache.a.resize(n);
  cache.d.resize(n);
  double score_scale = 0.0;
  for (int i = 0; i < n; ++i) {
    const double ai = a[i];
    const double p = fitted_probabilities[i];
    if (!std::isfinite(ai)) Rcpp::stop("a must be finite");
    if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
      Rcpp::stop(
          "fitted_probabilities must be finite and lie in [0, 1]");
    }
    score_scale = std::max(score_scale, std::abs(ai));
  }
  // T is unchanged by a common positive scale.  Work in max-|a| units so
  // small but nonzero scores retain their exact studentized geometry.
  if (score_scale == 0.0) score_scale = 1.0;
  for (int i = 0; i < n; ++i) {
    const double ai = a[i] / score_scale;
    const double p = fitted_probabilities[i];
    const double a2 = ai * ai;
    const double di = (1.0 - 2.0 * p) * a2;
    if (!std::isfinite(a2) || !std::isfinite(di)) {
      Rcpp::stop("squared score coefficients must be finite");
    }
    cache.a[i] = ai;
    cache.d[i] = di;
    cache.sc -= p * ai;
    cache.qc += p * p * a2;
  }
  if (!std::isfinite(cache.sc) || !std::isfinite(cache.qc)) {
    Rcpp::stop("empirical CRT sparse-cache constants must be finite");
  }
  return cache;
}

double statistic_from_zero_based_indices(
    const SparseEmpiricalStatisticCache& cache,
    const std::vector<int>& treated_indices) {
  double s = cache.sc;
  double q = cache.qc;
  for (std::size_t j = 0; j < treated_indices.size(); ++j) {
    const int index = treated_indices[j];
    if (index < 0 || index >= cache.n) {
      Rcpp::stop("a treated index lies outside the score-vector bounds");
    }
    s += cache.a[index];
    q += cache.d[index];
  }
  const double mean_correction = s * s / static_cast<double>(cache.n);
  const double variance = q - mean_correction;
  const double variance_scale =
      std::max(std::abs(q), mean_correction);
  if (!std::isfinite(s) || !std::isfinite(q) ||
      !std::isfinite(variance) || !std::isfinite(variance_scale) ||
      variance_scale <= 0.0 || variance <= 1e-14 * variance_scale) {
    return NA_REAL;
  }
  const double statistic = s / std::sqrt(variance);
  return std::isfinite(statistic) ? statistic : NA_REAL;
}

std::vector<int> observed_zero_based_indices(
    const IntegerVector& trt_idxs) {
  std::vector<int> out(trt_idxs.size());
  for (R_xlen_t j = 0; j < trt_idxs.size(); ++j) {
    if (trt_idxs[j] == NA_INTEGER || trt_idxs[j] < 1) {
      Rcpp::stop("trt_idxs must contain non-missing, positive R indices");
    }
    out[j] = trt_idxs[j] - 1;
  }
  return out;
}

void validate_observed_indices(const std::vector<int>& treated_indices,
                               const int n) {
  std::vector<int> sorted = treated_indices;
  for (std::size_t j = 0; j < sorted.size(); ++j) {
    if (sorted[j] < 0 || sorted[j] >= n) {
      Rcpp::stop("a treated index lies outside the score-vector bounds");
    }
  }
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    Rcpp::stop("trt_idxs cannot contain duplicates");
  }
}

std::vector<double> null_statistics_from_bank(
    const SparseEmpiricalStatisticCache& cache,
    const int start_pos,
    const int B,
    SEXP synthetic_idxs) {
  if (start_pos < 0) Rcpp::stop("start_pos must be nonnegative");
  if (B < 0) Rcpp::stop("B must be nonnegative");
  if (TYPEOF(synthetic_idxs) != EXTPTRSXP) {
    Rcpp::stop("synthetic_idxs must be an external-pointer resample bank");
  }
  Rcpp::XPtr<std::vector<std::vector<int>>> bank(synthetic_idxs);
  if (bank.get() == NULL) Rcpp::stop("synthetic_idxs is a null pointer");
  const std::size_t start = static_cast<std::size_t>(start_pos);
  const std::size_t count = static_cast<std::size_t>(B);
  if (start > bank->size() || count > bank->size() - start) {
    Rcpp::stop("the requested resample range exceeds synthetic_idxs");
  }

  std::vector<double> out(B);
  for (int j = 0; j < B; ++j) {
    // No address is taken from the treatment vector.  Empty Bernoulli draws
    // therefore use (sc, qc) safely rather than invoking legacy &v[0] UB.
    out[j] = statistic_from_zero_based_indices(
        cache, (*bank)[start + static_cast<std::size_t>(j)]);
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
  const double raw = compute_empirical_p_value(values, observed, side_code);
  if (!std::isfinite(raw)) return NA_REAL;
  // The legacy two-sided helper is 2*min(left,right) and can exceed one.
  return std::max(0.0, std::min(1.0, raw));
}

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

NumericVector list_numeric_vector_or_na(const List& values,
                                        const char* name,
                                        const int size) {
  NumericVector out(size, NA_REAL);
  if (!values.containsElementNamed(name)) return out;
  SEXP value = values[name];
  if (Rf_isNull(value) || Rf_length(value) != size) return out;
  return Rcpp::as<NumericVector>(value);
}

NumericMatrix list_numeric_matrix_or_na(const List& values,
                                        const char* name,
                                        const int nrow,
                                        const int ncol) {
  NumericMatrix out(nrow, ncol);
  std::fill(out.begin(), out.end(), NA_REAL);
  if (!values.containsElementNamed(name)) return out;
  SEXP value = values[name];
  if (Rf_isNull(value) || !Rf_isMatrix(value)) return out;
  NumericMatrix candidate(value);
  if (candidate.nrow() != nrow || candidate.ncol() != ncol) return out;
  return candidate;
}

}  // namespace

// Transparent observed-statistic test hook.  trt_idxs uses ordinary R
// one-based indices, unlike the zero-based entries in synthetic_idxs.
// [[Rcpp::export]]
double compute_observed_empirical_crt_statistic_v1(
    const NumericVector& a,
    const NumericVector& fitted_probabilities,
    const IntegerVector& trt_idxs) {
  const SparseEmpiricalStatisticCache cache =
      build_statistic_cache(a, fitted_probabilities);
  const std::vector<int> treated = observed_zero_based_indices(trt_idxs);
  validate_observed_indices(treated, cache.n);
  return statistic_from_zero_based_indices(cache, treated);
}

// Transparent resampling test hook.  The bank is generated by SCEPTRE's CRT
// index sampler and therefore stores zero-based cell indices.
// [[Rcpp::export]]
std::vector<double> compute_null_empirical_crt_statistics_v1(
    const NumericVector& a,
    const NumericVector& fitted_probabilities,
    const int start_pos,
    const int B,
    SEXP synthetic_idxs) {
  const SparseEmpiricalStatisticCache cache =
      build_statistic_cache(a, fitted_probabilities);
  return null_statistics_from_bank(cache, start_pos, B, synthetic_idxs);
}

// Adaptive empirically studentized CRT-SPA.  B1 is the central empirical
// bank.  A p-value at or below 0.02 triggers the exact full 2D saddle solve;
// any solver or directional-LR failure selects the independent B2 bank.
// [[Rcpp::export]]
SEXP run_low_level_test_full_crt_spa_empirical_v1(
    NumericVector y,
    NumericVector mu,
    NumericVector a,
    NumericVector fitted_probabilities,
    IntegerVector trt_idxs,
    int n_trt,
    SEXP synthetic_idxs,
    int B1,
    int B2,
    bool return_resampling_dist,
    int side_code,
    int max_iterations = 60) {
  if (B1 <= 0 || B2 <= 0) Rcpp::stop("B1 and B2 must be positive");
  if (side_code < -1 || side_code > 1) {
    Rcpp::stop("side_code must be -1, 0, or 1");
  }
  if (max_iterations < 0) {
    Rcpp::stop("max_iterations must be nonnegative");
  }
  if (y.size() != a.size() || mu.size() != a.size() ||
      fitted_probabilities.size() != a.size()) {
    Rcpp::stop("y, mu, a, and fitted_probabilities must have equal length");
  }
  if (n_trt < 0 || n_trt != trt_idxs.size()) {
    Rcpp::stop("n_trt must equal length(trt_idxs) and be nonnegative");
  }

  // Build and validate once; all observed/B1/B2 statistics use this exact
  // same sparse representation of R_i = a_i (x_i - p_i).
  const SparseEmpiricalStatisticCache statistic_cache =
      build_statistic_cache(a, fitted_probabilities);
  const std::vector<int> observed_indices =
      observed_zero_based_indices(trt_idxs);
  validate_observed_indices(observed_indices, statistic_cache.n);
  const double z_orig =
      statistic_from_zero_based_indices(statistic_cache, observed_indices);
  const double lfc = n_trt == 0
                         ? NA_REAL
                         : estimate_log_fold_change_v2(
                               y, mu, trt_idxs, n_trt);

  std::vector<double> null_statistics = null_statistics_from_bank(
      statistic_cache, 0, B1, synthetic_idxs);
  const bool observed_studentizer_valid = std::isfinite(z_orig);
  const bool B1_studentizers_valid = all_finite(null_statistics);
  double p = observed_studentizer_valid && B1_studentizers_valid
                 ? empirical_p_value_clamped(
                       null_statistics, z_orig, side_code)
                 : NA_REAL;
  int stage = 1;
  std::string p_value_source =
      !observed_studentizer_valid ? "invalid_observed_studentizer" :
      (!B1_studentizers_valid ? "invalid_B1_studentizer" : "B1_empirical");

  bool spa_attempted = false;
  LogicalVector spa_converged = LogicalVector::create(NA_LOGICAL);
  std::string spa_reason =
      !observed_studentizer_valid ? "invalid_observed_studentizer" :
      (!B1_studentizers_valid ? "invalid_B1_studentizer" : "not_attempted");
  int spa_iterations = NA_INTEGER;
  int spa_evaluations = NA_INTEGER;
  double spa_max_residual = NA_REAL;
  double spa_K = NA_REAL;
  double spa_rate = NA_REAL;
  double spa_r_lr = NA_REAL;
  double spa_q_lr = NA_REAL;
  double spa_target = NA_REAL;
  int spa_score_sign = NA_INTEGER;
  NumericVector spa_theta(2, NA_REAL);
  NumericVector spa_moment(2, NA_REAL);
  NumericVector spa_outer_residual(2, NA_REAL);
  NumericMatrix spa_hessian(2, 2);
  std::fill(spa_hessian.begin(), spa_hessian.end(), NA_REAL);
  List spa_diagnostics = List::create(
      Named("converged") = LogicalVector::create(NA_LOGICAL),
      Named("reason") = spa_reason);

  if (std::isfinite(p) && p <= kTailTrigger) {
    spa_attempted = true;
    int score_sign = side_code == -1 ? -1 : 1;
    double sidedness_multiplier = 1.0;
    if (side_code == 0) {
      const double p_left = empirical_p_value_clamped(
          null_statistics, z_orig, -1);
      const double p_right = empirical_p_value_clamped(
          null_statistics, z_orig, 1);
      if (p_right < p_left) {
        score_sign = 1;
      } else if (p_left < p_right) {
        score_sign = -1;
      } else {
        score_sign = z_orig >= 0.0 ? 1 : -1;
      }
      sidedness_multiplier = 2.0;
    }
    const double target = static_cast<double>(score_sign) * z_orig;
    spa_target = target;
    spa_score_sign = score_sign;

    bool converged = false;
    try {
      spa_diagnostics = sceptre::crt_empirical_spa_full(
          a, fitted_probabilities, target, score_sign, 1.0e-9,
          max_iterations, 24);
      converged = list_bool_or_false(spa_diagnostics, "converged");
      spa_converged[0] = converged;
      spa_reason = list_string_or(
          spa_diagnostics, "reason", "unknown_failure");
      spa_iterations = list_int_or_na(spa_diagnostics, "iterations");
      spa_evaluations = list_int_or_na(spa_diagnostics, "evaluations");
      spa_max_residual =
          list_double_or_na(spa_diagnostics, "max_residual");
      spa_K = list_double_or_na(spa_diagnostics, "K");
      spa_rate = list_double_or_na(spa_diagnostics, "rate");
      spa_r_lr = list_double_or_na(spa_diagnostics, "r_lr");
      spa_q_lr = list_double_or_na(spa_diagnostics, "q_lr");
      spa_theta = list_numeric_vector_or_na(
          spa_diagnostics, "theta", 2);
      spa_moment = list_numeric_vector_or_na(
          spa_diagnostics, "moment", 2);
      spa_outer_residual = list_numeric_vector_or_na(
          spa_diagnostics, "outer_residual", 2);
      spa_hessian = list_numeric_matrix_or_na(
          spa_diagnostics, "hessian", 2, 2);
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

    const double spa_p =
        list_double_or_na(spa_diagnostics, "p_value");
    if (converged && std::isfinite(spa_p) &&
        spa_p >= 0.0 && spa_p <= 1.0) {
      p = std::max(
          kMinimumPValue,
          std::min(1.0, sidedness_multiplier * spa_p));
      stage = 2;
      p_value_source = "crt_spa_empirical_directional";
    } else {
      null_statistics = null_statistics_from_bank(
          statistic_cache, B1, B2, synthetic_idxs);
      if (all_finite(null_statistics)) {
        p = empirical_p_value_clamped(
            null_statistics, z_orig, side_code);
      } else {
        p = NA_REAL;
        spa_reason = "invalid_B2_studentizer";
      }
      stage = 3;
      p_value_source = std::isfinite(p)
                           ? "B2_empirical"
                           : "invalid_B2_studentizer";
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
      Named("statistic_id") = kStatisticId,
      Named("equation_id") = kEquationId,
      Named("outer_dimension") = 2,
      Named("spa_tail_geometry") = kTailGeometry,
      Named("spa_experimental") = true,
      Named("spa_attempted") = spa_attempted,
      Named("spa_converged") = spa_converged,
      Named("spa_reason") = spa_reason,
      Named("spa_iterations") = spa_iterations,
      Named("spa_evaluations") = spa_evaluations,
      Named("spa_max_residual") = spa_max_residual,
      Named("spa_K") = spa_K,
      Named("spa_rate") = spa_rate,
      Named("spa_r_lr") = spa_r_lr,
      Named("spa_q_lr") = spa_q_lr,
      Named("spa_target") = spa_target,
      Named("spa_score_sign") = spa_score_sign,
      Named("spa_theta") = spa_theta,
      Named("spa_moment") = spa_moment,
      Named("spa_hessian") = spa_hessian,
      Named("spa_outer_residual") = spa_outer_residual,
      Named("spa_diagnostics") = spa_diagnostics);
  if (return_resampling_dist) {
    // This is the bank actually selected by the adaptive path: B1 for stage
    // 1 or a successful SPA stage, and B2 after a forced/organic fallback.
    out["resampling_dist"] = null_statistics;
  }
  return out;
}

// SPA-first empirically studentized CRT.  Unlike the adaptive B1 wrapper
// above, this entry point receives no resample bank and attempts the exact
// full-Newton saddlepoint calculation for every valid, noncentral observed
// statistic.  A failed attempt is returned as data (`needs_empirical_fallback`
// is true), allowing the R orchestration layer to construct one shared B2
// bank lazily only when at least one pair in a group needs it.
//
// For a nonzero observed statistic, the solver always works in its outward
// direction: score_sign is sign(z_orig) and target is abs(z_orig).  The
// opposite one-sided tail is obtained by complementation and the two-sided
// value is twice the smaller tail.  This keeps every solve on the regular
// positive-target branch without changing the requested target.  Exactly
// zero has no regular signed LR root, so it requests empirical fallback; no
// Gaussian or Taylor shortcut is used.
// [[Rcpp::export]]
SEXP run_low_level_test_full_crt_spa_empirical_always_v1(
    NumericVector y,
    NumericVector mu,
    NumericVector a,
    NumericVector fitted_probabilities,
    IntegerVector trt_idxs,
    int n_trt,
    int side_code,
    int max_iterations = 60) {
  if (side_code < -1 || side_code > 1) {
    Rcpp::stop("side_code must be -1, 0, or 1");
  }
  if (max_iterations < 0) {
    Rcpp::stop("max_iterations must be nonnegative");
  }
  if (y.size() != a.size() || mu.size() != a.size() ||
      fitted_probabilities.size() != a.size()) {
    Rcpp::stop("y, mu, a, and fitted_probabilities must have equal length");
  }
  if (n_trt < 0 || n_trt != trt_idxs.size()) {
    Rcpp::stop("n_trt must equal length(trt_idxs) and be nonnegative");
  }

  const SparseEmpiricalStatisticCache statistic_cache =
      build_statistic_cache(a, fitted_probabilities);
  const std::vector<int> observed_indices =
      observed_zero_based_indices(trt_idxs);
  validate_observed_indices(observed_indices, statistic_cache.n);
  const double z_orig =
      statistic_from_zero_based_indices(statistic_cache, observed_indices);
  const double lfc = n_trt == 0
                         ? NA_REAL
                         : estimate_log_fold_change_v2(
                               y, mu, trt_idxs, n_trt);

  double p = NA_REAL;
  std::string p_value_source = std::isfinite(z_orig)
                                   ? "B2_empirical_pending"
                                   : "invalid_observed_studentizer";
  // No resampling distribution can repair an undefined observed statistic.
  // Reserve the lazy-bank request for finite targets whose SPA calculation
  // is central, nonregular, or otherwise unsuccessful.
  bool needs_empirical_fallback = std::isfinite(z_orig);
  const bool spa_attempted = true;
  bool spa_root_attempted = false;
  LogicalVector spa_converged = LogicalVector::create(false);
  std::string spa_reason = "invalid_observed_studentizer";
  int spa_iterations = NA_INTEGER;
  int spa_evaluations = NA_INTEGER;
  double spa_max_residual = NA_REAL;
  double spa_K = NA_REAL;
  double spa_rate = NA_REAL;
  double spa_r_lr = NA_REAL;
  double spa_q_lr = NA_REAL;
  double spa_target = NA_REAL;
  int spa_score_sign = NA_INTEGER;
  double spa_directional_upper_tail = NA_REAL;
  bool spa_tail_was_complemented = false;
  NumericVector spa_theta(2, NA_REAL);
  NumericVector spa_moment(2, NA_REAL);
  NumericVector spa_outer_residual(2, NA_REAL);
  NumericMatrix spa_hessian(2, 2);
  std::fill(spa_hessian.begin(), spa_hessian.end(), NA_REAL);
  List spa_diagnostics = List::create(
      Named("converged") = false,
      Named("root_converged") = false,
      Named("reason") = spa_reason,
      Named("statistic_id") = kStatisticId,
      Named("equation_id") = kEquationId,
      Named("spa_tail_geometry") = kTailGeometry,
      Named("spa_experimental") = true);

  if (std::isfinite(z_orig) && z_orig == 0.0) {
    // The signed-root LR expression is singular at r=u=0.  Returning a
    // structured fallback is preferable to silently changing the target or
    // switching to a different central approximation.
    spa_reason = "central_target_requires_empirical_fallback";
    spa_diagnostics = List::create(
        Named("converged") = false,
        Named("root_converged") = false,
        Named("reason") = spa_reason,
        Named("target") = 0.0,
        Named("statistic_id") = kStatisticId,
        Named("equation_id") = kEquationId,
        Named("path") = "full_exact_bernoulli_2d_newton",
        Named("solver") = "full_line_search_newton",
        Named("spa_tail_geometry") = kTailGeometry,
        Named("spa_experimental") = true,
        Named("spa_diagnostic") =
            "the exact central target has no regular signed LR root; "
            "empirical B2 fallback requested without a Gaussian/Taylor "
            "shortcut");
  } else if (std::isfinite(z_orig)) {
    spa_root_attempted = true;
    const int score_sign = z_orig > 0.0 ? 1 : -1;
    const double target = std::abs(z_orig);
    spa_target = target;
    spa_score_sign = score_sign;
    bool converged = false;
    try {
      spa_diagnostics = sceptre::crt_empirical_spa_full(
          a, fitted_probabilities, target, score_sign, 1.0e-9,
          max_iterations, 24);
      converged = list_bool_or_false(spa_diagnostics, "converged");
      spa_converged[0] = converged;
      spa_reason = list_string_or(
          spa_diagnostics, "reason", "unknown_failure");
      spa_iterations = list_int_or_na(spa_diagnostics, "iterations");
      spa_evaluations = list_int_or_na(spa_diagnostics, "evaluations");
      spa_max_residual =
          list_double_or_na(spa_diagnostics, "max_residual");
      spa_K = list_double_or_na(spa_diagnostics, "K");
      spa_rate = list_double_or_na(spa_diagnostics, "rate");
      spa_r_lr = list_double_or_na(spa_diagnostics, "r_lr");
      spa_q_lr = list_double_or_na(spa_diagnostics, "q_lr");
      spa_theta = list_numeric_vector_or_na(
          spa_diagnostics, "theta", 2);
      spa_moment = list_numeric_vector_or_na(
          spa_diagnostics, "moment", 2);
      spa_outer_residual = list_numeric_vector_or_na(
          spa_diagnostics, "outer_residual", 2);
      spa_hessian = list_numeric_matrix_or_na(
          spa_diagnostics, "hessian", 2, 2);
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

    const double directional_tail =
        list_double_or_na(spa_diagnostics, "p_value");
    if (converged && std::isfinite(directional_tail) &&
        directional_tail >= 0.0 && directional_tail <= 1.0) {
      spa_directional_upper_tail = directional_tail;
      const double outward_tail = std::max(
          0.0, std::min(1.0, directional_tail));
      const double inward_tail = 1.0 - outward_tail;
      double requested_p = NA_REAL;
      if (side_code == 0) {
        requested_p = 2.0 * std::min(outward_tail, inward_tail);
      } else {
        const bool requested_is_outward = side_code == score_sign;
        requested_p = requested_is_outward ? outward_tail : inward_tail;
        spa_tail_was_complemented = !requested_is_outward;
      }
      if (std::isfinite(requested_p)) {
        p = std::max(
            kMinimumPValue, std::min(1.0, requested_p));
        p_value_source = "crt_spa_empirical_always_directional";
        needs_empirical_fallback = false;
      }
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
      Named("needs_empirical_fallback") = needs_empirical_fallback,
      Named("statistic_id") = kStatisticId,
      Named("equation_id") = kEquationId,
      Named("outer_dimension") = 2,
      Named("spa_tail_geometry") = kTailGeometry,
      Named("spa_experimental") = true,
      Named("spa_attempted") = spa_attempted,
      Named("spa_root_attempted") = spa_root_attempted,
      Named("spa_converged") = spa_converged,
      Named("spa_reason") = spa_reason,
      Named("spa_iterations") = spa_iterations,
      Named("spa_evaluations") = spa_evaluations,
      Named("spa_max_residual") = spa_max_residual,
      Named("spa_K") = spa_K,
      Named("spa_rate") = spa_rate,
      Named("spa_r_lr") = spa_r_lr,
      Named("spa_q_lr") = spa_q_lr,
      Named("spa_target") = spa_target,
      Named("spa_score_sign") = spa_score_sign,
      Named("spa_directional_upper_tail") =
          spa_directional_upper_tail,
      Named("spa_tail_was_complemented") =
          spa_tail_was_complemented,
      Named("spa_theta") = spa_theta,
      Named("spa_moment") = spa_moment,
      Named("spa_hessian") = spa_hessian,
      Named("spa_outer_residual") = spa_outer_residual,
      Named("spa_diagnostics") = spa_diagnostics,
      // There is intentionally no empirical bank on the SPA path.  Keep an
      // explicit zero-length vector so output_amount = 3 can assemble a
      // rectangular result without interpreting a missing field as NULL.
      Named("resampling_dist") = NumericVector(0));
}

// Complete a failed SPA-first attempt using a lazily generated, shared B2
// bank.  The attempt result is cloned so its original SPA reason, state, and
// audit diagnostics survive the fallback unchanged.
// [[Rcpp::export]]
SEXP finalize_low_level_test_empirical_crt_fallback_v1(
    NumericVector a,
    NumericVector fitted_probabilities,
    SEXP synthetic_idxs,
    int B2,
    bool return_resampling_dist,
    int side_code,
    List spa_attempt_result) {
  if (B2 <= 0) Rcpp::stop("B2 must be positive");
  if (side_code < -1 || side_code > 1) {
    Rcpp::stop("side_code must be -1, 0, or 1");
  }
  if (!spa_attempt_result.containsElementNamed("z_orig")) {
    Rcpp::stop("spa_attempt_result must contain z_orig");
  }
  const double z_orig =
      list_double_or_na(spa_attempt_result, "z_orig");
  const SparseEmpiricalStatisticCache statistic_cache =
      build_statistic_cache(a, fitted_probabilities);
  const std::vector<double> null_statistics = null_statistics_from_bank(
      statistic_cache, 0, B2, synthetic_idxs);
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
