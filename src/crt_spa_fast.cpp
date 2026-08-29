// fastSPA-style partial-normal acceleration for SCEPTRE's
// information-studentized CRT statistic.
//
// Original GWAS fastSPA can retain the fixed, sparse variant carriers.  For
// the negative-binomial SCEPTRE score, the analogous fixed exception set is
// supplied by outcome sparsity: rows with Y > 0 are retained as Bernoulli
// variables and rows with Y = 0 form the moment-matched Gaussian bulk.  This
// is the fixed observed-outcome partition from the method writeup; it does not
// depend on observed treatment.  After a provisional root solve, zero rows
// whose target-direction tilt is too large may be promoted and the root is
// re-solved, preserving the safety behavior of the partial-normal solver.
//
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>

#include "crt_spa_fast.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr double kTinyV = 1e-12;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kBerryEsseenConstant = 0.56;
constexpr double kBerryEsseenThreshold = 0.25;
constexpr double kMaxBulkTiltThreshold = 0.25;
constexpr double kPromotionTieTolerance = 1.0e-12;
constexpr int kMaxPromotionRounds = 4;

double expit_stable(const double x) {
  if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
  const double ex = std::exp(x);
  return ex / (1.0 + ex);
}

double max_abs(const VectorXd& x) {
  return x.size() == 0 ? 0.0 : x.cwiseAbs().maxCoeff();
}

void symmetrize_lower(MatrixXd& matrix) {
  for (Eigen::Index j = 0; j < matrix.rows(); ++j) {
    for (Eigen::Index k = 0; k < j; ++k) matrix(k, j) = matrix(j, k);
  }
}

bool solve_linear(const MatrixXd& matrix,
                  const VectorXd& rhs,
                  VectorXd& solution) {
  if (!matrix.allFinite() || !rhs.allFinite() ||
      matrix.rows() != matrix.cols() || matrix.rows() != rhs.size()) {
    return false;
  }
  try {
    Eigen::FullPivLU<MatrixXd> decomposition(matrix);
    if (!decomposition.isInvertible()) return false;
    solution = decomposition.solve(rhs);
    return solution.allFinite();
  } catch (...) {
    return false;
  }
}

struct FastInfoCache {
  int n = 0;
  int p = 0;
  int d = 0;
  int score_sign = 1;
  bool leading_intercept = true;
  bool information_invertible = false;
  bool partition_valid = false;
  MatrixXd G;  // columns are signed a followed by w * Z
  MatrixXd C_inv;
  VectorXd offsets;
  VectorXd propensity;
  VectorXd weights;
  int initial_exact_count = 0;
  int initial_bulk_count = 0;
  std::vector<int> exact_indices;
  std::vector<int> bulk_indices;
  VectorXd bulk_mean;
  MatrixXd bulk_covariance;
};

struct Boundary {
  bool valid = false;
  double V = kNaN;
  double b = kNaN;
  VectorXd grad;
  MatrixXd hess;
};

struct Evaluation {
  bool valid = false;
  VectorXd residual;
  MatrixXd jacobian;
  VectorXd moment;
  MatrixXd Sgg;
  Boundary boundary;
};

struct ApproximationDiagnostic {
  bool valid = false;
  bool safe = false;
  double berry_esseen_bound = kNaN;
  double max_bulk_tilt = kNaN;
};

Boundary boundary_terms(const VectorXd& moment, const MatrixXd& C_inv) {
  Boundary out;
  const int d = static_cast<int>(moment.size());
  const int p = static_cast<int>(C_inv.rows());
  out.grad.setZero(d);
  out.hess.setZero(d, d);
  if (d != p + 1 || C_inv.cols() != p || !moment.allFinite() ||
      !C_inv.allFinite()) {
    return out;
  }

  const double U = moment[0];
  const VectorXd B = moment.segment(1, d - 1);
  const VectorXd beta = C_inv * B;
  const double V = B[0] - B.dot(beta);
  if (!std::isfinite(V) || V <= kTinyV) return out;

  VectorXd grad_V = VectorXd::Zero(d);
  grad_V.segment(1, d - 1) = -2.0 * beta;
  grad_V[1] += 1.0;
  MatrixXd hess_V = MatrixXd::Zero(d, d);
  hess_V.block(1, 1, d - 1, d - 1) = -2.0 * C_inv;
  VectorXd e_U = VectorXd::Zero(d);
  e_U[0] = 1.0;

  const double f = std::pow(V, -0.5);
  const double fp = -0.5 * std::pow(V, -1.5);
  const double fpp = 0.75 * std::pow(V, -2.5);
  out.grad = f * e_U + U * fp * grad_V;
  out.hess = fp * (e_U * grad_V.transpose() +
                   grad_V * e_U.transpose()) +
             U * (fpp * grad_V * grad_V.transpose() + fp * hess_V);
  out.valid = std::isfinite(U) && std::isfinite(V) &&
              out.grad.allFinite() && out.hess.allFinite();
  out.V = V;
  out.b = U / std::sqrt(V);
  return out;
}

Evaluation evaluate_partial_normal(const FastInfoCache& cache,
                                   const VectorXd& state,
                                   const double target) {
  Evaluation out;
  if (state.size() != cache.d + 1 || !state.allFinite() ||
      !cache.partition_valid) {
    return out;
  }
  const VectorXd theta = state.head(cache.d);
  const double lambda = state[cache.d];
  out.moment = cache.bulk_mean + cache.bulk_covariance * theta;
  out.Sgg = cache.bulk_covariance;

  for (const int i : cache.exact_indices) {
    // Eigen stores MatrixXd column-major.  Traverse the retained row with a
    // fixed n-stride, matching the allocation-free loop in the exact solver.
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < cache.d; ++j) {
      tilt += theta[j] * g[j * cache.n];
    }
    if (!std::isfinite(tilt)) return out;
    const double tilted_probability =
        expit_stable(cache.offsets[i] + tilt);
    const double variance =
        tilted_probability * (1.0 - tilted_probability);
    for (int j = 0; j < cache.d; ++j) {
      const double gj = g[j * cache.n];
      out.moment[j] += tilted_probability * gj;
      for (int k = 0; k <= j; ++k) {
        out.Sgg(j, k) += variance * gj * g[k * cache.n];
      }
    }
  }

  symmetrize_lower(out.Sgg);
  if (!out.moment.allFinite() || !out.Sgg.allFinite()) return out;
  out.boundary = boundary_terms(out.moment, cache.C_inv);
  if (!out.boundary.valid) return out;

  out.residual.setZero(cache.d + 1);
  out.residual.head(cache.d) = theta - lambda * out.boundary.grad;
  out.residual[cache.d] = out.boundary.b - target;
  out.jacobian.setZero(cache.d + 1, cache.d + 1);
  out.jacobian.block(0, 0, cache.d, cache.d) =
      MatrixXd::Identity(cache.d, cache.d) -
      lambda * out.boundary.hess * out.Sgg;
  out.jacobian.block(0, cache.d, cache.d, 1) = -out.boundary.grad;
  out.jacobian.block(cache.d, 0, 1, cache.d) =
      (out.Sgg * out.boundary.grad).transpose();
  out.valid = out.residual.allFinite() && out.jacobian.allFinite();
  return out;
}

double xlog1px_minus_x(const double x) {
  if (!std::isfinite(x) || x < -1.0 - 1.0e-14) return kNaN;
  if (x <= -1.0) return 1.0;
  if (std::abs(x) >= 1.0e-3) return (1.0 + x) * std::log1p(x) - x;
  double power = x * x;
  double value = 0.5 * power;
  for (int k = 3; k <= 12; ++k) {
    power *= x;
    const double coefficient =
        (k & 1) ? -1.0 / (static_cast<double>(k) * (k - 1.0))
                : 1.0 / (static_cast<double>(k) * (k - 1.0));
    value += coefficient * power;
  }
  return value;
}

double bernoulli_kl_from_tilt(const double p, const double tilt) {
  if (!std::isfinite(p) || !std::isfinite(tilt) || p < 0.0 || p > 1.0) {
    return kNaN;
  }
  if (p == 0.0 || p == 1.0 || tilt == 0.0) return 0.0;

  double delta = kNaN;
  if (tilt > 0.0) {
    const double exp_negative = std::exp(-tilt);
    const double denominator = p + (1.0 - p) * exp_negative;
    if (!std::isfinite(denominator) || denominator <= 0.0) return kNaN;
    delta = p / denominator - p;
  } else {
    const double expm1_tilt = std::expm1(tilt);
    const double denominator = 1.0 + p * expm1_tilt;
    if (!std::isfinite(denominator) || denominator <= 0.0) return kNaN;
    delta = p * (1.0 - p) * expm1_tilt / denominator;
  }
  if (tilt > 0.0 && tilt < 700.0) {
    const double expm1_tilt = std::expm1(tilt);
    const double denominator = 1.0 + p * expm1_tilt;
    delta = p * (1.0 - p) * expm1_tilt / denominator;
  }

  const double hx = xlog1px_minus_x(delta / p);
  const double hy = xlog1px_minus_x(-delta / (1.0 - p));
  if (!std::isfinite(hx) || !std::isfinite(hy)) return kNaN;
  const double kl = p * hx + (1.0 - p) * hy;
  if (!std::isfinite(kl)) return kNaN;
  return kl >= 0.0 ? kl : (kl > -1.0e-15 ? 0.0 : kNaN);
}

double partial_normal_rate(const FastInfoCache& cache,
                           const VectorXd& theta) {
  double rate = 0.5 * theta.dot(cache.bulk_covariance * theta);
  if (!std::isfinite(rate) || rate < -1.0e-12) return kNaN;
  rate = std::max(0.0, rate);
  for (const int i : cache.exact_indices) {
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < cache.d; ++j) {
      tilt += theta[j] * g[j * cache.n];
    }
    const double term = bernoulli_kl_from_tilt(cache.propensity[i], tilt);
    if (!std::isfinite(term)) return kNaN;
    rate += term;
  }
  return rate;
}

double lr_upper_tail(const double rate,
                     const double q2,
                     const double direction) {
  if (!std::isfinite(rate) || !std::isfinite(q2) || rate <= 0.0 ||
      q2 <= 0.0) {
    return kNaN;
  }
  const long double sign = direction < 0.0 ? -1.0L : 1.0L;
  const long double r =
      sign * std::sqrt(static_cast<long double>(2.0) * rate);
  const long double q = sign * std::sqrt(static_cast<long double>(q2));
  if (!std::isfinite(r) || !std::isfinite(q) || r == 0.0L || q == 0.0L) {
    return kNaN;
  }
  const double r_double = static_cast<double>(r);
  const double correction = static_cast<double>((r - q) / (q * r));
  if (!std::isfinite(r_double) || !std::isfinite(correction)) return kNaN;
  return R::pnorm(r_double, 0.0, 1.0, false, false) +
         R::dnorm(r_double, 0.0, 1.0, false) * correction;
}

ApproximationDiagnostic approximation_diagnostic(
    const FastInfoCache& cache,
    const VectorXd& theta) {
  ApproximationDiagnostic out;
  if (!cache.partition_valid || theta.size() != cache.d ||
      !theta.allFinite()) {
    return out;
  }
  // With no Gaussian block the partial-normal evaluator is the exact
  // Bernoulli evaluator, so approximation diagnostics pass vacuously.
  if (cache.bulk_indices.empty()) {
    out.valid = true;
    out.safe = true;
    out.berry_esseen_bound = 0.0;
    out.max_bulk_tilt = 0.0;
    return out;
  }
  const double norm = theta.norm();
  if (!std::isfinite(norm) || norm <= 1.0e-14) return out;
  const VectorXd direction = theta / norm;
  const double variance =
      direction.dot(cache.bulk_covariance * direction);
  if (!std::isfinite(variance) || variance <= 1.0e-14) return out;

  double third_absolute_moment = 0.0;
  double max_tilt = 0.0;
  for (const int i : cache.bulk_indices) {
    const double* g = cache.G.data() + i;
    double projected = 0.0;
    double tilt = 0.0;
    for (int j = 0; j < cache.d; ++j) {
      projected += direction[j] * g[j * cache.n];
      tilt += theta[j] * g[j * cache.n];
    }
    const double p = cache.propensity[i];
    const double centered_third =
        p * (1.0 - p) * (p * p + (1.0 - p) * (1.0 - p));
    third_absolute_moment +=
        centered_third * std::pow(std::abs(projected), 3.0);
    max_tilt = std::max(max_tilt, std::abs(tilt));
  }
  const double bound = kBerryEsseenConstant * third_absolute_moment /
                       std::pow(variance, 1.5);
  if (!std::isfinite(bound) || !std::isfinite(max_tilt)) return out;
  out.valid = true;
  out.berry_esseen_bound = bound;
  out.max_bulk_tilt = max_tilt;
  out.safe = bound <= kBerryEsseenThreshold &&
             max_tilt <= kMaxBulkTiltThreshold;
  return out;
}

void add_fast_metadata(Rcpp::List& out,
                       const FastInfoCache& cache,
                       const double tolerance,
                       const ApproximationDiagnostic& diagnostic) {
  out["path"] = "partial_normal_fast";
  out["cgf_evaluation"] =
      "exact_nonzero_outcomes_plus_moment_matched_gaussian_zero_bulk";
  out["partition_rule"] =
      "initial_E_y_positive_B_y_zero_then_target_tilt_promotion";
  out["initial_partition_rule"] = "E={Y>0};B={Y=0}";
  out["initial_exact_count"] = cache.initial_exact_count;
  out["initial_bulk_count"] = cache.initial_bulk_count;
  out["nonzero_outcome_count"] = cache.initial_exact_count;
  out["positive_outcome_count"] = cache.initial_exact_count;
  out["zero_outcome_count"] = cache.initial_bulk_count;
  out["initial_exact_fraction"] = cache.n > 0
      ? static_cast<double>(cache.initial_exact_count) / cache.n
      : kNaN;
  out["initial_bulk_fraction"] = cache.n > 0
      ? static_cast<double>(cache.initial_bulk_count) / cache.n
      : kNaN;
  out["exact_count"] = static_cast<int>(cache.exact_indices.size());
  out["bulk_count"] = static_cast<int>(cache.bulk_indices.size());
  out["exact_fraction"] = cache.n > 0
      ? static_cast<double>(cache.exact_indices.size()) / cache.n
      : kNaN;
  out["bulk_fraction"] = cache.n > 0
      ? static_cast<double>(cache.bulk_indices.size()) / cache.n
      : kNaN;
  out["berry_esseen_bound"] = diagnostic.berry_esseen_bound;
  out["berry_esseen_constant"] = kBerryEsseenConstant;
  out["berry_esseen_threshold"] = kBerryEsseenThreshold;
  out["max_bulk_tilt"] = diagnostic.max_bulk_tilt;
  out["max_bulk_tilt_threshold"] = kMaxBulkTiltThreshold;
  out["promotion_rounds"] = 0;
  out["promoted_count"] = 0;
  out["promotion_threshold"] = kMaxBulkTiltThreshold;
  out["maximum_promotion_rounds"] = kMaxPromotionRounds;
  out["promotion_stop_reason"] = "not_needed";
  out["minimum_bulk_fraction"] = 0.0;
  out["approximation_safe"] = diagnostic.valid && diagnostic.safe;
  out["tolerance"] = tolerance;
}

Rcpp::List failure_result(const FastInfoCache& cache,
                          const std::string& reason,
                          const double target,
                          const double center,
                          const int iterations,
                          const double max_residual,
                          const std::vector<double>& history,
                          const VectorXd& state,
                          const int cgf_evaluations,
                          const double tolerance) {
  Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("converged") = false,
      Rcpp::Named("reason") = reason,
      Rcpp::Named("p_value") = kNaN,
      Rcpp::Named("raw_p_value") = kNaN,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("max_residual") = max_residual,
      Rcpp::Named("history") = Rcpp::wrap(history),
      Rcpp::Named("state") = Rcpp::wrap(state),
      Rcpp::Named("center") = center,
      Rcpp::Named("target") = target,
      Rcpp::Named("rate") = kNaN,
      Rcpp::Named("r_lr") = kNaN,
      Rcpp::Named("q_lr") = kNaN,
      Rcpp::Named("moment_dimension") = cache.d,
      Rcpp::Named("nuisance_dimension") = cache.p,
      Rcpp::Named("score_sign") = cache.score_sign,
      Rcpp::Named("exact_evaluations") = cgf_evaluations,
      Rcpp::Named("cgf_evaluations") = cgf_evaluations);
  add_fast_metadata(out, cache, tolerance, ApproximationDiagnostic());
  return out;
}

Rcpp::List finalize_result(const FastInfoCache& cache,
                           const VectorXd& state,
                           const Evaluation& evaluation,
                           const double center,
                           const double target,
                           const int iterations,
                           const int cgf_evaluations,
                           const std::vector<double>& history,
                           const double tolerance) {
  const VectorXd theta = state.head(cache.d);
  const double rate = partial_normal_rate(cache, theta);
  const double q2 = theta.dot(evaluation.Sgg * theta);
  const double raw_p = lr_upper_tail(rate, q2, theta[0]);
  const double lambda = state[cache.d];
  const ApproximationDiagnostic diagnostic =
      approximation_diagnostic(cache, theta);
  const bool regular = target > center && lambda > 0.0 && theta[0] > 0.0 &&
                       rate > 0.0 && q2 > 0.0;
  const bool in_range = std::isfinite(raw_p) && raw_p >= -1.0e-10 &&
                        raw_p <= 1.0 + 1.0e-10;
  const bool safe = diagnostic.valid && diagnostic.safe;
  const bool valid = evaluation.valid && regular && in_range && safe;
  const double sign = theta[0] < 0.0 ? -1.0 : 1.0;
  const std::string reason =
      !regular ? "nonregular_upper_root" :
      (!in_range ? "lr_out_of_range" :
       (!diagnostic.valid ? "invalid_partial_normal_diagnostic" :
        (diagnostic.berry_esseen_bound > kBerryEsseenThreshold
             ? "partial_normal_berry_esseen_unsafe" :
         (diagnostic.max_bulk_tilt > kMaxBulkTiltThreshold
              ? "partial_normal_bulk_tilt_unsafe" : "ok"))));

  Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("converged") = valid,
      Rcpp::Named("reason") = reason,
      Rcpp::Named("p_value") =
          valid ? std::max(0.0, std::min(1.0, raw_p)) : kNaN,
      Rcpp::Named("raw_p_value") = raw_p,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("max_residual") = max_abs(evaluation.residual),
      Rcpp::Named("history") = Rcpp::wrap(history),
      Rcpp::Named("state") = Rcpp::wrap(state),
      Rcpp::Named("center") = center,
      Rcpp::Named("target") = target,
      Rcpp::Named("rate") = rate,
      Rcpp::Named("r_lr") =
          sign * std::sqrt(2.0 * std::max(rate, 0.0)),
      Rcpp::Named("q_lr") = sign * std::sqrt(std::max(q2, 0.0)),
      Rcpp::Named("moment_dimension") = cache.d,
      Rcpp::Named("nuisance_dimension") = cache.p,
      Rcpp::Named("score_sign") = cache.score_sign,
      Rcpp::Named("exact_evaluations") = cgf_evaluations,
      Rcpp::Named("cgf_evaluations") = cgf_evaluations);
  add_fast_metadata(out, cache, tolerance, diagnostic);
  return out;
}

bool create_outcome_partition(FastInfoCache& cache,
                              const Rcpp::NumericVector& y) {
  if (y.size() != cache.n) return false;
  std::vector<unsigned char> is_exact(cache.n, 0);
  cache.exact_indices.clear();
  cache.bulk_indices.clear();
  cache.exact_indices.reserve(cache.n);
  cache.bulk_indices.reserve(cache.n);
  for (int i = 0; i < cache.n; ++i) {
    if (!std::isfinite(y[i]) || y[i] < 0.0) {
      Rcpp::stop("y must be finite and nonnegative");
    }
    if (y[i] > 0.0) {
      is_exact[i] = 1;
      cache.exact_indices.push_back(i);
    }
  }
  cache.initial_exact_count = static_cast<int>(cache.exact_indices.size());
  cache.initial_bulk_count = cache.n - cache.initial_exact_count;

  cache.bulk_mean = VectorXd::Zero(cache.d);
  cache.bulk_covariance = MatrixXd::Zero(cache.d, cache.d);
  for (int i = 0; i < cache.n; ++i) {
    if (is_exact[i]) continue;
    cache.bulk_indices.push_back(i);
    const double p = cache.propensity[i];
    const double variance = p * (1.0 - p);
    const double* g = cache.G.data() + i;
    for (int j = 0; j < cache.d; ++j) {
      const double gj = g[j * cache.n];
      cache.bulk_mean[j] += p * gj;
      for (int k = 0; k <= j; ++k) {
        cache.bulk_covariance(j, k) +=
            variance * gj * g[k * cache.n];
      }
    }
  }
  symmetrize_lower(cache.bulk_covariance);
  if (!cache.bulk_mean.allFinite() ||
      !cache.bulk_covariance.allFinite()) {
    return false;
  }
  cache.partition_valid = true;
  return true;
}

enum class PromotionStatus {
  promoted,
  no_progress,
  invalid_partition
};

bool recompute_bulk_moments(FastInfoCache& cache) {
  std::sort(cache.exact_indices.begin(), cache.exact_indices.end());
  if (std::adjacent_find(cache.exact_indices.begin(),
                         cache.exact_indices.end()) !=
      cache.exact_indices.end()) {
    return false;
  }

  std::vector<unsigned char> is_exact(cache.n, 0);
  for (const int i : cache.exact_indices) {
    if (i < 0 || i >= cache.n) return false;
    is_exact[i] = 1;
  }

  cache.bulk_indices.clear();
  cache.bulk_indices.reserve(cache.n - cache.exact_indices.size());
  cache.bulk_mean = VectorXd::Zero(cache.d);
  cache.bulk_covariance = MatrixXd::Zero(cache.d, cache.d);
  for (int i = 0; i < cache.n; ++i) {
    if (is_exact[i]) continue;
    cache.bulk_indices.push_back(i);
    const double p = cache.propensity[i];
    const double variance = p * (1.0 - p);
    const double* g = cache.G.data() + i;
    for (int j = 0; j < cache.d; ++j) {
      const double gj = g[j * cache.n];
      cache.bulk_mean[j] += p * gj;
      for (int k = 0; k <= j; ++k) {
        cache.bulk_covariance(j, k) +=
            variance * gj * g[k * cache.n];
      }
    }
  }
  symmetrize_lower(cache.bulk_covariance);
  cache.partition_valid =
      cache.bulk_mean.allFinite() && cache.bulk_covariance.allFinite();
  return cache.partition_valid;
}

PromotionStatus promote_large_bulk_tilts(FastInfoCache& cache,
                                         const VectorXd& theta,
                                         int& promoted_count) {
  promoted_count = 0;
  if (!cache.partition_valid || theta.size() != cache.d ||
      !theta.allFinite()) {
    return PromotionStatus::invalid_partition;
  }

  std::vector<int> promoted;
  promoted.reserve(cache.bulk_indices.size());
  const double threshold_tolerance =
      kPromotionTieTolerance * std::max(1.0, kMaxBulkTiltThreshold);
  for (const int i : cache.bulk_indices) {
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < cache.d; ++j) {
      tilt += theta[j] * g[j * cache.n];
    }
    if (!std::isfinite(tilt)) {
      return PromotionStatus::invalid_partition;
    }
    // Use a fixed, model-scale threshold and retain every boundary tie.
    // Consequently, the promoted feature rows do not depend on their input
    // order (or on an arbitrary tie-breaking row index).
    if (std::abs(tilt) + threshold_tolerance >=
        kMaxBulkTiltThreshold) {
      promoted.push_back(i);
    }
  }
  if (promoted.empty()) return PromotionStatus::no_progress;

  cache.exact_indices.insert(cache.exact_indices.end(), promoted.begin(),
                             promoted.end());
  if (!recompute_bulk_moments(cache)) {
    return PromotionStatus::invalid_partition;
  }
  promoted_count = static_cast<int>(promoted.size());
  return PromotionStatus::promoted;
}

FastInfoCache build_cache(const Rcpp::NumericVector& a,
                          const Rcpp::NumericVector& w,
                          const Rcpp::NumericVector& y,
                          const Rcpp::NumericMatrix& Z,
                          const Rcpp::NumericVector& propensity,
                          const int score_sign,
                          const bool require_interior_propensity = true,
                          const bool require_supported_geometry = true) {
  const int n = a.size();
  const int p = Z.ncol();
  if (n < 2 || p < 1 || Z.nrow() != n || w.size() != n || y.size() != n ||
      propensity.size() != n) {
    Rcpp::stop("input dimensions do not match");
  }
  if (score_sign != -1 && score_sign != 1) {
    Rcpp::stop("score_sign must be -1 or +1");
  }

  FastInfoCache cache;
  cache.n = n;
  cache.p = p;
  cache.d = p + 1;
  cache.score_sign = score_sign;
  cache.G.setZero(n, p + 1);
  cache.offsets.resize(n);
  cache.propensity.resize(n);
  cache.weights.resize(n);
  MatrixXd C = MatrixXd::Zero(p, p);
  VectorXd weighted_z = VectorXd::Zero(p);

  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(a[i])) Rcpp::stop("a must be finite");
    if (!std::isfinite(y[i]) || y[i] < 0.0) {
      Rcpp::stop("y must be finite and nonnegative");
    }
    if (!std::isfinite(w[i]) || w[i] <= 0.0) {
      Rcpp::stop("w must be positive and finite");
    }
    if (!std::isfinite(propensity[i]) || propensity[i] < 0.0 ||
        propensity[i] > 1.0 ||
        (require_interior_propensity &&
         (propensity[i] <= 0.0 || propensity[i] >= 1.0))) {
      Rcpp::stop(require_interior_propensity
                     ? "propensity must be finite and lie strictly between 0 and 1"
                     : "propensity must be finite and lie in [0, 1]");
    }
    cache.G(i, 0) = static_cast<double>(score_sign) * a[i];
    for (int j = 0; j < p; ++j) {
      if (!std::isfinite(Z(i, j))) Rcpp::stop("Z must be finite");
      weighted_z[j] = w[i] * Z(i, j);
      if (!std::isfinite(weighted_z[j])) {
        Rcpp::stop("w multiplied by Z must be finite");
      }
      cache.G(i, j + 1) = weighted_z[j];
    }
    for (int j = 0; j < p; ++j) {
      for (int k = 0; k <= j; ++k) {
        C(j, k) += weighted_z[j] * weighted_z[k] / w[i];
      }
    }
    cache.offsets[i] = std::log(propensity[i] / (1.0 - propensity[i]));
    cache.propensity[i] = propensity[i];
    cache.weights[i] = w[i];
    if (std::abs(Z(i, 0) - 1.0) > 1.0e-10) {
      cache.leading_intercept = false;
    }
  }

  if (require_supported_geometry && !cache.leading_intercept) {
    Rcpp::stop("the first column of Z must be an intercept");
  }
  symmetrize_lower(C);
  if (!C.allFinite()) Rcpp::stop("Z transpose diag(w) Z must be finite");
  bool invertible = true;
  try {
    Eigen::LLT<MatrixXd> decomposition(C);
    if (decomposition.info() != Eigen::Success) {
      invertible = false;
    } else {
      cache.C_inv = decomposition.solve(MatrixXd::Identity(p, p));
      invertible = decomposition.info() == Eigen::Success &&
                   cache.C_inv.allFinite();
    }
  } catch (...) {
    invertible = false;
  }
  if (!invertible) {
    if (require_supported_geometry) {
      Rcpp::stop("Z transpose diag(w) Z must be positive definite");
    }
    cache.C_inv = MatrixXd::Constant(p, p, kNaN);
  } else {
    cache.information_invertible = true;
  }

  create_outcome_partition(cache, y);
  return cache;
}

void validate_solver_controls(const double target,
                              const double tolerance,
                              const int max_iterations) {
  if (!std::isfinite(target)) Rcpp::stop("target must be finite");
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    Rcpp::stop("tolerance must be positive and finite");
  }
  if (max_iterations < 0) Rcpp::stop("max_iterations cannot be negative");
}

Rcpp::List run_fast_solver_once(const FastInfoCache& cache,
                                const double target,
                                const double tolerance,
                                const int max_iterations,
                                const int minimum_updates = 0) {
  VectorXd state = VectorXd::Zero(cache.d + 1);
  const std::vector<double> empty_history;
  if (!cache.partition_valid) {
    return failure_result(cache, "invalid_partial_normal_partition", target,
                          kNaN, 0, kNaN, empty_history, state, 0, tolerance);
  }
  Evaluation base = evaluate_partial_normal(cache, state, target);
  int cgf_evaluations = 1;
  if (!base.valid) {
    return failure_result(cache, "invalid_center", target, kNaN, 0, kNaN,
                          empty_history, state, 1, tolerance);
  }

  const double center = base.boundary.b;
  const double variance =
      base.boundary.grad.dot(base.Sgg * base.boundary.grad);
  if (!std::isfinite(variance) || variance <= 1.0e-14) {
    return failure_result(cache, "degenerate_center_variance", target,
                          center, 0, kNaN, empty_history, state, 1,
                          tolerance);
  }

  state[cache.d] = (target - center) / variance;
  state.head(cache.d) = state[cache.d] * base.boundary.grad;
  Evaluation current = evaluate_partial_normal(cache, state, target);
  ++cgf_evaluations;
  std::vector<double> history;
  int updates = 0;
  bool converged = false;

  for (int iteration = 0; iteration <= max_iterations; ++iteration) {
    if (!current.valid) break;
    const double residual = max_abs(current.residual);
    history.push_back(residual);
    if (residual <= tolerance && updates >= minimum_updates) {
      converged = true;
      break;
    }
    if (iteration == max_iterations) break;

    VectorXd step;
    if (!solve_linear(current.jacobian, -current.residual, step)) break;
    bool accepted = false;
    double scale = 1.0;
    const bool forced_refinement =
        updates < minimum_updates && residual <= tolerance;
    for (int backtrack = 0; backtrack < 24; ++backtrack) {
      const VectorXd candidate = state + scale * step;
      Evaluation trial = evaluate_partial_normal(cache, candidate, target);
      ++cgf_evaluations;
      const double trial_residual =
          trial.valid ? max_abs(trial.residual) : kNaN;
      if (trial.valid &&
          (trial_residual < residual ||
           (forced_refinement && trial_residual <= tolerance))) {
        state = candidate;
        current = std::move(trial);
        ++updates;
        accepted = true;
        break;
      }
      scale *= 0.5;
    }
    if (!accepted) break;
  }

  if (!converged || !current.valid) {
    return failure_result(
        cache, "newton_failed", target, center, updates,
        current.valid ? max_abs(current.residual) : kNaN, history, state,
        cgf_evaluations, tolerance);
  }
  return finalize_result(cache, state, current, center, target, updates,
                         cgf_evaluations, history, tolerance);
}

std::string result_reason(const Rcpp::List& result) {
  if (!result.containsElementNamed("reason")) return "missing_reason";
  SEXP value = result["reason"];
  if (Rf_isNull(value) || Rf_length(value) != 1) return "missing_reason";
  return Rcpp::as<std::string>(value);
}

bool result_theta(const Rcpp::List& result,
                  const int d,
                  VectorXd& theta) {
  if (!result.containsElementNamed("state")) return false;
  SEXP value = result["state"];
  if (Rf_isNull(value) || !Rf_isNumeric(value) || Rf_length(value) < d) {
    return false;
  }
  const Rcpp::NumericVector state(value);
  theta.resize(d);
  for (int j = 0; j < d; ++j) theta[j] = state[j];
  return theta.allFinite();
}

void set_promotion_metadata(Rcpp::List& result,
                            const int promotion_rounds,
                            const int promoted_count,
                            const std::string& stop_reason) {
  result["promotion_rounds"] = promotion_rounds;
  result["promoted_count"] = promoted_count;
  result["promotion_stop_reason"] = stop_reason;
}

Rcpp::List run_fast_solver(FastInfoCache& cache,
                           const double target,
                           const double tolerance,
                           const int max_iterations,
                           const int minimum_updates = 0) {
  int promotion_rounds = 0;
  int promoted_total = 0;

  while (true) {
    Rcpp::List result = run_fast_solver_once(
        cache, target, tolerance, max_iterations, minimum_updates);
    const std::string reason = result_reason(result);

    // Promotion is deliberately narrower than the overall safety gate.  It
    // is attempted only after Newton reached a mathematically regular LR
    // root and Berry--Esseen passed, leaving the Gaussian-block tilt as the
    // sole failed condition.  All other failures keep their existing
    // empirical-fallback behavior.
    if (reason != "partial_normal_bulk_tilt_unsafe") {
      const std::string stop_reason =
          promotion_rounds == 0
              ? "not_needed"
              : (reason == "ok" ? "safe_after_promotion"
                                 : (reason ==
                                            "partial_normal_berry_esseen_unsafe"
                                        ? "berry_esseen_gate"
                                        : "post_promotion_solver_or_lr_failure"));
      set_promotion_metadata(result, promotion_rounds, promoted_total,
                             stop_reason);
      return result;
    }
    if (promotion_rounds >= kMaxPromotionRounds) {
      set_promotion_metadata(result, promotion_rounds, promoted_total,
                             "maximum_rounds");
      return result;
    }

    VectorXd theta;
    if (!result_theta(result, cache.d, theta)) {
      set_promotion_metadata(result, promotion_rounds, promoted_total,
                             "no_progress");
      return result;
    }
    int promoted_this_round = 0;
    const PromotionStatus status = promote_large_bulk_tilts(
        cache, theta, promoted_this_round);
    if (status != PromotionStatus::promoted) {
      const std::string stop_reason =
          status == PromotionStatus::no_progress
              ? "no_progress"
              : "invalid_promoted_partition";
      set_promotion_metadata(result, promotion_rounds, promoted_total,
                             stop_reason);
      return result;
    }
    ++promotion_rounds;
    promoted_total += promoted_this_round;
  }
}

void add_outward_metadata(Rcpp::List& out,
                          const double base_center,
                          const double observed_target,
                          const int score_sign) {
  out["base_center"] = base_center;
  out["observed_target"] = observed_target;
  out["outward_score_sign"] = score_sign;
}

}  // namespace

namespace sceptre {

Rcpp::List crt_spa_full_fast(const Rcpp::NumericVector& a,
                             const Rcpp::NumericVector& w,
                             const Rcpp::NumericVector& y,
                             const Rcpp::NumericMatrix& Z,
                             const Rcpp::NumericVector& propensity,
                             const double target,
                             const int score_sign,
                             const double tolerance,
                             const int max_iterations) {
  validate_solver_controls(target, tolerance, max_iterations);
  FastInfoCache cache = build_cache(a, w, y, Z, propensity, score_sign);
  return run_fast_solver(cache, target, tolerance, max_iterations);
}

Rcpp::List crt_spa_full_outward_fast(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericVector& y,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::NumericVector& propensity,
    const Rcpp::IntegerVector& treated_indices,
    const double tolerance,
    const int max_iterations) {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    Rcpp::stop("tolerance must be positive and finite");
  }
  if (max_iterations < 0) Rcpp::stop("max_iterations cannot be negative");

  const FastInfoCache positive_cache =
      build_cache(a, w, y, Z, propensity, 1, false, false);
  std::vector<int> treated(treated_indices.size());
  for (R_xlen_t j = 0; j < treated_indices.size(); ++j) {
    if (treated_indices[j] == NA_INTEGER || treated_indices[j] < 1 ||
        treated_indices[j] > positive_cache.n) {
      Rcpp::stop("treated_indices contains an invalid one-based index");
    }
    treated[j] = treated_indices[j] - 1;
  }
  std::vector<int> sorted_treated = treated;
  std::sort(sorted_treated.begin(), sorted_treated.end());
  if (std::adjacent_find(sorted_treated.begin(), sorted_treated.end()) !=
      sorted_treated.end()) {
    Rcpp::stop("treated_indices cannot contain duplicates");
  }

  double observed_U = 0.0;
  double observed_weight = 0.0;
  VectorXd observed_B = VectorXd::Zero(positive_cache.p);
  for (const int index : treated) {
    observed_U += positive_cache.G(index, 0);
    observed_weight += positive_cache.weights[index];
    observed_B += positive_cache.G.row(index)
                      .segment(1, positive_cache.p)
                      .transpose();
  }
  const double projection = positive_cache.information_invertible
      ? observed_B.dot(positive_cache.C_inv * observed_B)
      : kNaN;
  const double observed_information = observed_weight - projection;
  const double information_scale =
      std::max(std::abs(observed_weight), std::abs(projection));
  const double observed_target =
      std::isfinite(observed_U) && std::isfinite(observed_information) &&
              std::isfinite(information_scale) && information_scale > 0.0 &&
              observed_information > 1.0e-14 * information_scale
          ? observed_U / std::sqrt(observed_information)
          : kNaN;
  const VectorXd zero_state = VectorXd::Zero(positive_cache.d + 1);

  if (!positive_cache.information_invertible) {
    Rcpp::List out = failure_result(
        positive_cache, "singular_information_geometry", kNaN, kNaN, 0,
        kNaN, std::vector<double>(), zero_state, 0, tolerance);
    add_outward_metadata(out, NA_REAL, NA_REAL, NA_INTEGER);
    return out;
  }
  if (!std::isfinite(observed_target)) {
    Rcpp::List out = failure_result(
        positive_cache, "invalid_observed_studentizer", kNaN, kNaN, 0,
        kNaN, std::vector<double>(), zero_state, 0, tolerance);
    add_outward_metadata(out, NA_REAL, NA_REAL, NA_INTEGER);
    return out;
  }
  if (!positive_cache.leading_intercept) {
    Rcpp::List out = failure_result(
        positive_cache, "unsupported_spa_geometry_no_intercept",
        observed_target, kNaN, 0, kNaN, std::vector<double>(), zero_state,
        0, tolerance);
    add_outward_metadata(out, NA_REAL, observed_target, NA_INTEGER);
    return out;
  }
  for (R_xlen_t i = 0; i < propensity.size(); ++i) {
    if (propensity[i] <= 0.0 || propensity[i] >= 1.0) {
      Rcpp::List out = failure_result(
          positive_cache, "non_interior_propensity", observed_target,
          kNaN, 0, kNaN, std::vector<double>(), zero_state, 0, tolerance);
      add_outward_metadata(out, NA_REAL, observed_target, NA_INTEGER);
      return out;
    }
  }
  if (!positive_cache.partition_valid) {
    Rcpp::List out = failure_result(
        positive_cache, "invalid_partial_normal_partition", observed_target,
        kNaN, 0, kNaN, std::vector<double>(), zero_state, 0, tolerance);
    add_outward_metadata(out, NA_REAL, observed_target, NA_INTEGER);
    return out;
  }

  const Evaluation base =
      evaluate_partial_normal(positive_cache, zero_state, observed_target);
  if (!base.valid || !std::isfinite(base.boundary.b)) {
    Rcpp::List out = failure_result(
        positive_cache, "invalid_center", observed_target, kNaN, 0, kNaN,
        std::vector<double>(), zero_state, 1, tolerance);
    add_outward_metadata(out, NA_REAL, observed_target, NA_INTEGER);
    return out;
  }
  const double base_center = base.boundary.b;
  const double displacement = observed_target - base_center;
  if (displacement == 0.0) {
    Rcpp::List out = failure_result(
        positive_cache, "central_target_requires_empirical_fallback",
        observed_target, base_center, 0, 0.0, std::vector<double>(),
        zero_state, 1, tolerance);
    add_outward_metadata(out, base_center, observed_target, NA_INTEGER);
    return out;
  }

  const int score_sign = displacement > 0.0 ? 1 : -1;
  const double outward_target = score_sign * observed_target;
  FastInfoCache outward_cache = score_sign == 1
      ? positive_cache
      : build_cache(a, w, y, Z, propensity, -1);
  if (max_iterations == 0) {
    const VectorXd state = VectorXd::Zero(outward_cache.d + 1);
    Rcpp::List out = failure_result(
        outward_cache, "solver_disabled_max_iterations_zero",
        outward_target, score_sign * base_center, 0, kNaN,
        std::vector<double>(), state, 1, tolerance);
    add_outward_metadata(out, base_center, observed_target, score_sign);
    return out;
  }

  Rcpp::List out = run_fast_solver(outward_cache, outward_target, tolerance,
                                   max_iterations, 1);
  add_outward_metadata(out, base_center, observed_target, score_sign);
  return out;
}

}  // namespace sceptre

// [[Rcpp::export]]
Rcpp::List crt_spa_full_fast_cpp(const Rcpp::NumericVector& a,
                                 const Rcpp::NumericVector& w,
                                 const Rcpp::NumericVector& y,
                                 const Rcpp::NumericMatrix& Z,
                                 const Rcpp::NumericVector& propensity,
                                 const double target,
                                 const int score_sign = 1,
                                 const double tolerance = 1e-5,
                                 const int max_iterations = 50) {
  return sceptre::crt_spa_full_fast(a, w, y, Z, propensity, target,
                                    score_sign, tolerance, max_iterations);
}

// [[Rcpp::export]]
Rcpp::List crt_spa_full_outward_fast_cpp(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericVector& y,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::NumericVector& propensity,
    const Rcpp::IntegerVector& treated_indices,
    const double tolerance = 1e-5,
    const int max_iterations = 50) {
  return sceptre::crt_spa_full_outward_fast(
      a, w, y, Z, propensity, treated_indices, tolerance, max_iterations);
}
