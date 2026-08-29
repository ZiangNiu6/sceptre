// Partial-normal fastSPA analogue for the empirically studentized CRT
// statistic S / sqrt(Q - S^2 / n).
//
// The fixed sparse-outcome partition follows the studentization writeup:
// positive-outcome rows are evaluated as exact Bernoulli terms and zero-
// outcome rows form the moment-matched Gaussian block.  This partition uses
// the fixed observed response, never the random CRT treatment assignment.
// After a provisional root solve, zero-outcome rows whose target-direction
// tilt is too large are promoted into the exact block and the root is
// re-solved.  Promotion is allowed to exhaust the Gaussian block, in which
// case the calculation is the exact Bernoulli solver.
//
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>

#include "crt_empirical_spa_fast.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using Eigen::Matrix2d;
using Eigen::Vector2d;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kBerryEsseenConstant = 0.56;
constexpr double kBerryEsseenThreshold = 0.25;
constexpr double kMaxBulkTiltThreshold = 0.25;
constexpr double kPromotionTieTolerance = 1.0e-12;
constexpr int kMaxPromotionRounds = 8;
constexpr const char* kStatisticId = "empirical_studentized_crt_v1";
constexpr const char* kEquationId = "crt_studentized_reduced_root_v1";
constexpr const char* kTailGeometry =
    "directional_tangent_halfspace_lugannani_rice";

double expit_stable(const double x) {
  if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
  const double ex = std::exp(x);
  return ex / (1.0 + ex);
}

double softplus_stable(const double x) {
  return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
}

double max_abs(const Vector2d& x) {
  return std::max(std::abs(x[0]), std::abs(x[1]));
}

bool solve_linear(const Matrix2d& A, const Vector2d& b, Vector2d& x) {
  if (!A.allFinite() || !b.allFinite()) return false;
  try {
    Eigen::FullPivLU<Matrix2d> decomposition(A);
    if (!decomposition.isInvertible()) return false;
    x = decomposition.solve(b);
    return x.allFinite();
  } catch (...) {
    return false;
  }
}

struct EmpiricalFastCache {
  int n = 0;
  bool partition_valid = false;
  double score_scale = 1.0;
  Eigen::VectorXd a;
  Eigen::VectorXd d;
  Eigen::VectorXd propensity;
  Eigen::VectorXd offset;
  double sc = 0.0;
  double qc = 0.0;
  std::vector<int> exact_indices;
  std::vector<int> bulk_indices;
  int initial_exact_count = 0;
  int initial_bulk_count = 0;
  // Stored for score_sign = +1. The sign is applied by congruence when the
  // CGF is evaluated, so the partition itself is independent of tail side.
  Vector2d bulk_mean = Vector2d::Zero();
  Matrix2d bulk_covariance = Matrix2d::Zero();
};

EmpiricalFastCache build_cache(const Rcpp::NumericVector& a,
                               const Rcpp::NumericVector& propensity,
                               const Rcpp::NumericVector& y) {
  const int n = a.size();
  if (n < 2 || propensity.size() != n || y.size() != n) {
    Rcpp::stop(
        "a, propensity, and y must have the same length of at least two");
  }

  EmpiricalFastCache cache;
  cache.n = n;
  cache.a.resize(n);
  cache.d.resize(n);
  cache.propensity.resize(n);
  cache.offset.resize(n);
  double score_scale = 0.0;
  for (int i = 0; i < n; ++i) {
    const double ai = a[i];
    const double p = propensity[i];
    const double yi = y[i];
    if (!std::isfinite(ai)) Rcpp::stop("a must be finite");
    if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
      Rcpp::stop("propensity must be finite and lie in [0, 1]");
    }
    if (!std::isfinite(yi) || yi < 0.0) {
      Rcpp::stop("y must be finite and nonnegative");
    }
    score_scale = std::max(score_scale, std::abs(ai));
  }

  cache.score_scale = score_scale > 0.0 ? score_scale : 1.0;
  for (int i = 0; i < n; ++i) {
    const double ai = a[i] / cache.score_scale;
    const double p = propensity[i];
    const double a2 = ai * ai;
    const double di = (1.0 - 2.0 * p) * a2;
    if (!std::isfinite(a2) || !std::isfinite(di)) {
      Rcpp::stop("squared score coefficients must be finite");
    }
    cache.a[i] = ai;
    cache.d[i] = di;
    cache.propensity[i] = p;
    cache.offset[i] =
        (p > 0.0 && p < 1.0) ? std::log(p) - std::log1p(-p) : 0.0;
    cache.sc -= p * ai;
    cache.qc += p * p * a2;
  }
  if (!std::isfinite(cache.sc) || !std::isfinite(cache.qc)) {
    Rcpp::stop("empirical CRT fast-cache quantities must be finite");
  }

  cache.exact_indices.reserve(n);
  cache.bulk_indices.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (y[i] > 0.0) {
      cache.exact_indices.push_back(i);
      continue;
    }
    cache.bulk_indices.push_back(i);
    const Vector2d g(cache.a[i], cache.d[i]);
    const double p = cache.propensity[i];
    cache.bulk_mean.noalias() += p * g;
    cache.bulk_covariance.noalias() +=
        p * (1.0 - p) * (g * g.transpose());
  }
  cache.initial_exact_count = static_cast<int>(cache.exact_indices.size());
  cache.initial_bulk_count = static_cast<int>(cache.bulk_indices.size());
  if (!cache.bulk_mean.allFinite() || !cache.bulk_covariance.allFinite()) {
    return cache;
  }
  cache.partition_valid = true;
  return cache;
}

struct BernoulliTilt {
  bool valid = false;
  double probability = kNaN;
  double log_normalizer = kNaN;
};

BernoulliTilt evaluate_bernoulli_tilt(const double p,
                                      const double offset,
                                      const double tilt) {
  BernoulliTilt out;
  if (!std::isfinite(tilt)) return out;
  if (p == 0.0) {
    out.valid = true;
    out.probability = 0.0;
    out.log_normalizer = 0.0;
    return out;
  }
  if (p == 1.0) {
    out.valid = true;
    out.probability = 1.0;
    out.log_normalizer = tilt;
    return out;
  }
  const double eta = offset + tilt;
  if (!std::isfinite(eta)) return out;
  out.probability = expit_stable(eta);
  out.log_normalizer =
      softplus_stable(eta) - softplus_stable(offset);
  out.valid = std::isfinite(out.probability) &&
              std::isfinite(out.log_normalizer);
  return out;
}

double xlog1px_minus_x(const double x) {
  if (!std::isfinite(x) || x < -1.0 - 1.0e-14) return kNaN;
  if (x <= -1.0) return 1.0;
  if (std::abs(x) >= 1.0e-3) {
    return (1.0 + x) * std::log1p(x) - x;
  }
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

Vector2d signed_bulk_mean(const EmpiricalFastCache& cache,
                          const int score_sign) {
  Vector2d mean = cache.bulk_mean;
  mean[0] *= static_cast<double>(score_sign);
  return mean;
}

Matrix2d signed_bulk_covariance(const EmpiricalFastCache& cache,
                                const int score_sign) {
  Matrix2d covariance = cache.bulk_covariance;
  covariance(0, 1) *= static_cast<double>(score_sign);
  covariance(1, 0) = covariance(0, 1);
  return covariance;
}

struct CgfEvaluation {
  bool valid = false;
  double K = kNaN;
  double rate = kNaN;
  Vector2d moment = Vector2d::Constant(kNaN);
  Matrix2d hessian = Matrix2d::Constant(kNaN);
};

CgfEvaluation evaluate_cgf(const EmpiricalFastCache& cache,
                           const Vector2d& theta,
                           const int score_sign) {
  CgfEvaluation out;
  if (!theta.allFinite() || (score_sign != -1 && score_sign != 1)) return out;

  const double signed_sc = static_cast<double>(score_sign) * cache.sc;
  const Vector2d bulk_mean = signed_bulk_mean(cache, score_sign);
  const Matrix2d bulk_covariance =
      signed_bulk_covariance(cache, score_sign);
  const Vector2d constants(signed_sc, cache.qc);
  out.K = theta.dot(constants + bulk_mean) +
          0.5 * theta.dot(bulk_covariance * theta);
  out.moment = constants + bulk_mean + bulk_covariance * theta;
  out.hessian = bulk_covariance;
  double stable_rate = 0.5 * theta.dot(bulk_covariance * theta);
  if (stable_rate < 0.0 && stable_rate > -1e-14) stable_rate = 0.0;

  for (const int i : cache.exact_indices) {
    const double g0 = static_cast<double>(score_sign) * cache.a[i];
    const double g1 = cache.d[i];
    const double tilt = theta[0] * g0 + theta[1] * g1;
    const BernoulliTilt tilted = evaluate_bernoulli_tilt(
        cache.propensity[i], cache.offset[i], tilt);
    if (!tilted.valid) return CgfEvaluation();
    const double rate_term =
        bernoulli_kl_from_tilt(cache.propensity[i], tilt);
    if (!std::isfinite(rate_term)) return CgfEvaluation();
    stable_rate += rate_term;
    const double pi = tilted.probability;
    const double h = pi * (1.0 - pi);
    out.K += tilted.log_normalizer;
    out.moment[0] += pi * g0;
    out.moment[1] += pi * g1;
    out.hessian(0, 0) += h * g0 * g0;
    out.hessian(0, 1) += h * g0 * g1;
    out.hessian(1, 1) += h * g1 * g1;
  }
  out.hessian(1, 0) = out.hessian(0, 1);
  out.rate = stable_rate;
  out.valid = std::isfinite(out.K) && std::isfinite(out.rate) &&
              out.rate >= 0.0 && out.moment.allFinite() &&
              out.hessian.allFinite();
  return out;
}

struct BoundaryEvaluation {
  bool valid = false;
  double value = kNaN;
  double variance = kNaN;
  Vector2d gradient = Vector2d::Constant(kNaN);
};

BoundaryEvaluation evaluate_boundary(const Vector2d& moment, const int n) {
  BoundaryEvaluation out;
  if (!moment.allFinite() || n < 2) return out;
  const double s = moment[0];
  const double q = moment[1];
  const double mean_correction = s * s / static_cast<double>(n);
  const double variance = q - mean_correction;
  const double scale = std::max(std::abs(q), mean_correction);
  if (!std::isfinite(variance) || !std::isfinite(scale) || scale <= 0.0 ||
      variance <= 1e-14 * scale) return out;
  const double root_v = std::sqrt(variance);
  const double v_three_halves = variance * root_v;
  out.value = s / root_v;
  out.variance = variance;
  out.gradient << q / v_three_halves, -0.5 * s / v_three_halves;
  out.valid = std::isfinite(out.value) && out.gradient.allFinite();
  return out;
}

struct RootEvaluation {
  bool valid = false;
  CgfEvaluation cgf;
  Vector2d residual = Vector2d::Constant(kNaN);
  Vector2d scaled_residual = Vector2d::Constant(kNaN);
  Matrix2d jacobian = Matrix2d::Constant(kNaN);
  Matrix2d scaled_jacobian = Matrix2d::Constant(kNaN);
  double s = kNaN;
  double q = kNaN;
  double variance = kNaN;
  double kappa = kNaN;
};

RootEvaluation evaluate_root(const EmpiricalFastCache& cache,
                             const Vector2d& theta,
                             const double target,
                             const int score_sign,
                             const double q_scale) {
  RootEvaluation out;
  out.cgf = evaluate_cgf(cache, theta, score_sign);
  if (!out.cgf.valid || !std::isfinite(target) || target <= 0.0 ||
      !std::isfinite(q_scale) || q_scale <= 0.0) return out;

  out.s = out.cgf.moment[0];
  out.q = out.cgf.moment[1];
  out.variance = out.q - out.s * out.s / static_cast<double>(cache.n);
  out.kappa = 1.0 / (target * target) +
              1.0 / static_cast<double>(cache.n);
  const Matrix2d& H = out.cgf.hessian;
  out.residual[0] = out.q - out.kappa * out.s * out.s;
  out.residual[1] = theta[0] + 2.0 * out.kappa * out.s * theta[1];
  out.jacobian.row(0) =
      H.row(1) - 2.0 * out.kappa * out.s * H.row(0);
  out.jacobian(1, 0) =
      1.0 + 2.0 * out.kappa * theta[1] * H(0, 0);
  out.jacobian(1, 1) =
      2.0 * out.kappa * out.s +
      2.0 * out.kappa * theta[1] * H(0, 1);
  out.scaled_residual << out.residual[0] / q_scale, out.residual[1];
  out.scaled_jacobian = out.jacobian;
  out.scaled_jacobian.row(0) /= q_scale;

  const double mean_correction =
      out.s * out.s / static_cast<double>(cache.n);
  const double variance_scale = std::max(std::abs(out.q), mean_correction);
  out.valid = out.residual.allFinite() && out.jacobian.allFinite() &&
              out.scaled_residual.allFinite() &&
              out.scaled_jacobian.allFinite() && out.s > 0.0 &&
              std::isfinite(out.variance) &&
              std::isfinite(variance_scale) && variance_scale > 0.0 &&
              out.variance > 1e-14 * variance_scale;
  return out;
}

struct CrtSolve {
  bool converged = false;
  std::string reason = "iteration_limit";
  Vector2d theta = Vector2d::Zero();
  RootEvaluation value;
  int iterations = 0;
  int evaluations = 0;
  double center = kNaN;
  double center_sd = kNaN;
  double q_scale = kNaN;
  std::vector<double> history;
};

CrtSolve solve_full(const EmpiricalFastCache& cache,
                    const double target,
                    const int score_sign,
                    const double tolerance,
                    const int max_iterations,
                    const int max_backtracks) {
  CrtSolve out;
  if (score_sign != -1 && score_sign != 1) {
    out.reason = "invalid_score_sign";
    return out;
  }
  if (!std::isfinite(target) || target <= 0.0) {
    out.reason = "invalid_target";
    return out;
  }
  if (!std::isfinite(tolerance) || tolerance <= 0.0 ||
      max_iterations < 0 || max_backtracks < 1) {
    out.reason = "invalid_controls";
    return out;
  }
  for (int i = 0; i < cache.n; ++i) {
    if (cache.propensity[i] <= 0.0 || cache.propensity[i] >= 1.0) {
      out.reason = "non_interior_propensity";
      return out;
    }
  }
  if (!cache.partition_valid) {
    out.reason = "invalid_partial_normal_partition";
    return out;
  }

  const CgfEvaluation base = evaluate_cgf(cache, out.theta, score_sign);
  ++out.evaluations;
  if (!base.valid) {
    out.reason = "invalid_untilted_state";
    return out;
  }
  const BoundaryEvaluation center = evaluate_boundary(base.moment, cache.n);
  if (!center.valid) {
    out.reason = "degenerate_untilted_variance";
    out.value.cgf = base;
    return out;
  }
  out.center = center.value;
  const double center_variance =
      center.gradient.dot(base.hessian * center.gradient);
  if (!std::isfinite(center_variance) || center_variance <= 0.0) {
    out.reason = "degenerate_untilted_delta_variance";
    out.value.cgf = base;
    return out;
  }
  out.center_sd = std::sqrt(center_variance);
  out.q_scale = std::abs(base.moment[1]);

  if (max_iterations == 0) {
    out.reason = "solver_disabled_max_iterations_zero";
    out.value = evaluate_root(cache, out.theta, target, score_sign,
                              out.q_scale);
    ++out.evaluations;
    return out;
  }

  const double kappa = 1.0 / (target * target) +
                       1.0 / static_cast<double>(cache.n);
  const double q0 = base.moment[1];
  if (!std::isfinite(q0) || q0 <= 0.0) {
    out.reason = "nonpositive_untilted_q";
    out.value.cgf = base;
    return out;
  }
  const double s_star = std::sqrt(q0 / kappa);
  Vector2d direction;
  direction << 1.0, -1.0 / (2.0 * kappa * s_star);
  const double response = base.hessian.row(0).dot(direction);
  const double amplitude =
      (std::isfinite(response) && std::abs(response) > 1e-14)
          ? (s_star - base.moment[0]) / response
          : target / std::sqrt(q0);
  out.theta = amplitude * direction;
  RootEvaluation current =
      evaluate_root(cache, out.theta, target, score_sign, out.q_scale);
  ++out.evaluations;

  if (!current.valid) {
    out.theta << target / std::sqrt(q0),
        -target * target / (2.0 * q0);
    current = evaluate_root(cache, out.theta, target, score_sign,
                            out.q_scale);
    ++out.evaluations;
  }

  while (true) {
    if (!current.valid) {
      out.reason = "invalid_outer_state";
      break;
    }
    const double max_residual = max_abs(current.scaled_residual);
    const double merit = current.scaled_residual.norm();
    out.history.push_back(merit);
    if (max_residual <= tolerance) {
      out.converged = true;
      out.reason = "ok";
      break;
    }
    if (out.iterations >= max_iterations) {
      out.reason = "iteration_limit";
      break;
    }

    Vector2d step;
    if (!solve_linear(current.scaled_jacobian,
                      -current.scaled_residual, step)) {
      out.reason = "singular_outer_jacobian";
      break;
    }
    const double relative = std::max(
        std::abs(step[0]) / (1.0 + std::abs(out.theta[0])),
        std::abs(step[1]) / (1.0 + std::abs(out.theta[1])));
    if (relative > 10.0) step *= 10.0 / relative;

    bool accepted = false;
    double scale = 1.0;
    for (int backtrack = 0; backtrack < max_backtracks; ++backtrack) {
      const Vector2d candidate_theta = out.theta + scale * step;
      RootEvaluation candidate = evaluate_root(
          cache, candidate_theta, target, score_sign, out.q_scale);
      ++out.evaluations;
      if (candidate.valid && candidate.scaled_residual.norm() < merit) {
        out.theta = candidate_theta;
        current = std::move(candidate);
        ++out.iterations;
        accepted = true;
        break;
      }
      scale *= 0.5;
    }
    if (!accepted) {
      out.reason = "outer_line_search_failed";
      break;
    }
    if ((out.iterations & 7) == 0) Rcpp::checkUserInterrupt();
  }

  out.value = current;
  return out;
}

struct NormalBulkDiagnostic {
  bool valid = false;
  bool safe = false;
  double directional_variance = kNaN;
  double absolute_third_moment_sum = kNaN;
  double berry_esseen_ratio = kNaN;
  double max_abs_tilt = kNaN;
};

NormalBulkDiagnostic diagnose_normal_bulk(const EmpiricalFastCache& cache,
                                          const Vector2d& theta,
                                          const int score_sign) {
  NormalBulkDiagnostic out;
  if (!cache.partition_valid || !theta.allFinite() ||
      (score_sign != -1 && score_sign != 1)) {
    return out;
  }
  double variance = 0.0;
  double third_moment_sum = 0.0;
  double max_tilt = 0.0;
  for (const int i : cache.bulk_indices) {
    const double projection =
        theta[0] * static_cast<double>(score_sign) * cache.a[i] +
        theta[1] * cache.d[i];
    if (!std::isfinite(projection)) return out;
    const double p = cache.propensity[i];
    const double variance_weight = p * (1.0 - p);
    variance += variance_weight * projection * projection;
    third_moment_sum +=
        variance_weight * (p * p + (1.0 - p) * (1.0 - p)) *
        std::pow(std::abs(projection), 3.0);
    max_tilt = std::max(max_tilt, std::abs(projection));
  }
  if (!std::isfinite(variance) || !std::isfinite(third_moment_sum) ||
      !std::isfinite(max_tilt) || variance < 0.0 || third_moment_sum < 0.0) {
    return out;
  }
  double ratio = 0.0;
  if (variance > 0.0) {
    ratio = kBerryEsseenConstant * third_moment_sum /
            (variance * std::sqrt(variance));
  } else if (third_moment_sum > 0.0) {
    return out;
  }
  if (!std::isfinite(ratio) || ratio < 0.0) return out;
  out.valid = true;
  out.safe = ratio <= kBerryEsseenThreshold &&
             max_tilt <= kMaxBulkTiltThreshold;
  out.directional_variance = variance;
  out.absolute_third_moment_sum = third_moment_sum;
  out.berry_esseen_ratio = ratio;
  out.max_abs_tilt = max_tilt;
  return out;
}

struct PromotionAttempt {
  bool valid = false;
  bool made_progress = false;
  int promoted_count = 0;
};

PromotionAttempt promote_unsafe_bulk_rows(EmpiricalFastCache& cache,
                                          const Vector2d& theta,
                                          const int score_sign) {
  PromotionAttempt out;
  if (!cache.partition_valid || cache.bulk_indices.empty() ||
      !theta.allFinite() || (score_sign != -1 && score_sign != 1)) {
    return out;
  }

  std::vector<unsigned char> promote(cache.n, 0);
  std::vector<int> promoted_indices;
  promoted_indices.reserve(cache.bulk_indices.size());
  const double threshold_tolerance =
      kPromotionTieTolerance * std::max(1.0, kMaxBulkTiltThreshold);
  for (const int i : cache.bulk_indices) {
    const double tilt =
        theta[0] * static_cast<double>(score_sign) * cache.a[i] +
        theta[1] * cache.d[i];
    if (!std::isfinite(tilt)) return out;
    // Promote all rows crossing the diagnostic threshold. The small symmetric
    // tolerance includes numerical ties at the boundary and keeps the rule
    // stable under row permutations and common rescalings of the score.
    if (std::abs(tilt) + threshold_tolerance < kMaxBulkTiltThreshold) {
      continue;
    }
    promote[i] = 1;
    promoted_indices.push_back(i);
  }

  out.valid = true;
  if (promoted_indices.empty()) return out;
  const std::size_t new_exact_count =
      cache.exact_indices.size() + promoted_indices.size();

  std::vector<int> new_exact_indices;
  new_exact_indices.reserve(new_exact_count);
  std::merge(cache.exact_indices.begin(), cache.exact_indices.end(),
             promoted_indices.begin(), promoted_indices.end(),
             std::back_inserter(new_exact_indices));
  std::vector<int> new_bulk_indices;
  new_bulk_indices.reserve(cache.bulk_indices.size() - promoted_indices.size());
  Vector2d new_bulk_mean = Vector2d::Zero();
  Matrix2d new_bulk_covariance = Matrix2d::Zero();
  for (const int i : cache.bulk_indices) {
    if (promote[i]) continue;
    new_bulk_indices.push_back(i);
    const Vector2d g(cache.a[i], cache.d[i]);
    const double p = cache.propensity[i];
    new_bulk_mean.noalias() += p * g;
    new_bulk_covariance.noalias() +=
        p * (1.0 - p) * (g * g.transpose());
  }
  if (!new_bulk_mean.allFinite() || !new_bulk_covariance.allFinite()) {
    out.valid = false;
    return out;
  }

  cache.exact_indices = std::move(new_exact_indices);
  cache.bulk_indices = std::move(new_bulk_indices);
  cache.bulk_mean = new_bulk_mean;
  cache.bulk_covariance = new_bulk_covariance;
  out.made_progress = true;
  out.promoted_count = static_cast<int>(promoted_indices.size());
  return out;
}

double directional_lr_upper_tail(const double rate, const double q2) {
  if (!std::isfinite(rate) || !std::isfinite(q2) ||
      rate <= 0.0 || q2 <= 0.0) return kNaN;
  const long double r_extended =
      std::sqrt(static_cast<long double>(2.0) * rate);
  const long double u_extended =
      std::sqrt(static_cast<long double>(q2));
  if (!std::isfinite(r_extended) || !std::isfinite(u_extended) ||
      r_extended <= 0.0L || u_extended <= 0.0L) return kNaN;
  const long double correction_extended =
      (r_extended - u_extended) / (u_extended * r_extended);
  const double r = static_cast<double>(r_extended);
  const double correction = static_cast<double>(correction_extended);
  if (!std::isfinite(r) || !std::isfinite(correction)) return kNaN;
  return R::pnorm(r, 0.0, 1.0, false, false) +
         R::dnorm(r, 0.0, 1.0, false) * correction;
}

struct DirectionalLrDiagnostic {
  bool has_cgf = false;
  bool regular = false;
  double q2 = kNaN;
  double raw_p = kNaN;
};

DirectionalLrDiagnostic diagnose_directional_lr(const CrtSolve& solve) {
  DirectionalLrDiagnostic out;
  out.has_cgf = solve.value.cgf.valid;
  if (out.has_cgf) {
    out.q2 = solve.theta.dot(solve.value.cgf.hessian * solve.theta);
  }
  if (solve.converged && out.has_cgf) {
    out.raw_p = directional_lr_upper_tail(solve.value.cgf.rate, out.q2);
  }
  out.regular = solve.converged && solve.value.valid && out.has_cgf &&
                solve.value.s > 0.0 && solve.theta[0] > 0.0 &&
                solve.theta[1] < 0.0 && solve.value.cgf.rate > 0.0 &&
                out.q2 > 0.0 && std::isfinite(out.raw_p) &&
                out.raw_p >= -1e-10 && out.raw_p <= 1.0 + 1e-10;
  return out;
}

Rcpp::List format_result(const EmpiricalFastCache& cache,
                         const CrtSolve& solve,
                         const double target,
                         const int score_sign,
                         const double tolerance,
                         const int promotion_rounds,
                         const int promoted_count,
                         const std::string& promotion_failure_reason) {
  const DirectionalLrDiagnostic lr = diagnose_directional_lr(solve);
  const bool has_cgf = lr.has_cgf;
  const NormalBulkDiagnostic diagnostic =
      diagnose_normal_bulk(cache, solve.theta, score_sign);
  const double score_scale = cache.score_scale;
  const double score_scale_squared = score_scale * score_scale;
  Vector2d theta_original;
  theta_original << solve.theta[0] / score_scale,
      solve.theta[1] / score_scale_squared;
  const double q2 = lr.q2;
  const double raw_p = lr.raw_p;
  const bool directional_regular = lr.regular;
  const bool promotion_failed = !promotion_failure_reason.empty();
  const bool regular = directional_regular && diagnostic.valid &&
                       diagnostic.safe && !promotion_failed;
  std::string reason;
  if (!solve.converged) {
    reason = solve.reason;
  } else if (!directional_regular) {
    reason = "nonregular_directional_lr";
  } else if (promotion_failed) {
    reason = promotion_failure_reason;
  } else if (!diagnostic.valid) {
    reason = "invalid_fast_normal_bulk_diagnostic";
  } else if (diagnostic.berry_esseen_ratio > kBerryEsseenThreshold) {
    reason = "fast_normal_bulk_berry_esseen_unsafe";
  } else if (diagnostic.max_abs_tilt > kMaxBulkTiltThreshold) {
    reason = "fast_normal_bulk_tilt_unsafe";
  } else {
    reason = "ok";
  }
  const std::string promotion_stop_reason =
      promotion_failed
          ? (promotion_failure_reason ==
                     "fast_normal_bulk_promotion_round_limit"
                 ? "maximum_rounds"
                 : (promotion_failure_reason ==
                            "fast_normal_bulk_promotion_no_progress"
                        ? "no_progress"
                        : "invalid_promoted_partition"))
          : (promotion_rounds == 0
                 ? "not_needed"
                 : (regular
                        ? "safe_after_promotion"
                        : (reason == "fast_normal_bulk_berry_esseen_unsafe"
                               ? "berry_esseen_gate"
                               : "post_promotion_solver_or_lr_failure")));
  const double p = regular
                       ? std::max(0.0, std::min(1.0, raw_p))
                       : kNaN;

  Rcpp::NumericVector moment(2, NA_REAL);
  Rcpp::NumericMatrix hessian(2, 2);
  std::fill(hessian.begin(), hessian.end(), NA_REAL);
  if (has_cgf) {
    Vector2d moment_original;
    moment_original << score_scale * solve.value.cgf.moment[0],
        score_scale_squared * solve.value.cgf.moment[1];
    Matrix2d scale_matrix = Matrix2d::Zero();
    scale_matrix(0, 0) = score_scale;
    scale_matrix(1, 1) = score_scale_squared;
    const Matrix2d hessian_original =
        scale_matrix * solve.value.cgf.hessian * scale_matrix;
    moment = Rcpp::wrap(moment_original);
    hessian = Rcpp::wrap(hessian_original);
  }
  Rcpp::NumericVector outer_residual(2, NA_REAL);
  Rcpp::NumericVector scaled_outer_residual(2, NA_REAL);
  if (solve.value.residual.allFinite()) {
    Vector2d residual_original;
    residual_original <<
        score_scale_squared * solve.value.residual[0],
        solve.value.residual[1] / score_scale;
    outer_residual = Rcpp::wrap(residual_original);
  }
  if (solve.value.scaled_residual.allFinite()) {
    scaled_outer_residual = Rcpp::wrap(solve.value.scaled_residual);
  }

  const int exact_count = static_cast<int>(cache.exact_indices.size());
  const int bulk_count = static_cast<int>(cache.bulk_indices.size());
  const double exact_fraction =
      static_cast<double>(exact_count) / static_cast<double>(cache.n);
  const double bulk_fraction =
      static_cast<double>(bulk_count) / static_cast<double>(cache.n);
  const double initial_exact_fraction =
      static_cast<double>(cache.initial_exact_count) /
      static_cast<double>(cache.n);
  const double initial_bulk_fraction =
      static_cast<double>(cache.initial_bulk_count) /
      static_cast<double>(cache.n);

  return Rcpp::List::create(
      Rcpp::Named("converged") = regular,
      Rcpp::Named("root_converged") = solve.converged,
      Rcpp::Named("reason") = reason,
      Rcpp::Named("p_value") = std::isfinite(p) ? p : NA_REAL,
      Rcpp::Named("raw_p_value") =
          std::isfinite(raw_p) ? raw_p : NA_REAL,
      Rcpp::Named("target") = target,
      Rcpp::Named("score_sign") = score_sign,
      Rcpp::Named("theta") = Rcpp::wrap(theta_original),
      Rcpp::Named("state") = Rcpp::wrap(theta_original),
      Rcpp::Named("normalized_state") = Rcpp::wrap(solve.theta),
      Rcpp::Named("moment") = moment,
      Rcpp::Named("hessian") = hessian,
      Rcpp::Named("K") =
          has_cgf && std::isfinite(solve.value.cgf.K)
              ? solve.value.cgf.K : NA_REAL,
      Rcpp::Named("rate") =
          has_cgf && std::isfinite(solve.value.cgf.rate)
              ? solve.value.cgf.rate : NA_REAL,
      Rcpp::Named("outer_residual") = outer_residual,
      Rcpp::Named("scaled_outer_residual") = scaled_outer_residual,
      Rcpp::Named("max_residual") =
          solve.value.scaled_residual.allFinite()
              ? max_abs(solve.value.scaled_residual) : NA_REAL,
      Rcpp::Named("root_tolerance") = tolerance,
      Rcpp::Named("q_scale") = solve.q_scale * score_scale_squared,
      Rcpp::Named("score_normalization_scale") = score_scale,
      Rcpp::Named("kappa") = solve.value.kappa,
      Rcpp::Named("center") = solve.center,
      Rcpp::Named("center_sd") = solve.center_sd,
      Rcpp::Named("iterations") = solve.iterations,
      Rcpp::Named("outer_iterations") = solve.iterations,
      Rcpp::Named("evaluations") = solve.evaluations,
      Rcpp::Named("history") = Rcpp::wrap(solve.history),
      Rcpp::Named("r_lr") =
          has_cgf && solve.value.cgf.rate > 0.0
              ? std::sqrt(2.0 * solve.value.cgf.rate) : NA_REAL,
      Rcpp::Named("q_lr") =
          std::isfinite(q2) && q2 > 0.0 ? std::sqrt(q2) : NA_REAL,
      Rcpp::Named("directional_q2") =
          std::isfinite(q2) ? q2 : NA_REAL,
      Rcpp::Named("outer_dimension") = 2,
      Rcpp::Named("inner_dimension") = 0,
      Rcpp::Named("statistic_id") = kStatisticId,
      Rcpp::Named("equation_id") = kEquationId,
      Rcpp::Named("path") = "partial_normal_fast_bernoulli_2d_newton",
      Rcpp::Named("solver") = "full_line_search_newton",
      Rcpp::Named("cgf_evaluation") =
          "exact_positive_outcome_plus_moment_matched_zero_outcome_gaussian_2d",
      Rcpp::Named("partition_rule") =
          "positive_outcome_exact_zero_outcome_gaussian_then_target_tilt_promotion",
      Rcpp::Named("initial_partition_rule") = "E={Y>0};B={Y=0}",
      Rcpp::Named("initial_exact_count") = cache.initial_exact_count,
      Rcpp::Named("initial_bulk_count") = cache.initial_bulk_count,
      Rcpp::Named("initial_exact_fraction") = initial_exact_fraction,
      Rcpp::Named("initial_bulk_fraction") = initial_bulk_fraction,
      Rcpp::Named("nonzero_outcome_count") = cache.initial_exact_count,
      Rcpp::Named("positive_outcome_count") = cache.initial_exact_count,
      Rcpp::Named("zero_outcome_count") = cache.initial_bulk_count,
      Rcpp::Named("exact_count") = exact_count,
      Rcpp::Named("bulk_count") = bulk_count,
      Rcpp::Named("exact_fraction") = exact_fraction,
      Rcpp::Named("bulk_fraction") = bulk_fraction,
      Rcpp::Named("fast_exact_count") = exact_count,
      Rcpp::Named("fast_bulk_count") = bulk_count,
      Rcpp::Named("fast_exact_fraction") = exact_fraction,
      Rcpp::Named("fast_bulk_fraction") = bulk_fraction,
      Rcpp::Named("promotion_rounds") = promotion_rounds,
      Rcpp::Named("promoted_count") = promoted_count,
      Rcpp::Named("maximum_promotion_rounds") = kMaxPromotionRounds,
      Rcpp::Named("promotion_failure_reason") =
          promotion_failed ? promotion_failure_reason : std::string("none"),
      Rcpp::Named("promotion_stop_reason") = promotion_stop_reason,
      Rcpp::Named("directional_berry_esseen") =
          diagnostic.valid ? diagnostic.berry_esseen_ratio : NA_REAL,
      Rcpp::Named("directional_berry_esseen_threshold") =
          kBerryEsseenThreshold,
      Rcpp::Named("berry_esseen_constant") = kBerryEsseenConstant,
      Rcpp::Named("normal_bulk_directional_variance") =
          diagnostic.valid ? diagnostic.directional_variance : NA_REAL,
      Rcpp::Named("normal_bulk_absolute_third_moment_sum") =
          diagnostic.valid ? diagnostic.absolute_third_moment_sum : NA_REAL,
      Rcpp::Named("max_bulk_tilt") =
          diagnostic.valid ? diagnostic.max_abs_tilt : NA_REAL,
      Rcpp::Named("max_bulk_tilt_threshold") = kMaxBulkTiltThreshold,
      Rcpp::Named("minimum_bulk_fraction") = 0.0,
      Rcpp::Named("empty_bulk_is_exact_safe") = true,
      Rcpp::Named("approximation_safe") = regular,
      Rcpp::Named("root_certified_to_solver_tolerance") = solve.converged,
      Rcpp::Named("spa_tail_geometry") = kTailGeometry,
      Rcpp::Named("spa_experimental") = true,
      Rcpp::Named("curved_boundary_tail_correction") = false,
      Rcpp::Named("spa_diagnostic") =
          "directional LR evaluates positive-outcome rows exactly, starts "
          "zero-outcome rows in the moment-matched Gaussian block, promotes "
          "zero-outcome rows with excessive target-direction tilt, and "
          "requests empirical B2 fallback when the remaining Gaussian block "
          "does not pass its diagnostics");
}

}  // namespace

namespace sceptre {

Rcpp::List crt_empirical_spa_full_fast(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& propensity,
    const Rcpp::NumericVector& y,
    const double target,
    const int score_sign,
    const double tolerance,
    const int max_iterations,
    const int max_backtracks) {
  if (score_sign != -1 && score_sign != 1) {
    Rcpp::stop("score_sign must be -1 or +1");
  }
  if (!std::isfinite(target) || target <= 0.0) {
    Rcpp::stop("target must be positive and finite");
  }
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    Rcpp::stop("tolerance must be positive and finite");
  }
  if (max_iterations < 0) {
    Rcpp::stop("max_iterations must be nonnegative");
  }
  if (max_backtracks < 1) {
    Rcpp::stop("max_backtracks must be positive");
  }
  EmpiricalFastCache cache = build_cache(a, propensity, y);
  CrtSolve solve = solve_full(cache, target, score_sign, tolerance,
                              max_iterations, max_backtracks);
  int promotion_rounds = 0;
  int promoted_count = 0;
  std::string promotion_failure_reason;

  while (solve.converged) {
    const DirectionalLrDiagnostic lr = diagnose_directional_lr(solve);
    if (!lr.regular) break;
    const NormalBulkDiagnostic diagnostic =
        diagnose_normal_bulk(cache, solve.theta, score_sign);
    if (!diagnostic.valid ||
        diagnostic.berry_esseen_ratio > kBerryEsseenThreshold ||
        diagnostic.max_abs_tilt <= kMaxBulkTiltThreshold) {
      break;
    }
    if (promotion_rounds >= kMaxPromotionRounds) {
      promotion_failure_reason = "fast_normal_bulk_promotion_round_limit";
      break;
    }

    const PromotionAttempt promotion =
        promote_unsafe_bulk_rows(cache, solve.theta, score_sign);
    if (!promotion.valid) {
      promotion_failure_reason = "fast_normal_bulk_promotion_invalid";
      break;
    }
    if (!promotion.made_progress || promotion.promoted_count <= 0) {
      promotion_failure_reason = "fast_normal_bulk_promotion_no_progress";
      break;
    }

    ++promotion_rounds;
    promoted_count += promotion.promoted_count;
    solve = solve_full(cache, target, score_sign, tolerance,
                       max_iterations, max_backtracks);
  }

  return format_result(cache, solve, target, score_sign, tolerance,
                       promotion_rounds, promoted_count,
                       promotion_failure_reason);
}

}  // namespace sceptre

// [[Rcpp::export]]
Rcpp::List crt_empirical_spa_full_fast_cpp(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& propensity,
    const Rcpp::NumericVector& y,
    const double target,
    const int score_sign = 1,
    const double tolerance = 1e-5,
    const int max_iterations = 60,
    const int max_backtracks = 24) {
  return sceptre::crt_empirical_spa_full_fast(
      a, propensity, y, target, score_sign, tolerance, max_iterations,
      max_backtracks);
}
