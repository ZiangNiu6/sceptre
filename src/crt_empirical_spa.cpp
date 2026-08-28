// Exact two-dimensional CRT saddlepoint approximation for the empirically
// studentized statistic S / sqrt(Q - S^2 / n).
//
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>

#include "crt_empirical_spa.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using Eigen::Matrix2d;
using Eigen::Vector2d;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
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

struct EmpiricalCrtCache {
  int n = 0;
  double score_scale = 1.0;
  Eigen::VectorXd a;
  Eigen::VectorXd d;
  Eigen::VectorXd propensity;
  Eigen::VectorXd offset;
  double sc = 0.0;
  double qc = 0.0;
};

EmpiricalCrtCache build_cache(const Rcpp::NumericVector& a,
                              const Rcpp::NumericVector& propensity) {
  const int n = a.size();
  if (n < 2 || propensity.size() != n) {
    Rcpp::stop("a and propensity must have the same length of at least two");
  }

  EmpiricalCrtCache cache;
  cache.n = n;
  cache.a.resize(n);
  cache.d.resize(n);
  cache.propensity.resize(n);
  cache.offset.resize(n);
  double score_scale = 0.0;
  for (int i = 0; i < n; ++i) {
    const double ai = a[i];
    const double p = propensity[i];
    if (!std::isfinite(ai)) Rcpp::stop("a must be finite");
    if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
      Rcpp::stop("propensity must be finite and lie in [0, 1]");
    }
    score_scale = std::max(score_scale, std::abs(ai));
  }
  // The statistic and its null law are invariant to a common positive scale
  // on a.  Solving in max-|a| units prevents small but perfectly valid score
  // vectors from being mistaken for degenerate ones.  Audit quantities are
  // transformed back to the caller's original units in format_result().
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
    Rcpp::stop("empirical CRT sparse-cache constants must be finite");
  }
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

// h(x) = (1 + x) log(1 + x) - x is the cancellation-free building block
// for a Bernoulli KL divergence.  The direct expression loses all useful
// digits near zero, so use its convergent power series there.  This is a
// stable evaluation of an exact identity, not an approximation to the CGF
// equations or to the Newton solve.
double xlog1px_minus_x(const double x) {
  if (!std::isfinite(x) || x < -1.0 - 1.0e-14) return kNaN;
  if (x <= -1.0) return 1.0;
  if (std::abs(x) >= 1.0e-3) {
    return (1.0 + x) * std::log1p(x) - x;
  }
  double power = x * x;
  double value = 0.5 * power;
  // h(x) = sum_{k=2}^infinity (-1)^k x^k / (k (k - 1)).
  // Twelve terms are ample on |x| < 1e-3 (the omitted term is < 1e-40).
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

  // Under exponential tilt t,
  //   pi - p = p(1-p) expm1(t) / (1 + p expm1(t)).
  // Expressing KL(Ber(pi) || Ber(p)) in terms of this difference avoids the
  // O(t) - O(t) cancellation in pi*t - log E exp(tX).
  double delta = kNaN;
  if (tilt > 0.0) {
    const double exp_negative = std::exp(-tilt);
    const double denominator = p + (1.0 - p) * exp_negative;
    if (!std::isfinite(denominator) || denominator <= 0.0) return kNaN;
    const double pi = p / denominator;
    delta = pi - p;
  } else {
    const double expm1_tilt = std::expm1(tilt);
    const double denominator = 1.0 + p * expm1_tilt;
    if (!std::isfinite(denominator) || denominator <= 0.0) return kNaN;
    delta = p * (1.0 - p) * expm1_tilt / denominator;
  }
  // The positive-tilt branch above avoids expm1 overflow, but pi-p can lose
  // relative precision for a very small positive tilt.  Use the expm1 form
  // whenever it is safely representable.
  if (tilt > 0.0 && tilt < 700.0) {
    const double expm1_tilt = std::expm1(tilt);
    const double denominator = 1.0 + p * expm1_tilt;
    delta = p * (1.0 - p) * expm1_tilt / denominator;
  }

  const double x = delta / p;
  const double y = -delta / (1.0 - p);
  const double hx = xlog1px_minus_x(x);
  const double hy = xlog1px_minus_x(y);
  if (!std::isfinite(hx) || !std::isfinite(hy)) return kNaN;
  const double kl = p * hx + (1.0 - p) * hy;
  if (!std::isfinite(kl)) return kNaN;
  // Roundoff can produce a negative subnormal at the exact centre only.
  return kl >= 0.0 ? kl : (kl > -1.0e-15 ? 0.0 : kNaN);
}

struct CgfEvaluation {
  bool valid = false;
  double K = kNaN;
  double rate = kNaN;
  Vector2d moment = Vector2d::Constant(kNaN);
  Matrix2d hessian = Matrix2d::Constant(kNaN);
};

CgfEvaluation evaluate_cgf(const EmpiricalCrtCache& cache,
                           const Vector2d& theta,
                           const int score_sign) {
  CgfEvaluation out;
  if (!theta.allFinite() || (score_sign != -1 && score_sign != 1)) return out;

  const double signed_sc = static_cast<double>(score_sign) * cache.sc;
  out.K = theta[0] * signed_sc + theta[1] * cache.qc;
  out.moment << signed_sc, cache.qc;
  out.hessian.setZero();
  double stable_rate = 0.0;
  for (int i = 0; i < cache.n; ++i) {
    const double g0 = static_cast<double>(score_sign) * cache.a[i];
    const double g1 = cache.d[i];
    const double tilt = theta[0] * g0 + theta[1] * g1;
    const BernoulliTilt tilted = evaluate_bernoulli_tilt(
        cache.propensity[i], cache.offset[i], tilt);
    if (!tilted.valid) return CgfEvaluation();
    const double rate_term = bernoulli_kl_from_tilt(
        cache.propensity[i], tilt);
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
  // theta^T K'(theta) - K(theta) is exactly the sum of these tilted
  // Bernoulli KL terms; the deterministic sparse-cache constants cancel.
  // This form remains accurate when the requested statistic is close to the
  // null centre and both terms in the naive subtraction are O(theta).
  out.rate = stable_rate;
  out.valid = std::isfinite(out.K) && std::isfinite(out.rate) &&
              out.moment.allFinite() && out.hessian.allFinite();
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
  Vector2d residual = Vector2d::Constant(kNaN);       // exact F1, F2
  Vector2d scaled_residual = Vector2d::Constant(kNaN);
  Matrix2d jacobian = Matrix2d::Constant(kNaN);       // exact Jacobian
  Matrix2d scaled_jacobian = Matrix2d::Constant(kNaN);
  double s = kNaN;
  double q = kNaN;
  double variance = kNaN;
  double kappa = kNaN;
};

RootEvaluation evaluate_root(const EmpiricalCrtCache& cache,
                             const Vector2d& theta,
                             const double target,
                             const int score_sign,
                             const double q_scale) {
  RootEvaluation out;
  out.cgf = evaluate_cgf(cache, theta, score_sign);
  if (!out.cgf.valid || !std::isfinite(target) || target <= 0.0 ||
      !std::isfinite(q_scale) || q_scale <= 0.0) {
    return out;
  }

  out.s = out.cgf.moment[0];
  out.q = out.cgf.moment[1];
  out.variance = out.q - out.s * out.s / static_cast<double>(cache.n);
  out.kappa = 1.0 / (target * target) +
              1.0 / static_cast<double>(cache.n);
  const Matrix2d& H = out.cgf.hessian;

  // These are the exact reduced KKT equations.  Scaling the first row below
  // changes neither their root nor the Newton step; it only makes the merit
  // function insensitive to the physical units of Q.
  out.residual[0] = out.q - out.kappa * out.s * out.s;
  out.residual[1] =
      theta[0] + 2.0 * out.kappa * out.s * theta[1];
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
  const double variance_scale =
      std::max(std::abs(out.q), mean_correction);
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

CrtSolve solve_full(const EmpiricalCrtCache& cache,
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
      // Observed and empirical-resample statistics remain valid at the
      // endpoints.  Only the regular SPA solve requires an interior law, so
      // the adaptive wrapper can diagnose this and select B2 without aborting.
      out.reason = "non_interior_propensity";
      return out;
    }
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

  // max_iterations == 0 is an intentional, deterministic way for the
  // adaptive wrapper (and its tests) to force the empirical B2 fallback.
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
  RootEvaluation current = evaluate_root(
      cache, out.theta, target, score_sign, out.q_scale);
  ++out.evaluations;

  if (!current.valid) {
    // This remains an exact full-row solve: the alternative is only a seed,
    // and every accepted update uses the exact CGF and exact root Jacobian.
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
      if (candidate.valid &&
          candidate.scaled_residual.norm() < merit) {
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

double directional_lr_upper_tail(const double rate, const double q2) {
  if (!std::isfinite(rate) || !std::isfinite(q2) ||
      rate <= 0.0 || q2 <= 0.0) {
    return kNaN;
  }
  // Algebraically this is the ordinary Lugannani--Rice correction
  // (1/u - 1/r).  Evaluating it as (r-u)/(u*r), and carrying the square
  // roots and correction in long-double precision, avoids subtracting two
  // large reciprocals when a nonzero observed statistic is near the null
  // centre.  This is only a stable evaluation of the same directional LR
  // formula: it neither perturbs the target nor introduces a Gaussian/Taylor
  // saddlepoint solver.  The exactly central target remains nonregular and is
  // handled as a structured empirical-fallback request by the SPA-first
  // wrapper.
  const long double r_extended =
      std::sqrt(static_cast<long double>(2.0) * rate);
  const long double u_extended =
      std::sqrt(static_cast<long double>(q2));
  if (!std::isfinite(r_extended) || !std::isfinite(u_extended) ||
      r_extended <= 0.0L || u_extended <= 0.0L) {
    return kNaN;
  }
  const long double correction_extended =
      (r_extended - u_extended) / (u_extended * r_extended);
  const double r = static_cast<double>(r_extended);
  const double correction = static_cast<double>(correction_extended);
  if (!std::isfinite(r) || !std::isfinite(correction)) return kNaN;
  return R::pnorm(r, 0.0, 1.0, false, false) +
         R::dnorm(r, 0.0, 1.0, false) * correction;
}

Rcpp::List format_result(const EmpiricalCrtCache& cache,
                         const CrtSolve& solve,
                         const double target,
                         const int score_sign,
                         const double tolerance) {
  const bool has_cgf = solve.value.cgf.valid;
  const double score_scale = cache.score_scale;
  const double score_scale_squared = score_scale * score_scale;
  Vector2d theta_original;
  theta_original << solve.theta[0] / score_scale,
      solve.theta[1] / score_scale_squared;
  const double q2 = has_cgf
                        ? solve.theta.dot(solve.value.cgf.hessian * solve.theta)
                        : kNaN;
  const double raw_p = (solve.converged && has_cgf)
                           ? directional_lr_upper_tail(
                                 solve.value.cgf.rate, q2)
                           : kNaN;
  const bool regular =
      solve.converged && solve.value.valid && has_cgf &&
      solve.value.s > 0.0 && solve.theta[0] > 0.0 &&
      solve.theta[1] < 0.0 && solve.value.cgf.rate > 0.0 && q2 > 0.0 &&
      std::isfinite(raw_p) && raw_p >= -1e-10 && raw_p <= 1.0 + 1e-10;
  const std::string reason =
      regular ? "ok" :
      (solve.converged ? "nonregular_directional_lr" : solve.reason);
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
      Rcpp::Named("path") = "full_exact_bernoulli_2d_newton",
      Rcpp::Named("solver") = "full_line_search_newton",
      Rcpp::Named("cgf_evaluation") = "exact_product_bernoulli_2d",
      Rcpp::Named("root_certified_to_solver_tolerance") = solve.converged,
      Rcpp::Named("spa_tail_geometry") = kTailGeometry,
      Rcpp::Named("spa_experimental") = true,
      Rcpp::Named("curved_boundary_tail_correction") = false,
      Rcpp::Named("spa_diagnostic") =
          "directional LR is for the tangent half-space, not a derived "
          "curved-boundary correction");
}

}  // namespace

namespace sceptre {

Rcpp::List crt_empirical_spa_full(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& propensity,
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
  const EmpiricalCrtCache cache = build_cache(a, propensity);
  const CrtSolve solve = solve_full(cache, target, score_sign, tolerance,
                                    max_iterations, max_backtracks);
  return format_result(cache, solve, target, score_sign, tolerance);
}

}  // namespace sceptre

// [[Rcpp::export]]
Rcpp::List crt_empirical_spa_full_cpp(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& propensity,
    const double target,
    const int score_sign = 1,
    const double tolerance = 1e-9,
    const int max_iterations = 60,
    const int max_backtracks = 24) {
  return sceptre::crt_empirical_spa_full(
      a, propensity, target, score_sign, tolerance, max_iterations,
      max_backtracks);
}
