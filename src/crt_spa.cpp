// Exact Full Newton CRT saddlepoint approximation for SCEPTRE's
// information-studentized statistic.
//
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>

#include "crt_spa.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr double kTinyV = 1e-12;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double expit_stable(const double x) {
  if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
  const double ex = std::exp(x);
  return ex / (1.0 + ex);
}

double max_abs(const VectorXd& x) {
  if (x.size() == 0) return 0.0;
  return x.cwiseAbs().maxCoeff();
}

bool solve_linear(const MatrixXd& A, const VectorXd& b, VectorXd& x) {
  if (!A.allFinite() || !b.allFinite() || A.rows() != A.cols() ||
      A.rows() != b.size()) {
    return false;
  }
  try {
    Eigen::FullPivLU<MatrixXd> decomposition(A);
    if (!decomposition.isInvertible()) return false;
    x = decomposition.solve(b);
    return x.allFinite();
  } catch (...) {
    return false;
  }
}

void symmetrize_lower(MatrixXd& A) {
  for (Eigen::Index j = 0; j < A.rows(); ++j) {
    for (Eigen::Index k = 0; k < j; ++k) A(k, j) = A(j, k);
  }
}

struct InfoCache {
  int n = 0;
  int p = 0;
  int d = 0;
  int score_sign = 1;
  bool leading_intercept = true;
  bool information_invertible = false;
  MatrixXd G;  // Columns: signed a, then w * Z.
  MatrixXd C_inv;
  VectorXd offsets;
  VectorXd propensity;
  VectorXd weights;
};

struct Boundary {
  bool valid = false;
  double V = kNaN;
  double b = kNaN;
  VectorXd grad;
  MatrixXd hess;
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

struct Evaluation {
  bool valid = false;
  VectorXd residual;
  MatrixXd jacobian;
  VectorXd moment;
  MatrixXd Sgg;
  Boundary boundary;
};

Evaluation evaluate_crt_exact(const InfoCache& cache,
                              const VectorXd& x,
                              const double target) {
  Evaluation out;
  const int d = cache.d;
  if (x.size() != d + 1 || !x.allFinite()) {
    return out;
  }
  const VectorXd theta = x.head(d);
  const double lambda = x[d];
  out.moment.setZero(d);
  out.Sgg.setZero(d, d);

  for (int i = 0; i < cache.n; ++i) {
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < d; ++j) tilt += theta[j] * g[j * cache.n];
    if (!std::isfinite(tilt)) return out;
    const double pi = expit_stable(cache.offsets[i] + tilt);
    const double h = pi * (1.0 - pi);
    for (int j = 0; j < d; ++j) {
      const double gj = g[j * cache.n];
      out.moment[j] += pi * gj;
      for (int k = 0; k <= j; ++k) {
        out.Sgg(j, k) += h * gj * g[k * cache.n];
      }
    }
  }

  symmetrize_lower(out.Sgg);
  if (!out.moment.allFinite() || !out.Sgg.allFinite()) return out;
  out.boundary = boundary_terms(out.moment, cache.C_inv);
  if (!out.boundary.valid) return out;

  out.residual.setZero(d + 1);
  out.residual.head(d) = theta - lambda * out.boundary.grad;
  out.residual[d] = out.boundary.b - target;
  out.jacobian.setZero(d + 1, d + 1);
  out.jacobian.block(0, 0, d, d) =
      MatrixXd::Identity(d, d) -
      lambda * out.boundary.hess * out.Sgg;
  out.jacobian.block(0, d, d, 1) = -out.boundary.grad;
  out.jacobian.block(d, 0, 1, d) =
      (out.Sgg * out.boundary.grad).transpose();
  out.valid = out.residual.allFinite() && out.jacobian.allFinite();
  return out;
}

Rcpp::List failure_result(const InfoCache& cache,
                          const std::string& reason,
                          const double target,
                          const double center,
                          const int iterations,
                          const double max_residual,
                          const std::vector<double>& history,
                          const VectorXd& state,
                          const int exact_evaluations) {
  return Rcpp::List::create(
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
      Rcpp::Named("path") = "full_exact",
      Rcpp::Named("exact_evaluations") = exact_evaluations);
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

// Exact KL(Ber(pi_tilt) || Ber(p)) evaluated without the cancellation in
// tilt*pi_tilt - log E exp(tilt*X).  The small-x series evaluates the exact
// xlog1px identity stably; it is not an alternative CGF or Newton solver.
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

double lr_upper_tail(const double rate, const double q2,
                     const double direction) {
  if (!std::isfinite(rate) || !std::isfinite(q2) ||
      rate <= 0.0 || q2 <= 0.0) return kNaN;
  const long double sign = direction < 0.0 ? -1.0L : 1.0L;
  const long double r_lr_extended =
      sign * std::sqrt(static_cast<long double>(2.0) * rate);
  const long double q_lr_extended =
      sign * std::sqrt(static_cast<long double>(q2));
  if (!std::isfinite(r_lr_extended) || !std::isfinite(q_lr_extended) ||
      r_lr_extended == 0.0L || q_lr_extended == 0.0L) return kNaN;
  // This is exactly (1/q - 1/r), written to avoid subtracting two large
  // reciprocals near the null center.  No Gaussian central shortcut is used.
  const long double correction_extended =
      (r_lr_extended - q_lr_extended) /
      (q_lr_extended * r_lr_extended);
  const double r_lr = static_cast<double>(r_lr_extended);
  const double correction = static_cast<double>(correction_extended);
  if (!std::isfinite(r_lr) || !std::isfinite(correction)) return kNaN;
  return R::pnorm(r_lr, 0.0, 1.0, false, false) +
         R::dnorm(r_lr, 0.0, 1.0, false) * correction;
}

double crt_rate(const InfoCache& cache, const VectorXd& theta) {
  double rate = 0.0;
  for (int i = 0; i < cache.n; ++i) {
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < cache.d; ++j) {
      tilt += theta[j] * g[j * cache.n];
    }
    if (!std::isfinite(tilt)) return kNaN;
    const double rate_term = bernoulli_kl_from_tilt(
        cache.propensity[i], tilt);
    if (!std::isfinite(rate_term)) return kNaN;
    rate += rate_term;
  }
  return rate;
}

Rcpp::List finalize_crt(const InfoCache& cache,
                        const VectorXd& x,
                        const Evaluation& evaluation,
                        const double center,
                        const double target,
                        const int iterations,
                        const std::vector<double>& history) {
  const VectorXd theta = x.head(cache.d);
  const double rate = crt_rate(cache, theta);
  const double q2 = theta.dot(evaluation.Sgg * theta);
  const double raw_p = lr_upper_tail(rate, q2, theta[0]);
  const double lambda = x[cache.d];
  const bool regular = target > center && lambda > 0.0 && theta[0] > 0.0 &&
                       rate > 0.0 && q2 > 0.0;
  const bool in_range = std::isfinite(raw_p) && raw_p >= -1e-10 &&
                        raw_p <= 1.0 + 1e-10;
  const bool valid = evaluation.valid && regular && in_range;
  const double sign = theta[0] < 0.0 ? -1.0 : 1.0;

  return Rcpp::List::create(
      Rcpp::Named("converged") = valid,
      Rcpp::Named("reason") =
          !regular ? "nonregular_upper_root" :
          (!in_range ? "lr_out_of_range" : "ok"),
      Rcpp::Named("p_value") =
          valid ? std::max(0.0, std::min(1.0, raw_p)) : kNaN,
      Rcpp::Named("raw_p_value") = raw_p,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("max_residual") = max_abs(evaluation.residual),
      Rcpp::Named("history") = Rcpp::wrap(history),
      Rcpp::Named("state") = Rcpp::wrap(x),
      Rcpp::Named("center") = center,
      Rcpp::Named("target") = target,
      Rcpp::Named("rate") = rate,
      Rcpp::Named("r_lr") =
          sign * std::sqrt(2.0 * std::max(rate, 0.0)),
      Rcpp::Named("q_lr") = sign * std::sqrt(std::max(q2, 0.0)),
      Rcpp::Named("moment_dimension") = cache.d,
      Rcpp::Named("nuisance_dimension") = cache.p,
      Rcpp::Named("score_sign") = cache.score_sign,
      Rcpp::Named("path") = "full_exact",
      Rcpp::Named("exact_evaluations") =
          static_cast<int>(history.size()) + 1);
}

InfoCache build_cache(const Rcpp::NumericVector& a,
                      const Rcpp::NumericVector& w,
                      const Rcpp::NumericMatrix& Z,
                      const Rcpp::NumericVector& propensity,
                      const int score_sign,
                      const bool require_interior_propensity = true,
                      const bool require_supported_geometry = true) {
  const int n = a.size();
  const int p = Z.ncol();
  if (n < 2 || p < 1 || Z.nrow() != n || w.size() != n ||
      propensity.size() != n) {
    Rcpp::stop("input dimensions do not match");
  }
  if (score_sign != -1 && score_sign != 1) {
    Rcpp::stop("score_sign must be -1 or +1");
  }

  InfoCache cache;
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
    cache.offsets[i] =
        std::log(propensity[i] / (1.0 - propensity[i]));
    cache.propensity[i] = propensity[i];
    cache.weights[i] = w[i];
    if (std::abs(Z(i, 0) - 1.0) > 1e-10) {
      cache.leading_intercept = false;
    }
  }

  if (require_supported_geometry && !cache.leading_intercept) {
    Rcpp::stop("the first column of Z must be an intercept");
  }

  symmetrize_lower(C);
  if (!C.allFinite()) {
    Rcpp::stop("Z transpose diag(w) Z must be finite");
  }
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
  return cache;
}

void validate_solver_controls(const double target,
                              const double tolerance,
                              const int max_iterations) {
  if (!std::isfinite(target)) Rcpp::stop("target must be finite");
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    Rcpp::stop("tolerance must be positive and finite");
  }
  if (max_iterations < 0) {
    Rcpp::stop("max_iterations cannot be negative");
  }
}

Rcpp::List run_crt_full(const InfoCache& cache,
                        const double target,
                        const double tolerance,
                        const int max_iterations,
                        const int minimum_updates = 0) {
  VectorXd x = VectorXd::Zero(cache.d + 1);
  const std::vector<double> empty_history;
  Evaluation base = evaluate_crt_exact(cache, x, target);
  if (!base.valid) {
    return failure_result(cache, "invalid_center", target, kNaN, 0, kNaN,
                          empty_history, x, 1);
  }

  const double center = base.boundary.b;
  const double variance =
      base.boundary.grad.dot(base.Sgg * base.boundary.grad);
  if (!std::isfinite(variance) || variance <= 1e-14) {
    return failure_result(cache, "degenerate_center_variance", target,
                          center, 0, kNaN, empty_history, x, 1);
  }

  x[cache.d] = (target - center) / variance;
  x.head(cache.d) = x[cache.d] * base.boundary.grad;
  Evaluation current = evaluate_crt_exact(cache, x, target);
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
      const VectorXd candidate = x + scale * step;
      Evaluation trial = evaluate_crt_exact(cache, candidate, target);
      const double trial_residual =
          trial.valid ? max_abs(trial.residual) : kNaN;
      if (trial.valid &&
          (trial_residual < residual ||
           (forced_refinement && trial_residual <= tolerance))) {
        x = candidate;
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
        current.valid ? max_abs(current.residual) : kNaN, history, x,
        static_cast<int>(history.size()) + 1);
  }
  return finalize_crt(cache, x, current, center, target, updates, history);
}

}  // namespace

namespace sceptre {

Rcpp::List crt_spa_full(const Rcpp::NumericVector& a,
                        const Rcpp::NumericVector& w,
                        const Rcpp::NumericMatrix& Z,
                        const Rcpp::NumericVector& propensity,
                        const double target,
                        const int score_sign,
                        const double tolerance,
                        const int max_iterations) {
  validate_solver_controls(target, tolerance, max_iterations);
  const InfoCache cache = build_cache(a, w, Z, propensity, score_sign);
  return run_crt_full(cache, target, tolerance, max_iterations);
}

Rcpp::List crt_spa_full_outward(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::NumericVector& propensity,
    const Rcpp::IntegerVector& treated_indices,
    const double tolerance,
    const int max_iterations) {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    Rcpp::stop("tolerance must be positive and finite");
  }
  if (max_iterations < 0) {
    Rcpp::stop("max_iterations cannot be negative");
  }

  // Learn the null center from the untilted state only; no throwaway Newton
  // solve is performed.  Reflection of the score coordinate negates both the
  // observed statistic and this center while leaving the nuisance geometry
  // unchanged.
  const InfoCache positive_cache = build_cache(
      a, w, Z, propensity, 1, false, false);
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
  for (std::size_t j = 0; j < treated.size(); ++j) {
    const int index = treated[j];
    observed_U += positive_cache.G(index, 0);
    observed_weight += positive_cache.weights[index];
    observed_B += positive_cache.G.row(index)
                      .segment(1, positive_cache.p)
                      .transpose();
  }
  const double projection = positive_cache.information_invertible
                                ? observed_B.dot(
                                      positive_cache.C_inv * observed_B)
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
        kNaN, std::vector<double>(), zero_state, 0);
    out["base_center"] = NA_REAL;
    out["observed_target"] = NA_REAL;
    out["outward_score_sign"] = NA_INTEGER;
    return out;
  }
  if (!std::isfinite(observed_target)) {
    Rcpp::List out = failure_result(
        positive_cache, "invalid_observed_studentizer", kNaN, kNaN, 0,
        kNaN, std::vector<double>(), zero_state, 0);
    out["base_center"] = NA_REAL;
    out["observed_target"] = NA_REAL;
    out["outward_score_sign"] = NA_INTEGER;
    return out;
  }
  if (!positive_cache.leading_intercept) {
    Rcpp::List out = failure_result(
        positive_cache, "unsupported_spa_geometry_no_intercept",
        observed_target, kNaN, 0, kNaN, std::vector<double>(), zero_state,
        0);
    out["base_center"] = NA_REAL;
    out["observed_target"] = observed_target;
    out["outward_score_sign"] = NA_INTEGER;
    return out;
  }
  const Evaluation base = evaluate_crt_exact(
      positive_cache, zero_state, observed_target);
  if (!base.valid || !std::isfinite(base.boundary.b)) {
    Rcpp::List out = failure_result(
        positive_cache, "invalid_center", observed_target, kNaN, 0, kNaN,
        std::vector<double>(), zero_state, 1);
    out["base_center"] = NA_REAL;
    out["observed_target"] = observed_target;
    out["outward_score_sign"] = NA_INTEGER;
    return out;
  }

  const double base_center = base.boundary.b;
  for (R_xlen_t i = 0; i < propensity.size(); ++i) {
    if (propensity[i] <= 0.0 || propensity[i] >= 1.0) {
      Rcpp::List out = failure_result(
          positive_cache, "non_interior_propensity", observed_target,
          base_center, 0, NA_REAL, std::vector<double>(), zero_state, 1);
      out["base_center"] = base_center;
      out["observed_target"] = observed_target;
      out["outward_score_sign"] = NA_INTEGER;
      return out;
    }
  }
  const double displacement = observed_target - base_center;
  if (displacement == 0.0) {
    Rcpp::List out = failure_result(
        positive_cache, "central_target_requires_empirical_fallback",
        observed_target, base_center, 0, 0.0, std::vector<double>(),
        zero_state, 1);
    out["base_center"] = base_center;
    out["observed_target"] = observed_target;
    out["outward_score_sign"] = NA_INTEGER;
    return out;
  }

  const int score_sign = displacement > 0.0 ? 1 : -1;
  const double outward_target =
      static_cast<double>(score_sign) * observed_target;
  const InfoCache outward_cache = score_sign == 1
                                      ? positive_cache
                                      : build_cache(a, w, Z, propensity, -1);
  if (max_iterations == 0) {
    const VectorXd state = VectorXd::Zero(outward_cache.d + 1);
    Rcpp::List out = failure_result(
        outward_cache, "solver_disabled_max_iterations_zero",
        outward_target, static_cast<double>(score_sign) * base_center,
        0, NA_REAL, std::vector<double>(), state, 1);
    out["base_center"] = base_center;
    out["observed_target"] = observed_target;
    out["outward_score_sign"] = score_sign;
    return out;
  }

  // Require at least one exact Newton update.  Very close to the center the
  // first-order seed can already satisfy an absolute residual tolerance, but
  // its second-order correction is still needed for a stable LR limit.
  Rcpp::List out = run_crt_full(
      outward_cache, outward_target, tolerance, max_iterations, 1);
  out["base_center"] = base_center;
  out["observed_target"] = observed_target;
  out["outward_score_sign"] = score_sign;
  return out;
}

}  // namespace sceptre

// [[Rcpp::export]]
Rcpp::List crt_spa_full_cpp(const Rcpp::NumericVector& a,
                            const Rcpp::NumericVector& w,
                            const Rcpp::NumericMatrix& Z,
                            const Rcpp::NumericVector& propensity,
                            const double target,
                            const int score_sign = 1,
                            const double tolerance = 1e-5,
                            const int max_iterations = 50) {
  return sceptre::crt_spa_full(a, w, Z, propensity, target, score_sign,
                               tolerance, max_iterations);
}
