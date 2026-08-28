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

double softplus_stable(const double x) {
  return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
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
  MatrixXd G;  // Columns: signed a, then w * Z.
  MatrixXd C_inv;
  VectorXd offsets;
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

double lr_upper_tail(const double rate, const double q2,
                     const double direction) {
  if (!std::isfinite(rate) || !std::isfinite(q2) || q2 <= 0.0) return kNaN;
  const double sign = direction < 0.0 ? -1.0 : 1.0;
  const double r_lr = sign * std::sqrt(2.0 * std::max(rate, 0.0));
  const double q_lr = sign * std::sqrt(q2);
  if (std::abs(r_lr) < 0.1 || std::abs(q_lr) < 1e-10) {
    return R::pnorm(r_lr, 0.0, 1.0, false, false);
  }
  return R::pnorm(r_lr, 0.0, 1.0, false, false) +
         R::dnorm(r_lr, 0.0, 1.0, false) *
             (1.0 / q_lr - 1.0 / r_lr);
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
    const double eta = cache.offsets[i] + tilt;
    const double pi = expit_stable(eta);
    rate += tilt * pi -
            (softplus_stable(eta) - softplus_stable(cache.offsets[i]));
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
                      const int score_sign) {
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
  MatrixXd C = MatrixXd::Zero(p, p);
  VectorXd weighted_z = VectorXd::Zero(p);

  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(a[i])) Rcpp::stop("a must be finite");
    if (!std::isfinite(w[i]) || w[i] <= 0.0) {
      Rcpp::stop("w must be positive and finite");
    }
    if (!std::isfinite(propensity[i]) || propensity[i] <= 0.0 ||
        propensity[i] >= 1.0) {
      Rcpp::stop("propensity must be finite and lie strictly between 0 and 1");
    }
    if (!std::isfinite(Z(i, 0)) || std::abs(Z(i, 0) - 1.0) > 1e-10) {
      Rcpp::stop("the first column of Z must be an intercept");
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
    Rcpp::stop("Z transpose diag(w) Z must be positive definite");
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
                        const int max_iterations) {
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
    if (residual <= tolerance) {
      converged = true;
      break;
    }
    if (iteration == max_iterations) break;

    VectorXd step;
    if (!solve_linear(current.jacobian, -current.residual, step)) break;
    bool accepted = false;
    double scale = 1.0;
    for (int backtrack = 0; backtrack < 24; ++backtrack) {
      const VectorXd candidate = x + scale * step;
      Evaluation trial = evaluate_crt_exact(cache, candidate, target);
      if (trial.valid && max_abs(trial.residual) < residual) {
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

}  // namespace sceptre

// [[Rcpp::export]]
Rcpp::List crt_spa_full_cpp(const Rcpp::NumericVector& a,
                            const Rcpp::NumericVector& w,
                            const Rcpp::NumericMatrix& Z,
                            const Rcpp::NumericVector& propensity,
                            const double target,
                            const int score_sign = 1,
                            const double tolerance = 1e-9,
                            const int max_iterations = 50) {
  return sceptre::crt_spa_full(a, w, Z, propensity, target, score_sign,
                               tolerance, max_iterations);
}
