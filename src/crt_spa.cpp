// Exact Full Newton CRT saddlepoint approximation for SCEPTRE's
// information-studentized statistic.
//
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>

#include "crt_spa.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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
  bool use_prepared_kernels = false;
  MatrixXd G;  // Columns: signed a, then w * Z.
  MatrixXd C_inv;
  VectorXd offsets;
  VectorXd propensity;
  VectorXd weights;

  // Reference caches own their Eigen vectors. Prepared caches instead retain
  // R owners and borrow read-only data for the duration of the native call.
  // Owned pointers are resolved on access, so default copies/moves cannot leave
  // pointers into another cache's Eigen storage. Borrowed pointers remain valid
  // because the corresponding R owners are copied/moved with the cache.
  const double* offsets_data() const {
    return offsets_view_ == nullptr ? offsets.data() : offsets_view_;
  }

  const double* propensity_data() const {
    return propensity_view_ == nullptr ? propensity.data() : propensity_view_;
  }

  const double* weights_data() const {
    return weights_view_ == nullptr ? weights.data() : weights_view_;
  }

  void bind_prepared_arrays(const Rcpp::NumericVector& offsets_r,
                            const Rcpp::NumericVector& propensity_r,
                            const Rcpp::NumericVector& weights_r) {
    offsets_owner_ = offsets_r;
    propensity_owner_ = propensity_r;
    weights_owner_ = weights_r;
    offsets_view_ = offsets_r.begin();
    propensity_view_ = propensity_r.begin();
    weights_view_ = weights_r.begin();
  }

 private:
  Rcpp::RObject offsets_owner_;
  Rcpp::RObject propensity_owner_;
  Rcpp::RObject weights_owner_;
  const double* offsets_view_ = nullptr;
  const double* propensity_view_ = nullptr;
  const double* weights_view_ = nullptr;
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
  // Prepared d=7 evaluations own their tilts and the exact theta that produced
  // them. They move with an accepted Evaluation; rejected trials retain no
  // shared scratch state. A partial buffer is never eligible for rate reuse.
  VectorXd retained_tilts;
  VectorXd retained_theta;
  bool retained_tilts_complete = false;
};

struct ZeroMoments {
  VectorXd moment;
  MatrixXd Sgg;
  bool all_propensities_interior = true;
};

Evaluation evaluate_crt_zero(const InfoCache& cache,
                             const ZeroMoments& moments,
                             const double target) {
  Evaluation out;
  out.moment = moments.moment;
  out.Sgg = moments.Sgg;
  if (!cache.information_invertible ||
      out.moment.size() != cache.d || out.Sgg.rows() != cache.d ||
      out.Sgg.cols() != cache.d || !out.moment.allFinite() ||
      !out.Sgg.allFinite()) {
    return out;
  }
  out.boundary = boundary_terms(out.moment, cache.C_inv);
  if (!out.boundary.valid) return out;
  out.residual.setZero(cache.d + 1);
  out.residual[cache.d] = out.boundary.b - target;
  out.jacobian.setZero(cache.d + 1, cache.d + 1);
  out.jacobian.block(0, 0, cache.d, cache.d) =
      MatrixXd::Identity(cache.d, cache.d);
  out.jacobian.block(0, cache.d, cache.d, 1) = -out.boundary.grad;
  out.jacobian.block(cache.d, 0, 1, cache.d) =
      (out.Sgg * out.boundary.grad).transpose();
  out.valid = out.residual.allFinite() && out.jacobian.allFinite();
  return out;
}

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
  const double* offsets = cache.offsets_data();

  for (int i = 0; i < cache.n; ++i) {
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < d; ++j) tilt += theta[j] * g[j * cache.n];
    if (!std::isfinite(tilt)) return out;
    const double pi = expit_stable(offsets[i] + tilt);
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

// The common six-covariate design has seven score/nuisance moments. Gather
// each feature once per cell and use fixed-size local lower-triangle sums.
// Every accumulator retains the reference cell order and h * gj * gk expression.
// No horizontal reductions, approximate functions, or new FMA calls are used.
bool accumulate_crt_moments_d7(const InfoCache& cache,
                                const VectorXd& theta,
                                VectorXd& moment,
                                MatrixXd& covariance,
                                VectorXd& retained_tilts) {
  std::array<double, 7> moment_sum{};
  std::array<double, 28> covariance_sum{};
  std::array<const double*, 7> columns;
  for (int j = 0; j < 7; ++j) {
    columns[j] = cache.G.data() + j * cache.n;
  }
  const double* offsets = cache.offsets_data();
  bool finite_tilts = true;
  for (int i = 0; i < cache.n; ++i) {
    std::array<double, 7> g;
    for (int j = 0; j < 7; ++j) g[j] = columns[j][i];
    double tilt = 0.0;
    for (int j = 0; j < 7; ++j) tilt += theta[j] * g[j];
    if (!std::isfinite(tilt)) {
      finite_tilts = false;
      break;
    }
    retained_tilts[i] = tilt;
    const double pi = expit_stable(offsets[i] + tilt);
    const double h = pi * (1.0 - pi);
    int packed = 0;
    for (int j = 0; j < 7; ++j) {
      const double gj = g[j];
      moment_sum[j] += pi * gj;
      for (int k = 0; k <= j; ++k) {
        covariance_sum[packed++] += h * gj * g[k];
      }
    }
  }
  int packed = 0;
  for (int j = 0; j < 7; ++j) {
    moment[j] = moment_sum[j];
    for (int k = 0; k <= j; ++k) {
      covariance(j, k) = covariance_sum[packed++];
    }
  }
  return finite_tilts;
}

Evaluation evaluate_crt_optimized(const InfoCache& cache,
                              const VectorXd& x,
                              const double target) {
  if (cache.d != 7) return evaluate_crt_exact(cache, x, target);
  Evaluation out;
  const int d = cache.d;
  if (x.size() != d + 1 || !x.allFinite()) {
    return out;
  }
  const VectorXd theta = x.head(d);
  const double lambda = x[d];
  out.moment.setZero(d);
  out.Sgg.setZero(d, d);
  out.retained_tilts.resize(cache.n);

  if (!accumulate_crt_moments_d7(
          cache, theta, out.moment, out.Sgg, out.retained_tilts)) {
    return out;
  }
  out.retained_theta = theta;
  out.retained_tilts_complete = true;

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

Evaluation evaluate_crt_dispatch(const InfoCache& cache,
                                 const VectorXd& x,
                                 const double target) {
  return cache.use_prepared_kernels
             ? evaluate_crt_optimized(cache, x, target)
             : evaluate_crt_exact(cache, x, target);
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

// Prepared-only equivalent. For 0 < tilt < 700 the reference's initial
// exp(-tilt) result is overwritten by this same expm1 expression. Its initial
// denominator is necessarily positive and finite for an interior p, so skip
// that discarded calculation while retaining all endpoint/large-tilt guards.
double bernoulli_kl_from_tilt_optimized(const double p, const double tilt) {
  if (!std::isfinite(p) || !std::isfinite(tilt) || p < 0.0 || p > 1.0) {
    return kNaN;
  }
  if (p == 0.0 || p == 1.0 || tilt == 0.0) return 0.0;

  double delta = kNaN;
  if (tilt > 0.0 && tilt < 700.0) {
    const double expm1_tilt = std::expm1(tilt);
    const double denominator = 1.0 + p * expm1_tilt;
    delta = p * (1.0 - p) * expm1_tilt / denominator;
  } else if (tilt > 0.0) {
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
  const double* propensity = cache.propensity_data();
  for (int i = 0; i < cache.n; ++i) {
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < cache.d; ++j) {
      tilt += theta[j] * g[j * cache.n];
    }
    if (!std::isfinite(tilt)) return kNaN;
    const double rate_term = bernoulli_kl_from_tilt(
        propensity[i], tilt);
    if (!std::isfinite(rate_term)) return kNaN;
    rate += rate_term;
  }
  return rate;
}


double crt_rate_optimized(const InfoCache& cache, const VectorXd& theta) {
  double rate = 0.0;
  const double* propensity = cache.propensity_data();
  for (int i = 0; i < cache.n; ++i) {
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < cache.d; ++j) {
      tilt += theta[j] * g[j * cache.n];
    }
    if (!std::isfinite(tilt)) return kNaN;
    const double rate_term = bernoulli_kl_from_tilt_optimized(
        propensity[i], tilt);
    if (!std::isfinite(rate_term)) return kNaN;
    rate += rate_term;
  }
  return rate;
}
bool can_reuse_retained_tilts(const InfoCache& cache,
                               const Evaluation& evaluation,
                               const VectorXd& theta) {
  if (!evaluation.retained_tilts_complete ||
      evaluation.retained_tilts.size() != cache.n ||
      evaluation.retained_theta.size() != cache.d || theta.size() != cache.d) {
    return false;
  }
  // In particular, do not reuse tilts from an earlier/rejected Newton state.
  // Bitwise comparison also distinguishes signed zero without rescanning G.
  return std::memcmp(evaluation.retained_theta.data(), theta.data(),
                     static_cast<std::size_t>(cache.d) * sizeof(double)) == 0;
}

double crt_rate_from_retained_tilts(const InfoCache& cache,
                                    const VectorXd& tilts) {
  const double* propensity = cache.propensity_data();
  double rate = 0.0;
  for (int i = 0; i < cache.n; ++i) {
    const double tilt = tilts[i];
    if (!std::isfinite(tilt)) return kNaN;
    const double rate_term = bernoulli_kl_from_tilt_optimized(
        propensity[i], tilt);
    if (!std::isfinite(rate_term)) return kNaN;
    rate += rate_term;
  }
  return rate;
}

double select_prepared_crt_rate(const InfoCache& cache,
                                 const Evaluation& evaluation,
                                 const VectorXd& theta,
                                 bool* used_retained_rate = nullptr) {
  const bool reuse = can_reuse_retained_tilts(cache, evaluation, theta);
  if (used_retained_rate != nullptr) *used_retained_rate = reuse;
  return reuse ? crt_rate_from_retained_tilts(cache, evaluation.retained_tilts)
               : crt_rate_optimized(cache, theta);
}

Rcpp::List finalize_crt(const InfoCache& cache,
                        const VectorXd& x,
                        const Evaluation& evaluation,
                        const double center,
                        const double target,
                        const int iterations,
                        const std::vector<double>& history) {
  const VectorXd theta = x.head(cache.d);
  const double rate = cache.use_prepared_kernels
                          ? select_prepared_crt_rate(cache, evaluation, theta)
                          : crt_rate(cache, theta);
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
                      const bool require_supported_geometry = true,
                      ZeroMoments* zero_moments = nullptr) {
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
  if (zero_moments != nullptr) {
    zero_moments->moment = VectorXd::Zero(cache.d);
    zero_moments->Sgg = MatrixXd::Zero(cache.d, cache.d);
    zero_moments->all_propensities_interior = true;
  }

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
    if (zero_moments != nullptr) {
      if (propensity[i] <= 0.0 || propensity[i] >= 1.0) {
        zero_moments->all_propensities_interior = false;
      }
      // Preserve the exact evaluator's logit/expit round trip and summation
      // order while the feature row is already being constructed.
      const double pi = expit_stable(cache.offsets[i]);
      const double h = pi * (1.0 - pi);
      const double* g = cache.G.data() + i;
      for (int j = 0; j < cache.d; ++j) {
        const double gj = g[j * n];
        zero_moments->moment[j] += pi * gj;
        for (int k = 0; k <= j; ++k) {
          zero_moments->Sgg(j, k) += h * gj * g[k * n];
        }
      }
    }
  }

  if (require_supported_geometry && !cache.leading_intercept) {
    Rcpp::stop("the first column of Z must be an intercept");
  }

  symmetrize_lower(C);
  if (zero_moments != nullptr) symmetrize_lower(zero_moments->Sgg);
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

// Prepared contexts are internal, run-local inputs. Their schema/dimensions are
// checked here; the caller must rebuild them whenever the corresponding target
// probabilities or gene weights/design change. No process-global cache is used.
SEXP prepared_field(const Rcpp::List& preparation,
                     const char* name,
                     const SEXPTYPE type,
                     const char* context) {
  if (!preparation.containsElementNamed(name)) {
    Rcpp::stop(std::string(context) + " is missing " + name);
  }
  SEXP field = preparation[name];
  if (TYPEOF(field) != type) {
    Rcpp::stop(std::string(context) + " has an invalid " + name + " type");
  }
  return field;
}

int prepared_integer(const Rcpp::List& preparation,
                       const char* name,
                       const char* context) {
  SEXP field = prepared_field(preparation, name, INTSXP, context);
  if (Rf_xlength(field) != 1 || INTEGER(field)[0] == NA_INTEGER) {
    Rcpp::stop(std::string(context) + " has an invalid " + name);
  }
  return INTEGER(field)[0];
}

bool prepared_boolean(const Rcpp::List& preparation,
                        const char* name,
                        const char* context) {
  SEXP field = prepared_field(preparation, name, LGLSXP, context);
  if (Rf_xlength(field) != 1 || LOGICAL(field)[0] == NA_LOGICAL) {
    Rcpp::stop(std::string(context) + " has an invalid " + name);
  }
  return LOGICAL(field)[0] != 0;
}

Rcpp::NumericVector prepared_vector(const Rcpp::List& preparation,
                                     const char* name,
                                     const int n,
                                     const char* context) {
  SEXP field = prepared_field(preparation, name, REALSXP, context);
  if (Rf_xlength(field) != n || Rf_isMatrix(field)) {
    Rcpp::stop(std::string(context) + " has an invalid " + name + " length");
  }
  return Rcpp::NumericVector(field);
}

Rcpp::List prepare_target(const Rcpp::NumericVector& propensity) {
  const int n = propensity.size();
  if (n < 2) Rcpp::stop("input dimensions do not match");
  Rcpp::NumericVector offsets(n), zero_probability(n), zero_variance(n);
  bool all_propensities_interior = true;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(propensity[i]) || propensity[i] < 0.0 ||
        propensity[i] > 1.0) {
      Rcpp::stop("propensity must be finite and lie in [0, 1]");
    }
    offsets[i] = std::log(propensity[i] / (1.0 - propensity[i]));
    const double pi = expit_stable(offsets[i]);
    zero_probability[i] = pi;
    zero_variance[i] = pi * (1.0 - pi);
    if (propensity[i] <= 0.0 || propensity[i] >= 1.0) {
      all_propensities_interior = false;
    }
  }
  return Rcpp::List::create(
      Rcpp::Named("schema_version") = 1,
      Rcpp::Named("n") = n,
      Rcpp::Named("propensity") = propensity,
      Rcpp::Named("offsets") = offsets,
      Rcpp::Named("zero_probability") = zero_probability,
      Rcpp::Named("zero_variance") = zero_variance,
      Rcpp::Named("all_propensities_interior") = all_propensities_interior);
}

Rcpp::List prepare_gene_geometry(const Rcpp::NumericVector& w,
                                  const Rcpp::NumericMatrix& Z) {
  const int n = w.size();
  const int p = Z.ncol();
  if (n < 2 || p < 1 || Z.nrow() != n) {
    Rcpp::stop("input dimensions do not match");
  }
  MatrixXd C = MatrixXd::Zero(p, p);
  VectorXd weighted_z = VectorXd::Zero(p);
  bool leading_intercept = true;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(w[i]) || w[i] <= 0.0) {
      Rcpp::stop("w must be positive and finite");
    }
    for (int j = 0; j < p; ++j) {
      if (!std::isfinite(Z(i, j))) Rcpp::stop("Z must be finite");
      weighted_z[j] = w[i] * Z(i, j);
      if (!std::isfinite(weighted_z[j])) {
        Rcpp::stop("w multiplied by Z must be finite");
      }
    }
    // This expression and its accumulation order intentionally match the
    // reference build_cache(), including the division by w[i].
    for (int j = 0; j < p; ++j) {
      for (int k = 0; k <= j; ++k) {
        C(j, k) += weighted_z[j] * weighted_z[k] / w[i];
      }
    }
    if (std::abs(Z(i, 0) - 1.0) > 1e-10) leading_intercept = false;
  }
  symmetrize_lower(C);
  if (!C.allFinite()) {
    Rcpp::stop("Z transpose diag(w) Z must be finite");
  }
  MatrixXd C_inv;
  bool information_invertible = true;
  try {
    Eigen::LLT<MatrixXd> decomposition(C);
    if (decomposition.info() != Eigen::Success) {
      information_invertible = false;
    } else {
      C_inv = decomposition.solve(MatrixXd::Identity(p, p));
      information_invertible = decomposition.info() == Eigen::Success &&
                               C_inv.allFinite();
    }
  } catch (...) {
    information_invertible = false;
  }
  if (!information_invertible) {
    C_inv = MatrixXd::Constant(p, p, kNaN);
  }
  return Rcpp::List::create(
      Rcpp::Named("schema_version") = 1,
      Rcpp::Named("n") = n,
      Rcpp::Named("p") = p,
      Rcpp::Named("C_inv") = Rcpp::wrap(C_inv),
      Rcpp::Named("information_invertible") = information_invertible,
      Rcpp::Named("leading_intercept") = leading_intercept);
}

InfoCache build_prepared_cache(const Rcpp::NumericVector& a,
                                 const Rcpp::NumericVector& w,
                                 const Rcpp::NumericMatrix& Z,
                                 const Rcpp::List& target_prepared,
                                 const Rcpp::List& gene_geometry,
                                 ZeroMoments& moments) {
  const int n = a.size();
  const int p = Z.ncol();
  if (n < 2 || p < 1 || Z.nrow() != n || w.size() != n) {
    Rcpp::stop("input dimensions do not match");
  }
  if (prepared_integer(target_prepared, "schema_version", "target_prepared") != 1 ||
      prepared_integer(gene_geometry, "schema_version", "gene_geometry") != 1) {
    Rcpp::stop("unsupported CRT-SPA preparation schema version");
  }
  if (prepared_integer(target_prepared, "n", "target_prepared") != n ||
      prepared_integer(gene_geometry, "n", "gene_geometry") != n ||
      prepared_integer(gene_geometry, "p", "gene_geometry") != p) {
    Rcpp::stop("prepared CRT-SPA input dimensions do not match");
  }
  const Rcpp::NumericVector propensity = prepared_vector(
      target_prepared, "propensity", n, "target_prepared");
  const Rcpp::NumericVector offsets = prepared_vector(
      target_prepared, "offsets", n, "target_prepared");
  const Rcpp::NumericVector zero_probability = prepared_vector(
      target_prepared, "zero_probability", n, "target_prepared");
  const Rcpp::NumericVector zero_variance = prepared_vector(
      target_prepared, "zero_variance", n, "target_prepared");
  moments.all_propensities_interior = prepared_boolean(
      target_prepared, "all_propensities_interior", "target_prepared");
  const bool expected_intercept = prepared_boolean(
      gene_geometry, "leading_intercept", "gene_geometry");
  SEXP C_inv_sexp = prepared_field(gene_geometry, "C_inv", REALSXP, "gene_geometry");
  if (!Rf_isMatrix(C_inv_sexp)) Rcpp::stop("gene_geometry C_inv must be a matrix");
  const Rcpp::NumericMatrix C_inv(C_inv_sexp);
  if (C_inv.nrow() != p || C_inv.ncol() != p) {
    Rcpp::stop("gene_geometry C_inv dimensions do not match");
  }

  InfoCache cache;
  cache.n = n;
  cache.p = p;
  cache.d = p + 1;
  cache.score_sign = 1;
  cache.use_prepared_kernels = true;
  cache.information_invertible = prepared_boolean(
      gene_geometry, "information_invertible", "gene_geometry");
  cache.C_inv = Rcpp::as<MatrixXd>(C_inv);
  if (cache.information_invertible && !cache.C_inv.allFinite()) {
    Rcpp::stop("gene_geometry C_inv must be finite when information is invertible");
  }
  // Each row's score and all p weighted features are initialized below before
  // its mixed moments are read. No zero fill or owned n-vector copies are needed.
  cache.G.resize(n, p + 1);
  cache.bind_prepared_arrays(offsets, propensity, w);
  moments.moment = VectorXd::Zero(cache.d);
  moments.Sgg = MatrixXd::Zero(cache.d, cache.d);
  bool all_propensities_interior = true;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(a[i])) Rcpp::stop("a must be finite");
    if (!std::isfinite(w[i]) || w[i] <= 0.0) {
      Rcpp::stop("w must be positive and finite");
    }
    if (!std::isfinite(propensity[i]) || propensity[i] < 0.0 ||
        propensity[i] > 1.0) {
      Rcpp::stop("propensity must be finite and lie in [0, 1]");
    }
    cache.G(i, 0) = a[i];
    for (int j = 0; j < p; ++j) {
      if (!std::isfinite(Z(i, j))) Rcpp::stop("Z must be finite");
      const double weighted_z = w[i] * Z(i, j);
      if (!std::isfinite(weighted_z)) {
        Rcpp::stop("w multiplied by Z must be finite");
      }
      cache.G(i, j + 1) = weighted_z;
    }
    if (std::isnan(offsets[i]) || !std::isfinite(zero_probability[i]) ||
        zero_probability[i] < 0.0 || zero_probability[i] > 1.0 ||
        !std::isfinite(zero_variance[i]) || zero_variance[i] < 0.0 ||
        zero_variance[i] > 0.25) {
      Rcpp::stop("target_prepared contains invalid derived values");
    }
    if (std::abs(Z(i, 0) - 1.0) > 1e-10) cache.leading_intercept = false;
    if (propensity[i] <= 0.0 || propensity[i] >= 1.0) {
      all_propensities_interior = false;
    }
    const double pi = zero_probability[i];
    const double h = zero_variance[i];
    const double* g = cache.G.data() + i;
    for (int j = 0; j < cache.d; ++j) {
      const double gj = g[j * n];
      moments.moment[j] += pi * gj;
      for (int k = 0; k <= j; ++k) {
        moments.Sgg(j, k) += h * gj * g[k * n];
      }
    }
  }
  if (cache.leading_intercept != expected_intercept) {
    Rcpp::stop("gene_geometry does not match the intercept geometry of Z");
  }
  if (all_propensities_interior != moments.all_propensities_interior) {
    Rcpp::stop("target_prepared has an inconsistent propensity interior flag");
  }
  symmetrize_lower(moments.Sgg);
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
                        const int minimum_updates = 0,
                        const Evaluation* zero_evaluation = nullptr) {
  VectorXd x = VectorXd::Zero(cache.d + 1);
  const std::vector<double> empty_history;
  Evaluation base = zero_evaluation == nullptr
                        ? evaluate_crt_dispatch(cache, x, target)
                        : *zero_evaluation;
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
  Evaluation current = evaluate_crt_dispatch(cache, x, target);
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
      Evaluation trial = evaluate_crt_dispatch(cache, candidate, target);
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

Rcpp::List mark_cached_exact(Rcpp::List out) {
  out["cgf_mode"] = "exact_bernoulli";
  out["cache_strategy"] = "cached_exact";
  return out;
}

Rcpp::List cached_outward_result(Rcpp::List out,
                                  const double base_center,
                                  const double observed_target,
                                  const int score_sign) {
  out["base_center"] = base_center;
  out["observed_target"] = observed_target;
  out["outward_score_sign"] = score_sign;
  return mark_cached_exact(out);
}

Rcpp::List cached_outward_failure(const InfoCache& cache,
                                   const std::string& reason,
                                   const double target,
                                   const double center,
                                   const double observed,
                                   const int score_sign,
                                   const int exact_evaluations) {
  const VectorXd state = VectorXd::Zero(cache.d + 1);
  const double residual = reason == "central_target_requires_empirical_fallback"
                              ? 0.0
                              : (reason == "non_interior_propensity"
                                     ? NA_REAL : kNaN);
  Rcpp::List out = failure_result(
      cache, reason, target, center, 0, residual,
      std::vector<double>(), state, exact_evaluations);
  return cached_outward_result(
      out, std::isfinite(center) ? center : NA_REAL,
      std::isfinite(observed) ? observed : NA_REAL, score_sign);
}

Rcpp::List prepared_outward_result(Rcpp::List out,
                                     const double base_center,
                                     const double observed_target,
                                     const int score_sign) {
  out = cached_outward_result(out, base_center, observed_target, score_sign);
  out["cache_strategy"] = "shared_preparation";
  out["shared_preparation"] = true;
  return out;
}

Rcpp::List prepared_outward_failure(const InfoCache& cache,
                                      const std::string& reason,
                                      const double target,
                                      const double center,
                                      const double observed,
                                      const int score_sign,
                                      const int exact_evaluations) {
  Rcpp::List out = cached_outward_failure(
      cache, reason, target, center, observed, score_sign, exact_evaluations);
  out["cache_strategy"] = "shared_preparation";
  out["shared_preparation"] = true;
  return out;
}

Rcpp::List kernel_evaluation_diagnostics(const Evaluation& evaluation,
                                         const double rate,
                                         const Rcpp::NumericVector& cell_rate) {
  return Rcpp::List::create(
      Rcpp::Named("valid") = evaluation.valid,
      Rcpp::Named("moment") = Rcpp::wrap(evaluation.moment),
      Rcpp::Named("Sgg") = Rcpp::wrap(evaluation.Sgg),
      Rcpp::Named("residual") = Rcpp::wrap(evaluation.residual),
      Rcpp::Named("jacobian") = Rcpp::wrap(evaluation.jacobian),
      Rcpp::Named("boundary") = Rcpp::List::create(
          Rcpp::Named("valid") = evaluation.boundary.valid,
          Rcpp::Named("V") = evaluation.boundary.V,
          Rcpp::Named("b") = evaluation.boundary.b,
          Rcpp::Named("grad") = Rcpp::wrap(evaluation.boundary.grad),
          Rcpp::Named("hess") = Rcpp::wrap(evaluation.boundary.hess)),
      Rcpp::Named("rate") = rate,
      Rcpp::Named("cell_rate") = cell_rate);
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

Rcpp::List crt_spa_full_cached(const Rcpp::NumericVector& a,
                               const Rcpp::NumericVector& w,
                               const Rcpp::NumericMatrix& Z,
                               const Rcpp::NumericVector& propensity,
                               const double target,
                               const int score_sign,
                               const double tolerance,
                               const int max_iterations) {
  validate_solver_controls(target, tolerance, max_iterations);
  ZeroMoments moments;
  const InfoCache cache = build_cache(
      a, w, Z, propensity, score_sign, true, true, &moments);
  const Evaluation base = evaluate_crt_zero(cache, moments, target);
  return mark_cached_exact(run_crt_full(
      cache, target, tolerance, max_iterations, 0, &base));
}

Rcpp::List crt_spa_full_outward_cached(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::NumericVector& propensity,
    const Rcpp::IntegerVector& treated_indices,
    const double tolerance,
    const int max_iterations) {
  validate_solver_controls(0.0, tolerance, max_iterations);
  ZeroMoments moments;
  InfoCache cache = build_cache(
      a, w, Z, propensity, 1, false, false, &moments);
  std::vector<int> treated(treated_indices.size());
  for (R_xlen_t j = 0; j < treated_indices.size(); ++j) {
    if (treated_indices[j] == NA_INTEGER || treated_indices[j] < 1 ||
        treated_indices[j] > cache.n) {
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
  VectorXd observed_B = VectorXd::Zero(cache.p);
  for (std::size_t j = 0; j < treated.size(); ++j) {
    const int index = treated[j];
    observed_U += cache.G(index, 0);
    observed_weight += cache.weights[index];
    observed_B += cache.G.row(index).segment(1, cache.p).transpose();
  }
  const double projection = cache.information_invertible
                                ? observed_B.dot(cache.C_inv * observed_B)
                                : kNaN;
  const double observed_information = observed_weight - projection;
  const double information_scale =
      std::max(std::abs(observed_weight), std::abs(projection));
  const double observed =
      std::isfinite(observed_U) && std::isfinite(observed_information) &&
              std::isfinite(information_scale) && information_scale > 0.0 &&
              observed_information > 1.0e-14 * information_scale
          ? observed_U / std::sqrt(observed_information)
          : kNaN;
  if (!cache.information_invertible) {
    return cached_outward_failure(
        cache, "singular_information_geometry", kNaN, kNaN, kNaN,
        NA_INTEGER, 0);
  }
  if (!std::isfinite(observed)) {
    return cached_outward_failure(
        cache, "invalid_observed_studentizer", kNaN, kNaN, kNaN,
        NA_INTEGER, 0);
  }
  if (!cache.leading_intercept) {
    return cached_outward_failure(
        cache, "unsupported_spa_geometry_no_intercept", observed, kNaN,
        observed, NA_INTEGER, 0);
  }
  const Evaluation positive_base = evaluate_crt_zero(cache, moments, observed);
  if (!positive_base.valid || !std::isfinite(positive_base.boundary.b)) {
    return cached_outward_failure(
        cache, "invalid_center", observed, kNaN, observed, NA_INTEGER, 1);
  }
  const double base_center = positive_base.boundary.b;
  if (!moments.all_propensities_interior) {
    return cached_outward_failure(
        cache, "non_interior_propensity", observed, base_center,
        observed, NA_INTEGER, 1);
  }
  const double displacement = observed - base_center;
  if (displacement == 0.0) {
    return cached_outward_failure(
        cache, "central_target_requires_empirical_fallback", observed,
        base_center, observed, NA_INTEGER, 1);
  }

  const int score_sign = displacement > 0.0 ? 1 : -1;
  const double target = static_cast<double>(score_sign) * observed;
  cache.score_sign = score_sign;
  if (score_sign == -1) {
    // Reflect only the score coordinate. The nuisance geometry and its
    // factorization are unchanged, so no second n-row cache is needed.
    for (int i = 0; i < cache.n; ++i) cache.G(i, 0) = -cache.G(i, 0);
    moments.moment[0] = -moments.moment[0];
    for (int j = 1; j < cache.d; ++j) {
      moments.Sgg(0, j) = -moments.Sgg(0, j);
      moments.Sgg(j, 0) = -moments.Sgg(j, 0);
    }
  }
  if (max_iterations == 0) {
    Rcpp::List out = failure_result(
        cache, "solver_disabled_max_iterations_zero", target,
        static_cast<double>(score_sign) * base_center, 0, NA_REAL,
        std::vector<double>(), VectorXd::Zero(cache.d + 1), 1);
    return cached_outward_result(out, base_center, observed, score_sign);
  }
  const Evaluation base = evaluate_crt_zero(cache, moments, target);
  Rcpp::List out = run_crt_full(
      cache, target, tolerance, max_iterations, 1, &base);
  return cached_outward_result(out, base_center, observed, score_sign);
}


Rcpp::List crt_spa_full_outward_prepared(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::List& target_prepared,
    const Rcpp::List& gene_geometry,
    const Rcpp::IntegerVector& treated_indices,
    const double tolerance,
    const int max_iterations) {
  validate_solver_controls(0.0, tolerance, max_iterations);
  ZeroMoments moments;
  InfoCache cache = build_prepared_cache(
      a, w, Z, target_prepared, gene_geometry, moments);
  const double* weights = cache.weights_data();
  std::vector<int> treated(treated_indices.size());
  for (R_xlen_t j = 0; j < treated_indices.size(); ++j) {
    if (treated_indices[j] == NA_INTEGER || treated_indices[j] < 1 ||
        treated_indices[j] > cache.n) {
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
  VectorXd observed_B = VectorXd::Zero(cache.p);
  for (std::size_t j = 0; j < treated.size(); ++j) {
    const int index = treated[j];
    observed_U += cache.G(index, 0);
    observed_weight += weights[index];
    observed_B += cache.G.row(index).segment(1, cache.p).transpose();
  }
  const double projection = cache.information_invertible
                                ? observed_B.dot(cache.C_inv * observed_B)
                                : kNaN;
  const double observed_information = observed_weight - projection;
  const double information_scale =
      std::max(std::abs(observed_weight), std::abs(projection));
  const double observed =
      std::isfinite(observed_U) && std::isfinite(observed_information) &&
              std::isfinite(information_scale) && information_scale > 0.0 &&
              observed_information > 1.0e-14 * information_scale
          ? observed_U / std::sqrt(observed_information)
          : kNaN;
  if (!cache.information_invertible) {
    return prepared_outward_failure(
        cache, "singular_information_geometry", kNaN, kNaN, kNaN,
        NA_INTEGER, 0);
  }
  if (!std::isfinite(observed)) {
    return prepared_outward_failure(
        cache, "invalid_observed_studentizer", kNaN, kNaN, kNaN,
        NA_INTEGER, 0);
  }
  if (!cache.leading_intercept) {
    return prepared_outward_failure(
        cache, "unsupported_spa_geometry_no_intercept", observed, kNaN,
        observed, NA_INTEGER, 0);
  }
  const Evaluation positive_base = evaluate_crt_zero(cache, moments, observed);
  if (!positive_base.valid || !std::isfinite(positive_base.boundary.b)) {
    return prepared_outward_failure(
        cache, "invalid_center", observed, kNaN, observed, NA_INTEGER, 1);
  }
  const double base_center = positive_base.boundary.b;
  if (!moments.all_propensities_interior) {
    return prepared_outward_failure(
        cache, "non_interior_propensity", observed, base_center,
        observed, NA_INTEGER, 1);
  }
  const double displacement = observed - base_center;
  if (displacement == 0.0) {
    return prepared_outward_failure(
        cache, "central_target_requires_empirical_fallback", observed,
        base_center, observed, NA_INTEGER, 1);
  }

  const int score_sign = displacement > 0.0 ? 1 : -1;
  const double target = static_cast<double>(score_sign) * observed;
  cache.score_sign = score_sign;
  if (score_sign == -1) {
    // Reflect only the score coordinate. The nuisance geometry and its
    // factorization are unchanged, so no second n-row cache is needed.
    for (int i = 0; i < cache.n; ++i) cache.G(i, 0) = -cache.G(i, 0);
    moments.moment[0] = -moments.moment[0];
    for (int j = 1; j < cache.d; ++j) {
      moments.Sgg(0, j) = -moments.Sgg(0, j);
      moments.Sgg(j, 0) = -moments.Sgg(j, 0);
    }
  }
  if (max_iterations == 0) {
    Rcpp::List out = failure_result(
        cache, "solver_disabled_max_iterations_zero", target,
        static_cast<double>(score_sign) * base_center, 0, NA_REAL,
        std::vector<double>(), VectorXd::Zero(cache.d + 1), 1);
    return prepared_outward_result(out, base_center, observed, score_sign);
  }
  const Evaluation base = evaluate_crt_zero(cache, moments, target);
  Rcpp::List out = run_crt_full(
      cache, target, tolerance, max_iterations, 1, &base);
  return prepared_outward_result(out, base_center, observed, score_sign);
}

}  // namespace sceptre

// [[Rcpp::export]]
Rcpp::List crt_spa_full_cpp(const Rcpp::NumericVector& a,
                            const Rcpp::NumericVector& w,
                            const Rcpp::NumericMatrix& Z,
                            const Rcpp::NumericVector& propensity,
                            const double target,
                            const int score_sign = 1,
                            const double tolerance = 1e-4,
                            const int max_iterations = 50) {
  return sceptre::crt_spa_full(a, w, Z, propensity, target, score_sign,
                               tolerance, max_iterations);
}

// [[Rcpp::export]]
Rcpp::List crt_spa_full_cached_cpp(const Rcpp::NumericVector& a,
                                    const Rcpp::NumericVector& w,
                                    const Rcpp::NumericMatrix& Z,
                                    const Rcpp::NumericVector& propensity,
                                    const double target,
                                    const int score_sign = 1,
                                    const double tolerance = 1e-4,
                                    const int max_iterations = 50) {
  return sceptre::crt_spa_full_cached(
      a, w, Z, propensity, target, score_sign, tolerance, max_iterations);
}

// [[Rcpp::export]]
Rcpp::List crt_spa_full_outward_cached_cpp(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::NumericVector& propensity,
    const Rcpp::IntegerVector& treated_indices,
    const double tolerance = 1e-4,
    const int max_iterations = 50) {
  return sceptre::crt_spa_full_outward_cached(
      a, w, Z, propensity, treated_indices, tolerance, max_iterations);
}

// Internal run-local preparation. These R-owned lists can be serialized or
// inherited by fork workers; the caller must not alter their derived values.
// [[Rcpp::export]]
Rcpp::List prepare_crt_spa_target_cpp(const Rcpp::NumericVector& propensity) {
  return prepare_target(propensity);
}

// [[Rcpp::export]]
Rcpp::List prepare_crt_spa_gene_geometry_cpp(
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z) {
  return prepare_gene_geometry(w, Z);
}

// [[Rcpp::export]]
Rcpp::List crt_spa_full_outward_prepared_cpp(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::List& target_prepared,
    const Rcpp::List& gene_geometry,
    const Rcpp::IntegerVector& treated_indices,
    const double tolerance = 1e-4,
    const int max_iterations = 50) {
  return sceptre::crt_spa_full_outward_prepared(
      a, w, Z, target_prepared, gene_geometry, treated_indices,
      tolerance, max_iterations);
}

// Internal numerical audit of reference and prepared kernels at a supplied
// state. This bypasses root selection so tests can examine extreme tilts,
// endpoint propensities, individual KL terms, and full moment/Hessian arrays.
// It is not a user-facing analysis method.
// [[Rcpp::export]]
Rcpp::List crt_spa_evaluate_kernels_cpp(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::NumericVector& propensity,
    const Rcpp::NumericVector& theta,
    const double lambda = 0.0,
    const double target = 0.0,
    const int score_sign = 1,
    Rcpp::Nullable<Rcpp::NumericVector> rate_theta = R_NilValue) {
  const InfoCache cache = build_cache(
      a, w, Z, propensity, score_sign, false, false);
  if (theta.size() != cache.d) {
    Rcpp::stop("theta must have one score and one coefficient per Z column");
  }
  VectorXd state(cache.d + 1);
  for (int j = 0; j < cache.d; ++j) state[j] = theta[j];
  state[cache.d] = lambda;
  const VectorXd tilt_coefficients = state.head(cache.d);
  VectorXd requested_rate_theta = tilt_coefficients;
  if (rate_theta.isNotNull()) {
    const Rcpp::NumericVector override_theta(rate_theta.get());
    if (override_theta.size() != cache.d) {
      Rcpp::stop("rate_theta must have one score and one coefficient per Z column");
    }
    for (int j = 0; j < cache.d; ++j) requested_rate_theta[j] = override_theta[j];
  }
  const Evaluation reference = evaluate_crt_exact(cache, state, target);
  const Evaluation optimized = evaluate_crt_optimized(cache, state, target);
  Rcpp::NumericVector reference_cell_rate(cache.n), optimized_cell_rate(cache.n);
  const double* probabilities = cache.propensity_data();
  for (int i = 0; i < cache.n; ++i) {
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < cache.d; ++j) {
      tilt += tilt_coefficients[j] * g[j * cache.n];
    }
    reference_cell_rate[i] = bernoulli_kl_from_tilt(probabilities[i], tilt);
    optimized_cell_rate[i] = bernoulli_kl_from_tilt_optimized(probabilities[i], tilt);
  }
  bool used_retained_rate = false;
  const double retained_rate = select_prepared_crt_rate(
      cache, optimized, requested_rate_theta, &used_retained_rate);
  const double recomputed_rate = crt_rate_optimized(cache, requested_rate_theta);
  Rcpp::NumericVector retained_cell_rate(cache.n), recomputed_cell_rate(cache.n);
  for (int i = 0; i < cache.n; ++i) {
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < cache.d; ++j) {
      tilt += requested_rate_theta[j] * g[j * cache.n];
    }
    recomputed_cell_rate[i] = bernoulli_kl_from_tilt_optimized(probabilities[i], tilt);
    // The unselected branch is not evaluated: an incomplete retained buffer
    // must never be read or exported, including in this diagnostic.
    const double selected_tilt = used_retained_rate ? optimized.retained_tilts[i] : tilt;
    retained_cell_rate[i] = bernoulli_kl_from_tilt_optimized(probabilities[i], selected_tilt);
  }
  return Rcpp::List::create(
      Rcpp::Named("reference") = kernel_evaluation_diagnostics(
          reference, crt_rate(cache, tilt_coefficients), reference_cell_rate),
      Rcpp::Named("optimized") = kernel_evaluation_diagnostics(
          optimized, crt_rate_optimized(cache, tilt_coefficients), optimized_cell_rate),
      Rcpp::Named("optimized_d7") = cache.d == 7,
      Rcpp::Named("retained") = Rcpp::List::create(
          Rcpp::Named("rate") = retained_rate,
          Rcpp::Named("recomputed_rate") = recomputed_rate,
          Rcpp::Named("cell_rate") = retained_cell_rate,
          Rcpp::Named("recomputed_cell_rate") = recomputed_cell_rate,
          Rcpp::Named("n_tilts_stored") =
              optimized.retained_tilts_complete ? cache.n : 0,
          Rcpp::Named("n_tilts_reused") = used_retained_rate ? cache.n : 0,
          Rcpp::Named("used_retained_rate") = used_retained_rate));
}
