// Information-studentized saddlepoint approximation for SCEPTRE's
// fixed-count randomization (permutation) test.
//
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>

#include "rpt_spa.h"

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
constexpr double kMinimumCountVarianceRatio = 1.0e-4;
constexpr double kMaximumCountBerryEsseenRatio = 1.0;
constexpr double kMinimumAbsoluteRlr = 0.1;

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
  int m = 0;
  int score_sign = 1;
  bool leading_intercept = true;
  bool information_invertible = false;
  double base_probability = kNaN;
  double base_count_variance = kNaN;
  double base_offset = kNaN;
  MatrixXd G;  // Columns: signed a, then w * Z.
  MatrixXd C_inv;
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
  const VectorXd B = moment.segment(1, p);
  const VectorXd beta = C_inv * B;
  const double V = B[0] - B.dot(beta);
  if (!std::isfinite(V) || V <= kTinyV) return out;

  VectorXd grad_V = VectorXd::Zero(d);
  grad_V.segment(1, p) = -2.0 * beta;
  grad_V[1] += 1.0;
  MatrixXd hess_V = MatrixXd::Zero(d, d);
  hess_V.block(1, 1, p, p) = -2.0 * C_inv;
  VectorXd e_U = VectorXd::Zero(d);
  e_U[0] = 1.0;

  const double f = std::pow(V, -0.5);
  const double fp = -0.5 * std::pow(V, -1.5);
  const double fpp = 0.75 * std::pow(V, -2.5);
  out.grad = f * e_U + U * fp * grad_V;
  out.hess = fp * (e_U * grad_V.transpose() +
                   grad_V * e_U.transpose()) +
             U * (fpp * grad_V * grad_V.transpose() + fp * hess_V);
  out.valid = std::isfinite(U) && out.grad.allFinite() &&
              out.hess.allFinite();
  out.V = V;
  out.b = U / std::sqrt(V);
  return out;
}

struct CountProfile {
  bool valid = false;
  double gamma = kNaN;
  double count = kNaN;
  double S0 = kNaN;
  int iterations = 0;
  VectorXd probabilities;
};

bool count_at_gamma(const VectorXd& linear_predictor,
                    const double gamma,
                    double& count,
                    double& variance,
                    VectorXd* probabilities = nullptr) {
  if (!std::isfinite(gamma)) return false;
  long double count_sum = 0.0L;
  long double variance_sum = 0.0L;
  if (probabilities != nullptr) {
    probabilities->resize(linear_predictor.size());
  }
  for (Eigen::Index i = 0; i < linear_predictor.size(); ++i) {
    const double value = linear_predictor[i] + gamma;
    if (std::isnan(value)) return false;
    const double probability = expit_stable(value);
    if (!std::isfinite(probability)) return false;
    const double h = probability * (1.0 - probability);
    count_sum += static_cast<long double>(probability);
    variance_sum += static_cast<long double>(h);
    if (probabilities != nullptr) (*probabilities)[i] = probability;
  }
  count = static_cast<double>(count_sum);
  variance = static_cast<double>(variance_sum);
  return std::isfinite(count) && std::isfinite(variance);
}

CountProfile profile_count_tilt(const VectorXd& linear_predictor,
                                const int m,
                                const double base_offset) {
  CountProfile out;
  const int n = static_cast<int>(linear_predictor.size());
  if (n < 2 || m <= 0 || m >= n || !linear_predictor.allFinite()) return out;

  long double mean_linear = 0.0L;
  for (int i = 0; i < n; ++i) {
    mean_linear += static_cast<long double>(linear_predictor[i]);
  }
  mean_linear /= static_cast<long double>(n);
  double initial = base_offset - static_cast<double>(mean_linear);
  if (!std::isfinite(initial)) initial = 0.0;

  double initial_count = kNaN;
  double initial_variance = kNaN;
  if (!count_at_gamma(linear_predictor, initial, initial_count,
                      initial_variance)) {
    return out;
  }
  const double target = static_cast<double>(m);
  const double count_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, static_cast<double>(n));
  if (std::abs(initial_count - target) <= count_tolerance) {
    if (!count_at_gamma(linear_predictor, initial, initial_count,
                        initial_variance, &out.probabilities) ||
        initial_variance <= 0.0) {
      return out;
    }
    out.valid = true;
    out.gamma = initial;
    out.count = initial_count;
    out.S0 = initial_variance;
    out.iterations = 1;
    return out;
  }

  double lower = initial;
  double upper = initial;
  double lower_count = initial_count;
  double upper_count = initial_count;
  double step = 1.0;
  if (initial_count > target + count_tolerance) {
    for (int expansion = 0; expansion < 80; ++expansion) {
      lower = initial - step;
      double ignored_variance = kNaN;
      if (!count_at_gamma(linear_predictor, lower, lower_count,
                          ignored_variance)) {
        return out;
      }
      if (lower_count <= target) break;
      step *= 2.0;
    }
  } else if (initial_count < target - count_tolerance) {
    for (int expansion = 0; expansion < 80; ++expansion) {
      upper = initial + step;
      double ignored_variance = kNaN;
      if (!count_at_gamma(linear_predictor, upper, upper_count,
                          ignored_variance)) {
        return out;
      }
      if (upper_count >= target) break;
      step *= 2.0;
    }
  }
  if (lower_count > target || upper_count < target ||
      !std::isfinite(lower) || !std::isfinite(upper)) {
    return out;
  }

  double gamma = initial;
  double count = initial_count;
  double S0 = initial_variance;
  for (int iteration = 0; iteration < 100; ++iteration) {
    out.iterations = iteration + 1;
    if (std::abs(count - target) <= count_tolerance) break;
    if (count < target) {
      lower = gamma;
      lower_count = count;
    } else {
      upper = gamma;
      upper_count = count;
    }

    double candidate = kNaN;
    if (std::isfinite(S0) && S0 > 0.0) {
      candidate = gamma - (count - target) / S0;
    }
    if (!std::isfinite(candidate) || candidate <= lower ||
        candidate >= upper) {
      candidate = lower + 0.5 * (upper - lower);
    }
    gamma = candidate;
    if (!count_at_gamma(linear_predictor, gamma, count, S0)) return out;
  }

  if (!count_at_gamma(linear_predictor, gamma, count, S0,
                      &out.probabilities) ||
      std::abs(count - target) > 4.0 * count_tolerance || S0 <= 0.0) {
    return out;
  }
  out.valid = true;
  out.gamma = gamma;
  out.count = count;
  out.S0 = S0;
  return out;
}

struct Evaluation {
  bool valid = false;
  VectorXd residual;
  MatrixXd jacobian;
  VectorXd moment;
  VectorXd Sg;
  MatrixXd Sgg;
  MatrixXd conditional_information;
  Boundary boundary;
  double gamma = kNaN;
  double tilted_count = kNaN;
  double S0 = kNaN;
  double count_variance_ratio = kNaN;
  double count_berry_esseen_ratio = kNaN;
  int gamma_iterations = 0;
};

Evaluation evaluate_rpt_exact(const InfoCache& cache,
                              const VectorXd& x,
                              const double target) {
  Evaluation out;
  const int d = cache.d;
  if (x.size() != d + 1 || !x.allFinite()) return out;
  const VectorXd theta = x.head(d);
  const double lambda = x[d];

  VectorXd linear_predictor(cache.n);
  for (int i = 0; i < cache.n; ++i) {
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < d; ++j) tilt += theta[j] * g[j * cache.n];
    if (!std::isfinite(tilt)) return out;
    linear_predictor[i] = cache.base_offset + tilt;
  }
  const CountProfile profile = profile_count_tilt(
      linear_predictor, cache.m, cache.base_offset);
  if (!profile.valid) return out;

  out.gamma = profile.gamma;
  out.tilted_count = profile.count;
  out.S0 = profile.S0;
  out.gamma_iterations = profile.iterations;
  out.moment.setZero(d);
  out.Sg.setZero(d);
  out.Sgg.setZero(d, d);
  double count_third_absolute_moment = 0.0;
  for (int i = 0; i < cache.n; ++i) {
    const double probability = profile.probabilities[i];
    const double h = probability * (1.0 - probability);
    count_third_absolute_moment +=
        h * ((1.0 - probability) * (1.0 - probability) +
             probability * probability);
    const double* g = cache.G.data() + i;
    for (int j = 0; j < d; ++j) {
      const double gj = g[j * cache.n];
      out.moment[j] += probability * gj;
      out.Sg[j] += h * gj;
      for (int k = 0; k <= j; ++k) {
        out.Sgg(j, k) += h * gj * g[k * cache.n];
      }
    }
  }
  symmetrize_lower(out.Sgg);
  if (!out.moment.allFinite() || !out.Sg.allFinite() ||
      !out.Sgg.allFinite() || !std::isfinite(out.S0) || out.S0 <= 0.0) {
    return out;
  }
  out.count_variance_ratio = out.S0 / cache.base_count_variance;
  out.count_berry_esseen_ratio =
      count_third_absolute_moment / std::pow(out.S0, 1.5);
  if (!std::isfinite(out.count_variance_ratio) ||
      !std::isfinite(out.count_berry_esseen_ratio)) {
    return out;
  }
  out.conditional_information =
      out.Sgg - out.Sg * out.Sg.transpose() / out.S0;
  out.conditional_information =
      0.5 * (out.conditional_information +
             out.conditional_information.transpose());
  if (!out.conditional_information.allFinite()) return out;

  out.boundary = boundary_terms(out.moment, cache.C_inv);
  if (!out.boundary.valid) return out;
  out.residual.setZero(d + 1);
  out.residual.head(d) = theta - lambda * out.boundary.grad;
  out.residual[d] = out.boundary.b - target;
  out.jacobian.setZero(d + 1, d + 1);
  out.jacobian.block(0, 0, d, d) =
      MatrixXd::Identity(d, d) - lambda * out.boundary.hess *
                                      out.conditional_information;
  out.jacobian.block(0, d, d, 1) = -out.boundary.grad;
  out.jacobian.block(d, 0, 1, d) =
      (out.conditional_information * out.boundary.grad).transpose();
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
                          const int exact_evaluations,
                          const Evaluation* evaluation = nullptr) {
  const double gamma = evaluation == nullptr ? kNaN : evaluation->gamma;
  const double tilted_count =
      evaluation == nullptr ? kNaN : evaluation->tilted_count;
  const double count_variance =
      evaluation == nullptr ? kNaN : evaluation->S0;
  const double count_variance_ratio =
      evaluation == nullptr ? kNaN : evaluation->count_variance_ratio;
  const double count_berry_esseen_ratio =
      evaluation == nullptr ? kNaN : evaluation->count_berry_esseen_ratio;
  const int gamma_iterations =
      evaluation == nullptr ? NA_INTEGER : evaluation->gamma_iterations;
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
      Rcpp::Named("kl_rate") = kNaN,
      Rcpp::Named("conditional_rate_correction") = kNaN,
      Rcpp::Named("r_lr") = kNaN,
      Rcpp::Named("q_lr") = kNaN,
      Rcpp::Named("q2") = kNaN,
      Rcpp::Named("moment_dimension") = cache.d,
      Rcpp::Named("nuisance_dimension") = cache.p,
      Rcpp::Named("score_sign") = cache.score_sign,
      Rcpp::Named("path") = "full_exact_conditional",
      Rcpp::Named("exact_evaluations") = exact_evaluations,
      Rcpp::Named("treated_count") = cache.m,
      Rcpp::Named("base_treatment_probability") = cache.base_probability,
      Rcpp::Named("count_tilt") = gamma,
      Rcpp::Named("tilted_count") = tilted_count,
      Rcpp::Named("count_residual") =
          std::isfinite(tilted_count) ? tilted_count - cache.m : kNaN,
      Rcpp::Named("count_variance") = count_variance,
      Rcpp::Named("count_variance_ratio") = count_variance_ratio,
      Rcpp::Named("count_berry_esseen_ratio") =
          count_berry_esseen_ratio,
      Rcpp::Named("base_count_variance") = cache.base_count_variance,
      Rcpp::Named("gamma_iterations") = gamma_iterations);
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
  if (!std::isfinite(p) || !std::isfinite(tilt) || p <= 0.0 || p >= 1.0) {
    return kNaN;
  }
  if (tilt == 0.0) return 0.0;

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
  if (!std::isfinite(rate) || !std::isfinite(q2) || rate <= 0.0 ||
      q2 <= 0.0) {
    return kNaN;
  }
  const long double sign = direction < 0.0 ? -1.0L : 1.0L;
  const long double r_lr_extended =
      sign * std::sqrt(static_cast<long double>(2.0) * rate);
  const long double q_lr_extended =
      sign * std::sqrt(static_cast<long double>(q2));
  if (!std::isfinite(r_lr_extended) || !std::isfinite(q_lr_extended) ||
      r_lr_extended == 0.0L || q_lr_extended == 0.0L) {
    return kNaN;
  }
  const long double correction_extended =
      (r_lr_extended - q_lr_extended) /
      (q_lr_extended * r_lr_extended);
  const double r_lr = static_cast<double>(r_lr_extended);
  const double correction = static_cast<double>(correction_extended);
  if (!std::isfinite(r_lr) || !std::isfinite(correction)) return kNaN;
  return R::pnorm(r_lr, 0.0, 1.0, false, false) +
         R::dnorm(r_lr, 0.0, 1.0, false) * correction;
}

double rpt_kl_rate(const InfoCache& cache,
                   const VectorXd& theta,
                   const double gamma) {
  double rate = 0.0;
  for (int i = 0; i < cache.n; ++i) {
    const double* g = cache.G.data() + i;
    double tilt = gamma;
    for (int j = 0; j < cache.d; ++j) {
      tilt += theta[j] * g[j * cache.n];
    }
    if (!std::isfinite(tilt)) return kNaN;
    const double term = bernoulli_kl_from_tilt(
        cache.base_probability, tilt);
    if (!std::isfinite(term)) return kNaN;
    rate += term;
  }
  return rate;
}

Rcpp::List finalize_rpt(const InfoCache& cache,
                        const VectorXd& x,
                        const Evaluation& evaluation,
                        const double center,
                        const double target,
                        const int iterations,
                        const std::vector<double>& history) {
  const VectorXd theta = x.head(cache.d);
  const double kl_rate = rpt_kl_rate(cache, theta, evaluation.gamma);
  const double conditional_correction =
      0.5 * std::log(evaluation.S0 / cache.base_count_variance);
  const double rate = kl_rate + conditional_correction;
  double q2 = theta.dot(evaluation.conditional_information * theta);
  if (q2 < 0.0 && q2 > -1.0e-12) q2 = 0.0;
  const double raw_p = lr_upper_tail(rate, q2, theta[0]);
  const double r_lr = rate > 0.0
                          ? std::sqrt(2.0 * rate)
                          : kNaN;
  const double q_lr = q2 > 0.0 ? std::sqrt(q2) : kNaN;
  const double lambda = x[cache.d];
  const double count_residual = evaluation.tilted_count - cache.m;
  const bool count_ok =
      std::isfinite(count_residual) &&
      std::abs(count_residual) <=
          256.0 * std::numeric_limits<double>::epsilon() *
              std::max(1.0, static_cast<double>(cache.n));
  // The coefficient-saddle normalization for conditioning on N=m relies on
  // a regular local count distribution. Near a finite support boundary the
  // tilted Bernoulli probabilities become almost deterministic, S0 collapses,
  // and the continuous conditional LR formula can be grossly inaccurate even
  // though the KKT residual is small. Such cases must use the exact-margin
  // empirical fallback.
  const bool count_conditioning_regular =
      evaluation.count_variance_ratio >= kMinimumCountVarianceRatio &&
      evaluation.count_berry_esseen_ratio <=
          kMaximumCountBerryEsseenRatio;
  // The conditional rate correction is only first-order accurate. Its small
  // error is magnified by the reciprocal LR correction as r and q approach
  // zero, so SPA-always delegates a narrow central neighborhood to the
  // empirical fallback. The screened method never reaches this branch for a
  // central statistic.
  const bool near_center =
      std::isfinite(r_lr) && std::abs(r_lr) < kMinimumAbsoluteRlr;
  const bool regular = target > center && lambda > 0.0 && theta[0] > 0.0 &&
                       kl_rate > 0.0 && rate > 0.0 && q2 > 0.0 && count_ok &&
                       count_conditioning_regular && !near_center;
  const bool in_range = std::isfinite(raw_p) && raw_p >= -1e-10 &&
                        raw_p <= 1.0 + 1e-10;
  const bool valid = evaluation.valid && regular && in_range;
  const double sign = theta[0] < 0.0 ? -1.0 : 1.0;

  return Rcpp::List::create(
      Rcpp::Named("converged") = valid,
      Rcpp::Named("reason") =
          !count_conditioning_regular ? "nonregular_count_conditioning" :
          (near_center ? "near_center_requires_empirical_fallback" :
          (!regular ? "nonregular_upper_root" :
          (!in_range ? "lr_out_of_range" : "ok"))),
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
      Rcpp::Named("kl_rate") = kl_rate,
      Rcpp::Named("conditional_rate_correction") = conditional_correction,
      Rcpp::Named("r_lr") = sign * r_lr,
      Rcpp::Named("q_lr") = sign * q_lr,
      Rcpp::Named("q2") = q2,
      Rcpp::Named("moment_dimension") = cache.d,
      Rcpp::Named("nuisance_dimension") = cache.p,
      Rcpp::Named("score_sign") = cache.score_sign,
      Rcpp::Named("path") = "full_exact_conditional",
      Rcpp::Named("exact_evaluations") =
          static_cast<int>(history.size()) + 1,
      Rcpp::Named("treated_count") = cache.m,
      Rcpp::Named("base_treatment_probability") = cache.base_probability,
      Rcpp::Named("count_tilt") = evaluation.gamma,
      Rcpp::Named("tilted_count") = evaluation.tilted_count,
      Rcpp::Named("count_residual") = count_residual,
      Rcpp::Named("count_variance") = evaluation.S0,
      Rcpp::Named("count_variance_ratio") =
          evaluation.count_variance_ratio,
      Rcpp::Named("count_berry_esseen_ratio") =
          evaluation.count_berry_esseen_ratio,
      Rcpp::Named("base_count_variance") = cache.base_count_variance,
      Rcpp::Named("gamma_iterations") = evaluation.gamma_iterations);
}

InfoCache build_cache(const Rcpp::NumericVector& a,
                      const Rcpp::NumericVector& w,
                      const Rcpp::NumericMatrix& Z,
                      const int m,
                      const int score_sign,
                      const bool require_supported_geometry = true) {
  const int n = a.size();
  const int p = Z.ncol();
  if (n < 2 || p < 1 || Z.nrow() != n || w.size() != n) {
    Rcpp::stop("input dimensions do not match");
  }
  if (m <= 0 || m >= n) {
    Rcpp::stop("m must be strictly between zero and the number of cells");
  }
  if (score_sign != -1 && score_sign != 1) {
    Rcpp::stop("score_sign must be -1 or +1");
  }

  InfoCache cache;
  cache.n = n;
  cache.p = p;
  cache.d = p + 1;
  cache.m = m;
  cache.score_sign = score_sign;
  cache.base_probability = static_cast<double>(m) / n;
  cache.base_count_variance =
      n * cache.base_probability * (1.0 - cache.base_probability);
  cache.base_offset = std::log(cache.base_probability /
                               (1.0 - cache.base_probability));
  cache.G.setZero(n, p + 1);
  cache.weights.resize(n);
  MatrixXd C = MatrixXd::Zero(p, p);
  VectorXd weighted_z = VectorXd::Zero(p);

  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(a[i])) Rcpp::stop("a must be finite");
    if (!std::isfinite(w[i]) || w[i] <= 0.0) {
      Rcpp::stop("w must be positive and finite");
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

Rcpp::List run_rpt_full(const InfoCache& cache,
                        const double target,
                        const double tolerance,
                        const int max_iterations,
                        const int minimum_updates = 0) {
  VectorXd x = VectorXd::Zero(cache.d + 1);
  const std::vector<double> empty_history;
  Evaluation base = evaluate_rpt_exact(cache, x, target);
  if (!base.valid) {
    return failure_result(cache, "invalid_center", target, kNaN, 0, kNaN,
                          empty_history, x, 1, &base);
  }

  const double center = base.boundary.b;
  const double variance = base.boundary.grad.dot(
      base.conditional_information * base.boundary.grad);
  if (!std::isfinite(variance) || variance <= 1e-14) {
    return failure_result(cache, "degenerate_center_variance", target,
                          center, 0, kNaN, empty_history, x, 1, &base);
  }

  x[cache.d] = (target - center) / variance;
  x.head(cache.d) = x[cache.d] * base.boundary.grad;
  Evaluation current = evaluate_rpt_exact(cache, x, target);
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
      Evaluation trial = evaluate_rpt_exact(cache, candidate, target);
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
        static_cast<int>(history.size()) + 1, &current);
  }
  return finalize_rpt(cache, x, current, center, target, updates, history);
}

}  // namespace

namespace sceptre {

Rcpp::List rpt_spa_full(const Rcpp::NumericVector& a,
                        const Rcpp::NumericVector& w,
                        const Rcpp::NumericMatrix& Z,
                        const int m,
                        const double target,
                        const int score_sign,
                        const double tolerance,
                        const int max_iterations) {
  validate_solver_controls(target, tolerance, max_iterations);
  const InfoCache cache = build_cache(a, w, Z, m, score_sign);
  return run_rpt_full(cache, target, tolerance, max_iterations);
}

Rcpp::List rpt_spa_full_outward(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::IntegerVector& treated_indices,
    const double tolerance,
    const int max_iterations) {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    Rcpp::stop("tolerance must be positive and finite");
  }
  if (max_iterations < 0) {
    Rcpp::stop("max_iterations cannot be negative");
  }
  const int n = a.size();
  const int m = treated_indices.size();
  if (m <= 0 || m >= n) {
    Rcpp::stop(
        "treated_indices must contain between 1 and n - 1 indices");
  }

  const InfoCache positive_cache = build_cache(a, w, Z, m, 1, false);
  std::vector<int> treated(m);
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
  const Evaluation base = evaluate_rpt_exact(
      positive_cache, zero_state, observed_target);
  if (!base.valid || !std::isfinite(base.boundary.b)) {
    Rcpp::List out = failure_result(
        positive_cache, "invalid_center", observed_target, kNaN, 0, kNaN,
        std::vector<double>(), zero_state, 1, &base);
    out["base_center"] = NA_REAL;
    out["observed_target"] = observed_target;
    out["outward_score_sign"] = NA_INTEGER;
    return out;
  }

  const double base_center = base.boundary.b;
  const double displacement = observed_target - base_center;
  if (displacement == 0.0) {
    Rcpp::List out = failure_result(
        positive_cache, "central_target_requires_empirical_fallback",
        observed_target, base_center, 0, 0.0, std::vector<double>(),
        zero_state, 1, &base);
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
                                      : build_cache(a, w, Z, m, -1);
  if (max_iterations == 0) {
    const VectorXd state = VectorXd::Zero(outward_cache.d + 1);
    Rcpp::List out = failure_result(
        outward_cache, "solver_disabled_max_iterations_zero",
        outward_target, static_cast<double>(score_sign) * base_center, 0,
        kNaN, std::vector<double>(), state, 1);
    out["base_center"] = base_center;
    out["observed_target"] = observed_target;
    out["outward_score_sign"] = score_sign;
    return out;
  }

  Rcpp::List out = run_rpt_full(
      outward_cache, outward_target, tolerance, max_iterations, 1);
  out["base_center"] = base_center;
  out["observed_target"] = observed_target;
  out["outward_score_sign"] = score_sign;
  return out;
}

}  // namespace sceptre

// [[Rcpp::export]]
Rcpp::List rpt_spa_full_cpp(const Rcpp::NumericVector& a,
                            const Rcpp::NumericVector& w,
                            const Rcpp::NumericMatrix& Z,
                            const int m,
                            const double target,
                            const int score_sign = 1,
                            const double tolerance = 1e-5,
                            const int max_iterations = 50) {
  return sceptre::rpt_spa_full(a, w, Z, m, target, score_sign, tolerance,
                               max_iterations);
}
