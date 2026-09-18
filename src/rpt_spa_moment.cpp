// Cached response geometry and guarded sparse-moment RPT SPA.
// The numerical engine is ported from the validated standalone prototype.
#include "rpt_spa_moment.h"
#include <RcppEigen.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
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
  } catch (const Rcpp::internal::InterruptedException&) {
    throw;
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
  VectorXd positive_feature_sums;
  MatrixXd positive_feature_crossproducts;
  MatrixXd positive_feature_centered_crossproducts;
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
};

struct SolverCounters {
  int analytic_center_evaluations = 0;
  int exact_evaluations = 0;
  int count_passes = 0;
  int count_stagnations = 0;
  int gamma_iterations = 0;
  int gamma_warm_starts = 0;
  int gamma_cold_retries = 0;
  int compressed_evaluations = 0;
  int compressed_count_passes = 0;
  int compressed_count_stagnations = 0;
  int compressed_exception_passes = 0;
  int compressed_gamma_iterations = 0;
  int compressed_gamma_warm_starts = 0;
  int compressed_gamma_cold_retries = 0;
};

struct MomentDiagnostics {
  std::string path = "uninitialized";
  std::string fallback_reason = "none";
  bool eligible = false;
  bool synopsis_built = false;
  bool exact_audit_passed = false;
  int n_zero = 0;
  int n_exception = 0;
  int feature_dimension = 0;
  int maximum_moment_degree = 0;
  int packed_moment_count = 0;
  int compressed_updates = 0;
  int exact_polish_updates = 0;
  double zero_fraction = kNaN;
  double exact_audit_tolerance = kNaN;
  double effective_exact_audit_tolerance = kNaN;
  double maximum_locality_bound = 0.0;
};

struct EvaluationWorkspace {
  explicit EvaluationWorkspace(const int n) : linear_predictor(n) {}
  VectorXd linear_predictor;
};

struct MomentSynopsis {
  bool valid = false;
  int p = 0;
  int maximum_moment_degree = 0;
  int n_zero = 0;
  int packed_moment_count = 0;
  double M0 = 0.0;
  VectorXd M1;
  MatrixXd M2;
  std::vector<double> M3;
  std::vector<double> M4;
  VectorXd max_abs;
  std::vector<int> exceptions;
};

struct PreparedMomentContext {
  InfoCache cache;
  MomentSynopsis synopsis;
  VectorXd positive_feature_sums;
  bool eligible = false;
  int maximum_moment_degree = 0;
  int n_zero = 0;
  double zero_fraction = kNaN;
};

std::size_t index3(const int p, const int i, const int j, const int k) {
  return (static_cast<std::size_t>(i) * p + j) * p + k;
}

std::size_t index4(const int p, const int i, const int j, const int k,
                   const int ell) {
  return ((static_cast<std::size_t>(i) * p + j) * p + k) * p + ell;
}

void validate_maximum_moment_degree(const int maximum_moment_degree) {
  if (maximum_moment_degree < 2 || maximum_moment_degree > 4) {
    Rcpp::stop("maximum_moment_degree must be 2, 3, or 4");
  }
}

void validate_moment_dimensions(const int p, const int degree) {
  validate_maximum_moment_degree(degree);
  if (p < 1) Rcpp::stop("Z must have at least one column");
  long double entries = 1.0L;
  long double power = 1.0L;
  for (int k = 1; k <= degree; ++k) {
    power *= p;
    entries += power;
  }
  if (entries > std::numeric_limits<int>::max()) {
    Rcpp::stop("the requested moment dimension exceeds supported storage");
  }
}

MomentSynopsis build_moment_synopsis(const InfoCache& cache,
                                     const Rcpp::LogicalVector& zero_mask,
                                     const int maximum_moment_degree) {
  if (zero_mask.size() != cache.n) {
    Rcpp::stop("zero_mask must have one value per cell");
  }
  validate_moment_dimensions(cache.p, maximum_moment_degree);
  MomentSynopsis out;
  out.p = cache.p;
  out.maximum_moment_degree = maximum_moment_degree;
  out.M1.setZero(cache.p);
  out.M2.setZero(cache.p, cache.p);
  if (maximum_moment_degree >= 3) {
    out.M3.assign(static_cast<std::size_t>(cache.p) * cache.p * cache.p, 0.0);
  }
  if (maximum_moment_degree == 4) {
    out.M4.assign(static_cast<std::size_t>(cache.p) * cache.p * cache.p *
                      cache.p,
                  0.0);
  }
  out.max_abs.setZero(cache.p);

  const std::size_t p_size = static_cast<std::size_t>(cache.p);
  const int n2 = static_cast<int>(p_size * (p_size + 1) / 2);
  const int n3 = maximum_moment_degree >= 3
      ? static_cast<int>(p_size * (p_size + 1) * (p_size + 2) / 6) : 0;
  const int n4 = maximum_moment_degree == 4
      ? static_cast<int>(p_size * (p_size + 1) * (p_size + 2) *
                         (p_size + 3) / 24) : 0;
  std::vector<double> packed2(n2, 0.0);
  std::vector<double> packed3(maximum_moment_degree >= 3 ? n3 : 0, 0.0);
  std::vector<double> packed4(
      maximum_moment_degree == 4 ? n4 : 0, 0.0);

  for (int i = 0; i < cache.n; ++i) {
    if ((i & 8191) == 0) Rcpp::checkUserInterrupt();
    if (zero_mask[i] == NA_LOGICAL) {
      Rcpp::stop("zero_mask cannot contain missing values");
    }
    if (!zero_mask[i]) {
      out.exceptions.push_back(i);
      continue;
    }
    const double raw_score = cache.G(i, 0) / cache.score_sign;
    const double intercept_feature = cache.G(i, 1);
    const double identity_tolerance =
        128.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::max(std::abs(raw_score),
                               std::abs(intercept_feature)));
    if (std::abs(raw_score + intercept_feature) > identity_tolerance) {
      Rcpp::stop("zero_mask is inconsistent with the NB zero-row identity");
    }

    ++out.n_zero;
    out.M0 += 1.0;
    int position2 = 0;
    int position3 = 0;
    int position4 = 0;
    for (int a = 0; a < cache.p; ++a) {
      const double fa = cache.G(i, a + 1);
      out.M1[a] += fa;
      out.max_abs[a] = std::max(out.max_abs[a], std::abs(fa));
      for (int b = a; b < cache.p; ++b) {
        const double fab = fa * cache.G(i, b + 1);
        packed2[position2++] += fab;
        if (maximum_moment_degree < 3) continue;
        for (int c = b; c < cache.p; ++c) {
          const double fabc = fab * cache.G(i, c + 1);
          packed3[position3++] += fabc;
          if (maximum_moment_degree == 4) {
            for (int ell = c; ell < cache.p; ++ell) {
              packed4[position4++] += fabc * cache.G(i, ell + 1);
            }
          }
        }
      }
    }
  }

  int position2 = 0;
  int position3 = 0;
  int position4 = 0;
  for (int a = 0; a < cache.p; ++a) {
    for (int b = a; b < cache.p; ++b) {
      std::array<int, 2> indices = {{a, b}};
      do {
        out.M2(indices[0], indices[1]) = packed2[position2];
      } while (std::next_permutation(indices.begin(), indices.end()));
      ++position2;
      if (maximum_moment_degree < 3) continue;
      for (int c = b; c < cache.p; ++c) {
        std::array<int, 3> indices3 = {{a, b, c}};
        do {
          out.M3[index3(cache.p, indices3[0], indices3[1], indices3[2])] =
              packed3[position3];
        } while (std::next_permutation(indices3.begin(), indices3.end()));
        ++position3;
        if (maximum_moment_degree == 4) {
          for (int ell = c; ell < cache.p; ++ell) {
            std::array<int, 4> indices4 = {{a, b, c, ell}};
            do {
              out.M4[index4(cache.p, indices4[0], indices4[1], indices4[2],
                                  indices4[3])] = packed4[position4];
            } while (std::next_permutation(indices4.begin(), indices4.end()));
            ++position4;
          }
        }
      }
    }
  }
  out.packed_moment_count = 1 + cache.p + n2 +
                            (maximum_moment_degree >= 3 ? n3 : 0) +
                            (maximum_moment_degree == 4 ? n4 : 0);
  const bool fourth_moment_finite = maximum_moment_degree < 4 ||
      std::all_of(out.M4.begin(), out.M4.end(),
                  [](const double value) { return std::isfinite(value); });
  out.valid = out.n_zero > 0 && out.M1.allFinite() && out.M2.allFinite() &&
              out.max_abs.allFinite() &&
              std::all_of(out.M3.begin(), out.M3.end(),
                          [](const double value) { return std::isfinite(value); }) &&
              fourth_moment_finite;
  return out;
}

struct MomentContractions {
  bool valid = false;
  double max_delta = kNaN;
  double L0 = 0.0;
  double Q0 = 0.0;
  VectorXd L1;
  VectorXd Q1;
  MatrixXd L2;
  MatrixXd Q2;
};

MomentContractions contract_moments(const MomentSynopsis& synopsis,
                                    const VectorXd& psi) {
  MomentContractions out;
  const int p = synopsis.p;
  if (!synopsis.valid || psi.size() != p || !psi.allFinite()) return out;
  out.L1.setZero(p);
  out.Q1.setZero(p);
  out.L2.setZero(p, p);
  out.Q2.setZero(p, p);
  out.max_delta = psi.cwiseAbs().dot(synopsis.max_abs);
  out.L0 = psi.dot(synopsis.M1);
  out.Q0 = psi.dot(synopsis.M2 * psi);
  if (synopsis.maximum_moment_degree == 2) {
    out.L1.noalias() = synopsis.M2 * psi;
    out.valid = std::isfinite(out.max_delta) && std::isfinite(out.L0) &&
                std::isfinite(out.Q0) && out.L1.allFinite();
    return out;
  }
  for (int j = 0; j < p; ++j) {
    for (int a = 0; a < p; ++a) {
      out.L1[j] += psi[a] * synopsis.M2(j, a);
      for (int b = 0; b < p; ++b) {
        out.Q1[j] += psi[a] * psi[b] *
                     synopsis.M3[index3(p, j, a, b)];
      }
    }
    for (int k = 0; k < p; ++k) {
      for (int a = 0; a < p; ++a) {
        out.L2(j, k) +=
            psi[a] * synopsis.M3[index3(p, j, k, a)];
        if (synopsis.maximum_moment_degree == 4) {
          for (int b = 0; b < p; ++b) {
            out.Q2(j, k) += psi[a] * psi[b] *
                            synopsis.M4[index4(p, j, k, a, b)];
          }
        }
      }
    }
  }
  out.valid = std::isfinite(out.max_delta) && std::isfinite(out.L0) &&
              std::isfinite(out.Q0) && out.L1.allFinite() &&
              out.Q1.allFinite() && out.L2.allFinite() && out.Q2.allFinite();
  return out;
}

struct ZeroAggregate {
  bool valid = false;
  double count = kNaN;
  double S0 = kNaN;
  VectorXd moment_f;
  VectorXd Sf;
  MatrixXd Sff;
};

ZeroAggregate evaluate_zero_aggregate(const MomentSynopsis& synopsis,
                                      const MomentContractions& contractions,
                                      const double baseline_logit) {
  ZeroAggregate out;
  const int p = synopsis.p;
  if (!contractions.valid || !std::isfinite(baseline_logit)) return out;
  const double probability = expit_stable(baseline_logit);
  const double h = probability * (1.0 - probability);
  const double hp = h * (1.0 - 2.0 * probability);
  const double hpp = h *
      (1.0 - 6.0 * probability + 6.0 * probability * probability);
  out.count = probability * synopsis.M0 + h * contractions.L0 +
              0.5 * hp * contractions.Q0;
  out.S0 = h * synopsis.M0 + hp * contractions.L0 +
           0.5 * hpp * contractions.Q0;
  // Degree two differentiates a quadratic CGF surrogate: quadratic count,
  // linear feature mean, and constant raw Sff. Accepted roots receive a
  // full exact-CGF audit before the saddlepoint p-value is evaluated.
  out.moment_f = probability * synopsis.M1 + h * contractions.L1 +
                 0.5 * hp * contractions.Q1;
  out.Sf = h * synopsis.M1 + hp * contractions.L1 +
           0.5 * hpp * contractions.Q1;
  out.Sff = h * synopsis.M2 + hp * contractions.L2;
  if (synopsis.maximum_moment_degree == 4) {
    out.Sff += 0.5 * hpp * contractions.Q2;
  }
  const double count_tolerance =
      1.0e-10 * std::max(1.0, synopsis.M0);
  out.valid = std::isfinite(out.count) &&
              out.count >= -count_tolerance &&
              out.count <= synopsis.M0 + count_tolerance &&
              std::isfinite(out.S0) &&
              out.S0 > 0.0 && out.moment_f.allFinite() &&
              out.Sf.allFinite() && out.Sff.allFinite();
  return out;
}

// Short nonnegative blocks plus compensated reduction avoid platform-dependent
// long-double precision without a serial compensation dependency at every cell.
class CountSum {
 public:
  void add(const double value) {
    block_sum_ += value;
    if (++block_size_ == 32) {
      add_block(block_sum_);
      block_sum_ = 0.0;
      block_size_ = 0;
    }
  }
  double value() const {
    return sum_ + (block_sum_ - correction_);
  }

 private:
  void add_block(const double value) {
    const double corrected = value - correction_;
    const double next = sum_ + corrected;
    correction_ = (next - sum_) - corrected;
    sum_ = next;
  }
  double sum_ = 0.0;
  double correction_ = 0.0;
  double block_sum_ = 0.0;
  int block_size_ = 0;
};

bool count_at_gamma(const VectorXd& linear_predictor,
                    const double gamma,
                    double& count,
                    double& variance,
                    SolverCounters& counters) {
  if (!std::isfinite(gamma)) return false;
  ++counters.count_passes;

  CountSum count_sum;
  CountSum variance_sum;
  for (Eigen::Index i = 0; i < linear_predictor.size(); ++i) {
    if ((i & 8191) == 0) Rcpp::checkUserInterrupt();
    const double value = linear_predictor[i] + gamma;
    if (std::isnan(value)) return false;
    const double probability = expit_stable(value);
    if (!std::isfinite(probability)) return false;
    const double h = probability * (1.0 - probability);
    count_sum.add(probability);
    variance_sum.add(h);
  }
  count = count_sum.value();
  variance = variance_sum.value();
  return std::isfinite(count) && std::isfinite(variance);
}

CountProfile profile_count_tilt(const VectorXd& linear_predictor,
                                const int m,
                                const double default_initial,
                                const double warm_initial,
                                SolverCounters& counters) {
  CountProfile out;
  const int n = static_cast<int>(linear_predictor.size());
  if (n < 2 || m <= 0 || m >= n) return out;

  double initial = warm_initial;
  if (std::isfinite(initial)) {
    ++counters.gamma_warm_starts;

  } else {
    initial = default_initial;
  }
  if (!std::isfinite(initial)) initial = 0.0;

  double initial_count = kNaN;
  double initial_variance = kNaN;
  if (!count_at_gamma(linear_predictor, initial, initial_count,
                      initial_variance, counters)) {
    return out;
  }
  const double target = static_cast<double>(m);
  const double count_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, static_cast<double>(n));
  if (std::abs(initial_count - target) <= count_tolerance) {
    if (initial_variance <= 0.0) return out;
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
      Rcpp::checkUserInterrupt();
      lower = initial - step;
      double ignored_variance = kNaN;
      if (!count_at_gamma(linear_predictor, lower, lower_count,
                          ignored_variance, counters)) {
        return out;
      }
      if (lower_count <= target) break;
      step *= 2.0;
    }
  } else if (initial_count < target - count_tolerance) {
    for (int expansion = 0; expansion < 80; ++expansion) {
      Rcpp::checkUserInterrupt();
      upper = initial + step;
      double ignored_variance = kNaN;
      if (!count_at_gamma(linear_predictor, upper, upper_count,
                          ignored_variance, counters)) {
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
    Rcpp::checkUserInterrupt();
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
    // Do not repeatedly rescan cells when no representable update remains.
    // The unchanged residual gate below still decides whether this is valid.
    if (candidate == gamma || candidate <= lower || candidate >= upper) {
      ++counters.count_stagnations;
      const bool use_lower = std::abs(lower_count - target) <
                             std::abs(upper_count - target);
      const double best_count = use_lower ? lower_count : upper_count;
      if (std::abs(best_count - target) < std::abs(count - target)) {
        gamma = use_lower ? lower : upper;
        if (!count_at_gamma(linear_predictor, gamma, count, S0, counters)) {
          return out;
        }
      }
      break;
    }
    gamma = candidate;
    if (!count_at_gamma(linear_predictor, gamma, count, S0, counters)) {
      return out;
    }
  }

  if (std::abs(count - target) > 4.0 * count_tolerance || S0 <= 0.0) {
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
  bool locality_ok = true;
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
  double max_delta = 0.0;
  int gamma_iterations = 0;
};

Evaluation evaluate_rpt_exact(const InfoCache& cache,
                              const VectorXd& x,
                              const double target,
                              const double gamma_initial,
                              EvaluationWorkspace& workspace,
                              SolverCounters& counters) {
  Evaluation out;
  ++counters.exact_evaluations;
  const int d = cache.d;
  if (x.size() != d + 1 || !x.allFinite()) return out;
  const VectorXd theta = x.head(d);
  const double lambda = x[d];

  VectorXd& linear_predictor = workspace.linear_predictor;
  if (linear_predictor.size() != cache.n) linear_predictor.resize(cache.n);
  long double mean_tilt = 0.0L;
  for (int i = 0; i < cache.n; ++i) {
    if ((i & 8191) == 0) Rcpp::checkUserInterrupt();
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < d; ++j) tilt += theta[j] * g[j * cache.n];
    if (!std::isfinite(tilt)) return out;
    mean_tilt += static_cast<long double>(tilt);
    linear_predictor[i] = cache.base_offset + tilt;
  }
  const double default_gamma =
      -static_cast<double>(mean_tilt / static_cast<long double>(cache.n));

  CountProfile profile = profile_count_tilt(
      linear_predictor, cache.m, default_gamma, gamma_initial, counters);
  if (!profile.valid && std::isfinite(gamma_initial)) {
    ++counters.gamma_cold_retries;

    profile = profile_count_tilt(
        linear_predictor, cache.m, default_gamma, kNaN, counters);
  }
  if (!profile.valid) return out;
  counters.gamma_iterations += profile.iterations;

  out.gamma = profile.gamma;
  out.gamma_iterations = profile.iterations;
  out.moment.setZero(d);
  out.Sg.setZero(d);
  out.Sgg.setZero(d, d);
  CountSum count_sum;
  CountSum variance_sum;
  double count_third_absolute_moment = 0.0;
  for (int i = 0; i < cache.n; ++i) {
    if ((i & 8191) == 0) Rcpp::checkUserInterrupt();
    const double probability = expit_stable(
        linear_predictor[i] + profile.gamma);
    if (!std::isfinite(probability)) return out;
    const double h = probability * (1.0 - probability);
    count_sum.add(probability);
    variance_sum.add(h);
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
  out.tilted_count = count_sum.value();
  out.S0 = variance_sum.value();
  symmetrize_lower(out.Sgg);

  if (!out.moment.allFinite() || !out.Sg.allFinite() ||
      !out.Sgg.allFinite() || !std::isfinite(out.S0) || out.S0 <= 0.0) {
    return out;
  }
  const double count_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, static_cast<double>(cache.n));
  if (std::abs(out.tilted_count - cache.m) > 4.0 * count_tolerance) {
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

struct CompressedWorkspace {
  VectorXd exception_linear;
};

bool zero_count_variance(const MomentSynopsis& synopsis,
                         const MomentContractions& contractions,
                         const double baseline_logit,
                         double& count,
                         double& variance) {
  if (!contractions.valid || !std::isfinite(baseline_logit)) return false;
  const double probability = expit_stable(baseline_logit);
  const double h = probability * (1.0 - probability);
  const double hp = h * (1.0 - 2.0 * probability);
  const double hpp = h *
      (1.0 - 6.0 * probability + 6.0 * probability * probability);
  count = probability * synopsis.M0 + h * contractions.L0 +
          0.5 * hp * contractions.Q0;
  variance = h * synopsis.M0 + hp * contractions.L0 +
             0.5 * hpp * contractions.Q0;
  return std::isfinite(count) && std::isfinite(variance) && variance > 0.0;
}

bool compressed_count_at_gamma(const InfoCache& cache,
                               const MomentSynopsis& synopsis,
                               const MomentContractions& contractions,
                               const VectorXd& exception_linear,
                               const double gamma,
                               double& count,
                               double& variance,
                               SolverCounters& counters) {
  if (!std::isfinite(gamma) ||
      exception_linear.size() !=
          static_cast<Eigen::Index>(synopsis.exceptions.size())) {
    return false;
  }
  ++counters.compressed_count_passes;
  ++counters.compressed_exception_passes;

  double zero_count = kNaN;
  double zero_variance = kNaN;
  if (!zero_count_variance(synopsis, contractions,
                           cache.base_offset + gamma,
                           zero_count, zero_variance)) {
    return false;
  }
  CountSum count_sum;
  CountSum variance_sum;
  count_sum.add(zero_count);
  variance_sum.add(zero_variance);
  for (Eigen::Index k = 0; k < exception_linear.size(); ++k) {
    if ((k & 8191) == 0) Rcpp::checkUserInterrupt();
    const double probability = expit_stable(
        cache.base_offset + gamma + exception_linear[k]);
    if (!std::isfinite(probability)) return false;
    const double h = probability * (1.0 - probability);
    count_sum.add(probability);
    variance_sum.add(h);
  }
  count = count_sum.value();
  variance = variance_sum.value();
  return std::isfinite(count) && std::isfinite(variance) && variance > 0.0;
}

CountProfile profile_count_tilt_compressed(
    const InfoCache& cache,
    const MomentSynopsis& synopsis,
    const MomentContractions& contractions,
    const VectorXd& exception_linear,
    const double default_initial,
    const double warm_initial,
    SolverCounters& counters) {
  CountProfile out;
  double initial = warm_initial;
  if (std::isfinite(initial)) {
    ++counters.compressed_gamma_warm_starts;

  } else {
    initial = default_initial;
  }
  if (!std::isfinite(initial)) initial = 0.0;
  double initial_count = kNaN;
  double initial_variance = kNaN;
  if (!compressed_count_at_gamma(cache, synopsis, contractions,
                                 exception_linear, initial, initial_count,
                                 initial_variance, counters)) {
    return out;
  }
  const double target = static_cast<double>(cache.m);
  const double count_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, static_cast<double>(cache.n));
  if (std::abs(initial_count - target) <= count_tolerance) {
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
      Rcpp::checkUserInterrupt();
      lower = initial - step;
      double ignored_variance = kNaN;
      if (!compressed_count_at_gamma(cache, synopsis, contractions,
                                     exception_linear, lower, lower_count,
                                     ignored_variance, counters)) {
        return out;
      }
      if (lower_count <= target) break;
      step *= 2.0;
    }
  } else {
    for (int expansion = 0; expansion < 80; ++expansion) {
      Rcpp::checkUserInterrupt();
      upper = initial + step;
      double ignored_variance = kNaN;
      if (!compressed_count_at_gamma(cache, synopsis, contractions,
                                     exception_linear, upper, upper_count,
                                     ignored_variance, counters)) {
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
    Rcpp::checkUserInterrupt();
    out.iterations = iteration + 1;
    if (std::abs(count - target) <= count_tolerance) break;
    if (count < target) {
      lower = gamma;
      lower_count = count;
    } else {
      upper = gamma;
      upper_count = count;
    }
    double candidate = gamma - (count - target) / S0;
    if (!std::isfinite(candidate) || candidate <= lower ||
        candidate >= upper) {
      candidate = lower + 0.5 * (upper - lower);
    }
    if (candidate == gamma || candidate <= lower || candidate >= upper) {
      ++counters.compressed_count_stagnations;
      const bool use_lower = std::abs(lower_count - target) <
                             std::abs(upper_count - target);
      const double best_count = use_lower ? lower_count : upper_count;
      if (std::abs(best_count - target) < std::abs(count - target)) {
        gamma = use_lower ? lower : upper;
        if (!compressed_count_at_gamma(cache, synopsis, contractions,
                                       exception_linear, gamma, count, S0,
                                       counters)) {
          return out;
        }
      }
      break;
    }
    gamma = candidate;
    if (!compressed_count_at_gamma(cache, synopsis, contractions,
                                   exception_linear, gamma, count, S0,
                                   counters)) {
      return out;
    }
  }
  if (std::abs(count - target) > 4.0 * count_tolerance || S0 <= 0.0) {
    return out;
  }
  out.valid = true;
  out.gamma = gamma;
  out.count = count;
  out.S0 = S0;
  return out;
}

Evaluation evaluate_rpt_compressed(const InfoCache& cache,
                                   const MomentSynopsis& synopsis,
                                   const VectorXd& x,
                                   const double target,
                                   const double gamma_initial,
                                   const double delta_limit,
                                   CompressedWorkspace& workspace,
                                   SolverCounters& counters) {
  Evaluation out;
  ++counters.compressed_evaluations;
  const int d = cache.d;
  const int p = cache.p;
  if (x.size() != d + 1 || !x.allFinite() || !synopsis.valid) return out;
  const VectorXd theta = x.head(d);
  const double lambda = x[d];
  VectorXd psi = theta.segment(1, p);
  psi[0] -= static_cast<double>(cache.score_sign) * theta[0];

  const MomentContractions contractions = contract_moments(synopsis, psi);

  out.max_delta = contractions.max_delta;
  out.locality_ok = contractions.valid &&
                    contractions.max_delta <= delta_limit;
  if (!out.locality_ok) return out;

  workspace.exception_linear.resize(synopsis.exceptions.size());
  for (std::size_t k = 0; k < synopsis.exceptions.size(); ++k) {
    const int i = synopsis.exceptions[k];
    const double* g = cache.G.data() + i;
    double tilt = 0.0;
    for (int j = 0; j < d; ++j) tilt += theta[j] * g[j * cache.n];
    if (!std::isfinite(tilt)) return out;
    workspace.exception_linear[static_cast<Eigen::Index>(k)] = tilt;
  }
  ++counters.compressed_exception_passes;

  const CountProfile profile = profile_count_tilt_compressed(
      cache, synopsis, contractions, workspace.exception_linear,
      -(psi.dot(synopsis.M1) + workspace.exception_linear.sum()) /
          static_cast<double>(cache.n),
      gamma_initial, counters);
  CountProfile accepted_profile = profile;
  if (!accepted_profile.valid && std::isfinite(gamma_initial)) {
    ++counters.compressed_gamma_cold_retries;

    accepted_profile = profile_count_tilt_compressed(
        cache, synopsis, contractions, workspace.exception_linear,
        -(psi.dot(synopsis.M1) + workspace.exception_linear.sum()) /
            static_cast<double>(cache.n),
        kNaN, counters);
  }
  if (!accepted_profile.valid) return out;
  counters.compressed_gamma_iterations += accepted_profile.iterations;

  const ZeroAggregate zero = evaluate_zero_aggregate(
      synopsis, contractions, cache.base_offset + accepted_profile.gamma);
  if (!zero.valid) return out;

  out.gamma = accepted_profile.gamma;
  out.gamma_iterations = accepted_profile.iterations;
  CountSum count_sum;
  CountSum variance_sum;
  count_sum.add(zero.count);
  variance_sum.add(zero.S0);
  out.moment.setZero(d);
  out.Sg.setZero(d);
  out.Sgg.setZero(d, d);
  out.moment[0] = -static_cast<double>(cache.score_sign) * zero.moment_f[0];
  out.Sg[0] = -static_cast<double>(cache.score_sign) * zero.Sf[0];
  for (int j = 0; j < p; ++j) {
    out.moment[j + 1] = zero.moment_f[j];
    out.Sg[j + 1] = zero.Sf[j];
  }
  out.Sgg(0, 0) = zero.Sff(0, 0);
  for (int j = 0; j < p; ++j) {
    const double cross = -static_cast<double>(cache.score_sign) *
                         zero.Sff(0, j);
    out.Sgg(0, j + 1) = cross;
    out.Sgg(j + 1, 0) = cross;
    for (int k = 0; k < p; ++k) {
      out.Sgg(j + 1, k + 1) = zero.Sff(j, k);
    }
  }

  for (std::size_t position = 0;
       position < synopsis.exceptions.size(); ++position) {
    if ((position & 8191) == 0) Rcpp::checkUserInterrupt();
    const int i = synopsis.exceptions[position];
    const double probability = expit_stable(
        cache.base_offset + accepted_profile.gamma +
        workspace.exception_linear[static_cast<Eigen::Index>(position)]);
    if (!std::isfinite(probability)) return out;
    const double h = probability * (1.0 - probability);
    count_sum.add(probability);
    variance_sum.add(h);
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
  out.tilted_count = count_sum.value();
  out.S0 = variance_sum.value();
  ++counters.compressed_exception_passes;
  symmetrize_lower(out.Sgg);

  const double count_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, static_cast<double>(cache.n));
  if (std::abs(out.tilted_count - cache.m) > 4.0 * count_tolerance ||
      !out.moment.allFinite() || !out.Sg.allFinite() ||
      !out.Sgg.allFinite() || !std::isfinite(out.S0) || out.S0 <= 0.0) {
    return out;
  }
  out.count_variance_ratio = out.S0 / cache.base_count_variance;
  out.conditional_information =
      out.Sgg - out.Sg * out.Sg.transpose() / out.S0;
  out.conditional_information =
      0.5 * (out.conditional_information +
             out.conditional_information.transpose());
  if (!out.conditional_information.allFinite()) return out;
  Eigen::SelfAdjointEigenSolver<MatrixXd> information_eigenvalues(
      out.conditional_information, Eigen::EigenvaluesOnly);
  if (information_eigenvalues.info() != Eigen::Success) return out;
  const double information_scale =
      std::max(1.0, out.conditional_information.cwiseAbs().maxCoeff());
  if (information_eigenvalues.eigenvalues().minCoeff() <
      -1.0e-8 * information_scale) {
    return out;
  }
  out.boundary = boundary_terms(out.moment, cache.C_inv);
  if (!out.boundary.valid) return out;
  out.residual.setZero(d + 1);
  out.residual.head(d) = theta - lambda * out.boundary.grad;
  out.residual[d] = out.boundary.b - target;
  out.jacobian.setZero(d + 1, d + 1);
  out.jacobian.block(0, 0, d, d) =
      MatrixXd::Identity(d, d) -
      lambda * out.boundary.hess * out.conditional_information;
  out.jacobian.block(0, d, d, 1) = -out.boundary.grad;
  out.jacobian.block(d, 0, 1, d) =
      (out.conditional_information * out.boundary.grad).transpose();
  out.valid = out.residual.allFinite() && out.jacobian.allFinite();
  return out;
}

struct CompressedSolve {
  bool converged = false;
  std::string reason = "not_started";
  VectorXd state;
  Evaluation evaluation;
  std::vector<double> history;
  int updates = 0;
  double maximum_locality_bound = 0.0;
};

CompressedSolve run_rpt_compressed(const InfoCache& cache,
                                   const MomentSynopsis& synopsis,
                                   const double target,
                                   const double tolerance,
                                   const int max_iterations,
                                   const double delta_limit,
                                   const Evaluation& exact_base,
                                   CompressedWorkspace& workspace,
                                   SolverCounters& counters) {
  CompressedSolve out;
  out.state = VectorXd::Zero(cache.d + 1);
  if (!exact_base.valid) {
    out.reason = "invalid_exact_center";
    return out;
  }
  const double center = exact_base.boundary.b;
  const double variance = exact_base.boundary.grad.dot(
      exact_base.conditional_information * exact_base.boundary.grad);
  if (!std::isfinite(variance) || variance <= 1e-14) {
    out.reason = "degenerate_center_variance";
    return out;
  }

  out.state[cache.d] = (target - center) / variance;
  out.state.head(cache.d) =
      out.state[cache.d] * exact_base.boundary.grad;
  const double initial_gamma =
      exact_base.gamma -
      exact_base.Sg.dot(out.state.head(cache.d)) / exact_base.S0;
  out.evaluation = evaluate_rpt_compressed(
      cache, synopsis, out.state, target, initial_gamma, delta_limit,
      workspace, counters);
  if (std::isfinite(out.evaluation.max_delta)) {
    out.maximum_locality_bound = out.evaluation.max_delta;
  }

  for (int iteration = 0; iteration <= max_iterations; ++iteration) {
    Rcpp::checkUserInterrupt();
    if (!out.evaluation.valid) {
      out.reason = out.evaluation.locality_ok
                       ? "invalid_compressed_evaluation"
                       : "locality_guard";
      break;
    }
    const double residual = max_abs(out.evaluation.residual);
    out.history.push_back(residual);
    if (residual <= tolerance) {
      out.converged = true;
      out.reason = "ok";
      break;
    }
    if (iteration == max_iterations) {
      out.reason = "compressed_iteration_limit";
      break;
    }

    VectorXd step;
    if (!solve_linear(out.evaluation.jacobian,
                      -out.evaluation.residual, step)) {
      out.reason = "compressed_singular_jacobian";
      break;
    }
    bool accepted = false;
    double scale = 1.0;
    bool saw_locality_rejection = false;
    for (int backtrack = 0; backtrack < 24; ++backtrack) {
      Rcpp::checkUserInterrupt();
      const VectorXd candidate = out.state + scale * step;
      const VectorXd theta_delta =
          candidate.head(cache.d) - out.state.head(cache.d);
      const double gamma_initial =
          out.evaluation.gamma -
          out.evaluation.Sg.dot(theta_delta) / out.evaluation.S0;
      Evaluation trial = evaluate_rpt_compressed(
          cache, synopsis, candidate, target, gamma_initial, delta_limit,
          workspace, counters);
      if (std::isfinite(trial.max_delta)) {
        out.maximum_locality_bound =
            std::max(out.maximum_locality_bound, trial.max_delta);
      }
      if (!trial.locality_ok) saw_locality_rejection = true;
      const double trial_residual =
          trial.valid ? max_abs(trial.residual) : kNaN;
      if (trial.valid && trial_residual < residual) {
        out.state = candidate;
        out.evaluation = std::move(trial);
        ++out.updates;
        accepted = true;
        break;
      }
      scale *= 0.5;
    }
    if (!accepted) {
      out.reason = saw_locality_rejection
                       ? "locality_guard_or_line_search"
                       : "compressed_line_search_failed";
      break;
    }
  }
  return out;
}

bool polish_rpt_exact(const InfoCache& cache,
                      const double target,
                      const double tolerance,
                      const int maximum_updates,
                      VectorXd& state,
                      Evaluation& current,
                      EvaluationWorkspace& workspace,
                      SolverCounters& counters,
                      std::vector<double>& history,
                      int& updates) {
  updates = 0;
  for (int iteration = 0; iteration <= maximum_updates; ++iteration) {
    Rcpp::checkUserInterrupt();
    if (!current.valid) return false;
    const double residual = max_abs(current.residual);
    history.push_back(residual);
    if (residual <= tolerance) return true;
    if (iteration == maximum_updates) return false;

    VectorXd step;
    if (!solve_linear(current.jacobian, -current.residual, step)) {
      return false;
    }
    bool accepted = false;
    double scale = 1.0;
    for (int backtrack = 0; backtrack < 24; ++backtrack) {
      Rcpp::checkUserInterrupt();
      const VectorXd candidate = state + scale * step;
      const VectorXd theta_delta =
          candidate.head(cache.d) - state.head(cache.d);
      const double gamma_initial =
          current.gamma - current.Sg.dot(theta_delta) / current.S0;
      Evaluation trial = evaluate_rpt_exact(
          cache, candidate, target, gamma_initial, workspace, counters);
      const double trial_residual =
          trial.valid ? max_abs(trial.residual) : kNaN;
      if (trial.valid && trial_residual < residual) {
        state = candidate;
        current = std::move(trial);
        ++updates;
        accepted = true;
        break;
      }
      scale *= 0.5;
    }
    if (!accepted) return false;
  }
  return false;
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
                   const double gamma,
                   const VectorXd* accepted_linear_predictor = nullptr) {
  if (accepted_linear_predictor != nullptr &&
      accepted_linear_predictor->size() != cache.n) {
    return kNaN;
  }
  double rate = 0.0;
  for (int i = 0; i < cache.n; ++i) {
    if ((i & 8191) == 0) Rcpp::checkUserInterrupt();
    double tilt = gamma;
    if (accepted_linear_predictor == nullptr) {
      const double* g = cache.G.data() + i;
      for (int j = 0; j < cache.d; ++j) {
        tilt += theta[j] * g[j * cache.n];
      }
    } else {
      tilt += (*accepted_linear_predictor)[i] - cache.base_offset;
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
                        const std::vector<double>& history,
                        const VectorXd* accepted_linear_predictor = nullptr) {
  const VectorXd theta = x.head(cache.d);
  const double kl_rate = rpt_kl_rate(
      cache, theta, evaluation.gamma, accepted_linear_predictor);
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

Rcpp::List add_acceleration_diagnostics(Rcpp::List out,
                                        const SolverCounters& counters) {
  out["path"] = "full_exact_conditional_prepared";
  out["analytic_center_evaluations"] = counters.analytic_center_evaluations;
  out["exact_evaluations"] = counters.exact_evaluations;
  out["count_passes"] = counters.count_passes;
  out["count_stagnations"] = counters.count_stagnations;
  out["gamma_iterations_total"] = counters.gamma_iterations;
  out["gamma_warm_starts"] = counters.gamma_warm_starts;
  out["gamma_cold_retries"] = counters.gamma_cold_retries;
  out["compressed_evaluations"] = counters.compressed_evaluations;
  out["compressed_count_passes"] = counters.compressed_count_passes;
  out["compressed_count_stagnations"] = counters.compressed_count_stagnations;
  out["compressed_exception_passes"] =
      counters.compressed_exception_passes;
  out["compressed_gamma_iterations_total"] =
      counters.compressed_gamma_iterations;
  out["compressed_gamma_warm_starts"] =
      counters.compressed_gamma_warm_starts;
  out["compressed_gamma_cold_retries"] =
      counters.compressed_gamma_cold_retries;
  out["acceleration_id"] = "rpt_exact_warm_gamma_fused_v1";
  return out;
}

Rcpp::List add_moment_diagnostics(Rcpp::List out,
                                  const SolverCounters& counters,
                                  const MomentDiagnostics& diagnostics) {
  out = add_acceleration_diagnostics(out, counters);
  out["path"] = diagnostics.path;
  out["acceleration_id"] =
      diagnostics.maximum_moment_degree == 2
          ? "rpt_zero_moments_28_exact_audit_v1"
          : (diagnostics.maximum_moment_degree == 3
                 ? "rpt_zero_moments_84_exact_audit_v1"
                 : "rpt_zero_moments_210_exact_audit_v1");
  out["moment_approximation"] =
      diagnostics.maximum_moment_degree == 2
          ? "linear_mean_constant_raw_sff_surrogate_newton"
          : (diagnostics.maximum_moment_degree == 3
                 ? "quadratic_mean_linear_raw_sff_quasi_newton"
                 : "second_order_probability_variance_quasi_newton");
  out["mean_taylor_order"] = diagnostics.maximum_moment_degree == 2 ? 1 : 2;
  out["raw_sff_taylor_order"] = diagnostics.maximum_moment_degree - 2;
  out["fixed_gamma_mean_derivative_consistent"] =
      diagnostics.maximum_moment_degree <= 3;
  out["profiled_jacobian_derivative_consistent"] =
      diagnostics.maximum_moment_degree == 2;
  out["profiled_information_role"] = diagnostics.maximum_moment_degree == 2
      ? "surrogate_cgf_newton" : "symmetric_quasi_newton";
  out["highest_derivative_moment"] = diagnostics.maximum_moment_degree;
  out["final_result_source"] =
      diagnostics.exact_audit_passed
          ? "full_exact_audit_or_polish"
          : (diagnostics.path == "moment_geometry_empirical_fallback" ||
                     diagnostics.path == "moment_solver_disabled_empirical_fallback"
                 ? "empirical_fallback_required"
                 : "full_exact_fallback");
  out["moment_fallback_reason"] = diagnostics.fallback_reason;
  out["moment_eligible"] = diagnostics.eligible;
  out["synopsis_built"] = diagnostics.synopsis_built;
  out["exact_audit_passed"] = diagnostics.exact_audit_passed;
  out["exact_audit_tolerance"] = diagnostics.exact_audit_tolerance;
  out["effective_exact_audit_tolerance"] =
      diagnostics.effective_exact_audit_tolerance;
  out["zero_count"] = diagnostics.n_zero;
  out["exception_count"] = diagnostics.n_exception;
  out["zero_fraction"] = diagnostics.zero_fraction;
  out["packed_moment_count"] = diagnostics.packed_moment_count;
  out["maximum_moment_degree"] = diagnostics.maximum_moment_degree;
  out["expanded_moment_entry_count"] = diagnostics.synopsis_built
      ? 1 + diagnostics.feature_dimension +
            diagnostics.feature_dimension * diagnostics.feature_dimension +
            (diagnostics.maximum_moment_degree >= 3
                 ? diagnostics.feature_dimension * diagnostics.feature_dimension *
                       diagnostics.feature_dimension
                 : 0) +
            (diagnostics.maximum_moment_degree == 4
                 ? diagnostics.feature_dimension *
                       diagnostics.feature_dimension *
                       diagnostics.feature_dimension *
                       diagnostics.feature_dimension
                 : 0)
      : 0;
  out["compressed_updates"] = diagnostics.compressed_updates;
  out["exact_polish_updates"] = diagnostics.exact_polish_updates;
  out["maximum_locality_bound"] = diagnostics.maximum_locality_bound;
  return out;
}

void cache_compensated_add(const double value, double& sum, double& correction) {
  const double next = sum + value;
  correction += std::abs(sum) >= std::abs(value)
      ? (sum - next) + value : (value - next) + sum;
  sum = next;
}

InfoCache build_cache(const Rcpp::NumericVector& a,
                      const Rcpp::NumericVector& w,
                      const Rcpp::NumericMatrix& Z,
                      const int m,
                      const int score_sign,
                      const bool require_supported_geometry = true) {
  if (a.size() > std::numeric_limits<int>::max()) {
    Rcpp::stop("the cell count exceeds the supported integer range");
  }
  const int n = static_cast<int>(a.size());
  const int p = Z.ncol();
  if (n < 2 || p < 1 || Z.nrow() != n || w.size() != n) {
    Rcpp::stop("input dimensions do not match");
  }
  if (p == std::numeric_limits<int>::max() ||
      static_cast<long double>(n) * (static_cast<long double>(p) + 1.0L) >
          std::numeric_limits<Eigen::Index>::max() / sizeof(double)) {
    Rcpp::stop("the response geometry exceeds supported storage");
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
  cache.positive_feature_sums = VectorXd::Zero(cache.d);
  cache.positive_feature_centered_crossproducts = MatrixXd::Zero(cache.d, cache.d);
  VectorXd sum_correction = VectorXd::Zero(cache.d);
  VectorXd feature_anchor = VectorXd::Zero(cache.d);
  VectorXd shifted_sum = VectorXd::Zero(cache.d);
  VectorXd shifted_correction = VectorXd::Zero(cache.d);
  VectorXd positive_features(cache.d);
  MatrixXd C = MatrixXd::Zero(p, p);
  VectorXd weighted_z = VectorXd::Zero(p);

  for (int i = 0; i < n; ++i) {
    if ((i & 8191) == 0) Rcpp::checkUserInterrupt();
    if (!std::isfinite(a[i])) Rcpp::stop("a must be finite");
    if (!std::isfinite(w[i]) || w[i] <= 0.0) {
      Rcpp::stop("w must be positive and finite");
    }
    cache.G(i, 0) = static_cast<double>(score_sign) * a[i];
    positive_features[0] = a[i];
    for (int j = 0; j < p; ++j) {
      if (!std::isfinite(Z(i, j))) Rcpp::stop("Z must be finite");
      weighted_z[j] = w[i] * Z(i, j);
      if (!std::isfinite(weighted_z[j])) {
        Rcpp::stop("w multiplied by Z must be finite");
      }
      cache.G(i, j + 1) = weighted_z[j];
      positive_features[j + 1] = weighted_z[j];
    }
    if (i == 0) feature_anchor = positive_features;
    for (int j = 0; j < cache.d; ++j) {
      cache_compensated_add(positive_features[j],
          cache.positive_feature_sums[j], sum_correction[j]);
      cache_compensated_add(positive_features[j] - feature_anchor[j],
          shifted_sum[j], shifted_correction[j]);
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

  cache.positive_feature_sums += sum_correction;
  const VectorXd mean_shift =
      (shifted_sum + shifted_correction) / static_cast<double>(n);
  if (!cache.positive_feature_sums.allFinite() || !mean_shift.allFinite()) {
    Rcpp::stop("response null moments must be finite");
  }
  // Keep the mean split so centering retains small variation on large offsets.
  // The bounded workspace lets Eigen form crossproducts without per-cell
  // compensated rank updates or another full response-sized allocation.
  const int block_rows = std::max(1, std::min(2048, 65536 / cache.d));
  MatrixXd centered_block(block_rows, cache.d);
  MatrixXd block_crossproducts(cache.d, cache.d);
  MatrixXd centered_correction = MatrixXd::Zero(cache.d, cache.d);
  for (int start = 0; start < n;) {
    Rcpp::checkUserInterrupt();
    const int count = std::min(block_rows, n - start);
    auto block = centered_block.topRows(count);
    for (int j = 0; j < cache.d; ++j) {
      block.col(j) = cache.G.col(j).segment(start, count);
      if (j == 0 && score_sign == -1) block.col(j) *= -1.0;
      block.col(j).array() -= feature_anchor[j];
      block.col(j).array() -= mean_shift[j];
    }
    block_crossproducts.setZero();
    block_crossproducts.selfadjointView<Eigen::Lower>().rankUpdate(block.transpose());
    for (int j = 0; j < cache.d; ++j) {
      for (int k = 0; k <= j; ++k) {
        cache_compensated_add(block_crossproducts(j, k),
            cache.positive_feature_centered_crossproducts(j, k),
            centered_correction(j, k));
      }
    }
    start += count;
  }
  cache.positive_feature_centered_crossproducts += centered_correction;
  symmetrize_lower(cache.positive_feature_centered_crossproducts);
  cache.positive_feature_crossproducts =
      cache.positive_feature_centered_crossproducts +
      cache.positive_feature_sums * cache.positive_feature_sums.transpose() /
          static_cast<double>(n);
  if (!cache.positive_feature_sums.allFinite() ||
      !cache.positive_feature_centered_crossproducts.allFinite() ||
      !cache.positive_feature_crossproducts.allFinite()) {
    Rcpp::stop("response null moments must be finite");
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
  } catch (const Rcpp::internal::InterruptedException&) {
    throw;
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

void validate_exact_audit_tolerance(const double exact_audit_tolerance) {
  if (!std::isfinite(exact_audit_tolerance) || exact_audit_tolerance <= 0.0) {
    Rcpp::stop("exact_audit_tolerance must be positive and finite");
  }
}

void configure_prepared_pair(PreparedMomentContext& context,
                             const int m,
                             const int score_sign) {
  // Prepared contexts are worker-local; each pair reuses response geometry.
  InfoCache& cache = context.cache;
  if (m <= 0 || m >= cache.n) {
    Rcpp::stop("m must be strictly between zero and the number of cells");
  }
  if (score_sign != -1 && score_sign != 1) {
    Rcpp::stop("score_sign must be -1 or +1");
  }
  if (score_sign != cache.score_sign) {
    cache.G.col(0) *= -1.0;
    cache.score_sign = score_sign;
  }
  cache.m = m;
  cache.base_probability = static_cast<double>(m) / cache.n;
  cache.base_count_variance =
      cache.n * cache.base_probability * (1.0 - cache.base_probability);
  cache.base_offset = std::log(cache.base_probability /
                               (1.0 - cache.base_probability));
}

Evaluation evaluate_rpt_null(const InfoCache& cache, const double target,
                             SolverCounters& counters) {
  ++counters.analytic_center_evaluations;
  Evaluation out;
  const double probability = cache.base_probability;
  const double h = probability * (1.0 - probability);
  if (!std::isfinite(target) || !std::isfinite(h) || h <= 0.0) return out;

  out.gamma = 0.0;
  out.tilted_count = static_cast<double>(cache.m);
  out.S0 = static_cast<double>(cache.n) * h;
  out.moment = probability * cache.positive_feature_sums;
  out.Sg = h * cache.positive_feature_sums;
  out.Sgg = h * cache.positive_feature_crossproducts;
  // This is the profiled Bernoulli CGF curvature, without an N/(N-1) factor.
  out.conditional_information = h * cache.positive_feature_centered_crossproducts;
  if (cache.score_sign == -1) {
    out.moment[0] *= -1.0;
    out.Sg[0] *= -1.0;
    out.Sgg.row(0) *= -1.0;
    out.Sgg.col(0) *= -1.0;
    out.conditional_information.row(0) *= -1.0;
    out.conditional_information.col(0) *= -1.0;
  }
  out.count_variance_ratio = out.S0 / cache.base_count_variance;
  out.count_berry_esseen_ratio =
      ((1.0 - probability) * (1.0 - probability) + probability * probability) /
      std::sqrt(out.S0);
  if (!out.moment.allFinite() || !out.Sg.allFinite() || !out.Sgg.allFinite() ||
      !out.conditional_information.allFinite() || !std::isfinite(out.S0) ||
      out.S0 <= 0.0 || !std::isfinite(out.count_variance_ratio) ||
      !std::isfinite(out.count_berry_esseen_ratio)) {
    return out;
  }

  out.boundary = boundary_terms(out.moment, cache.C_inv);
  if (!out.boundary.valid) return out;
  out.residual = VectorXd::Zero(cache.d + 1);
  out.residual[cache.d] = out.boundary.b - target;
  out.jacobian = MatrixXd::Zero(cache.d + 1, cache.d + 1);
  out.jacobian.topLeftCorner(cache.d, cache.d).setIdentity();
  out.jacobian.block(0, cache.d, cache.d, 1) = -out.boundary.grad;
  out.jacobian.block(cache.d, 0, 1, cache.d) =
      (out.conditional_information * out.boundary.grad).transpose();
  out.valid = out.residual.allFinite() && out.jacobian.allFinite();
  return out;
}

Rcpp::List run_rpt_full(const InfoCache& cache,
                        const double target,
                        const double tolerance,
                        const int max_iterations,
                        EvaluationWorkspace& workspace,
                        SolverCounters& counters,
                        const int minimum_updates = 0,
                        const Evaluation* supplied_base = nullptr) {
  VectorXd x = VectorXd::Zero(cache.d + 1);
  const std::vector<double> empty_history;
  Evaluation base = supplied_base == nullptr
                        ? evaluate_rpt_null(cache, target, counters)
                        : *supplied_base;
  if (!base.valid) {
    return add_acceleration_diagnostics(
        failure_result(cache, "invalid_center", target, kNaN, 0, kNaN,
                       empty_history, x, counters.exact_evaluations, &base),
        counters);
  }

  const double center = base.boundary.b;
  const double variance = base.boundary.grad.dot(
      base.conditional_information * base.boundary.grad);
  if (!std::isfinite(variance) || variance <= 1e-14) {
    return add_acceleration_diagnostics(
        failure_result(cache, "degenerate_center_variance", target,
                       center, 0, kNaN, empty_history, x,
                       counters.exact_evaluations, &base),
        counters);
  }

  x[cache.d] = (target - center) / variance;
  x.head(cache.d) = x[cache.d] * base.boundary.grad;
  const double initial_gamma =
      base.gamma - base.Sg.dot(x.head(cache.d)) / base.S0;
  Evaluation current = evaluate_rpt_exact(
      cache, x, target, initial_gamma, workspace, counters);
  std::vector<double> history;
  int updates = 0;
  bool converged = false;

  for (int iteration = 0; iteration <= max_iterations; ++iteration) {
    Rcpp::checkUserInterrupt();
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
      Rcpp::checkUserInterrupt();
      const VectorXd candidate = x + scale * step;
      const VectorXd theta_delta =
          candidate.head(cache.d) - x.head(cache.d);
      const double gamma_initial =
          current.gamma - current.Sg.dot(theta_delta) / current.S0;
      Evaluation trial = evaluate_rpt_exact(
          cache, candidate, target, gamma_initial, workspace, counters);
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
    return add_acceleration_diagnostics(
        failure_result(
            cache, "newton_failed", target, center, updates,
            current.valid ? max_abs(current.residual) : kNaN, history, x,
            counters.exact_evaluations, &current),
        counters);
  }
  return add_acceleration_diagnostics(
      finalize_rpt(cache, x, current, center, target, updates, history,
                   &workspace.linear_predictor),
      counters);
}

Rcpp::List run_rpt_moment(const InfoCache& cache,
                          const Rcpp::LogicalVector& zero_mask,
                          const double target,
                          const double tolerance,
                          const double compressed_tolerance,
                          const int max_iterations,
                          const int maximum_polish_updates,
                          const double delta_limit,
                          const double minimum_zero_fraction,
                          const int maximum_moment_degree,
                          const double exact_audit_tolerance,
                          MomentDiagnostics& diagnostics,
                          const MomentSynopsis* supplied_synopsis = nullptr,
                          const bool prepared_context = false) {
  SolverCounters counters;
  EvaluationWorkspace exact_workspace(cache.n);
  CompressedWorkspace compressed_workspace;
  const VectorXd zero_state = VectorXd::Zero(cache.d + 1);
  validate_maximum_moment_degree(maximum_moment_degree);
  diagnostics.feature_dimension = cache.p;
  diagnostics.maximum_moment_degree = maximum_moment_degree;
  diagnostics.exact_audit_tolerance = exact_audit_tolerance;
  diagnostics.effective_exact_audit_tolerance =
      std::min(tolerance, exact_audit_tolerance);

  if (!prepared_context) {
    if (zero_mask.size() != cache.n) {
      Rcpp::stop("zero_mask must have one value per cell");
    }
    int n_zero = 0;
    for (int i = 0; i < cache.n; ++i) {
      if ((i & 8191) == 0) Rcpp::checkUserInterrupt();
      if (zero_mask[i] == NA_LOGICAL) {
        Rcpp::stop("zero_mask cannot contain missing values");
      }
      n_zero += zero_mask[i] ? 1 : 0;
    }
    diagnostics.n_zero = n_zero;
    diagnostics.n_exception = cache.n - n_zero;
    diagnostics.zero_fraction = static_cast<double>(n_zero) / cache.n;
    diagnostics.eligible =
        diagnostics.zero_fraction >= minimum_zero_fraction;
  }

  auto annotate = [&](Rcpp::List result) {
    return add_moment_diagnostics(result, counters, diagnostics);
  };
  auto exact_fallback = [&](const std::string& reason,
                            const Evaluation* supplied_base) {
    diagnostics.path = "moment_full_exact_fallback";
    diagnostics.fallback_reason = reason;

    Rcpp::List result = run_rpt_full(
        cache, target, tolerance, max_iterations, exact_workspace, counters,
        0, supplied_base);

    return annotate(result);
  };

  if (max_iterations == 0) {
    diagnostics.path = "moment_solver_disabled_empirical_fallback";
    diagnostics.fallback_reason = "solver_disabled_max_iterations_zero";
    return annotate(failure_result(
        cache, diagnostics.fallback_reason, target, kNaN, 0, kNaN,
        std::vector<double>(), zero_state, 0));
  }
  if (!diagnostics.eligible) {
    return exact_fallback("insufficient_zero_fraction", nullptr);
  }

  MomentSynopsis built_synopsis;
  const MomentSynopsis* synopsis = supplied_synopsis;
  if (synopsis == nullptr) {
    built_synopsis = build_moment_synopsis(
        cache, zero_mask, maximum_moment_degree);

    diagnostics.synopsis_built = true;
    diagnostics.packed_moment_count = built_synopsis.packed_moment_count;
    synopsis = &built_synopsis;
  }
  if (!synopsis->valid) {
    return exact_fallback("invalid_moment_synopsis", nullptr);
  }

  // Zero tilt uses the cached full-CGF null moments for either solve path.
  const Evaluation null_base = evaluate_rpt_null(cache, target, counters);
  if (!null_base.valid) {
    return exact_fallback("invalid_analytic_center", nullptr);
  }
  const double center = null_base.boundary.b;

  CompressedSolve compressed = run_rpt_compressed(
      cache, *synopsis, target, compressed_tolerance, max_iterations,
      delta_limit, null_base, compressed_workspace, counters);

  diagnostics.compressed_updates = compressed.updates;
  diagnostics.maximum_locality_bound = compressed.maximum_locality_bound;
  if (!compressed.converged) {
    return exact_fallback(compressed.reason, &null_base);
  }

  VectorXd exact_state = compressed.state;

  Evaluation exact_current = evaluate_rpt_exact(
      cache, exact_state, target, compressed.evaluation.gamma,
      exact_workspace, counters);

  if (!exact_current.valid) {
    return exact_fallback("invalid_exact_audit", &null_base);
  }

  std::vector<double> history = compressed.history;

  const bool exact_converged = polish_rpt_exact(
      cache, target, diagnostics.effective_exact_audit_tolerance,
      maximum_polish_updates, exact_state,
      exact_current, exact_workspace, counters, history,
      diagnostics.exact_polish_updates);

  if (!exact_converged) {
    return exact_fallback("exact_polish_failed", &null_base);
  }

  diagnostics.exact_audit_passed = true;
  diagnostics.path = diagnostics.exact_polish_updates == 0
                         ? "moment_compressed_exact_audit"
                         : "moment_compressed_exact_polish";
  diagnostics.fallback_reason = "none";
  Rcpp::List result = finalize_rpt(
      cache, exact_state, exact_current, center, target,
      compressed.updates + diagnostics.exact_polish_updates, history,
      &exact_workspace.linear_predictor);
  return annotate(result);
}

}  // namespace

namespace {

SEXP prepared_context_tag() {
  return Rf_install("sceptre_rpt_spa_response_v1");
}

PreparedMomentContext* checked_prepared_context(SEXP pointer,
                                               const bool allow_released = false) {
  if (TYPEOF(pointer) != EXTPTRSXP ||
      R_ExternalPtrTag(pointer) != prepared_context_tag() ||
      !Rf_inherits(pointer, "sceptre_rpt_spa_response_v1")) {
    Rcpp::stop("expected an RPT prepared response context");
  }
  auto* context = static_cast<PreparedMomentContext*>(R_ExternalPtrAddr(pointer));
  if (context == nullptr) {
    if (allow_released) return nullptr;
    Rcpp::stop("prepared context has been released or is null");
  }
  SEXP protected_metadata = R_ExternalPtrProtected(pointer);
  if (TYPEOF(protected_metadata) != INTSXP || XLENGTH(protected_metadata) != 3) {
    Rcpp::stop("prepared context has invalid dimension metadata");
  }
  const Rcpp::IntegerVector metadata(protected_metadata);
  const InfoCache& cache = context->cache;
  if (cache.n < 2 || cache.p < 1 || cache.d != cache.p + 1 ||
      metadata[0] != cache.n || metadata[1] != cache.p ||
      metadata[2] != context->maximum_moment_degree ||
      cache.G.rows() != cache.n || cache.G.cols() != cache.d ||
      cache.weights.size() != cache.n ||
      cache.C_inv.rows() != cache.p || cache.C_inv.cols() != cache.p ||
      cache.positive_feature_sums.size() != cache.d ||
      cache.positive_feature_crossproducts.rows() != cache.d ||
      cache.positive_feature_crossproducts.cols() != cache.d ||
      cache.positive_feature_centered_crossproducts.rows() != cache.d ||
      cache.positive_feature_centered_crossproducts.cols() != cache.d ||
      context->positive_feature_sums.size() != cache.d ||
      context->n_zero < 0 || context->n_zero > cache.n ||
      (context->eligible && (!context->synopsis.valid ||
          context->synopsis.p != cache.p ||
          context->synopsis.n_zero != context->n_zero ||
          context->synopsis.maximum_moment_degree != context->maximum_moment_degree))) {
    Rcpp::stop("prepared context has inconsistent dimensions");
  }
  return context;
}

void validate_moment_solver_controls(const double target,
                                    const double tolerance,
                                    const double compressed_tolerance,
                                    const int max_iterations,
                                    const int maximum_polish_updates,
                                    const double delta_limit,
                                    const double exact_audit_tolerance) {
  validate_solver_controls(target, tolerance, max_iterations);
  validate_exact_audit_tolerance(exact_audit_tolerance);
  if (!std::isfinite(compressed_tolerance) || compressed_tolerance <= 0.0) {
    Rcpp::stop("compressed_tolerance must be positive and finite");
  }
  if (maximum_polish_updates < 0) {
    Rcpp::stop("maximum_polish_updates cannot be negative");
  }
  if (!std::isfinite(delta_limit) || delta_limit <= 0.0) {
    Rcpp::stop("delta_limit must be positive and finite");
  }
}

}  // namespace

namespace sceptre {

SEXP prepare_rpt_spa_response(
    const Rcpp::NumericVector& a,
    const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z,
    const Rcpp::LogicalVector& zero_mask,
    const double minimum_zero_fraction,
    const int maximum_moment_degree) {
  if (!std::isfinite(minimum_zero_fraction) ||
      minimum_zero_fraction < 0.0 || minimum_zero_fraction > 1.0) {
    Rcpp::stop("minimum_zero_fraction must be between zero and one");
  }
  if (zero_mask.size() != a.size()) {
    Rcpp::stop("zero_mask must have one value per cell");
  }
  validate_moment_dimensions(Z.ncol(), maximum_moment_degree);
  std::unique_ptr<PreparedMomentContext> context(new PreparedMomentContext());
  context->cache = build_cache(a, w, Z, 1, 1);
  context->maximum_moment_degree = maximum_moment_degree;
  context->positive_feature_sums = context->cache.positive_feature_sums;
  if (!context->positive_feature_sums.allFinite()) {
    Rcpp::stop("response feature sums must be finite");
  }
  for (int i = 0; i < context->cache.n; ++i) {
    if ((i & 8191) == 0) Rcpp::checkUserInterrupt();
    if (zero_mask[i] == NA_LOGICAL) {
      Rcpp::stop("zero_mask cannot contain missing values");
    }
    if (!zero_mask[i]) continue;
    const double identity_tolerance =
        128.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::max(std::abs(a[i]), std::abs(w[i])));
    if (std::abs(a[i] + w[i]) > identity_tolerance) {
      Rcpp::stop("zero_mask is inconsistent with the NB zero-row identity");
    }
    ++context->n_zero;
  }
  context->zero_fraction = static_cast<double>(context->n_zero) / context->cache.n;
  context->eligible = context->zero_fraction >= minimum_zero_fraction;
  if (context->eligible) {
    context->synopsis = build_moment_synopsis(
        context->cache, zero_mask, maximum_moment_degree);
    if (!context->synopsis.valid) {
      Rcpp::stop("failed to construct a valid moment synopsis");
    }
  }
  Rcpp::IntegerVector metadata = Rcpp::IntegerVector::create(
      Rcpp::Named("n_cells") = context->cache.n,
      Rcpp::Named("feature_dimension") = context->cache.p,
      Rcpp::Named("maximum_moment_degree") = maximum_moment_degree);
  Rcpp::XPtr<PreparedMomentContext> pointer(
      context.get(), true, prepared_context_tag(), metadata);
  context.release();
  pointer.attr("class") = "sceptre_rpt_spa_response_v1";
  pointer.attr("n_cells") = metadata[0];
  pointer.attr("feature_dimension") = metadata[1];
  pointer.attr("maximum_moment_degree") = metadata[2];
  return pointer;
}

void release_rpt_spa_response(SEXP prepared_context) {
  PreparedMomentContext* context = checked_prepared_context(prepared_context, true);
  R_ClearExternalPtr(prepared_context);
  delete context;
}

Rcpp::List rpt_spa_moment_prepared(
    SEXP prepared_context, const int m, const double target, const int score_sign,
    const double tolerance, const double compressed_tolerance,
    const int max_iterations, const int maximum_polish_updates,
    const double delta_limit, const double exact_audit_tolerance) {
  validate_moment_solver_controls(target, tolerance, compressed_tolerance,
      max_iterations, maximum_polish_updates, delta_limit, exact_audit_tolerance);
  PreparedMomentContext* context = checked_prepared_context(prepared_context);
  configure_prepared_pair(*context, m, score_sign);
  MomentDiagnostics diagnostics;
  diagnostics.eligible = context->eligible;
  diagnostics.synopsis_built = context->eligible;
  diagnostics.n_zero = context->n_zero;
  diagnostics.n_exception = context->cache.n - context->n_zero;
  diagnostics.zero_fraction = context->zero_fraction;
  diagnostics.packed_moment_count =
      context->eligible ? context->synopsis.packed_moment_count : 0;
  const MomentSynopsis* synopsis = context->eligible ? &context->synopsis : nullptr;
  return run_rpt_moment(
      context->cache, Rcpp::LogicalVector(), target, tolerance,
      compressed_tolerance, max_iterations, maximum_polish_updates,
      delta_limit, context->zero_fraction, context->maximum_moment_degree,
      exact_audit_tolerance, diagnostics, synopsis, true);
}

Rcpp::List rpt_spa_moment_outward_prepared(
    SEXP prepared_context_pointer,
    const Rcpp::IntegerVector& treated_indices,
    const double tolerance,
    const double compressed_tolerance,
    const int max_iterations,
    const int maximum_polish_updates,
    const double delta_limit,
    const double exact_audit_tolerance) {
  validate_moment_solver_controls(0.0, tolerance, compressed_tolerance,
      max_iterations, maximum_polish_updates, delta_limit, exact_audit_tolerance);
  PreparedMomentContext* context = checked_prepared_context(prepared_context_pointer);
  if (treated_indices.size() < 1 || treated_indices.size() >= context->cache.n) {
    Rcpp::stop("treated_indices must contain between 1 and n - 1 indices");
  }
  const int m = treated_indices.size();
  configure_prepared_pair(*context, m, 1);
  InfoCache& cache = context->cache;

  MomentDiagnostics diagnostics;
  diagnostics.eligible = context->eligible;
  diagnostics.synopsis_built = context->eligible;
  diagnostics.n_zero = context->n_zero;
  diagnostics.n_exception = cache.n - context->n_zero;
  diagnostics.zero_fraction = context->zero_fraction;
  diagnostics.feature_dimension = cache.p;
  diagnostics.maximum_moment_degree = context->maximum_moment_degree;
  diagnostics.exact_audit_tolerance = exact_audit_tolerance;
  diagnostics.effective_exact_audit_tolerance =
      std::min(tolerance, exact_audit_tolerance);
  diagnostics.packed_moment_count =
      context->eligible ? context->synopsis.packed_moment_count : 0;

  std::vector<int> treated(m);
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
  const double observed_target =
      std::isfinite(observed_U) && std::isfinite(observed_information) &&
              std::isfinite(information_scale) && information_scale > 0.0 &&
              observed_information > 1.0e-14 * information_scale
          ? observed_U / std::sqrt(observed_information)
          : kNaN;

  auto geometry_failure = [&](const std::string& reason,
                              const double target,
                              const double base_center) {
    diagnostics.path = "moment_geometry_empirical_fallback";
    diagnostics.fallback_reason = reason;

    const VectorXd zero_state = VectorXd::Zero(cache.d + 1);
    const SolverCounters counters;
    Rcpp::List result = failure_result(
        cache, reason, target, base_center, 0, kNaN,
        std::vector<double>(), zero_state, 0);
    result["base_center"] = std::isfinite(base_center)
                                ? Rcpp::wrap(base_center)
                                : Rcpp::wrap(NA_REAL);
    result["observed_target"] = std::isfinite(observed_target)
                                    ? Rcpp::wrap(observed_target)
                                    : Rcpp::wrap(NA_REAL);
    result["outward_score_sign"] = NA_INTEGER;
    result = add_moment_diagnostics(result, counters, diagnostics);

    return result;
  };

  if (!cache.information_invertible) {
    return geometry_failure("singular_information_geometry", kNaN, kNaN);
  }
  if (!std::isfinite(observed_target)) {
    return geometry_failure("invalid_observed_studentizer", kNaN, kNaN);
  }

  VectorXd base_moment =
      cache.base_probability * context->positive_feature_sums;
  const Boundary base_boundary = boundary_terms(base_moment, cache.C_inv);
  if (!base_boundary.valid || !std::isfinite(base_boundary.b)) {
    return geometry_failure("invalid_center", observed_target, kNaN);
  }
  const double base_center = base_boundary.b;
  const double displacement = observed_target - base_center;
  if (displacement == 0.0) {
    return geometry_failure(
        "central_target_requires_empirical_fallback", observed_target,
        base_center);
  }
  const int score_sign = displacement > 0.0 ? 1 : -1;
  const double outward_target =
      static_cast<double>(score_sign) * observed_target;
  configure_prepared_pair(*context, m, score_sign);

  const MomentSynopsis* synopsis =
      context->eligible ? &context->synopsis : nullptr;
  Rcpp::List result = run_rpt_moment(
      cache, Rcpp::LogicalVector(), outward_target, tolerance,
      compressed_tolerance, max_iterations, maximum_polish_updates,
      delta_limit, context->zero_fraction, context->maximum_moment_degree,
      exact_audit_tolerance, diagnostics, synopsis, true);
  result["base_center"] = base_center;
  result["observed_target"] = observed_target;
  result["outward_score_sign"] = score_sign;

  return result;
}

}  // namespace sceptre

// [[Rcpp::export]]
SEXP prepare_rpt_spa_response_cpp(
    const Rcpp::NumericVector& a, const Rcpp::NumericVector& w,
    const Rcpp::NumericMatrix& Z, const Rcpp::LogicalVector& zero_mask,
    const double minimum_zero_fraction = 0.8,
    const int maximum_moment_degree = 2) {
  return sceptre::prepare_rpt_spa_response(
      a, w, Z, zero_mask, minimum_zero_fraction, maximum_moment_degree);
}

// [[Rcpp::export]]
void release_rpt_spa_response_cpp(SEXP prepared_context) {
  sceptre::release_rpt_spa_response(prepared_context);
}

// Internal diagnostic surface for checking the cached zero-tilt identities.
// [[Rcpp::export]]
Rcpp::List rpt_spa_moment_null_evaluation_cpp(
    SEXP prepared_context, const int m, const int score_sign = 1,
    const double target = 0.0) {
  if (!std::isfinite(target)) Rcpp::stop("target must be finite");
  PreparedMomentContext* context = checked_prepared_context(prepared_context);
  configure_prepared_pair(*context, m, score_sign);
  SolverCounters counters;
  const Evaluation out = evaluate_rpt_null(context->cache, target, counters);
  return Rcpp::List::create(
      Rcpp::Named("valid") = out.valid,
      Rcpp::Named("moment") = Rcpp::wrap(out.moment),
      Rcpp::Named("S0") = out.S0,
      Rcpp::Named("Sg") = Rcpp::wrap(out.Sg),
      Rcpp::Named("Sgg") = Rcpp::wrap(out.Sgg),
      Rcpp::Named("conditional_information") = Rcpp::wrap(out.conditional_information),
      Rcpp::Named("gamma") = out.gamma,
      Rcpp::Named("count_tilt") = out.gamma,
      Rcpp::Named("tilted_count") = out.tilted_count,
      Rcpp::Named("count_residual") = out.tilted_count - m,
      Rcpp::Named("count_variance_ratio") = out.count_variance_ratio,
      Rcpp::Named("count_berry_esseen_ratio") = out.count_berry_esseen_ratio,
      Rcpp::Named("center") = out.boundary.b,
      Rcpp::Named("boundary_grad") = Rcpp::wrap(out.boundary.grad),
      Rcpp::Named("boundary_hess") = Rcpp::wrap(out.boundary.hess),
      Rcpp::Named("residual") = Rcpp::wrap(out.residual),
      Rcpp::Named("jacobian") = Rcpp::wrap(out.jacobian),
      Rcpp::Named("analytic_center_evaluations") = counters.analytic_center_evaluations,
      Rcpp::Named("count_passes") = counters.count_passes,
      Rcpp::Named("gamma_iterations_total") = counters.gamma_iterations);
}

// [[Rcpp::export]]
Rcpp::List rpt_spa_moment_prepared_cpp(
    SEXP prepared_context, const int m, const double target,
    const int score_sign = 1, const double tolerance = 1e-4,
    const double compressed_tolerance = 1e-4, const int max_iterations = 50,
    const int maximum_polish_updates = 3, const double delta_limit = 0.06,
    const double exact_audit_tolerance = 1e-4) {
  return sceptre::rpt_spa_moment_prepared(
      prepared_context, m, target, score_sign, tolerance, compressed_tolerance,
      max_iterations, maximum_polish_updates, delta_limit, exact_audit_tolerance);
}

// [[Rcpp::export]]
Rcpp::List rpt_spa_moment_outward_prepared_cpp(
    SEXP prepared_context, const Rcpp::IntegerVector& treated_indices,
    const double tolerance = 1e-4, const double compressed_tolerance = 1e-4,
    const int max_iterations = 50, const int maximum_polish_updates = 3,
    const double delta_limit = 0.06, const double exact_audit_tolerance = 1e-4) {
  return sceptre::rpt_spa_moment_outward_prepared(
      prepared_context, treated_indices, tolerance, compressed_tolerance,
      max_iterations, maximum_polish_updates, delta_limit, exact_audit_tolerance);
}

// Internal count-root diagnostic; the predictor includes its baseline offset.
// [[Rcpp::export]]
Rcpp::List rpt_spa_count_profile_cpp(
    const Rcpp::NumericVector& linear_predictor,
    const int m,
    const double initial = NA_REAL) {
  const R_xlen_t n = linear_predictor.size();
  if (n < 2 || n > std::numeric_limits<int>::max()) {
    Rcpp::stop("linear_predictor must contain between 2 and INT_MAX cells");
  }
  if (m <= 0 || m >= n) Rcpp::stop("m must lie strictly between zero and the number of cells");
  if (!std::isfinite(initial) && !R_IsNA(initial)) {
    Rcpp::stop("initial must be finite or NA");
  }
  const Eigen::Map<const VectorXd> predictor(linear_predictor.begin(), n);
  if (!predictor.allFinite()) Rcpp::stop("linear_predictor must be finite");
  SolverCounters counters;
  const CountProfile result = profile_count_tilt(predictor, m, 0.0, initial, counters);
  return Rcpp::List::create(
      Rcpp::Named("valid") = result.valid,
      Rcpp::Named("gamma") = result.gamma,
      Rcpp::Named("count") = result.count,
      Rcpp::Named("S0") = result.S0,
      Rcpp::Named("iterations") = result.iterations,
      Rcpp::Named("count_passes") = counters.count_passes,
      Rcpp::Named("count_residual") = result.count - m,
      Rcpp::Named("count_tolerance") = 64.0 * std::numeric_limits<double>::epsilon() * n,
      Rcpp::Named("stagnations") = counters.count_stagnations,
      Rcpp::Named("gamma_warm_starts") = counters.gamma_warm_starts,
      Rcpp::Named("gamma_cold_retries") = counters.gamma_cold_retries);
}
