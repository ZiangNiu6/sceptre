// Structured GLM normal equations inspired by broadinstitute/pysceptre,
// commit 7adaa9f5fda12e0cf87df972c8fe73b73ef13e7d, glm/design.py and glm/irls.py.
// GPL-3.0-only. This native implementation retains safeguarded, unpenalized
// likelihood optimization and does not require a Python runtime.
//
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {
using Eigen::MatrixXd;
using Eigen::VectorXd;
constexpr double kMissing = std::numeric_limits<double>::quiet_NaN();

double max_abs(const VectorXd& x) {
  return x.size() ? x.cwiseAbs().maxCoeff() : 0.0;
}

struct StructuredDesign {
  Rcpp::NumericMatrix raw;
  Rcpp::NumericMatrix dense;
  Rcpp::IntegerVector dense_columns;
  Rcpp::IntegerMatrix codes;
  Rcpp::List groups;
  Rcpp::NumericVector centers;
  Rcpp::NumericVector scales;
  int n;
  int p;
  int d;
  int ng;
  int rank;
  std::vector<std::vector<int> > group_columns;

  explicit StructuredDesign(const Rcpp::List& prepared)
      : raw(prepared["raw_matrix"]), dense(prepared["dense"]),
        dense_columns(prepared["dense_columns"]), codes(prepared["codes"]),
        groups(prepared["groups"]), centers(prepared["centers"]),
        scales(prepared["scales"]), n(raw.nrow()), p(raw.ncol()),
        d(dense.ncol()), ng(groups.size()), rank(Rcpp::as<int>(prepared["rank"])) {
    if (n < 1 || p < 1 || dense.nrow() != n || dense_columns.size() != d ||
        codes.nrow() != n || codes.ncol() != ng || centers.size() != p ||
        scales.size() != p) {
      Rcpp::stop("malformed prepared design: incompatible component dimensions");
    }
    if (Rcpp::as<int>(prepared["n"]) != n ||
        Rcpp::as<int>(prepared["p"]) != p || rank < 0 || rank > std::min(n, p)) {
      Rcpp::stop("malformed prepared design: invalid dimension or rank metadata");
    }
    const Rcpp::LogicalVector valid = prepared["valid"];
    if (valid.size() != 1 || valid[0] == NA_LOGICAL) {
      Rcpp::stop("malformed prepared design: invalid validity flag");
    }
    for (int j = 0; j < p; ++j) {
      if (!std::isfinite(centers[j]) || !std::isfinite(scales[j]) || scales[j] <= 0.0) {
        Rcpp::stop("malformed prepared design: centers and positive scales must be finite");
      }
    }
    if (centers[0] != 0.0 || scales[0] != 1.0) {
      Rcpp::stop("malformed prepared design: intercept must remain untransformed");
    }
    std::vector<unsigned char> present(p, 0);
    for (int k = 0; k < d; ++k) {
      const int column = dense_columns[k];
      if (column < 0 || column >= p || present[column]) {
        Rcpp::stop("malformed prepared design: dense columns must be distinct valid indices");
      }
      present[column] = 1;
    }
    if (!present[0]) {
      Rcpp::stop("malformed prepared design: intercept must be in the dense block");
    }
    if (valid[0]) {
      for (int i = 0; i < n; ++i) {
        if (raw(i, 0) != 1.0) {
          Rcpp::stop("malformed prepared design: valid design needs a leading intercept");
        }
      }
    }
    group_columns.reserve(ng);
    for (int g = 0; g < ng; ++g) {
      const std::vector<int> columns = Rcpp::as<std::vector<int> >(groups[g]);
      if (columns.empty()) Rcpp::stop("malformed prepared design: empty indicator group");
      for (int column : columns) {
        if (column < 0 || column >= p || present[column]) {
          Rcpp::stop("malformed prepared design: indicator columns must form a distinct partition");
        }
        if (centers[column] != 0.0 || scales[column] != 1.0) {
          Rcpp::stop("malformed prepared design: indicator columns must remain untransformed");
        }
        present[column] = 1;
      }
      for (int i = 0; i < n; ++i) {
        const int code = codes(i, g);
        if (code < 0 || code > static_cast<int>(columns.size())) {
          Rcpp::stop("malformed prepared design: indicator codes are outside their group");
        }
      }
      group_columns.push_back(columns);
    }
    if (std::find(present.begin(), present.end(), 0) != present.end()) {
      Rcpp::stop("malformed prepared design: columns must partition the full design");
    }
  }

  VectorXd linear_predictor(const VectorXd& beta) const {
    const Eigen::Map<const MatrixXd> xd(dense.begin(), n, d);
    VectorXd bd(d);
    for (int j = 0; j < d; ++j) bd[j] = beta[dense_columns[j]];
    VectorXd eta = xd * bd;
    for (int g = 0; g < ng; ++g) {
      const std::vector<int>& columns = group_columns[g];
      for (int i = 0; i < n; ++i) {
        const int level = codes(i, g);
        if (level < columns.size()) eta[i] += beta[columns[level]];
      }
    }
    return eta;
  }

  // X' diag(w) X and X' r, with one dense block and exact grouped sums.
  // No n-by-p-squared outer-product array is constructed or retained.
  void equations(const VectorXd& weights, const VectorXd& residual,
                 MatrixXd& information, VectorXd& score) const {
    information.setZero(p, p);
    score.setZero(p);
    const Eigen::Map<const MatrixXd> xd(dense.begin(), n, d);
    const MatrixXd wx = xd.array().colwise() * weights.array();
    const MatrixXd dd = xd.transpose() * wx;
    const VectorXd dr = xd.transpose() * residual;
    for (int j = 0; j < d; ++j) {
      const int cj = dense_columns[j];
      score[cj] = dr[j];
      for (int k = 0; k < d; ++k) information(cj, dense_columns[k]) = dd(j, k);
    }
    for (int g = 0; g < ng; ++g) {
      const std::vector<int>& columns = group_columns[g];
      for (int i = 0; i < n; ++i) {
        const int level = codes(i, g);
        if (level >= columns.size()) continue;
        const int column = columns[level];
        information(column, column) += weights[i];
        score[column] += residual[i];
        for (int j = 0; j < d; ++j) {
          information(column, dense_columns[j]) += wx(i, j);
        }
        for (int h = 0; h < g; ++h) {
          const std::vector<int>& previous = group_columns[h];
          const int other = codes(i, h);
          if (other < previous.size()) information(column, previous[other]) += weights[i];
        }
      }
      for (int level = 0; level < columns.size(); ++level) {
        const int column = columns[level];
        for (int j = 0; j < d; ++j) {
          information(dense_columns[j], column) = information(column, dense_columns[j]);
        }
        for (int h = 0; h < g; ++h) {
          const std::vector<int>& previous = group_columns[h];
          for (int other = 0; other < previous.size(); ++other) {
            information(previous[other], column) = information(column, previous[other]);
          }
        }
      }
    }
  }

  VectorXd score_only(const VectorXd& residual) const {
    VectorXd score = VectorXd::Zero(p);
    const Eigen::Map<const MatrixXd> xd(dense.begin(), n, d);
    const VectorXd dr = xd.transpose() * residual;
    for (int j = 0; j < d; ++j) score[dense_columns[j]] = dr[j];
    for (int g = 0; g < ng; ++g) {
      const std::vector<int>& columns = group_columns[g];
      for (int i = 0; i < n; ++i) {
        const int level = codes(i, g);
        if (level < columns.size()) score[columns[level]] += residual[i];
      }
    }
    return score;
  }

  VectorXd original_coefficients(const VectorXd& beta) const {
    VectorXd original = beta;
    for (int j = 1; j < p; ++j) {
      original[j] = beta[j] / scales[j];
      original[0] -= centers[j] * original[j];
    }
    return original;
  }

  VectorXd original_score(const VectorXd& score) const {
    VectorXd original = score;
    for (int j = 1; j < p; ++j) original[j] = scales[j] * score[j] + centers[j] * score[0];
    return original;
  }
};

struct Evaluation {
  bool valid = false;
  double loglik = kMissing;
  double deviance = kMissing;
  double max_eta = kMissing;
  VectorXd linear_predictors;
  VectorXd fitted;
  VectorXd gradient;
  MatrixXd information;
};

Evaluation evaluate_glm(const StructuredDesign& design, const VectorXd& beta,
                        const VectorXd& y, const bool binomial) {
  Evaluation out;
  if (!beta.allFinite()) return out;
  const VectorXd eta = design.linear_predictor(beta);
  if (!eta.allFinite()) return out;
  out.linear_predictors = eta;
  out.fitted = VectorXd::Constant(design.n, kMissing);
  VectorXd weights(design.n);
  double loglik = 0.0;
  double deviance = 0.0;
  for (int i = 0; i < design.n; ++i) {
    if (binomial) {
      const double exponential = std::exp(-std::abs(eta[i]));
      const double mu = eta[i] >= 0.0 ? 1.0 / (1.0 + exponential)
                                      : exponential / (1.0 + exponential);
      if (!(mu > 0.0 && mu < 1.0)) return out;
      const double log_partition = std::log1p(exponential);
      const double log_mu = eta[i] >= 0.0 ? -log_partition : eta[i] - log_partition;
      const double log_complement = eta[i] >= 0.0 ? -eta[i] - log_partition : -log_partition;
      const double ll = y[i] * log_mu + (1.0 - y[i]) * log_complement;
      double saturated = 0.0;
      if (y[i] > 0.0) saturated += y[i] * std::log(y[i]);
      if (y[i] < 1.0) saturated += (1.0 - y[i]) * std::log1p(-y[i]);
      out.fitted[i] = mu;
      weights[i] = mu * (1.0 - mu);
      loglik += ll;
      deviance += 2.0 * std::max(0.0, saturated - ll);
    } else {
      const double mu = std::exp(eta[i]);
      if (!std::isfinite(mu) || mu <= 0.0) return out;
      out.fitted[i] = mu;
      weights[i] = mu;
      loglik += y[i] * eta[i] - mu;
      const double term = y[i] == 0.0 ? mu : y[i] * (std::log(y[i]) - eta[i]) - (y[i] - mu);
      deviance += 2.0 * std::max(0.0, term);
    }
  }
  if (!std::isfinite(loglik) || !std::isfinite(deviance)) return out;
  const VectorXd residual = y - out.fitted;
  design.equations(weights, residual, out.information, out.gradient);
  out.loglik = loglik;
  out.deviance = deviance;
  out.max_eta = max_abs(eta);
  out.valid = out.gradient.allFinite() && out.information.allFinite();
  return out;
}

bool completely_separated(const VectorXd& y, const VectorXd& fitted) {
  double minimum_treated = 1.0;
  double maximum_control = 0.0;
  for (Eigen::Index i = 0; i < y.size(); ++i) {
    if (y[i] == 1.0) minimum_treated = std::min(minimum_treated, fitted[i]);
    else if (y[i] == 0.0) maximum_control = std::max(maximum_control, fitted[i]);
    else return false;  // A fractional binomial response is not a 0/1 assignment.
  }
  return minimum_treated > 0.5 && maximum_control < 0.5;
}

bool solve_step(const MatrixXd& information, const VectorXd& gradient,
                VectorXd& step, double& condition_ratio) {
  Eigen::LDLT<MatrixXd> decomposition(information);
  if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) return false;
  const VectorXd diag = decomposition.vectorD().cwiseAbs();
  if (!diag.allFinite() || diag.size() == 0) return false;
  const double largest = diag.maxCoeff();
  const double smallest = diag.minCoeff();
  condition_ratio = largest > 0.0 ? smallest / largest : 0.0;
  if (!(largest > 0.0) || !(smallest > 1e-12 * largest)) return false;
  step = decomposition.solve(gradient);
  return decomposition.info() == Eigen::Success && step.allFinite();
}

Rcpp::List fit_result(const StructuredDesign& design, const VectorXd& beta,
                      const Evaluation& evaluation, bool converged,
                      const std::string& reason, int iterations,
                      int halvings, double condition_ratio,
                      double score_tolerance, double relative_change) {
  const VectorXd score = evaluation.gradient.size() == design.p
      ? design.original_score(evaluation.gradient) : VectorXd::Constant(design.p, kMissing);
  return Rcpp::List::create(
      Rcpp::Named("coefficients") = Rcpp::wrap(design.original_coefficients(beta)),
      Rcpp::Named("fitted.values") = Rcpp::wrap(evaluation.fitted),
      Rcpp::Named("linear.predictors") = Rcpp::wrap(evaluation.linear_predictors),
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("reason") = reason,
      Rcpp::Named("deviance") = evaluation.deviance,
      Rcpp::Named("loglik") = evaluation.loglik,
      Rcpp::Named("score") = Rcpp::wrap(score),
      Rcpp::Named("conditioned_gradient_max") = evaluation.gradient.size() ? max_abs(evaluation.gradient) : kMissing,
      Rcpp::Named("conditioned_score_tolerance") = score_tolerance,
      Rcpp::Named("information_condition_ratio") = condition_ratio,
      Rcpp::Named("relative_deviance_change") = relative_change,
      Rcpp::Named("step_halvings") = halvings,
      Rcpp::Named("rank") = design.rank,
      Rcpp::Named("df.residual") = design.n - design.p,
      Rcpp::Named("n_indicator_groups") = design.ng);
}
}  // namespace

// [[Rcpp::export]]
Rcpp::List prepare_pysceptre_design_cpp(const Rcpp::NumericMatrix& Z) {
  const int n = Z.nrow();
  const int p = Z.ncol();
  if (n < 1 || p < 1) Rcpp::stop("covariate matrix must have at least one row and column");
  bool finite = true;
  bool leading_intercept = true;
  std::vector<int> dense_columns;
  std::vector<std::vector<int> > groups;
  std::vector<std::vector<unsigned char> > occupied;
  for (int i = 0; i < n; ++i) leading_intercept = leading_intercept && Z(i, 0) == 1.0;
  for (int j = 0; j < p; ++j) {
    bool binary = true;
    int ones = 0;
    for (int i = 0; i < n; ++i) {
      const double value = Z(i, j);
      finite = finite && std::isfinite(value);
      binary = binary && (value == 0.0 || value == 1.0);
      ones += value == 1.0;
    }
    if (!binary || ones == 0 || ones == n || j == 0) {
      dense_columns.push_back(j);
      continue;
    }
    int group = -1;
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
      bool overlaps = false;
      for (int i = 0; i < n && !overlaps; ++i) overlaps = occupied[g][i] && Z(i, j) == 1.0;
      if (!overlaps) { group = g; break; }
    }
    if (group < 0) {
      group = groups.size();
      groups.push_back(std::vector<int>());
      occupied.push_back(std::vector<unsigned char>(n, 0));
    }
    groups[group].push_back(j);
    for (int i = 0; i < n; ++i) if (Z(i, j) == 1.0) occupied[group][i] = 1;
  }
  Rcpp::IntegerVector dense_cols = Rcpp::wrap(dense_columns);
  Rcpp::NumericMatrix dense(n, dense_columns.size());
  Rcpp::NumericVector centers(p, 0.0), scales(p, 1.0);
  std::string reason = !finite ? "nonfinite_design" : (!leading_intercept ? "missing_leading_intercept" : "ok");
  for (int k = 0; k < static_cast<int>(dense_columns.size()); ++k) {
    const int j = dense_columns[k];
    if (j == 0 || !finite || !leading_intercept) {
      for (int i = 0; i < n; ++i) dense(i, k) = Z(i, j);
      continue;
    }
    double center = 0.0;
    for (int i = 0; i < n; ++i) center += Z(i, j) / static_cast<double>(n);
    double sum_square = 0.0;
    for (int i = 0; i < n; ++i) sum_square += std::pow(Z(i, j) - center, 2);
    const double scale = std::sqrt(sum_square / static_cast<double>(n));
    if (!(scale > 0.0) || !std::isfinite(scale)) {
      reason = "rank_deficient_design";
      for (int i = 0; i < n; ++i) dense(i, k) = Z(i, j);
      continue;
    }
    centers[j] = center;
    scales[j] = scale;
    for (int i = 0; i < n; ++i) dense(i, k) = (Z(i, j) - center) / scale;
  }
  Rcpp::List group_list(groups.size());
  Rcpp::IntegerMatrix codes(n, groups.size());
  for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
    group_list[g] = Rcpp::wrap(groups[g]);
    for (int i = 0; i < n; ++i) codes(i, g) = groups[g].size();
    for (int k = 0; k < static_cast<int>(groups[g].size()); ++k) {
      for (int i = 0; i < n; ++i) if (Z(i, groups[g][k]) == 1.0) codes(i, g) = k;
    }
  }
  Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("raw_matrix") = Z, Rcpp::Named("dense") = dense,
      Rcpp::Named("dense_columns") = dense_cols, Rcpp::Named("groups") = group_list,
      Rcpp::Named("codes") = codes, Rcpp::Named("centers") = centers,
      Rcpp::Named("scales") = scales, Rcpp::Named("n") = n,
      Rcpp::Named("p") = p, Rcpp::Named("valid") = reason == "ok",
      Rcpp::Named("reason") = reason, Rcpp::Named("rank") = 0);
  if (reason == "ok") {
    const StructuredDesign design(out);
    MatrixXd gram;
    VectorXd ignored;
    design.equations(VectorXd::Ones(n), VectorXd::Zero(n), gram, ignored);
    Eigen::SelfAdjointEigenSolver<MatrixXd> decomposition(gram);
    int rank = 0;
    if (decomposition.info() != Eigen::Success || !decomposition.eigenvalues().allFinite()) {
      reason = "rank_check_failed";
    } else {
      const double largest = decomposition.eigenvalues().maxCoeff();
      rank = (decomposition.eigenvalues().array() > 1e-12 * std::max(1.0, largest)).count();
      if (rank != p) reason = "rank_deficient_design";
    }
    out["rank"] = rank;
    out["reason"] = reason;
    out["valid"] = reason == "ok";
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::List fit_pysceptre_glm_cpp(const Rcpp::List& prepared,
                                const Rcpp::NumericVector& response,
                                const std::string& family = "poisson",
                                const double tolerance = 1e-8,
                                const int max_iterations = 25) {
  if (family != "poisson" && family != "binomial") Rcpp::stop("unsupported family");
  const bool binomial = family == "binomial";
  if (!std::isfinite(tolerance) || tolerance <= 0.0 || max_iterations < 0) {
    Rcpp::stop("invalid solver controls");
  }
  const StructuredDesign design(prepared);
  if (response.size() != design.n) Rcpp::stop("response and covariate matrix have incompatible dimensions");
  const Eigen::Map<const VectorXd> y(response.begin(), response.size());
  if (!y.allFinite() || (y.array() < 0.0).any()) Rcpp::stop("response must be finite and nonnegative");
  if (binomial && (y.array() > 1.0).any()) Rcpp::stop("binomial response must lie between zero and one");
  VectorXd beta = VectorXd::Zero(design.p);
  Evaluation current;
  if (!Rcpp::as<bool>(prepared["valid"])) {
    return fit_result(design, beta, current, false, Rcpp::as<std::string>(prepared["reason"]), 0, 0, kMissing, kMissing, kMissing);
  }
  const double average = y.mean();
  if (!(average > 0.0) || !std::isfinite(average) || (binomial && average >= 1.0)) {
    return fit_result(design, beta, current, false, "degenerate_response", 0, 0, kMissing, kMissing, kMissing);
  }
  beta[0] = binomial ? std::log(average) - std::log1p(-average) : std::log(average);
  current = evaluate_glm(design, beta, y, binomial);
  const VectorXd xty = design.score_only(y);
  const double score_tolerance = tolerance * std::max(1.0, max_abs(xty));
  double condition_ratio = kMissing;
  double relative_change = kMissing;
  int iterations = 0;
  int halvings = 0;
  bool converged = false;
  std::string reason = current.valid ? "maximum_iterations_reached" : "invalid_initial_fit";
  for (int iteration = 1; current.valid && iteration <= max_iterations; ++iteration) {
    VectorXd step;
    if (!solve_step(current.information, current.gradient, step, condition_ratio)) {
      reason = "rank_deficient_information";
      break;
    }
    bool accepted = false;
    VectorXd candidate_beta;
    Evaluation candidate;
    double scale = 1.0;
    for (int halving = 0; halving <= 20; ++halving) {
      candidate_beta = beta + scale * step;
      candidate = evaluate_glm(design, candidate_beta, y, binomial);
      const double allowance = 1e-12 * (1.0 + std::abs(current.loglik));
      if (candidate.valid && candidate.loglik + allowance >= current.loglik) {
        accepted = true;
        break;
      }
      if (halving < 20) { scale *= 0.5; ++halvings; }
    }
    if (!accepted) { reason = "step_halving_failed"; break; }
    relative_change = std::abs(candidate.deviance - current.deviance) / (0.1 + std::abs(candidate.deviance));
    beta.swap(candidate_beta);
    current = std::move(candidate);
    iterations = iteration;
    if (binomial && (current.max_eta > 30.0 || max_abs(beta) > 30.0)) {
      reason = "suspected_separation";
      break;
    }
    converged = relative_change < tolerance && max_abs(current.gradient) <= score_tolerance;
    if (converged) { reason = "ok"; break; }
  }
  if (binomial && current.valid && completely_separated(y, current.fitted)) {
    converged = false;
    reason = "complete_separation";
  }
  return fit_result(design, beta, current, converged, reason, iterations, halvings,
                    condition_ratio, score_tolerance, relative_change);
}
