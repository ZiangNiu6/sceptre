.rpt_null_reference <- function(a, w, Z, m, score_sign, target) {
  n <- length(a)
  u <- m / n
  h <- u * (1 - u)
  G <- unname(cbind(score_sign * a, w * Z))
  moment <- u * colSums(G)
  S0 <- n * h
  Sg <- h * colSums(G)
  Sgg <- h * crossprod(G)
  conditional_information <- Sgg - tcrossprod(Sg) / S0
  C_inv <- solve(crossprod(Z, w * Z))
  d <- ncol(G)
  U <- moment[1L]
  B <- moment[-1L]
  beta <- drop(C_inv %*% B)
  V <- B[1L] - sum(B * beta)
  e <- c(1, rep(0, d - 1L))
  grad_V <- c(0, -2 * beta)
  grad_V[2L] <- grad_V[2L] + 1
  hess_V <- matrix(0, d, d)
  hess_V[-1L, -1L] <- -2 * C_inv
  grad <- e / sqrt(V) - 0.5 * U * V^(-1.5) * grad_V
  hess <- -0.5 * V^(-1.5) * (tcrossprod(e, grad_V) + tcrossprod(grad_V, e)) +
    U * (0.75 * V^(-2.5) * tcrossprod(grad_V) - 0.5 * V^(-1.5) * hess_V)
  jacobian <- matrix(0, d + 1L, d + 1L)
  jacobian[seq_len(d), seq_len(d)] <- diag(d)
  jacobian[seq_len(d), d + 1L] <- -grad
  jacobian[d + 1L, seq_len(d)] <- drop(conditional_information %*% grad)
  list(moment = moment, S0 = S0, Sg = Sg, Sgg = Sgg,
       conditional_information = conditional_information,
       center = U / sqrt(V), boundary_grad = grad, boundary_hess = hess,
       residual = c(rep(0, d), U / sqrt(V) - target), jacobian = jacobian,
       count_berry_esseen_ratio = ((1 - u)^2 + u^2) / sqrt(S0))
}

test_that("analytic RPT null evaluation matches independent profiled CGF formulas", {
  n <- 257L
  z <- seq(-1, 1, length.out = n)
  Z <- cbind(1, z, sin(2 * z))
  w <- 0.3 + exp(z) / 5
  a <- cos(3 * z) + z / 2
  context <- prepare_rpt_spa_response_cpp(a, w, Z, rep(FALSE, n))
  on.exit(release_rpt_spa_response_cpp(context))
  first <- NULL
  for (case in list(c(37L, 1L), c(1L, -1L), c(256L, 1L),
                    c(37L, -1L), c(37L, 1L))) {
    m <- case[1L]
    sign <- case[2L]
    target <- 0.4
    actual <- rpt_spa_moment_null_evaluation_cpp(context, m, sign, target)
    reference <- .rpt_null_reference(a, w, Z, m, sign, target)
    expect_true(actual$valid)
    for (field in c("moment", "Sg", "boundary_grad", "residual")) {
      expect_equal(as.numeric(actual[[field]]), as.numeric(reference[[field]]),
                   tolerance = 2e-11, info = field)
    }
    for (field in c("S0", "Sgg", "conditional_information", "center",
                    "boundary_hess", "jacobian", "count_berry_esseen_ratio")) {
      expect_equal(unname(actual[[field]]), unname(reference[[field]]),
                   tolerance = 2e-11, info = field)
    }
    expect_equal(actual$gamma, 0)
    expect_equal(actual$count_tilt, 0)
    expect_equal(actual$tilted_count, m)
    expect_equal(actual$count_residual, 0)
    expect_equal(actual$count_variance_ratio, 1)
    expect_equal(actual$analytic_center_evaluations, 1L)
    expect_equal(actual$count_passes, 0L)
    expect_equal(actual$gamma_iterations_total, 0L)
    if (m == 37L && sign == 1L) {
      if (is.null(first)) first <- actual else expect_equal(actual, first)
    }
  }
  expect_error(rpt_spa_moment_null_evaluation_cpp(context, 0L), "m must be")
  expect_error(rpt_spa_moment_null_evaluation_cpp(context, n), "m must be")
  expect_error(rpt_spa_moment_null_evaluation_cpp(context, 37L, 0L), "score_sign")
  expect_error(rpt_spa_moment_null_evaluation_cpp(context, 37L, target = Inf), "target")
})

test_that("analytic conditional information preserves small centered variation", {
  # The weights differ by only a few ulps: subtracting raw second moments or
  # dropping the compensated running mean loses their centered crossproduct.
  a <- c(-1, 0, 1)
  w <- c(1e16, 1e16 + 2, 1e16 + 4)
  context <- prepare_rpt_spa_response_cpp(a, w, matrix(1, 3L, 1L), rep(FALSE, 3L))
  on.exit(release_rpt_spa_response_cpp(context))
  actual <- rpt_spa_moment_null_evaluation_cpp(context, 1L)
  centered_features <- cbind(a, c(-2, 0, 2))
  expected <- (1 / 3) * (2 / 3) * crossprod(centered_features)
  expect_true(actual$valid)
  expect_equal(unname(actual$conditional_information), unname(expected), tolerance = 1e-12)
  reflected <- rpt_spa_moment_null_evaluation_cpp(context, 1L, score_sign = -1L)
  expect_equal(unname(reflected$conditional_information),
               unname(expected * tcrossprod(c(-1, 1))), tolerance = 1e-12)
})

.rpt_pairwise_sum <- function(values) {
  if (!length(values)) return(0)
  # Avoid sequential double accumulation on platforms without wider long
  # doubles, independently of the production kernel's compensated summation.
  while (length(values) > 1L) {
    n <- length(values)
    pair_end <- 2L * seq_len(n %/% 2L)
    reduced <- values[pair_end - 1L] + values[pair_end]
    if (n %% 2L) reduced <- c(reduced, values[n])
    values <- reduced
  }
  values[[1L]]
}

.rpt_count_reference <- function(linear_predictor, m) {
  baseline <- qlogis(m / length(linear_predictor))
  uniroot(function(gamma) .rpt_pairwise_sum(plogis(linear_predictor + gamma)) - m,
          interval = c(baseline - max(linear_predictor) - 1,
                       baseline - min(linear_predictor) + 1),
          tol = 1e-12)$root
}

.rpt_expect_count_precision <- function(result, linear_predictor, m) {
  gate <- 64 * .Machine$double.eps * length(linear_predictor)
  expect_true(result$valid)
  probabilities <- plogis(linear_predictor + result$gamma)
  count <- .rpt_pairwise_sum(probabilities)
  variance <- .rpt_pairwise_sum(probabilities * (1 - probabilities))
  expect_true(is.finite(result$gamma))
  expect_gt(result$S0, 0)
  expect_lte(abs(result$count - m), 4 * gate)
  expect_lte(abs(count - m), 4 * gate)
  expect_lte(abs(result$count - count), gate)
  expect_lte(abs(result$S0 - variance), 2 * gate)
}

test_that("large constant-count reproducers finish in a single count pass", {
  n <- 205797L
  for (m in c(1L, 864L, 1500L, n - 1L)) {
    predictor <- rep(qlogis(m / n), n)
    result <- rpt_spa_count_profile_cpp(predictor, m)
    .rpt_expect_count_precision(result, predictor, m)
    expect_equal(result$gamma, 0)
    expect_equal(result$count_passes, 1L)
    expect_equal(result$iterations, 1L)
    expect_equal(result$count_tolerance, 64 * .Machine$double.eps * n)
  }
})

test_that("constant predictors have the closed-form count tilt across count extremes", {
  n <- 257L
  for (m in c(1L, 37L, n - 1L)) {
    for (shift in c(-30, 0, 30)) {
      predictor <- rep(shift, n)
      result <- rpt_spa_count_profile_cpp(predictor, m)
      .rpt_expect_count_precision(result, predictor, m)
      expect_equal(result$gamma, qlogis(m / n) - shift, tolerance = 1e-11)
    }
  }
  # Large offsets can lose low bits when gamma cancels the predictor. The
  # larger cell-count gate still resolves the requested non-extreme counts.
  n <- 205797L
  for (shift in c(-700, 700)) {
    predictor <- rep(shift, n)
    result <- rpt_spa_count_profile_cpp(predictor, 864L)
    .rpt_expect_count_precision(result, predictor, 864L)
    expect_equal(result$gamma, qlogis(864 / n) - shift, tolerance = 1e-11)
  }
})

test_that("nonzero heterogeneous tilts agree with an independent count root", {
  n <- 513L
  predictor <- seq(-2.1, 1.7, length.out = n) +
    0.25 * sin(seq_len(n) / 17)
  for (m in c(1L, 61L, 256L, n - 1L)) {
    for (sign in c(-1, 1)) {
      current <- sign * predictor
      result <- rpt_spa_count_profile_cpp(current, m)
      .rpt_expect_count_precision(result, current, m)
      expect_equal(result$gamma, .rpt_count_reference(current, m), tolerance = 1e-10)
      warm <- rpt_spa_count_profile_cpp(current, m, initial = result$gamma)
      .rpt_expect_count_precision(warm, current, m)
      expect_equal(warm$gamma, result$gamma, tolerance = 1e-12)
      expect_equal(warm$gamma_warm_starts, 1L)
      expect_equal(warm$count_passes, 1L)
      shifted <- rpt_spa_count_profile_cpp(current + 12, m,
                                           initial = result$gamma - 12)
      .rpt_expect_count_precision(shifted, current + 12, m)
      expect_equal(shifted$gamma, result$gamma - 12, tolerance = 1e-12)
      expect_equal(shifted$S0, result$S0, tolerance = 1e-12)
    }
  }
})

test_that("large nonconstant count profiles retain the existing acceptance gate", {
  n <- 205797L
  for (m in c(864L, 1500L)) {
    predictor <- qlogis(m / n) + rep(c(-1.2, -0.1, 0.2, 0.8), length.out = n)
    result <- rpt_spa_count_profile_cpp(predictor, m)
    .rpt_expect_count_precision(result, predictor, m)
    expect_equal(result$gamma, .rpt_count_reference(predictor, m), tolerance = 1e-10)
    expect_equal(result$count_tolerance, 64 * .Machine$double.eps * n)
  }
})

test_that("count reduction retains partial blocks and count extremes", {
  for (n in c(31L, 32L, 33L, 63L, 64L, 65L)) {
    predictor <- rep(c(-5, -0.7, 0.2, 3), length.out = n) + seq_len(n) / (10 * n)
    for (m in c(1L, n %/% 3L, n - 1L)) {
      result <- rpt_spa_count_profile_cpp(predictor, m)
      .rpt_expect_count_precision(result, predictor, m)
      expect_equal(result$gamma, .rpt_count_reference(predictor, m), tolerance = 1e-10)
      reverse <- rpt_spa_count_profile_cpp(rev(predictor), m, initial = result$gamma)
      .rpt_expect_count_precision(reverse, rev(predictor), m)
      expect_equal(reverse$gamma, result$gamma, tolerance = 1e-12)
      expect_equal(reverse$S0, result$S0, tolerance = 1e-12)
      expect_equal(result$count_tolerance, 64 * .Machine$double.eps * n)
    }
  }
})

test_that("an unusable warm start can be retried cold without false convergence", {
  predictor <- seq(-1, 1, length.out = 33L)
  warm <- rpt_spa_count_profile_cpp(predictor, 9L, initial = 1e300)
  expect_false(warm$valid)
  expect_equal(warm$gamma_warm_starts, 1L)
  expect_equal(warm$gamma_cold_retries, 0L)
  cold <- rpt_spa_count_profile_cpp(predictor, 9L)
  .rpt_expect_count_precision(cold, predictor, 9L)
  expect_equal(cold$gamma_warm_starts, 0L)
  expect_equal(cold$gamma, .rpt_count_reference(predictor, 9L), tolerance = 1e-10)
})

test_that("representable stagnation never accepts an inaccurate count", {
  # Adjacent representable gammas cannot produce probability 1/5 here.
  predictor <- rep(1e20, 5L)
  result <- rpt_spa_count_profile_cpp(predictor, 1L, initial = -1e20)
  expect_false(result$valid)
  expect_gte(result$stagnations, 1L)
  expect_equal(result$count_tolerance, 64 * .Machine$double.eps * 5)
  saturated <- rpt_spa_count_profile_cpp(c(-1000, 1000), 1L)
  expect_false(saturated$valid)
})

test_that("count diagnostics reject malformed dimensions and nonfinite controls", {
  expect_error(rpt_spa_count_profile_cpp(0, 1L), "(count|cells|length|dimension|m)")
  for (m in c(0L, 3L)) {
    expect_error(rpt_spa_count_profile_cpp(c(-1, 0, 1), m), "m")
  }
  for (value in c(NA_real_, NaN, Inf, -Inf)) {
    expect_error(rpt_spa_count_profile_cpp(c(-1, value, 1), 1L), "finite")
  }
  for (initial in c(Inf, -Inf)) {
    expect_error(rpt_spa_count_profile_cpp(c(-1, 0, 1), 1L, initial), "initial")
  }
})
