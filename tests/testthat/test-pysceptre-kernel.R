test_that("structured Poisson fits crossed indicator designs in original units", {
  set.seed(731)
  n <- 2400L
  dat <- data.frame(
    depth = rnorm(n, 1000, 30),
    batch = factor(sample(letters[1:9], n, replace = TRUE)),
    replicate = factor(sample(1:3, n, replace = TRUE)),
    binary = rbinom(n, 1, 0.23)
  )
  Z <- model.matrix(~ depth + batch + replicate + binary, dat)
  beta <- c(-3.5, 0.004, rep(0.08, ncol(Z) - 2L))
  y <- rpois(n, exp(drop(Z %*% beta)))
  prepared <- prepare_pysceptre_glm_design(Z)
  expect_true(prepared$valid)
  expect_gte(length(prepared$groups), 2L)
  expect_lt(ncol(prepared$dense), ncol(Z))
  expect_identical(prepared$raw_matrix, Z)
  fit <- fit_pysceptre_glm(y, Z, prepared_design = prepared)
  reference <- stats::glm.fit(Z, y, family = stats::poisson())
  expect_identical(fit$method_used, "pysceptre")
  expect_true(fit$converged)
  expect_equal(drop(Z %*% fit$coefficients), drop(Z %*% reference$coefficients), tolerance = 1e-7)
  expect_equal(fit$fitted.values, reference$fitted.values, tolerance = 1e-7)
  expect_equal(fit$deviance, reference$deviance, tolerance = 1e-9)
  expect_identical(names(fit$coefficients), colnames(Z))
  expect_identical(fit$df.residual, n - ncol(Z))
})


test_that("structured designs survive serialization and reject another matrix", {
  set.seed(732)
  Z <- cbind(`(Intercept)` = 1, x = rnorm(200), b = rep(c(0, 1), 100))
  prepared <- prepare_pysceptre_glm_design(Z)
  restored <- unserialize(serialize(prepared, NULL))
  y <- rpois(200, exp(0.2 + 0.3 * Z[, "x"]))
  direct <- fit_pysceptre_glm(y, Z, prepared_design = prepared)
  recovered <- fit_pysceptre_glm(y, Z, prepared_design = restored)
  expect_identical(direct$coefficients, recovered$coefficients)
  expect_identical(direct$fitted.values, recovered$fitted.values)
  changed <- Z
  changed[1, "x"] <- changed[1, "x"] + 1
  expect_error(fit_pysceptre_glm(y, changed, prepared_design = prepared), "not made from this covariate")
})


test_that("unsupported numerical cases retain the reference Poisson fallback", {
  set.seed(733)
  Z <- cbind(`(Intercept)` = 1, x = rnorm(150))
  y <- rpois(150, exp(Z %*% c(0.4, 0.2)))
  forced <- fit_pysceptre_glm(y, Z, max_iterations = 0L)
  reference <- stats::glm.fit(Z, y, family = stats::poisson())
  expect_identical(forced$method_used, "glm.fit")
  expect_identical(forced$fallback_reason, "maximum_iterations_reached")
  expect_identical(forced$fitted.values, reference$fitted.values)
  rank_deficient <- cbind(Z, duplicate = Z[, "x"])
  fit <- fit_pysceptre_glm(y, rank_deficient)
  expect_identical(fit$method_used, "glm.fit")
  expect_identical(fit$fallback_reason, "rank_deficient_design")
  expect_true(anyNA(fit$coefficients))
  zero <- fit_pysceptre_glm(numeric(nrow(Z)), Z)
  zero_reference <- stats::glm.fit(Z, numeric(nrow(Z)), family = stats::poisson())
  expect_identical(zero$method_used, "glm.fit")
  expect_identical(zero$fallback_reason, "degenerate_response")
  expect_identical(zero$fitted.values, zero_reference$fitted.values)
})


test_that("the structured engine also fits rare binary assignments", {
  set.seed(734)
  n <- 5000L
  Z <- model.matrix(
    ~ x + batch + replicate,
    data.frame(x = rnorm(n), batch = factor(sample(letters[1:7], n, TRUE)),
               replicate = factor(sample(1:3, n, TRUE)))
  )
  beta <- c(-3.8, 0.25, rep(0.08, ncol(Z) - 2L))
  y <- rbinom(n, 1, plogis(drop(Z %*% beta)))
  prepared <- prepare_pysceptre_glm_design(Z)
  fit <- fit_pysceptre_glm(y, Z, family = "binomial", prepared_design = prepared)
  reference <- stats::glm.fit(Z, y, family = stats::binomial(),
                              control = stats::glm.control(epsilon = 1e-10))
  expect_identical(fit$method_used, "pysceptre")
  expect_true(fit$converged)
  expect_equal(fit$fitted.values, reference$fitted.values, tolerance = 1e-7)
  expect_equal(drop(Z %*% fit$coefficients), drop(Z %*% reference$coefficients), tolerance = 1e-7)
  expect_equal(fit$deviance, reference$deviance, tolerance = 1e-9)
  expect_true(all(fit$fitted.values > 0 & fit$fitted.values < 1))
  expect_error(fit_pysceptre_glm(replace(y, 1L, 2), Z, "binomial"), "between zero and one")
})


test_that("separated logistic data uses the reference fallback", {
  z <- seq(-2, 2, length.out = 200L)
  Z <- cbind(`(Intercept)` = 1, z = z)
  y <- as.integer(z > 0)
  fit <- suppressWarnings(fit_pysceptre_glm(y, Z, family = "binomial"))
  reference <- suppressWarnings(stats::glm.fit(Z, y, family = stats::binomial()))
  expect_identical(fit$method_used, "glm.fit")
  expect_match(fit$fallback_reason, "separation")
  expect_identical(fit$fitted.values, reference$fitted.values)
})


test_that("malformed prepared indices and transformations are rejected natively", {
  Z <- cbind(`(Intercept)` = 1, x = seq(-1, 1, length.out = 100L),
             b = rep(c(0, 1), 50L))
  prepared <- prepare_pysceptre_glm_design(Z)
  y <- rep(1, nrow(Z))
  check <- function(change) {
    corrupted <- change(prepared)
    expect_error(
      fit_pysceptre_glm_cpp(corrupted, y, "poisson"),
      "malformed prepared design"
    )
  }
  check(function(p) { p$dense_columns[1] <- ncol(Z); p })
  check(function(p) { p$dense_columns[1] <- NA_integer_; p })
  check(function(p) { p$dense_columns[2] <- p$dense_columns[1]; p })
  check(function(p) { p$groups[[1]][1] <- ncol(Z); p })
  check(function(p) { p$groups[[1]][1] <- p$dense_columns[2]; p })
  check(function(p) { p$groups[[1]][1] <- NA_integer_; p })
  check(function(p) { p$codes[1, 1] <- -1L; p })
  check(function(p) { p$codes[1, 1] <- NA_integer_; p })
  check(function(p) { p$codes[1, 1] <- length(p$groups[[1]]) + 1L; p })
  check(function(p) { p$scales[2] <- 0; p })
  check(function(p) { p$centers[2] <- NA_real_; p })
  check(function(p) { p$centers[1] <- 1; p })
  check(function(p) { p$scales[3] <- 2; p })
  check(function(p) { p$raw_matrix[1, 1] <- 0; p })
  check(function(p) { p$n <- p$n + 1L; p })
  check(function(p) { p$rank <- NA_integer_; p })
})


test_that("coefficient reconstruction cancellation uses the reference fallback", {
  set.seed(735)
  centered <- seq(-2, 2, length.out = 300L)
  Z <- cbind(`(Intercept)` = 1, x = 1e12 + centered)
  y <- rpois(nrow(Z), exp(0.4 + 0.3 * centered))
  prepared <- prepare_pysceptre_glm_design(Z)
  native <- fit_pysceptre_glm_cpp(prepared, y, "poisson")
  expect_true(native$converged)
  expect_gt(max(abs(expm1(drop(Z %*% native$coefficients) - native$linear.predictors))), 1e-8)
  fit <- fit_pysceptre_glm(y, Z, prepared_design = prepared)
  reference <- stats::glm.fit(Z, y, family = stats::poisson())
  expect_identical(fit$method_used, "glm.fit")
  expect_identical(fit$fallback_reason, "coefficient_reconstruction_mismatch")
  expect_identical(fit$coefficients, reference$coefficients)
  expect_identical(fit$fitted.values, reference$fitted.values)
})
