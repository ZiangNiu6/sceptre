test_that("all SPA solver interfaces default to 1e-4", {
  solvers <- c(
    "crt_spa_full_cpp", "crt_spa_full_cached_cpp",
    "crt_spa_full_outward_cached_cpp", "crt_spa_full_fast_cpp",
    "crt_spa_full_outward_fast_cpp", "crt_empirical_spa_full_cpp",
    "crt_empirical_spa_full_fast_cpp", "rpt_spa_full_cpp",
    "rpt_spa_moment_prepared_cpp", "rpt_spa_moment_outward_prepared_cpp"
  )
  for (solver in solvers) {
    defaults <- formals(get(solver, envir = asNamespace("sceptre")))
    expect_identical(defaults$tolerance, 1e-4, info = solver)
    if ("compressed_tolerance" %in% names(defaults)) {
      expect_identical(defaults$compressed_tolerance, 1e-4, info = solver)
      expect_identical(defaults$exact_audit_tolerance, 1e-4, info = solver)
    }
  }
})

.uniform_spa_fixture <- local({
  fixture <- NULL
  function() {
    if (!is.null(fixture)) return(fixture)
    set.seed(20260901)
    n <- 400L
    z <- runif(n, -1, 1)
    Z <- cbind(1, z)
    mu <- exp(-1.4 + 0.2 * z)
    y <- rnbinom(n, mu = mu, size = 2)
    w <- mu / (1 + mu / 2)
    a <- (y - mu) / (1 + mu / 2)
    D <- compute_D_matrix(crossprod(Z, Z * w), w * Z)
    propensity <- rep(0.2, n)
    B1 <- 99L
    B2 <- 999L
    rpt_bank <- fisher_yates_samlper(n_tot = n, M = 80L, B = B1 + B2)
    crt_bank <- crt_index_sampler_fast(propensity, B1 + B2)
    information_statistic <- function(indices) {
      indices <- indices + 1L
      projection <- rowSums(D[, indices, drop = FALSE])
      sum(a[indices]) / sqrt(sum(w[indices]) - sum(projection^2))
    }
    select_tail <- function(bank, statistic) {
      draws <- synth_idx_list_to_r_list(bank)
      values <- vapply(draws, statistic, numeric(1))
      initial <- values[seq_len(B1)]
      later <- values[B1 + seq_len(B2)]
      right <- which(later > max(initial))
      left <- which(later < min(initial))
      if (length(right)) {
        selected <- right[which.min(later[right] - max(initial))]
        side <- 1L
      } else {
        stopifnot(length(left) > 0L)
        selected <- left[which.min(min(initial) - later[left])]
        side <- -1L
      }
      list(bank = bank, indices = draws[[B1 + selected]] + 1L, side = side)
    }
    empirical_statistic <- function(indices) {
      compute_observed_empirical_crt_statistic_v1(a, propensity, indices + 1L)
    }
    fixture <<- list(
      a = a, w = w, y = y, mu = mu, Z = Z, D = D,
      propensity = propensity, B1 = B1, B2 = B2,
      rpt = select_tail(rpt_bank, information_statistic),
      crt = select_tail(crt_bank, information_statistic),
      empirical = select_tail(crt_bank, empirical_statistic)
    )
    fixture
  }
})

.expect_uniform_spa_root <- function(result, reference) {
  expect_true(result$spa_converged)
  expect_true(reference$converged)
  expect_identical(result$spa_iterations, reference$iterations)
  expect_equal(result$spa_max_residual, reference$max_residual, tolerance = 1e-12)
  expect_lte(result$spa_max_residual, 1e-4)
  expect_equal(result$spa_diagnostics$p_value, reference$p_value, tolerance = 1e-12)
}

test_that("screened and always CRT information routes use the 1e-4 root", {
  f <- .uniform_spa_fixture()
  target <- f$crt
  common <- list(
    y = f$y, mu = f$mu, a = f$a, w = f$w, Z = f$Z,
    fitted_probabilities = f$propensity, trt_idxs = target$indices,
    n_trt = length(target$indices), side_code = target$side
  )
  for (fast in c(FALSE, TRUE)) {
    solver <- if (fast) crt_spa_full_cached_cpp else crt_spa_full_cpp
    screened <- do.call(run_low_level_test_full_crt_spa_v1, c(common, list(
      D = f$D, use_all_cells = TRUE, synthetic_idxs = target$bank,
      B1 = f$B1, B2 = f$B2, return_resampling_dist = FALSE, use_fast = fast
    )))
    reference <- solver(f$a, f$w, f$Z, f$propensity,
      target$side * screened$z_orig, score_sign = target$side, tolerance = 1e-4)
    .expect_uniform_spa_root(screened, reference)
    always <- do.call(run_low_level_test_full_crt_spa_always_v1,
                     c(common, list(use_fast = fast)))
    reference <- solver(f$a, f$w, f$Z, f$propensity,
      always$spa_target, score_sign = always$spa_score_sign, tolerance = 1e-4)
    .expect_uniform_spa_root(always, reference)
  }
})

test_that("screened and always CRT empirical routes use the 1e-4 root", {
  f <- .uniform_spa_fixture()
  target <- f$empirical
  common <- list(
    y = f$y, mu = f$mu, a = f$a, fitted_probabilities = f$propensity,
    trt_idxs = target$indices, n_trt = length(target$indices),
    side_code = target$side
  )
  for (fast in c(FALSE, TRUE)) {
    solver <- if (fast) crt_empirical_spa_full_fast_cpp else crt_empirical_spa_full_cpp
    solver_arguments <- list(a = f$a, propensity = f$propensity, tolerance = 1e-4)
    if (fast) solver_arguments$y <- f$y
    screened <- do.call(run_low_level_test_full_crt_spa_empirical_v1, c(common, list(
      synthetic_idxs = target$bank, B1 = f$B1, B2 = f$B2,
      return_resampling_dist = FALSE, use_fast = fast
    )))
    always <- do.call(run_low_level_test_full_crt_spa_empirical_always_v1,
                     c(common, list(use_fast = fast)))
    for (result in list(screened, always)) {
      reference <- do.call(solver, c(solver_arguments, list(
        target = result$spa_target, score_sign = result$spa_score_sign
      )))
      .expect_uniform_spa_root(result, reference)
    }
  }
})

test_that("screened and always RPT routes use matching exact and moment tolerances", {
  f <- .uniform_spa_fixture()
  target <- f$rpt
  context <- prepare_rpt_spa_response_cpp(f$a, f$w, f$Z, f$y == 0,
                                        minimum_zero_fraction = 0)
  on.exit(release_rpt_spa_response_cpp(context))
  fraction <- length(target$indices) / length(f$a)
  B <- fraction * colSums(f$w * f$Z)
  variance <- fraction * sum(f$w) - drop(crossprod(
    B, solve(crossprod(f$Z, f$w * f$Z), B)
  ))
  compressed_target <- fraction * sum(f$a) / sqrt(variance) + 0.5
  compressed <- rpt_spa_moment_prepared_cpp(
    context, length(target$indices), compressed_target
  )
  explicit <- rpt_spa_moment_prepared_cpp(
    context, length(target$indices), compressed_target,
    tolerance = 1e-4, compressed_tolerance = 1e-4, exact_audit_tolerance = 1e-4
  )
  expect_identical(compressed$path, "moment_compressed_exact_polish")
  expect_true(compressed$converged)
  expect_true(compressed$exact_audit_passed)
  expect_identical(compressed, explicit)
  common <- list(
    y = f$y, mu = f$mu, a = f$a, w = f$w, Z = f$Z,
    trt_idxs = target$indices, n_trt = length(target$indices),
    side_code = target$side
  )
  for (moment in c(FALSE, TRUE)) {
    controls <- list(use_moment = moment, prepared_context = context)
    screened <- do.call(run_low_level_test_full_rpt_spa_v1, c(common, controls, list(
      D = f$D, use_all_cells = FALSE, synthetic_idxs = target$bank,
      B1 = f$B1, B2 = f$B2, return_resampling_dist = FALSE
    )))
    always <- do.call(run_low_level_test_full_rpt_spa_always_v1, c(common, controls))
    for (result in list(screened, always)) {
      sign <- if (is.null(result$spa_score_sign)) target$side else result$spa_score_sign
      if (moment) {
        reference <- rpt_spa_moment_prepared_cpp(
          context, length(target$indices), sign * result$z_orig,
          score_sign = sign, tolerance = 1e-4,
          compressed_tolerance = 1e-4, exact_audit_tolerance = 1e-4
        )
        expect_identical(result$spa_solver_tolerance, 1e-4)
        expect_identical(result$spa_compressed_tolerance, 1e-4)
        expect_identical(result$spa_exact_audit_tolerance, 1e-4)
      } else {
        reference <- rpt_spa_full_cpp(
          f$a, f$w, f$Z, length(target$indices), sign * result$z_orig,
          score_sign = sign, tolerance = 1e-4
        )
      }
      .expect_uniform_spa_root(result, reference)
    }
  }
})
