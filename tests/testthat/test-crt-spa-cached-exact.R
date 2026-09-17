.cached_crt_fixture <- local({
  fixtures <- list()

  function(p = 2L) {
    key <- as.character(p)
    if (!is.null(fixtures[[key]])) return(fixtures[[key]])
    set.seed(20260828)
    n <- 500L
    Z <- cbind(1, matrix(runif(n * (p - 1L), -1, 1), nrow = n))
    fitted_coefs <- c(-2, 0.35, rep(0.08, p - 2L))
    y <- rnbinom(n, size = 0.7, mu = exp(drop(Z %*% fitted_coefs)))
    pieces <- compute_precomputation_pieces(
      expression_vector = y,
      covariate_matrix = Z,
      fitted_coefs = fitted_coefs,
      theta = 0.7,
      full_test_stat = TRUE
    )
    propensity <- plogis(-1.5 + 0.4 * Z[, 2L])
    B1 <- 99L
    B2 <- 999L
    synthetic_idxs <- crt_index_sampler_fast(propensity, B1 + B2)
    synthetic_list <- synth_idx_list_to_r_list(synthetic_idxs)
    statistics <- vapply(synthetic_list, function(zero_based_indices) {
      indices <- zero_based_indices + 1L
      projection <- rowSums(pieces$D[, indices, drop = FALSE])
      sum(pieces$a[indices]) / sqrt(sum(pieces$w[indices]) - sum(projection^2))
    }, numeric(1L))
    stage_1 <- statistics[seq_len(B1)]
    stage_2 <- statistics[B1 + seq_len(B2)]
    right <- which(stage_2 > max(stage_1))
    left <- which(stage_2 < min(stage_1))
    if (length(right)) {
      candidate <- right[which.min(stage_2[right] - max(stage_1))]
      side <- 1L
    } else if (length(left)) {
      candidate <- left[which.min(min(stage_1) - stage_2[left])]
      side <- -1L
    } else {
      stop("The cached CRT fixture has no screened tail candidate.")
    }
    central_position <- which.min(abs(stage_1 - median(stage_1)))
    inverse <- solve(crossprod(Z, Z * pieces$w))
    features <- cbind(pieces$a, pieces$w * Z)
    moment <- colSums(features * propensity)
    nuisance_moment <- moment[-1L]
    beta <- drop(inverse %*% nuisance_moment)
    variance <- nuisance_moment[[1L]] - sum(nuisance_moment * beta)
    gradient_variance <- c(0, c(1, rep(0, p - 1L)) - 2 * beta)
    gradient <- c(1, rep(0, p)) / sqrt(variance) -
      0.5 * moment[[1L]] * gradient_variance / variance^(3 / 2)
    covariance <- crossprod(features, features * propensity * (1 - propensity))
    fixtures[[key]] <<- list(
      y = y, mu = pieces$mu, a = pieces$a, w = pieces$w,
      D = pieces$D, Z = Z, propensity = propensity,
      center = moment[[1L]] / sqrt(variance),
      standard_deviation = sqrt(drop(crossprod(gradient, covariance %*% gradient))),
      B1 = B1, B2 = B2, synthetic_idxs = synthetic_idxs,
      stage_1_statistics = stage_1, stage_2_statistics = stage_2,
      tail_side = side,
      trt_idxs = synthetic_list[[B1 + candidate]] + 1L,
      central_trt_idxs = synthetic_list[[central_position]] + 1L
    )
    fixtures[[key]]
  }
})

.cached_crt_always <- function(fixture, ...) {
  arguments <- list(
    y = fixture$y, mu = fixture$mu, a = fixture$a, w = fixture$w,
    Z = fixture$Z, fitted_probabilities = fixture$propensity,
    trt_idxs = fixture$trt_idxs, n_trt = length(fixture$trt_idxs),
    side_code = fixture$tail_side, max_iterations = 50L,
    use_fast = TRUE
  )
  do.call(run_low_level_test_full_crt_spa_always_v1,
          utils::modifyList(arguments, list(...)))
}

.cached_crt_screened <- function(fixture, ...) {
  arguments <- list(
    y = fixture$y, mu = fixture$mu, a = fixture$a, w = fixture$w,
    D = fixture$D, Z = fixture$Z,
    fitted_probabilities = fixture$propensity,
    trt_idxs = fixture$trt_idxs, n_trt = length(fixture$trt_idxs),
    use_all_cells = TRUE, synthetic_idxs = fixture$synthetic_idxs,
    B1 = fixture$B1, B2 = fixture$B2, return_resampling_dist = TRUE,
    side_code = fixture$tail_side, use_fast = TRUE
  )
  do.call(run_low_level_test_full_crt_spa_v1,
          utils::modifyList(arguments, list(...)))
}

.expect_cached_crt_core <- function(cached, native, tolerance = 1e-10) {
  expect_equal(cached[names(native)], native, tolerance = tolerance)
  expect_identical(cached$converged, native$converged)
  expect_identical(cached$reason, native$reason)
  expect_identical(cached$iterations, native$iterations)
  expect_identical(cached$path, "full_exact")
  expect_identical(cached$cgf_mode, "exact_bernoulli")
  expect_identical(cached$cache_strategy, "cached_exact")
  expect_null(cached$initial_partition_rule)
}

test_that("cached exact fixed-target fits preserve full Newton solutions", {
  for (p in c(2L, 6L)) {
    fixture <- .cached_crt_fixture(p)
    for (sign in c(-1L, 1L)) {
      for (standardized_target in c(0.5, 2.2)) {
        arguments <- list(
          a = fixture$a, w = fixture$w, Z = fixture$Z,
          propensity = fixture$propensity,
          target = sign * fixture$center +
            standardized_target * fixture$standard_deviation,
          score_sign = sign
        )
        native <- do.call(crt_spa_full_cpp, arguments)
        cached <- do.call(crt_spa_full_cached_cpp, arguments)
        expect_true(native$converged)
        .expect_cached_crt_core(cached, native)
        for (limit in c(0L, 1L)) {
          arguments$max_iterations <- limit
          stopped_native <- do.call(crt_spa_full_cpp, arguments)
          stopped_cached <- do.call(crt_spa_full_cached_cpp, arguments)
          .expect_cached_crt_core(stopped_cached, stopped_native)
        }
      }
    }
  }
})

test_that("cached always-SPA preserves all sidedness mappings and full diagnostics", {
  fixture <- .cached_crt_fixture()
  for (score_reflection in c(-1, 1)) {
    for (side in c(-1L, 0L, 1L)) {
      native <- .cached_crt_always(
        fixture, a = score_reflection * fixture$a,
        side_code = side, use_fast = FALSE
      )
      cached <- .cached_crt_always(
        fixture, a = score_reflection * fixture$a, side_code = side
      )
      expect_true(native$spa_converged)
      expect_true(cached$spa_fast)
      expect_identical(cached$p_value_source, "crt_spa_always_fast")
      comparable <- setdiff(names(native), c("spa_fast", "p_value_source", "spa_diagnostics"))
      expect_equal(cached[comparable], native[comparable], tolerance = 1e-10)
      .expect_cached_crt_core(cached$spa_diagnostics, native$spa_diagnostics)
      expect_length(cached$resampling_dist, 0L)
    }
  }
})

test_that("cached exact information SPA does not partition on observed outcomes", {
  fixture <- .cached_crt_fixture()
  for (runner in list(.cached_crt_always, .cached_crt_screened)) {
    sparse <- runner(fixture, y = rep(0, length(fixture$y)))
    dense <- runner(fixture, y = rep(2, length(fixture$y)))
    mixed <- runner(fixture)
    for (result in list(sparse, dense)) {
      expect_equal(result$p, mixed$p, tolerance = 0)
      expect_equal(result$z_orig, mixed$z_orig, tolerance = 0)
      expect_identical(result$spa_diagnostics, mixed$spa_diagnostics)
      expect_identical(result$spa_diagnostics$cgf_mode, "exact_bernoulli")
      expect_null(result$spa_diagnostics$initial_exact_count)
      expect_null(result$spa_diagnostics$initial_bulk_count)
    }
  }
})

test_that("cached screening preserves B1 skipping and the successful SPA path", {
  fixture <- .cached_crt_fixture()
  central_arguments <- list(
    trt_idxs = fixture$central_trt_idxs,
    n_trt = length(fixture$central_trt_idxs)
  )
  central_cached <- do.call(.cached_crt_screened, c(list(fixture), central_arguments))
  central_native <- do.call(.cached_crt_screened,
                           c(list(fixture, use_fast = FALSE), central_arguments))
  expect_identical(central_cached$stage, 1L)
  expect_identical(central_cached$spa_reason, "not_attempted")
  expect_true(is.na(central_cached$spa_converged))
  expect_equal(central_cached$p, central_native$p, tolerance = 0)
  expect_equal(central_cached$resampling_dist, fixture$stage_1_statistics)
  expect_identical(central_cached$spa_diagnostics, central_native$spa_diagnostics)

  native <- .cached_crt_screened(fixture, use_fast = FALSE)
  cached <- .cached_crt_screened(fixture)
  expect_identical(cached$stage, 2L)
  expect_true(cached$spa_converged)
  expect_identical(cached$p_value_source, "crt_spa_fast")
  expect_equal(cached$p, native$p, tolerance = 1e-10)
  expect_equal(cached$resampling_dist, native$resampling_dist, tolerance = 0)
  .expect_cached_crt_core(cached$spa_diagnostics, native$spa_diagnostics)
})

test_that("cached screening failures use the unchanged independent B2 bank", {
  fixture <- .cached_crt_fixture()
  unsupported_design <- fixture$Z
  unsupported_design[, 1L] <- 2
  native <- .cached_crt_screened(fixture, Z = unsupported_design, use_fast = FALSE)
  cached <- .cached_crt_screened(fixture, Z = unsupported_design)
  expect_identical(cached$stage, 3L)
  expect_identical(cached$p_value_source, "B2_empirical")
  expect_identical(cached$spa_reason, native$spa_reason)
  expect_match(cached$spa_reason, "input_or_solver_error.*intercept")
  expect_equal(cached$p, native$p, tolerance = 0)
  expect_equal(cached$resampling_dist, fixture$stage_2_statistics)
  expect_equal(cached$resampling_dist, native$resampling_dist, tolerance = 0)
})

test_that("cached always-SPA failure preserves B2 finalization", {
  fixture <- .cached_crt_fixture()
  native <- .cached_crt_always(fixture, max_iterations = 0L, use_fast = FALSE)
  cached <- .cached_crt_always(fixture, max_iterations = 0L)
  expect_identical(cached$spa_reason, "solver_disabled_max_iterations_zero")
  expect_true(cached$needs_empirical_fallback)
  .expect_cached_crt_core(cached$spa_diagnostics, native$spa_diagnostics)
  arguments <- list(
    a = fixture$a, w = fixture$w, Z = fixture$Z,
    synthetic_idxs = fixture$synthetic_idxs, B2 = fixture$B1 + fixture$B2,
    return_resampling_dist = TRUE, side_code = fixture$tail_side
  )
  finalized_native <- do.call(finalize_low_level_test_crt_spa_fallback_v1,
    c(arguments, list(spa_attempt_result = native)))
  finalized_cached <- do.call(finalize_low_level_test_crt_spa_fallback_v1,
    c(arguments, list(spa_attempt_result = cached)))
  expect_identical(finalized_cached$stage, 3L)
  expect_identical(finalized_cached$p_value_source, "B2_empirical")
  expect_false(finalized_cached$needs_empirical_fallback)
  expect_equal(finalized_cached$p, finalized_native$p, tolerance = 0)
  expect_equal(finalized_cached$resampling_dist,
               finalized_native$resampling_dist, tolerance = 0)
  expect_identical(finalized_cached$spa_diagnostics, cached$spa_diagnostics)
})

test_that("the cached outward entry point matches public always-SPA diagnostics", {
  fixture <- .cached_crt_fixture()
  direct <- crt_spa_full_outward_cached_cpp(
    fixture$a, fixture$w, fixture$Z, fixture$propensity, fixture$trt_idxs
  )
  public <- .cached_crt_always(fixture)
  expect_identical(direct, public$spa_diagnostics)
  expect_error(crt_spa_full_outward_cached_cpp(
    fixture$a, fixture$w, fixture$Z, fixture$propensity, c(0L, 2L)
  ), "invalid one-based index")
  expect_error(crt_spa_full_outward_cached_cpp(
    fixture$a, fixture$w, fixture$Z, fixture$propensity, c(1L, 1L)
  ), "cannot contain duplicates")
})

test_that("cached outward fits retain central fallback and minimum Newton updates", {
  fixture <- list(
    y = rep(2, 4L), mu = rep(1, 4L), a = c(1, -1, 1, -1),
    w = rep(1, 4L), Z = matrix(1, nrow = 4L),
    propensity = rep(0.5, 4L), trt_idxs = c(1L, 2L), tail_side = 1L
  )
  native <- .cached_crt_always(fixture, use_fast = FALSE)
  cached <- .cached_crt_always(fixture)
  expect_identical(cached$spa_reason, "central_target_requires_empirical_fallback")
  .expect_cached_crt_core(cached$spa_diagnostics, native$spa_diagnostics)

  fixture$a[[1L]] <- fixture$a[[1L]] + 2^-20
  fixture$a[[4L]] <- fixture$a[[4L]] - 2^-20
  for (sign in c(-1, 1)) {
    native <- .cached_crt_always(fixture, a = sign * fixture$a, use_fast = FALSE)
    cached <- .cached_crt_always(fixture, a = sign * fixture$a)
    expect_true(cached$spa_converged)
    expect_gte(cached$spa_iterations, 1L)
    expect_lt(abs(cached$p - 0.5), 1e-3)
    .expect_cached_crt_core(cached$spa_diagnostics, native$spa_diagnostics)
  }
})

test_that("cached exact input validation agrees with ordinary fixed-target fits", {
  fixture <- .cached_crt_fixture()
  arguments <- list(
    a = fixture$a, w = fixture$w, Z = fixture$Z,
    propensity = fixture$propensity,
    target = fixture$center + 2.2 * fixture$standard_deviation
  )
  invalid <- list(
    list(change = list(a = fixture$a[-1L]), reason = "dimensions"),
    list(change = list(a = replace(fixture$a, 1L, Inf)), reason = "a must be finite"),
    list(change = list(w = replace(fixture$w, 1L, 0)), reason = "w must be positive"),
    list(change = list(propensity = replace(fixture$propensity, 1L, 0)), reason = "propensity"),
    list(change = list(propensity = replace(fixture$propensity, 1L, 1)), reason = "propensity"),
    list(change = list(Z = cbind(fixture$Z[, 1L], 0)), reason = "positive definite"),
    list(change = list(score_sign = 0L), reason = "score_sign"),
    list(change = list(target = Inf), reason = "target must be finite"),
    list(change = list(tolerance = 0), reason = "tolerance must be positive"),
    list(change = list(max_iterations = -1L), reason = "max_iterations")
  )
  for (entry in invalid) {
    for (solver in list(crt_spa_full_cpp, crt_spa_full_cached_cpp)) {
      expect_error(do.call(solver, utils::modifyList(arguments, entry$change)),
                   entry$reason)
    }
  }
})

test_that("cached outward failure diagnostics preserve unsupported geometry and endpoints", {
  fixture <- .cached_crt_fixture()
  unsupported_design <- fixture$Z
  unsupported_design[, 1L] <- 2
  cases <- list(
    list(Z = unsupported_design),
    list(Z = cbind(fixture$Z[, 1L], 0)),
    list(fitted_probabilities = replace(fixture$propensity, 1L, 0)),
    list(fitted_probabilities = replace(fixture$propensity, 1L, 1)),
    list(trt_idxs = integer(), n_trt = 0L),
    list(trt_idxs = seq_along(fixture$a), n_trt = length(fixture$a))
  )
  for (change in cases) {
    native <- do.call(.cached_crt_always, c(list(fixture, use_fast = FALSE), change))
    cached <- do.call(.cached_crt_always, c(list(fixture), change))
    expect_false(cached$spa_converged)
    expect_identical(cached$needs_empirical_fallback, native$needs_empirical_fallback)
    expect_identical(cached$p_value_source, native$p_value_source)
    expect_identical(cached$spa_reason, native$spa_reason)
    expect_equal(cached$z_orig, native$z_orig, tolerance = 1e-10)
    .expect_cached_crt_core(cached$spa_diagnostics, native$spa_diagnostics)
  }
  for (fast in c(FALSE, TRUE)) {
    for (invalid_indices in list(c(0L, 2L), c(1L, 501L), c(1L, NA_integer_))) {
      expect_error(.cached_crt_always(fixture, trt_idxs = invalid_indices,
                                      n_trt = 2L, use_fast = fast),
                   "valid one-based indices")
    }
    expect_error(.cached_crt_always(fixture, trt_idxs = c(1L, 1L),
                                    n_trt = 2L, use_fast = fast),
                 "cannot contain duplicates")
  }
})
