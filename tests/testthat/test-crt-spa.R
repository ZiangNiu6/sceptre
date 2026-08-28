.crt_spa_full_statistic <- function(zero_based_indices, a, w, D) {
  indices <- zero_based_indices + 1L
  if (!length(indices)) return(NaN)
  projection <- rowSums(D[, indices, drop = FALSE])
  variance <- sum(w[indices]) - sum(projection^2)
  if (!is.finite(variance) || variance <= 0) return(NaN)
  sum(a[indices]) / sqrt(variance)
}

.crt_spa_tail_geometry <- function(a, w, Z, propensity, score_sign = 1L) {
  C_inv <- solve(crossprod(Z, Z * w))
  G <- cbind(score_sign * a, w * Z)
  tilted_variance <- propensity * (1 - propensity)
  moment <- colSums(G * propensity)
  B <- moment[-1L]
  beta <- drop(C_inv %*% B)
  V <- B[[1L]] - drop(crossprod(B, beta))
  grad_V <- c(
    0,
    c(1, rep(0, ncol(Z) - 1L)) - 2 * beta
  )
  gradient <- c(1, rep(0, ncol(Z))) / sqrt(V) -
    0.5 * moment[[1L]] * grad_V / V^(3 / 2)
  Sgg <- crossprod(G, G * tilted_variance)
  list(
    center = moment[[1L]] / sqrt(V),
    standard_deviation = sqrt(drop(crossprod(gradient, Sgg %*% gradient)))
  )
}

.make_crt_spa_fixture <- local({
  fixture <- NULL

  function() {
    if (!is.null(fixture)) return(fixture)

    set.seed(20260828)
    n <- 500L
    z <- runif(n, -1, 1)
    Z <- cbind(`(Intercept)` = 1, z = z)
    fitted_coefs <- c(-2, 0.35)
    theta <- 0.7
    mu <- exp(drop(Z %*% fitted_coefs))
    y <- rnbinom(n, size = theta, mu = mu)
    pieces <- compute_precomputation_pieces(
      expression_vector = y,
      covariate_matrix = Z,
      fitted_coefs = fitted_coefs,
      theta = theta,
      full_test_stat = TRUE
    )
    propensity <- plogis(-1.5 + 0.4 * z)

    # B1 = 99 makes the minimum two-sided plus-one p-value exactly 0.02,
    # allowing the deterministic extreme to exercise the inclusive trigger.
    B1 <- 99L
    B2 <- 999L
    synthetic_idxs <- crt_index_sampler_fast(propensity, B1 + B2)
    synthetic_list <- synth_idx_list_to_r_list(synthetic_idxs)
    statistics <- vapply(
      synthetic_list,
      .crt_spa_full_statistic,
      numeric(1),
      a = pieces$a,
      w = pieces$w,
      D = pieces$D
    )
    stopifnot(all(is.finite(statistics)))

    stage_1_statistics <- statistics[seq_len(B1)]
    stage_2_statistics <- statistics[B1 + seq_len(B2)]
    right_candidates <- which(stage_2_statistics > max(stage_1_statistics))
    left_candidates <- which(stage_2_statistics < min(stage_1_statistics))
    if (length(right_candidates)) {
      candidate <- right_candidates[
        which.min(stage_2_statistics[right_candidates] - max(stage_1_statistics))
      ]
      tail_side <- 1L
    } else if (length(left_candidates)) {
      candidate <- left_candidates[
        which.min(min(stage_1_statistics) - stage_2_statistics[left_candidates])
      ]
      tail_side <- -1L
    } else {
      stop("The deterministic CRT-SPA fixture has no stage-2 tail candidate.")
    }

    tail_position <- B1 + candidate
    central_position <- which.min(
      abs(stage_1_statistics - stats::median(stage_1_statistics))
    )
    geometry <- .crt_spa_tail_geometry(
      pieces$a, pieces$w, Z, propensity
    )

    fixture <<- list(
      y = y,
      mu = pieces$mu,
      a = pieces$a,
      w = pieces$w,
      D = pieces$D,
      Z = Z,
      propensity = propensity,
      B1 = B1,
      B2 = B2,
      synthetic_idxs = synthetic_idxs,
      synthetic_list = synthetic_list,
      statistics = statistics,
      stage_1_statistics = stage_1_statistics,
      stage_2_statistics = stage_2_statistics,
      tail_side = tail_side,
      tail_trt_idxs = synthetic_list[[tail_position]] + 1L,
      central_trt_idxs = synthetic_list[[central_position]] + 1L,
      core_target = geometry$center + 2.2 * geometry$standard_deviation
    )
    fixture
  }
})

.run_crt_spa_low_level <- function(fixture, ...) {
  arguments <- list(
    y = fixture$y,
    mu = fixture$mu,
    a = fixture$a,
    w = fixture$w,
    D = fixture$D,
    Z = fixture$Z,
    fitted_probabilities = fixture$propensity,
    trt_idxs = fixture$tail_trt_idxs,
    n_trt = length(fixture$tail_trt_idxs),
    use_all_cells = TRUE,
    synthetic_idxs = fixture$synthetic_idxs,
    B1 = fixture$B1,
    B2 = fixture$B2,
    return_resampling_dist = TRUE,
    side_code = fixture$tail_side
  )
  do.call(
    run_low_level_test_full_crt_spa_v1,
    utils::modifyList(arguments, list(...))
  )
}

test_that("empirical p-values retain sidedness, ties, and plus-one correction", {
  null_statistics <- c(-2, -1, 0, 1, 2)

  expect_equal(
    compute_empirical_p_value(null_statistics, z_orig = 1, side = -1L),
    5 / 6
  )
  expect_equal(
    compute_empirical_p_value(null_statistics, z_orig = 1, side = 1L),
    3 / 6
  )
  expect_equal(
    compute_empirical_p_value(null_statistics, z_orig = 1, side = 0L),
    1
  )

  expect_equal(
    compute_empirical_p_value(-null_statistics, z_orig = -1, side = 1L),
    compute_empirical_p_value(null_statistics, z_orig = 1, side = -1L)
  )
  expect_equal(
    compute_empirical_p_value(-null_statistics, z_orig = -1, side = 0L),
    compute_empirical_p_value(null_statistics, z_orig = 1, side = 0L)
  )
})

test_that("the SPA boundary is the native SCEPTRE full statistic", {
  fixture <- .make_crt_spa_fixture()
  indices <- fixture$tail_trt_idxs
  weighted_design_sum <- colSums(
    fixture$w[indices] * fixture$Z[indices, , drop = FALSE]
  )
  information_inverse <- solve(crossprod(fixture$Z, fixture$Z * fixture$w))
  boundary_variance <- weighted_design_sum[[1L]] - drop(crossprod(
    weighted_design_sum,
    information_inverse %*% weighted_design_sum
  ))
  spa_boundary <- sum(fixture$a[indices]) / sqrt(boundary_variance)
  native_statistic <- .crt_spa_full_statistic(
    indices - 1L, fixture$a, fixture$w, fixture$D
  )

  expect_equal(spa_boundary, native_statistic, tolerance = 1e-10)
})

test_that("the legacy low-level state machine is unchanged", {
  fixture <- .make_crt_spa_fixture()

  central <- run_low_level_test_full_v4(
    y = fixture$y,
    mu = fixture$mu,
    a = fixture$a,
    w = fixture$w,
    D = fixture$D,
    trt_idxs = fixture$central_trt_idxs,
    n_trt = length(fixture$central_trt_idxs),
    use_all_cells = TRUE,
    synthetic_idxs = fixture$synthetic_idxs,
    B1 = fixture$B1,
    B2 = fixture$B2,
    B3 = 0L,
    fit_parametric_curve = FALSE,
    return_resampling_dist = TRUE,
    side_code = fixture$tail_side
  )
  expect_identical(central$stage, 1L)
  expect_length(central$resampling_dist, fixture$B1)
  expect_equal(
    central$p,
    compute_empirical_p_value(
      central$resampling_dist, central$z_orig, fixture$tail_side
    )
  )

  tail <- run_low_level_test_full_v4(
    y = fixture$y,
    mu = fixture$mu,
    a = fixture$a,
    w = fixture$w,
    D = fixture$D,
    trt_idxs = fixture$tail_trt_idxs,
    n_trt = length(fixture$tail_trt_idxs),
    use_all_cells = TRUE,
    synthetic_idxs = fixture$synthetic_idxs,
    B1 = fixture$B1,
    B2 = fixture$B2,
    B3 = 0L,
    fit_parametric_curve = FALSE,
    return_resampling_dist = TRUE,
    side_code = fixture$tail_side
  )
  expect_identical(tail$stage, 3L)
  expect_length(tail$resampling_dist, fixture$B1)
  expect_equal(tail$p, 1 / (fixture$B1 + 1))
})

test_that("Full Newton CRT-SPA returns a certified in-range tail probability", {
  fixture <- .make_crt_spa_fixture()
  result <- crt_spa_full_cpp(
    fixture$a,
    fixture$w,
    fixture$Z,
    fixture$propensity,
    fixture$core_target
  )

  expect_true(result$converged)
  expect_identical(result$reason, "ok")
  expect_identical(result$path, "full_exact")
  expect_true(is.finite(result$p_value))
  expect_gte(result$p_value, 0)
  expect_lte(result$p_value, 1)
  expect_gt(result$target, result$center)
  expect_lte(result$max_residual, 1e-8)
  expect_identical(result$moment_dimension, ncol(fixture$Z) + 1L)
  expect_identical(result$nuisance_dimension, ncol(fixture$Z))
})

test_that("Full Newton CRT-SPA is invariant under the left-tail reflection", {
  fixture <- .make_crt_spa_fixture()
  right <- crt_spa_full_cpp(
    fixture$a,
    fixture$w,
    fixture$Z,
    fixture$propensity,
    fixture$core_target,
    score_sign = 1L
  )
  reflected_left <- crt_spa_full_cpp(
    -fixture$a,
    fixture$w,
    fixture$Z,
    fixture$propensity,
    fixture$core_target,
    score_sign = -1L
  )

  expect_true(right$converged)
  expect_true(reflected_left$converged)
  expect_equal(reflected_left$p_value, right$p_value, tolerance = 1e-13)
  expect_equal(reflected_left$rate, right$rate, tolerance = 1e-12)
  expect_equal(reflected_left$r_lr, right$r_lr, tolerance = 1e-12)
  expect_equal(reflected_left$q_lr, right$q_lr, tolerance = 1e-12)
})

test_that("Full Newton CRT-SPA validates inputs and reports nonconvergence", {
  fixture <- .make_crt_spa_fixture()

  expect_error(
    crt_spa_full_cpp(
      fixture$a[-1L], fixture$w, fixture$Z, fixture$propensity,
      fixture$core_target
    ),
    "dimensions"
  )
  bad_propensity <- fixture$propensity
  bad_propensity[[1L]] <- 0
  expect_error(
    crt_spa_full_cpp(
      fixture$a, fixture$w, fixture$Z, bad_propensity,
      fixture$core_target
    ),
    "propensity"
  )
  bad_design <- fixture$Z
  bad_design[, 1L] <- 2
  expect_error(
    crt_spa_full_cpp(
      fixture$a, fixture$w, bad_design, fixture$propensity,
      fixture$core_target
    ),
    "intercept"
  )
  expect_error(
    crt_spa_full_cpp(
      fixture$a, fixture$w, fixture$Z, fixture$propensity,
      fixture$core_target, score_sign = 0L
    ),
    "score_sign"
  )

  stopped <- crt_spa_full_cpp(
    fixture$a,
    fixture$w,
    fixture$Z,
    fixture$propensity,
    fixture$core_target,
    max_iterations = 0L
  )
  expect_false(stopped$converged)
  expect_identical(stopped$reason, "newton_failed")
  expect_true(is.na(stopped$p_value))
})

test_that("the low-level CRT-SPA path preserves reflected one-sided tests", {
  fixture <- .make_crt_spa_fixture()
  forward <- .run_crt_spa_low_level(fixture)
  reflected <- .run_crt_spa_low_level(
    fixture,
    a = -fixture$a,
    side_code = -fixture$tail_side
  )
  two_sided <- .run_crt_spa_low_level(fixture, side_code = 0L)

  expect_identical(forward$stage, 2L)
  expect_identical(forward$p_value_source, "crt_spa")
  expect_true(forward$spa_converged)
  expect_identical(forward$spa_reason, "ok")
  expect_true(is.finite(forward$p))
  expect_true(forward$p >= 0 && forward$p <= 1)
  expect_length(forward$resampling_dist, fixture$B1)

  expect_identical(reflected$stage, 2L)
  expect_identical(reflected$p_value_source, "crt_spa")
  expect_true(reflected$spa_converged)
  expect_equal(reflected$z_orig, -forward$z_orig, tolerance = 1e-13)
  expect_equal(reflected$p, forward$p, tolerance = 1e-13)
  expect_equal(reflected$spa_rate, forward$spa_rate, tolerance = 1e-12)

  expect_identical(two_sided$stage, 2L)
  expect_equal(two_sided$p, min(1, 2 * forward$p), tolerance = 1e-13)
})

test_that("invalid SPA inputs fall back to the independent B2 bank", {
  fixture <- .make_crt_spa_fixture()
  bad_design <- fixture$Z
  bad_design[, 1L] <- 2
  result <- .run_crt_spa_low_level(fixture, Z = bad_design)

  expect_identical(result$stage, 3L)
  expect_identical(result$p_value_source, "B2_empirical")
  expect_false(result$spa_converged)
  expect_match(result$spa_reason, "input_or_solver_error.*intercept")
  expect_length(result$resampling_dist, fixture$B2)
  expect_equal(result$resampling_dist, fixture$stage_2_statistics)
  expect_equal(
    result$p,
    compute_empirical_p_value(
      fixture$stage_2_statistics, result$z_orig, fixture$tail_side
    )
  )
})

test_that("ordinary Full Newton failure falls back to B2 empirically", {
  fixture <- .make_crt_spa_fixture()
  indices <- fixture$tail_trt_idxs
  lower_left <- sum(fixture$w[indices])
  lower_right <- sum(rowSums(fixture$D[, indices, drop = FALSE])^2)
  expect_gt(lower_right, 0)
  boundary_scale <- sqrt((1 - 1e-10) * lower_left / lower_right)
  near_boundary_D <- fixture$D * boundary_scale

  result <- .run_crt_spa_low_level(fixture, D = near_boundary_D)

  expect_identical(result$stage, 3L)
  expect_identical(result$p_value_source, "B2_empirical")
  expect_false(result$spa_converged)
  expect_true(result$spa_reason %in% c(
    "newton_failed", "nonregular_upper_root", "lr_out_of_range"
  ))
  expect_true(is.finite(result$p))
  expect_true(result$p > 0 && result$p <= 1)
  expect_length(result$resampling_dist, fixture$B2)
})

test_that("crt_spa is opt-in, CRT-only, and retains the planned banks", {
  set.seed(20260829)
  n <- 35L
  grna_target_data_frame <- make_mock_grna_target_data(
    num_guides_per_target = 1,
    chr_distances = 1,
    chr_starts = 1,
    num_nt_guides = 2
  )
  response_matrix <- matrix(rpois(3L * n, lambda = 1), nrow = 3L)
  grna_matrix <- matrix(
    rpois(nrow(grna_target_data_frame) * n, lambda = 1),
    nrow = nrow(grna_target_data_frame),
    dimnames = list(grna_target_data_frame$grna_id, NULL)
  )
  empty_pairs <- data.frame(
    grna_target = character(),
    response_id = character()
  )
  imported <- import_data(
    response_matrix = response_matrix,
    grna_matrix = grna_matrix,
    grna_target_data_frame = grna_target_data_frame,
    moi = "high"
  )

  configured <- set_analysis_parameters(
    imported,
    discovery_pairs = empty_pairs,
    resampling_mechanism = "crt",
    resampling_approximation = "crt_spa"
  )
  expect_identical(configured@resampling_approximation, "crt_spa")
  expect_identical(configured@B1, 499L)
  expect_identical(configured@B2, 4999L)
  expect_identical(configured@B3, 0L)
  expect_false(configured@run_permutations)

  expect_error(
    set_analysis_parameters(
      imported,
      discovery_pairs = empty_pairs,
      resampling_mechanism = "permutations",
      resampling_approximation = "crt_spa"
    ),
    "available only.*resampling_mechanism = 'crt'"
  )
})
