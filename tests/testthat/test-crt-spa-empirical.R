.empirical_crt_dense_components <- function(a, propensity, zero_based_indices) {
  stopifnot(length(a) == length(propensity))
  exposure <- numeric(length(a))
  exposure[zero_based_indices + 1L] <- 1
  contribution <- a * (exposure - propensity)
  score <- sum(contribution)
  variance <- sum(contribution^2) - score^2 / length(contribution)
  list(
    score = score,
    variance = variance,
    statistic = if (is.finite(variance) && variance > 0) {
      score / sqrt(variance)
    } else {
      NaN
    }
  )
}

.empirical_crt_dense_statistic <- function(a, propensity, zero_based_indices) {
  .empirical_crt_dense_components(
    a, propensity, zero_based_indices
  )$statistic
}

.empirical_crt_bank_statistics <- function(a, propensity, synthetic_list) {
  vapply(
    synthetic_list,
    .empirical_crt_dense_statistic,
    numeric(1),
    a = a,
    propensity = propensity
  )
}

.empirical_crt_softplus <- function(x) {
  pmax(x, 0) + log1p(exp(-abs(x)))
}

.empirical_crt_cgf <- function(theta, a, propensity, score_sign = 1L) {
  oriented_a <- score_sign * a
  quadratic_increment <- (1 - 2 * propensity) * oriented_a^2
  feature <- unname(cbind(oriented_a, quadratic_increment))
  constant <- c(
    -sum(propensity * oriented_a),
    sum(propensity^2 * oriented_a^2)
  )
  offset <- qlogis(propensity)
  linear_predictor <- drop(feature %*% theta)
  tilted_propensity <- plogis(offset + linear_predictor)
  moment <- constant + colSums(feature * tilted_propensity)
  hessian <- crossprod(
    feature,
    feature * (tilted_propensity * (1 - tilted_propensity))
  )
  cgf <- sum(theta * constant) + sum(
    .empirical_crt_softplus(offset + linear_predictor) -
      .empirical_crt_softplus(offset)
  )
  list(
    cgf = cgf,
    moment = moment,
    hessian = hessian,
    rate = sum(theta * moment) - cgf
  )
}

.empirical_crt_center_geometry <- function(a, propensity) {
  second_moment <- sum(a^2 * propensity * (1 - propensity))
  list(
    center = 0,
    standard_deviation = sqrt(second_moment / second_moment)
  )
}

.make_empirical_crt_fixture <- local({
  fixture <- NULL

  function() {
    if (!is.null(fixture)) return(fixture)

    set.seed(20260830)
    n <- 127L
    z <- as.numeric(scale(seq_len(n) + rnorm(n, sd = 7)))
    propensity <- plogis(-0.85 + 0.55 * z)
    a <- 0.45 * z + sin(seq_len(n) / 5) + rnorm(n, sd = 0.55)
    mu <- exp(0.25 + 0.12 * z)
    y <- mu + 0.5 + abs(a)

    B1 <- 49L
    B2 <- 399L
    synthetic_idxs <- crt_index_sampler_fast(propensity, B1 + B2)
    synthetic_list <- synth_idx_list_to_r_list(synthetic_idxs)
    statistics <- .empirical_crt_bank_statistics(
      a, propensity, synthetic_list
    )
    stopifnot(all(is.finite(statistics)))

    stage_1_statistics <- statistics[seq_len(B1)]
    stage_2_positions <- B1 + seq_len(B2)
    nonempty_stage_2 <- stage_2_positions[
      lengths(synthetic_list[stage_2_positions]) > 0L
    ]
    right_candidates <- nonempty_stage_2[
      statistics[nonempty_stage_2] > max(stage_1_statistics)
    ]
    left_candidates <- nonempty_stage_2[
      statistics[nonempty_stage_2] < min(stage_1_statistics)
    ]
    if (length(right_candidates)) {
      tail_position <- right_candidates[
        which.max(statistics[right_candidates])
      ]
      tail_side <- 1L
    } else if (length(left_candidates)) {
      tail_position <- left_candidates[
        which.min(statistics[left_candidates])
      ]
      tail_side <- -1L
    } else {
      stop("The deterministic empirical CRT fixture has no B2 tail draw.")
    }

    nonempty_stage_1 <- which(lengths(synthetic_list[seq_len(B1)]) > 0L)
    central_position <- nonempty_stage_1[
      which.min(abs(
        stage_1_statistics[nonempty_stage_1] -
          stats::median(stage_1_statistics)
      ))
    ]

    fixture <<- list(
      n = n,
      y = y,
      mu = mu,
      a = a,
      propensity = propensity,
      B1 = B1,
      B2 = B2,
      synthetic_idxs = synthetic_idxs,
      synthetic_list = synthetic_list,
      statistics = statistics,
      stage_1_statistics = stage_1_statistics,
      stage_2_statistics = statistics[stage_2_positions],
      central_trt_idxs = synthetic_list[[central_position]] + 1L,
      tail_trt_idxs = synthetic_list[[tail_position]] + 1L,
      tail_side = tail_side,
      geometry = .empirical_crt_center_geometry(a, propensity)
    )
    fixture
  }
})

.run_empirical_crt_low_level <- function(fixture, ...) {
  arguments <- list(
    y = fixture$y,
    mu = fixture$mu,
    a = fixture$a,
    fitted_probabilities = fixture$propensity,
    trt_idxs = fixture$tail_trt_idxs,
    n_trt = length(fixture$tail_trt_idxs),
    synthetic_idxs = fixture$synthetic_idxs,
    B1 = fixture$B1,
    B2 = fixture$B2,
    return_resampling_dist = TRUE,
    side_code = fixture$tail_side,
    max_iterations = 60L
  )
  do.call(
    run_low_level_test_full_crt_spa_empirical_v1,
    utils::modifyList(arguments, list(...))
  )
}

test_that("empirical CRT sparse statistics equal the dense definition", {
  set.seed(20260831)
  n <- 41L
  a <- rnorm(n) + seq_len(n) / n
  propensity <- plogis(seq(-1.7, 0.8, length.out = n))
  zero_based_indices <- sort(sample.int(n, 13L)) - 1L

  expected <- .empirical_crt_dense_statistic(
    a, propensity, zero_based_indices
  )
  observed <- compute_observed_empirical_crt_statistic_v1(
    a, propensity, zero_based_indices + 1L
  )
  empty_expected <- .empirical_crt_dense_statistic(
    a, propensity, integer()
  )
  empty_observed <- compute_observed_empirical_crt_statistic_v1(
    a, propensity, integer()
  )

  expect_equal(observed, expected, tolerance = 2e-13)
  expect_true(is.finite(empty_expected))
  expect_equal(empty_observed, empty_expected, tolerance = 2e-13)
})

test_that("empirical CRT statistic respects scale, reflection, and row order", {
  set.seed(20260901)
  n <- 53L
  a <- rnorm(n)
  propensity <- plogis(rnorm(n, mean = -0.7, sd = 0.45))
  trt_idxs <- sort(sample.int(n, 17L))
  baseline <- compute_observed_empirical_crt_statistic_v1(
    a, propensity, trt_idxs
  )

  expect_equal(
    compute_observed_empirical_crt_statistic_v1(
      7.25 * a, propensity, trt_idxs
    ),
    baseline,
    tolerance = 2e-13
  )
  expect_equal(
    compute_observed_empirical_crt_statistic_v1(
      1e-8 * a, propensity, trt_idxs
    ),
    baseline,
    tolerance = 2e-13
  )
  expect_equal(
    compute_observed_empirical_crt_statistic_v1(
      -a, propensity, trt_idxs
    ),
    -baseline,
    tolerance = 2e-13
  )

  permutation <- sample.int(n)
  permuted_trt_idxs <- match(trt_idxs, permutation)
  expect_equal(
    compute_observed_empirical_crt_statistic_v1(
      a[permutation], propensity[permutation], permuted_trt_idxs
    ),
    baseline,
    tolerance = 2e-13
  )
})

test_that("every sparse CRT resample is standardized by its own variance", {
  set.seed(20260902)
  n <- 73L
  a <- rnorm(n) + 0.35 * sin(seq_len(n) / 3)
  propensity <- plogis(seq(-1.9, 0.4, length.out = n))
  B <- 67L
  synthetic_idxs <- crt_index_sampler_fast(propensity, B)
  synthetic_list <- synth_idx_list_to_r_list(synthetic_idxs)
  expected <- .empirical_crt_bank_statistics(
    a, propensity, synthetic_list
  )
  own_variances <- vapply(
    synthetic_list,
    function(indices) {
      .empirical_crt_dense_components(a, propensity, indices)$variance
    },
    numeric(1)
  )

  observed <- compute_null_empirical_crt_statistics_v1(
    a, propensity, 0L, B, synthetic_idxs
  )
  start_pos <- 11L
  subset_B <- 23L
  observed_subset <- compute_null_empirical_crt_statistics_v1(
    a, propensity, start_pos, subset_B, synthetic_idxs
  )

  expect_true(all(is.finite(expected)))
  expect_gt(length(unique(signif(own_variances, 12))), B / 2)
  expect_equal(observed, expected, tolerance = 3e-13)
  expect_equal(
    observed_subset,
    expected[start_pos + seq_len(subset_B)],
    tolerance = 3e-13
  )
})

test_that("sparse CRT resampling safely handles empty treatment draws", {
  n <- 29L
  a <- cos(seq_len(n) / 4) + seq_len(n) / n
  propensity <- rep(1e-6, n)
  B <- 16L
  synthetic_idxs <- crt_index_sampler_fast(propensity, B)
  synthetic_list <- synth_idx_list_to_r_list(synthetic_idxs)
  expect_true(any(lengths(synthetic_list) == 0L))

  expected <- .empirical_crt_bank_statistics(
    a, propensity, synthetic_list
  )
  observed <- compute_null_empirical_crt_statistics_v1(
    a, propensity, 0L, B, synthetic_idxs
  )
  expect_true(all(is.finite(expected)))
  expect_equal(observed, expected, tolerance = 3e-13)
})

test_that("empirical CRT-SPA exposes a certified two-moment root", {
  fixture <- .make_empirical_crt_fixture()
  target <- fixture$geometry$center +
    1.5 * fixture$geometry$standard_deviation
  result <- crt_empirical_spa_full_cpp(
    fixture$a,
    fixture$propensity,
    target,
    tolerance = 1e-10,
    max_iterations = 80L
  )

  expect_true(result$converged)
  expect_identical(result$reason, "ok")
  expect_identical(result$statistic_id, "empirical_studentized_crt_v1")
  expect_identical(result$equation_id, "crt_studentized_reduced_root_v1")
  expect_identical(result$outer_dimension, 2L)
  expect_equal(result$target, target, tolerance = 1e-14)
  expect_equal(result$center, fixture$geometry$center, tolerance = 2e-13)
  expect_equal(
    result$center_sd,
    fixture$geometry$standard_deviation,
    tolerance = 2e-12
  )
  expect_length(result$state, 2L)
  expect_length(result$moment, 2L)
  expect_true(is.finite(result$p_value))
  expect_gte(result$p_value, 0)
  expect_lte(result$p_value, 1)
  expect_lte(result$max_residual, 1e-8)

  independent <- .empirical_crt_cgf(
    result$state, fixture$a, fixture$propensity
  )
  s <- independent$moment[[1L]]
  q <- independent$moment[[2L]]
  kappa <- 1 / target^2 + 1 / fixture$n
  root_residual <- c(
    q - kappa * s^2,
    result$state[[1L]] + 2 * kappa * s * result$state[[2L]]
  )
  expect_equal(result$moment, independent$moment, tolerance = 2e-10)
  expect_equal(result$hessian, independent$hessian, tolerance = 2e-10)
  expect_equal(result$K, independent$cgf, tolerance = 2e-10)
  expect_equal(result$rate, independent$rate, tolerance = 2e-10)
  expect_equal(result$outer_residual, root_residual, tolerance = 2e-10)
  expect_equal(
    s / sqrt(q - s^2 / fixture$n),
    target,
    tolerance = 2e-9
  )
  expect_lt(max(abs(root_residual)), 2e-8 * max(1, abs(q)))

  q_lr <- sqrt(drop(crossprod(
    result$state,
    independent$hessian %*% result$state
  )))
  r_lr <- sqrt(2 * independent$rate)
  independent_p <- pnorm(r_lr, lower.tail = FALSE) +
    dnorm(r_lr) * (1 / q_lr - 1 / r_lr)
  expect_equal(result$r_lr, r_lr, tolerance = 2e-10)
  expect_equal(result$q_lr, q_lr, tolerance = 2e-10)
  expect_equal(result$p_value, independent_p, tolerance = 2e-10)
})

test_that("empirical CRT-SPA is invariant to reflection, scale, and order", {
  fixture <- .make_empirical_crt_fixture()
  target <- fixture$geometry$center +
    1.35 * fixture$geometry$standard_deviation
  baseline <- crt_empirical_spa_full_cpp(
    fixture$a, fixture$propensity, target,
    tolerance = 1e-10, max_iterations = 80L
  )
  reflected <- crt_empirical_spa_full_cpp(
    -fixture$a, fixture$propensity, target,
    score_sign = -1L, tolerance = 1e-10, max_iterations = 80L
  )
  scaled <- crt_empirical_spa_full_cpp(
    3.5 * fixture$a, fixture$propensity, target,
    tolerance = 1e-10, max_iterations = 80L
  )
  tiny_scaled <- crt_empirical_spa_full_cpp(
    1e-8 * fixture$a, fixture$propensity, target,
    tolerance = 1e-10, max_iterations = 80L
  )
  set.seed(20260903)
  permutation <- sample.int(fixture$n)
  reordered <- crt_empirical_spa_full_cpp(
    fixture$a[permutation],
    fixture$propensity[permutation],
    target,
    tolerance = 1e-10,
    max_iterations = 80L
  )

  expect_true(all(vapply(
    list(baseline, reflected, scaled, tiny_scaled, reordered),
    function(result) isTRUE(result$converged),
    logical(1)
  )))
  for (candidate in list(reflected, scaled, tiny_scaled, reordered)) {
    expect_equal(candidate$p_value, baseline$p_value, tolerance = 2e-9)
    expect_equal(candidate$rate, baseline$rate, tolerance = 2e-9)
    expect_equal(candidate$r_lr, baseline$r_lr, tolerance = 2e-9)
    expect_equal(candidate$q_lr, baseline$q_lr, tolerance = 2e-9)
  }
})

test_that("empirical CRT-SPA diagnoses deterministic-propensity cells", {
  fixture <- .make_empirical_crt_fixture()
  mixed_propensity <- fixture$propensity
  mixed_propensity[c(1L, 2L)] <- c(0, 1)
  geometry <- .empirical_crt_center_geometry(
    fixture$a, mixed_propensity
  )
  target <- geometry$center + 1.35 * geometry$standard_deviation
  result <- crt_empirical_spa_full_cpp(
    fixture$a,
    mixed_propensity,
    target,
    tolerance = 1e-10,
    max_iterations = 80L
  )

  expect_false(result$converged)
  expect_false(result$root_converged)
  expect_identical(result$reason, "non_interior_propensity")
  expect_true(is.na(result$p_value))
  expect_identical(result$statistic_id, "empirical_studentized_crt_v1")
})

test_that("empirical CRT native entry points reject malformed inputs", {
  a <- c(-1.2, 0.4, 1.1, 2.3)
  propensity <- c(0.1, 0.25, 0.6, 0.8)

  expect_error(
    compute_observed_empirical_crt_statistic_v1(
      a[-1L], propensity, 1L
    ),
    "length|dimension"
  )
  endpoint_propensity <- c(0, 0.25, 0.6, 1)
  expect_equal(
    compute_observed_empirical_crt_statistic_v1(
      a, endpoint_propensity, c(2L, 4L)
    ),
    .empirical_crt_dense_statistic(
      a, endpoint_propensity, c(1L, 3L)
    ),
    tolerance = 2e-13
  )
  endpoint_bank <- crt_index_sampler_fast(endpoint_propensity, 7L)
  endpoint_list <- synth_idx_list_to_r_list(endpoint_bank)
  expect_equal(
    compute_null_empirical_crt_statistics_v1(
      a, endpoint_propensity, 0L, 7L, endpoint_bank
    ),
    .empirical_crt_bank_statistics(
      a, endpoint_propensity, endpoint_list
    ),
    tolerance = 2e-13
  )
  expect_error(
    compute_observed_empirical_crt_statistic_v1(a, propensity, 0L),
    "index|indices"
  )
  expect_error(
    compute_observed_empirical_crt_statistic_v1(a, propensity, 5L),
    "index|indices"
  )
  expect_error(
    crt_empirical_spa_full_cpp(
      a, propensity, 1.5, score_sign = 0L
    ),
    "score_sign"
  )
  expect_error(
    crt_empirical_spa_full_cpp(
      a, replace(propensity, 1L, -0.01), 1.5
    ),
    "probab|propensity"
  )
  expect_error(
    crt_empirical_spa_full_cpp(a, propensity, -1),
    "target"
  )
  expect_error(
    crt_empirical_spa_full_cpp(a, propensity, 1.5, tolerance = 0),
    "tolerance"
  )
  expect_error(
    crt_empirical_spa_full_cpp(
      a, propensity, 1.5, max_iterations = -1L
    ),
    "max_iterations"
  )

  degenerate <- compute_observed_empirical_crt_statistic_v1(
    rep(1, 8L), rep(0.5, 8L), seq_len(8L)
  )
  expect_true(is.na(degenerate) || is.nan(degenerate))
})

test_that("the empirical low-level path uses its statistic at stage one", {
  fixture <- .make_empirical_crt_fixture()
  result <- .run_empirical_crt_low_level(
    fixture,
    trt_idxs = fixture$central_trt_idxs,
    n_trt = length(fixture$central_trt_idxs),
    side_code = 1L
  )
  expected_observed <- .empirical_crt_dense_statistic(
    fixture$a,
    fixture$propensity,
    fixture$central_trt_idxs - 1L
  )

  expect_identical(result$stage, 1L)
  expect_identical(result$p_value_source, "B1_empirical")
  expect_equal(result$z_orig, expected_observed, tolerance = 3e-13)
  expect_equal(
    result$resampling_dist,
    fixture$stage_1_statistics,
    tolerance = 3e-13
  )
  expect_equal(
    result$p,
    compute_empirical_p_value(
      fixture$stage_1_statistics, expected_observed, 1L
    )
  )
  expect_false(result$spa_attempted)
  expect_true(is.na(result$spa_converged))
  expect_identical(result$spa_reason, "not_attempted")
  expect_identical(result$statistic_id, "empirical_studentized_crt_v1")
})

test_that("the empirical low-level path uses Full Newton at stage two", {
  fixture <- .make_empirical_crt_fixture()
  result <- .run_empirical_crt_low_level(fixture)

  expect_identical(result$stage, 2L)
  expect_identical(
    result$p_value_source,
    "crt_spa_empirical_directional"
  )
  expect_true(result$spa_attempted)
  expect_true(result$spa_converged)
  expect_identical(result$spa_reason, "ok")
  expect_identical(result$spa_tail_geometry,
    "directional_tangent_halfspace_lugannani_rice"
  )
  expect_true(result$spa_experimental)
  expect_length(result$resampling_dist, fixture$B1)
  expect_equal(
    result$resampling_dist,
    fixture$stage_1_statistics,
    tolerance = 3e-13
  )
  expect_true(is.finite(result$p))
  expect_gte(result$p, 0)
  expect_lte(result$p, 1)
  expect_lte(result$spa_max_residual, 1e-5)
})

test_that("forced empirical CRT-SPA failure uses the independent B2 bank", {
  fixture <- .make_empirical_crt_fixture()
  result <- .run_empirical_crt_low_level(
    fixture,
    max_iterations = 0L
  )
  expected_observed <- .empirical_crt_dense_statistic(
    fixture$a,
    fixture$propensity,
    fixture$tail_trt_idxs - 1L
  )

  expect_identical(result$stage, 3L)
  expect_identical(result$p_value_source, "B2_empirical")
  expect_true(result$spa_attempted)
  expect_false(result$spa_converged)
  expect_identical(
    result$spa_reason,
    "solver_disabled_max_iterations_zero"
  )
  expect_equal(result$z_orig, expected_observed, tolerance = 3e-13)
  expect_equal(
    result$resampling_dist,
    fixture$stage_2_statistics,
    tolerance = 3e-13
  )
  expect_equal(
    result$p,
    compute_empirical_p_value(
      fixture$stage_2_statistics,
      expected_observed,
      fixture$tail_side
    )
  )

  reflected <- .run_empirical_crt_low_level(
    fixture,
    a = -fixture$a,
    side_code = -fixture$tail_side,
    max_iterations = 0L
  )
  expect_identical(reflected$stage, 3L)
  expect_identical(reflected$p_value_source, "B2_empirical")
  expect_equal(reflected$z_orig, -result$z_orig, tolerance = 3e-13)
  expect_equal(
    reflected$resampling_dist,
    -result$resampling_dist,
    tolerance = 3e-13
  )
  expect_equal(reflected$p, result$p, tolerance = 3e-13)
})

test_that("non-interior propensity falls back with empirical statistics", {
  fixture <- .make_empirical_crt_fixture()
  propensity <- fixture$propensity
  propensity[c(1L, 2L)] <- c(0, 1)
  synthetic_idxs <- crt_index_sampler_fast(
    propensity, fixture$B1 + fixture$B2
  )
  synthetic_list <- synth_idx_list_to_r_list(synthetic_idxs)
  statistics <- .empirical_crt_bank_statistics(
    fixture$a, propensity, synthetic_list
  )
  stage_1 <- statistics[seq_len(fixture$B1)]
  stage_2_positions <- fixture$B1 + seq_len(fixture$B2)
  right_candidates <- stage_2_positions[
    statistics[stage_2_positions] > max(stage_1)
  ]
  left_candidates <- stage_2_positions[
    statistics[stage_2_positions] < min(stage_1)
  ]
  if (length(right_candidates)) {
    observed_position <- right_candidates[
      which.max(statistics[right_candidates])
    ]
    side <- 1L
  } else if (length(left_candidates)) {
    observed_position <- left_candidates[
      which.min(statistics[left_candidates])
    ]
    side <- -1L
  } else {
    stop("The deterministic endpoint fixture has no B2 tail draw.")
  }
  trt_idxs <- synthetic_list[[observed_position]] + 1L

  result <- run_low_level_test_full_crt_spa_empirical_v1(
    y = fixture$y,
    mu = fixture$mu,
    a = fixture$a,
    fitted_probabilities = propensity,
    trt_idxs = trt_idxs,
    n_trt = length(trt_idxs),
    synthetic_idxs = synthetic_idxs,
    B1 = fixture$B1,
    B2 = fixture$B2,
    return_resampling_dist = TRUE,
    side_code = side,
    max_iterations = 60L
  )
  expected_observed <- .empirical_crt_dense_statistic(
    fixture$a, propensity, trt_idxs - 1L
  )
  stage_2 <- statistics[stage_2_positions]

  expect_identical(result$stage, 3L)
  expect_identical(result$p_value_source, "B2_empirical")
  expect_true(result$spa_attempted)
  expect_false(result$spa_converged)
  expect_identical(result$spa_reason, "non_interior_propensity")
  expect_equal(result$z_orig, expected_observed, tolerance = 3e-13)
  expect_equal(result$resampling_dist, stage_2, tolerance = 3e-13)
  expect_equal(
    result$p,
    compute_empirical_p_value(stage_2, expected_observed, side)
  )
})

test_that("the low-level empirical path reports a degenerate observed V", {
  n <- 20L
  a <- rep(1, n)
  propensity <- rep(0.5, n)
  trt_idxs <- seq_len(n)
  synthetic_idxs <- crt_index_sampler_fast(propensity, 12L)

  result <- run_low_level_test_full_crt_spa_empirical_v1(
    y = rep(2, n),
    mu = rep(1, n),
    a = a,
    fitted_probabilities = propensity,
    trt_idxs = trt_idxs,
    n_trt = length(trt_idxs),
    synthetic_idxs = synthetic_idxs,
    B1 = 6L,
    B2 = 6L,
    return_resampling_dist = TRUE,
    side_code = 1L,
    max_iterations = 60L
  )

  expect_true(is.na(result$z_orig))
  expect_true(is.na(result$p))
  expect_identical(result$stage, 1L)
  expect_identical(
    result$p_value_source,
    "invalid_observed_studentizer"
  )
  expect_false(result$spa_attempted)
  expect_true(is.na(result$spa_converged))
  expect_identical(result$spa_reason, "invalid_observed_studentizer")
})

test_that("crt_spa_empirical is opt-in and leaves defaults unchanged", {
  expect_identical(
    formals(set_analysis_parameters)$resampling_approximation,
    "skew_normal"
  )

  set.seed(20260904)
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

  defaults <- set_analysis_parameters(
    imported,
    discovery_pairs = empty_pairs,
    resampling_mechanism = "crt"
  )
  expect_identical(defaults@resampling_approximation, "skew_normal")

  configured <- set_analysis_parameters(
    imported,
    discovery_pairs = empty_pairs,
    resampling_mechanism = "crt",
    resampling_approximation = "crt_spa_empirical"
  )
  expect_identical(
    configured@resampling_approximation,
    "crt_spa_empirical"
  )
  expect_identical(configured@B1, 499L)
  expect_identical(configured@B2, 4999L)
  expect_identical(configured@B3, 0L)
  expect_false(configured@run_permutations)

  expect_error(
    set_analysis_parameters(
      imported,
      discovery_pairs = empty_pairs,
      resampling_mechanism = "permutations",
      resampling_approximation = "crt_spa_empirical"
    ),
    "available only.*resampling_mechanism = 'crt'"
  )
})
