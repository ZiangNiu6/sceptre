.rpt_spa_information_statistic <- function(
    zero_based_indices, a, w, Z, information_inverse = NULL) {
  indices <- zero_based_indices + 1L
  if (!length(indices)) return(NaN)
  if (is.null(information_inverse)) {
    information_inverse <- solve(crossprod(Z, Z * w))
  }
  weighted_design_sum <- colSums(
    w[indices] * Z[indices, , drop = FALSE]
  )
  variance <- sum(w[indices]) - drop(crossprod(
    weighted_design_sum,
    information_inverse %*% weighted_design_sum
  ))
  if (!is.finite(variance) || variance <= 0) return(NaN)
  sum(a[indices]) / sqrt(variance)
}

.make_rpt_spa_fixture <- local({
  fixture <- NULL

  function() {
    if (!is.null(fixture)) return(fixture)

    set.seed(20260901)
    n <- 400L
    m <- 80L
    z <- runif(n, -1, 1)
    Z <- cbind(`(Intercept)` = 1, z = z)
    w <- 0.45 + 0.2 * (z + 1) / 2
    raw_score <- sin(5 * z) + 0.35 * cos(11 * z) + rnorm(n, sd = 0.2)
    a <- raw_score - drop(Z %*% solve(crossprod(Z), crossprod(Z, raw_score)))
    mu <- rep(2, n)
    y <- rpois(n, mu)
    information_inverse <- solve(crossprod(Z, Z * w))
    D <- compute_D_matrix(crossprod(Z, Z * w), w * Z)

    # B1 = 99 gives an inclusive two-sided tail trigger at p = 0.02.
    B1 <- 99L
    B2 <- 999L
    synthetic_idxs <- fisher_yates_samlper(n_tot = n, M = m, B = B1 + B2)
    synthetic_list <- synth_idx_list_to_r_list(synthetic_idxs)
    statistics <- vapply(
      synthetic_list,
      .rpt_spa_information_statistic,
      numeric(1),
      a = a,
      w = w,
      Z = Z,
      information_inverse = information_inverse
    )
    stopifnot(all(is.finite(statistics)))
    stopifnot(all(lengths(synthetic_list) == m))

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
      stop("The deterministic RPT-SPA fixture has no stage-2 tail candidate.")
    }

    tail_position <- B1 + candidate
    central_position <- which.min(
      abs(stage_1_statistics - stats::median(stage_1_statistics))
    )
    fixture <<- list(
      n = n,
      m = m,
      y = y,
      mu = mu,
      a = a,
      w = w,
      D = D,
      Z = Z,
      information_inverse = information_inverse,
      B1 = B1,
      B2 = B2,
      synthetic_idxs = synthetic_idxs,
      synthetic_list = synthetic_list,
      statistics = statistics,
      stage_1_statistics = stage_1_statistics,
      stage_2_statistics = stage_2_statistics,
      tail_side = tail_side,
      tail_trt_idxs = synthetic_list[[tail_position]] + 1L,
      tail_target = tail_side * statistics[[tail_position]],
      central_trt_idxs = synthetic_list[[central_position]] + 1L
    )
    fixture
  }
})

.run_rpt_spa_adaptive <- function(fixture, ...) {
  arguments <- list(
    y = fixture$y,
    mu = fixture$mu,
    a = fixture$a,
    w = fixture$w,
    D = fixture$D,
    Z = fixture$Z,
    trt_idxs = fixture$tail_trt_idxs,
    n_trt = fixture$m,
    use_all_cells = FALSE,
    synthetic_idxs = fixture$synthetic_idxs,
    B1 = fixture$B1,
    B2 = fixture$B2,
    return_resampling_dist = TRUE,
    side_code = fixture$tail_side
  )
  do.call(
    run_low_level_test_full_rpt_spa_v1,
    utils::modifyList(arguments, list(...))
  )
}

.run_rpt_spa_always <- function(fixture, ...) {
  arguments <- list(
    y = fixture$y,
    mu = fixture$mu,
    a = fixture$a,
    w = fixture$w,
    Z = fixture$Z,
    trt_idxs = fixture$tail_trt_idxs,
    n_trt = fixture$m,
    side_code = fixture$tail_side,
    max_iterations = 50L
  )
  do.call(
    run_low_level_test_full_rpt_spa_always_v1,
    utils::modifyList(arguments, list(...))
  )
}

.rpt_spa_mock_attempt <- function(needs_empirical_fallback) {
  list(
    p = if (needs_empirical_fallback) NA_real_ else 0.012,
    z_orig = 2.1,
    lfc = 0.2,
    stage = 2L,
    sn_params = rep(NA_real_, 3L),
    p_value_source = if (needs_empirical_fallback) {
      "B2_empirical_pending"
    } else {
      "rpt_spa_always"
    },
    spa_attempted = TRUE,
    spa_converged = !needs_empirical_fallback,
    spa_reason = if (needs_empirical_fallback) "forced_failure" else "ok",
    needs_empirical_fallback = needs_empirical_fallback,
    resampling_dist = numeric()
  )
}

test_that("the RPT SPA boundary equals the native full statistic", {
  fixture <- .make_rpt_spa_fixture()
  indices <- fixture$tail_trt_idxs
  weighted_design_sum <- colSums(
    fixture$w[indices] * fixture$Z[indices, , drop = FALSE]
  )
  boundary_variance <- sum(fixture$w[indices]) - drop(crossprod(
    weighted_design_sum,
    fixture$information_inverse %*% weighted_design_sum
  ))
  boundary_statistic <- sum(fixture$a[indices]) / sqrt(boundary_variance)
  projection <- rowSums(fixture$D[, indices, drop = FALSE])
  native_statistic <- sum(fixture$a[indices]) / sqrt(
    sum(fixture$w[indices]) - sum(projection^2)
  )

  expect_equal(boundary_statistic, native_statistic, tolerance = 3e-12)
  expect_equal(
    boundary_statistic,
    .rpt_spa_information_statistic(
      indices - 1L,
      fixture$a,
      fixture$w,
      fixture$Z,
      fixture$information_inverse
    ),
    tolerance = 3e-12
  )
})

test_that("RPT Full Newton enforces the count constraint and returns a probability", {
  fixture <- .make_rpt_spa_fixture()
  result <- rpt_spa_full_cpp(
    fixture$a,
    fixture$w,
    fixture$Z,
    fixture$m,
    fixture$tail_target,
    score_sign = fixture$tail_side
  )

  expect_true(result$converged)
  expect_identical(result$reason, "ok")
  expect_identical(result$path, "full_exact_conditional")
  expect_true(is.finite(result$p_value))
  expect_gte(result$p_value, 0)
  expect_lte(result$p_value, 1)
  expect_lte(result$max_residual, 1e-5)

  # The implementation profiles the count tilt and uses the intercept entry
  # of w * Z for q, avoiding a duplicate moment coordinate.
  d <- ncol(fixture$Z) + 1L
  expect_identical(result$moment_dimension, d)
  expect_identical(result$nuisance_dimension, ncol(fixture$Z))
  expect_length(result$state, d + 1L)
  gamma <- result$count_tilt
  theta <- result$state[seq_len(d)]
  features <- cbind(
    fixture$tail_side * fixture$a,
    fixture$w * fixture$Z
  )
  tilted_probabilities <- plogis(
    qlogis(fixture$m / fixture$n) + gamma + drop(features %*% theta)
  )
  expect_equal(sum(tilted_probabilities), fixture$m, tolerance = 2e-5)
  expect_equal(result$treated_count, fixture$m)
  expect_equal(result$tilted_count, fixture$m, tolerance = 2e-5)
  expect_lte(abs(result$count_residual), 2e-5)

  u <- fixture$m / fixture$n
  kl_rate <- sum(
    tilted_probabilities * log(tilted_probabilities / u) +
      (1 - tilted_probabilities) *
        log((1 - tilted_probabilities) / (1 - u))
  )
  conditional_correction <- 0.5 * log(
    sum(tilted_probabilities * (1 - tilted_probabilities)) /
      (fixture$n * u * (1 - u))
  )
  expect_equal(result$kl_rate, kl_rate, tolerance = 2e-11)
  expect_equal(
    result$conditional_rate_correction,
    conditional_correction,
    tolerance = 2e-12
  )
  expect_equal(
    result$rate,
    kl_rate + conditional_correction,
    tolerance = 2e-11
  )
})

test_that("RPT Full Newton is invariant under left-tail reflection", {
  fixture <- .make_rpt_spa_fixture()
  forward <- rpt_spa_full_cpp(
    fixture$a,
    fixture$w,
    fixture$Z,
    fixture$m,
    fixture$tail_target,
    score_sign = fixture$tail_side
  )
  reflected <- rpt_spa_full_cpp(
    -fixture$a,
    fixture$w,
    fixture$Z,
    fixture$m,
    fixture$tail_target,
    score_sign = -fixture$tail_side
  )

  expect_true(forward$converged)
  expect_true(reflected$converged)
  expect_equal(reflected$p_value, forward$p_value, tolerance = 1e-13)
  expect_equal(reflected$rate, forward$rate, tolerance = 1e-12)
  expect_equal(reflected$r_lr, forward$r_lr, tolerance = 1e-12)
  expect_equal(reflected$q_lr, forward$q_lr, tolerance = 1e-12)
})

test_that("RPT Full Newton rejects invalid fixed-count margins", {
  fixture <- .make_rpt_spa_fixture()
  arguments <- list(
    a = fixture$a,
    w = fixture$w,
    Z = fixture$Z,
    target = fixture$tail_target
  )

  for (bad_m in c(-1L, 0L, fixture$n, fixture$n + 1L)) {
    expect_error(
      do.call(rpt_spa_full_cpp, c(arguments, list(m = bad_m))),
      "strictly between zero|fixed-count margin"
    )
  }
  expect_error(
    rpt_spa_full_cpp(
      fixture$a[-1L], fixture$w, fixture$Z, fixture$m,
      fixture$tail_target
    ),
    "dimensions"
  )
  expect_error(
    rpt_spa_full_cpp(
      fixture$a, fixture$w, fixture$Z, fixture$m,
      fixture$tail_target, score_sign = 0L
    ),
    "score_sign"
  )
})

test_that("support-boundary saddles request an empirical fallback", {
  set.seed(1)
  n <- 20L
  m <- 4L
  z <- seq(-1, 1, length.out = n)
  Z <- cbind(`(Intercept)` = 1, z = z)
  w <- exp(0.2 * z)
  a <- stats::resid(stats::lm(stats::rnorm(n) ~ z))
  information_inverse <- solve(crossprod(Z, Z * w))
  assignments <- combn(n, m)
  statistics <- apply(assignments, 2L, function(indices) {
    weighted_design_sum <- colSums(
      w[indices] * Z[indices, , drop = FALSE]
    )
    variance <- sum(w[indices]) - drop(crossprod(
      weighted_design_sum,
      information_inverse %*% weighted_design_sum
    ))
    sum(a[indices]) / sqrt(variance)
  })
  maximum_position <- which.max(statistics)
  maximum_indices <- assignments[, maximum_position]
  exact_right_tail <- mean(statistics >= statistics[[maximum_position]])

  core <- rpt_spa_full_cpp(
    a, w, Z, m, statistics[[maximum_position]], score_sign = 1L
  )
  expect_equal(exact_right_tail, 1 / choose(n, m))
  expect_false(core$converged)
  expect_identical(core$reason, "nonregular_count_conditioning")
  expect_lt(core$count_variance_ratio, 1e-4)
  expect_gt(core$count_berry_esseen_ratio, 1)
  expect_true(is.na(core$p_value))

  attempt <- run_low_level_test_full_rpt_spa_always_v1(
    y = rep(1, n),
    mu = rep(1, n),
    a = a,
    w = w,
    Z = Z,
    trt_idxs = maximum_indices,
    n_trt = m,
    side_code = 1L,
    max_iterations = 50L
  )
  expect_true(attempt$needs_empirical_fallback)
  expect_identical(attempt$p_value_source, "B2_empirical_pending")
  expect_identical(attempt$spa_reason, "nonregular_count_conditioning")

  median_position <- which.min(abs(statistics - stats::median(statistics)))
  median_indices <- assignments[, median_position]
  median_attempt <- run_low_level_test_full_rpt_spa_always_v1(
    y = rep(1, n),
    mu = rep(1, n),
    a = a,
    w = w,
    Z = Z,
    trt_idxs = median_indices,
    n_trt = m,
    side_code = if (statistics[[median_position]] >= 0) 1L else -1L,
    max_iterations = 50L
  )
  expect_lt(abs(median_attempt$spa_r_lr), 0.1)
  expect_true(median_attempt$needs_empirical_fallback)
  expect_identical(
    median_attempt$spa_reason,
    "near_center_requires_empirical_fallback"
  )
})

test_that("screened RPT SPA preserves the empirical center and uses SPA in the tail", {
  fixture <- .make_rpt_spa_fixture()
  central <- .run_rpt_spa_adaptive(
    fixture,
    trt_idxs = fixture$central_trt_idxs
  )
  tail <- .run_rpt_spa_adaptive(fixture)

  expect_identical(central$stage, 1L)
  expect_identical(central$p_value_source, "B1_empirical")
  expect_length(central$resampling_dist, fixture$B1)
  expect_equal(
    central$p,
    compute_empirical_p_value(
      central$resampling_dist, central$z_orig, fixture$tail_side
    )
  )

  expect_identical(tail$stage, 2L)
  expect_identical(tail$p_value_source, "rpt_spa")
  expect_true(tail$spa_converged)
  expect_identical(tail$spa_reason, "ok")
  expect_length(tail$resampling_dist, fixture$B1)
  expect_true(is.finite(tail$p))
  expect_gte(tail$p, 0)
  expect_lte(tail$p, 1)
  expect_identical(tail$statistic_id, "information_studentized_rpt_v1")
  expect_identical(tail$equation_id, "rpt_information_conditional_kkt_v1")
})

test_that("screened RPT SPA falls back to an independent B2 bank", {
  fixture <- .make_rpt_spa_fixture()
  bad_design <- fixture$Z
  bad_design[, 1L] <- 2
  result <- .run_rpt_spa_adaptive(fixture, Z = bad_design)

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

test_that("RPT SPA-always succeeds without constructing a resampling distribution", {
  fixture <- .make_rpt_spa_fixture()
  result <- .run_rpt_spa_always(fixture)

  expect_identical(result$stage, 2L)
  expect_identical(result$p_value_source, "rpt_spa_always")
  expect_true(result$spa_attempted)
  expect_true(result$spa_converged)
  expect_identical(result$spa_reason, "ok")
  expect_false(result$needs_empirical_fallback)
  expect_length(result$resampling_dist, 0L)
  expect_true(is.finite(result$p))
  expect_gte(result$p, 0)
  expect_lte(result$p, 1)
  expect_identical(result$statistic_id, "information_studentized_rpt_v1")
  expect_identical(result$equation_id, "rpt_information_conditional_kkt_v1")
  expect_identical(
    result$spa_tail_geometry,
    "information_studentized_conditional_lugannani_rice"
  )
})

test_that("RPT SPA maps left, right, and two-sided tails consistently", {
  fixture <- .make_rpt_spa_fixture()
  right <- .run_rpt_spa_always(fixture, side_code = 1L)
  left <- .run_rpt_spa_always(fixture, side_code = -1L)
  both <- .run_rpt_spa_always(fixture, side_code = 0L)

  expect_true(all(vapply(
    list(right, left, both),
    function(result) isTRUE(result$spa_converged),
    logical(1L)
  )))
  expect_equal(right$p + left$p, 1, tolerance = 1e-12)
  expect_equal(
    both$p,
    min(1, 2 * min(left$p, right$p)),
    tolerance = 1e-12
  )

  screened_both <- .run_rpt_spa_adaptive(fixture, side_code = 0L)
  expect_identical(screened_both$stage, 2L)
  expect_identical(screened_both$p_value_source, "rpt_spa")
  expect_equal(screened_both$p, both$p, tolerance = 1e-12)
})

test_that("failed RPT SPA-always attempts finalize with a fresh B2 bank", {
  fixture <- .make_rpt_spa_fixture()
  attempt <- .run_rpt_spa_always(fixture, max_iterations = 0L)

  expect_identical(attempt$stage, 2L)
  expect_identical(attempt$p_value_source, "B2_empirical_pending")
  expect_true(attempt$needs_empirical_fallback)
  expect_false(attempt$spa_converged)
  expect_true(is.na(attempt$p))
  expect_length(attempt$resampling_dist, 0L)

  set.seed(20260902)
  B2 <- 399L
  synthetic_idxs <- fisher_yates_samlper(
    n_tot = fixture$n,
    M = fixture$m,
    B = B2
  )
  synthetic_list <- synth_idx_list_to_r_list(synthetic_idxs)
  expected_statistics <- vapply(
    synthetic_list,
    .rpt_spa_information_statistic,
    numeric(1),
    a = fixture$a,
    w = fixture$w,
    Z = fixture$Z,
    information_inverse = fixture$information_inverse
  )
  result <- finalize_low_level_test_rpt_spa_fallback_v1(
    a = fixture$a,
    w = fixture$w,
    Z = fixture$Z,
    n_trt = fixture$m,
    synthetic_idxs = synthetic_idxs,
    B2 = B2,
    return_resampling_dist = TRUE,
    side_code = fixture$tail_side,
    spa_attempt_result = attempt
  )

  expect_identical(result$stage, 3L)
  expect_identical(result$p_value_source, "B2_empirical")
  expect_false(result$needs_empirical_fallback)
  expect_false(result$spa_converged)
  expect_equal(result$resampling_dist, expected_statistics, tolerance = 3e-13)
  expect_equal(
    result$p,
    compute_empirical_p_value(
      expected_statistics, attempt$z_orig, fixture$tail_side
    )
  )
})

test_that("RPT SPA low-level entry points validate observed margins", {
  fixture <- .make_rpt_spa_fixture()

  expect_error(
    .run_rpt_spa_adaptive(
      fixture,
      trt_idxs = integer(),
      n_trt = 0L
    ),
    "between 1 and n - 1"
  )
  expect_error(
    .run_rpt_spa_adaptive(
      fixture,
      trt_idxs = seq_len(fixture$n),
      n_trt = fixture$n
    ),
    "between 1 and n - 1"
  )
  expect_error(
    .run_rpt_spa_always(
      fixture,
      trt_idxs = integer(),
      n_trt = 0L
    ),
    "between 1 and n - 1|fixed-count margin"
  )
  expect_error(
    .run_rpt_spa_always(
      fixture,
      trt_idxs = seq_len(fixture$n),
      n_trt = fixture$n
    ),
    "between 1 and n - 1|fixed-count margin"
  )
  expect_error(
    .run_rpt_spa_always(
      fixture,
      trt_idxs = c(1L, 1L),
      n_trt = 2L
    ),
    "duplicates"
  )
  expect_error(
    .run_rpt_spa_always(
      fixture,
      trt_idxs = c(0L, 2L),
      n_trt = 2L
    ),
    "one-based indices"
  )
})

test_that("screened RPT SPA rejects invalid observed studentizers", {
  n <- 20L
  m <- 5L
  observed_assignment <- c(rep(1, m), rep(0, n - m))
  Z <- cbind(`(Intercept)` = 1, assignment = observed_assignment)
  w <- rep(1, n)
  a <- seq(-1, 1, length.out = n)
  D <- compute_D_matrix(crossprod(Z, Z * w), w * Z)
  synthetic_idxs <- fisher_yates_samlper(n_tot = n, M = m, B = 28L)

  result <- run_low_level_test_full_rpt_spa_v1(
    y = rep(1, n),
    mu = rep(1, n),
    a = a,
    w = w,
    D = D,
    Z = Z,
    trt_idxs = seq_len(m),
    n_trt = m,
    use_all_cells = FALSE,
    synthetic_idxs = synthetic_idxs,
    B1 = 9L,
    B2 = 19L,
    return_resampling_dist = TRUE,
    side_code = 1L
  )

  expect_true(is.nan(result$z_orig))
  expect_true(is.na(result$p))
  expect_identical(result$stage, 1L)
  expect_identical(result$p_value_source, "invalid_observed_studentizer")
  expect_identical(result$spa_reason, "not_attempted")
  expect_length(result$resampling_dist, 9L)
})

test_that("RPT SPA options are permutation-only and use their planned banks", {
  set.seed(20260903)
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

  adaptive <- set_analysis_parameters(
    imported,
    discovery_pairs = empty_pairs,
    resampling_mechanism = "permutations",
    resampling_approximation = "rpt_spa"
  )
  expect_identical(adaptive@resampling_approximation, "rpt_spa")
  expect_identical(adaptive@B1, 499L)
  expect_identical(adaptive@B2, 4999L)
  expect_identical(adaptive@B3, 0L)
  expect_true(adaptive@run_permutations)

  always <- set_analysis_parameters(
    imported,
    discovery_pairs = empty_pairs,
    resampling_mechanism = "permutations",
    resampling_approximation = "rpt_spa_always"
  )
  expect_identical(always@resampling_approximation, "rpt_spa_always")
  expect_identical(always@B1, 0L)
  expect_identical(always@B2, 4999L)
  expect_identical(always@B3, 0L)
  expect_true(always@run_permutations)

  for (method in c("rpt_spa", "rpt_spa_always")) {
    expect_error(
      set_analysis_parameters(
        imported,
        discovery_pairs = empty_pairs,
        resampling_mechanism = "crt",
        resampling_approximation = method
      ),
      "available only.*resampling_mechanism = 'permutations'"
    )
  }
})

test_that("RPT SPA-always routing samples one cached fallback bank only on failure", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  attempt_calls <- 0L
  sampler_calls <- 0L
  finalizer_calls <- 0L
  sampled_B <- integer()
  attempt_weights <- list()
  finalized_banks <- list()
  bank <- new.env(parent = emptyenv())
  expected_weights <- c(0.5, 0.6, 0.7, 0.8)
  testthat::local_mocked_bindings(
    run_low_level_test_full_rpt_spa_always_v1 = function(...) {
      attempt_calls <<- attempt_calls + 1L
      attempt_weights[[attempt_calls]] <<- list(...)$w
      .rpt_spa_mock_attempt(attempt_calls >= 2L)
    },
    fisher_yates_samlper = function(n_tot, M, B) {
      sampler_calls <<- sampler_calls + 1L
      sampled_B <<- c(sampled_B, B)
      bank
    },
    finalize_low_level_test_rpt_spa_fallback_v1 = function(
        ..., synthetic_idxs, spa_attempt_result) {
      finalizer_calls <<- finalizer_calls + 1L
      finalized_banks[[finalizer_calls]] <<- synthetic_idxs
      spa_attempt_result$p <- 0.025
      spa_attempt_result$stage <- 3L
      spa_attempt_result$p_value_source <- "B2_empirical"
      spa_attempt_result$needs_empirical_fallback <- FALSE
      spa_attempt_result
    },
    run_low_level_test_full_v4 = function(...) {
      stop("The legacy RPT path was selected unexpectedly.")
    },
    .package = "sceptre"
  )

  grna_groups <- c("target_1", "target_2", "target_3")
  result <- sceptre:::perm_test_glm_factored_out(
    synthetic_idxs = NULL,
    B1 = 0L,
    B2 = 23L,
    B3 = 0L,
    fit_parametric_curve = FALSE,
    output_amount = 3L,
    grna_groups = grna_groups,
    expression_vector = c(1, 2, 3, 4),
    pieces_precomp = list(
      mu = rep(2, 4L),
      a = c(-1, -0.2, 0.3, 1.1),
      b = expected_weights
    ),
    get_idx_f = function(...) list(trt_idxs = c(1L, 2L), n_trt = 2L),
    side_code = 1L,
    covariate_matrix = matrix(1, nrow = 4L, ncol = 1L),
    use_rpt_spa = FALSE,
    use_rpt_spa_always = TRUE
  )

  expect_identical(attempt_calls, length(grna_groups))
  expect_identical(sampler_calls, 1L)
  expect_identical(finalizer_calls, 2L)
  expect_identical(sampled_B, 23L)
  expect_true(all(vapply(
    attempt_weights,
    identical,
    logical(1),
    expected_weights
  )))
  expect_true(all(vapply(
    finalized_banks,
    identical,
    logical(1),
    bank
  )))
  expect_identical(vapply(result, `[[`, integer(1), "stage"), c(2L, 3L, 3L))
})

test_that("RPT fallback factories keep one growing coupled bank", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  fisher_margins <- integer()
  hybrid_ranges <- list()
  testthat::local_mocked_bindings(
    fisher_yates_samlper = function(n_tot, M, B) {
      fisher_margins <<- c(fisher_margins, M)
      structure(new.env(parent = emptyenv()), margin = M)
    },
    hybrid_fisher_iwor_sampler = function(N, m, M, B) {
      hybrid_ranges[[length(hybrid_ranges) + 1L]] <<- c(N, m, M)
      structure(new.env(parent = emptyenv()), range = c(m, M))
    },
    .package = "sceptre"
  )

  fixed_factory <- sceptre:::make_rpt_spa_fallback_bank_factory(B2 = 23L)
  fixed_3 <- fixed_factory(n_cells = 100L, n_trt = 3L)
  fixed_2 <- fixed_factory(n_cells = 100L, n_trt = 2L)
  fixed_5 <- fixed_factory(n_cells = 100L, n_trt = 5L)
  fixed_4 <- fixed_factory(n_cells = 100L, n_trt = 4L)
  expect_identical(fisher_margins, c(3L, 5L))
  expect_identical(fixed_3, fixed_2)
  expect_identical(fixed_5, fixed_4)

  varying_factory <- sceptre:::make_rpt_spa_fallback_bank_factory(
    B2 = 23L, varying_n_with_fixed_controls = TRUE
  )
  varying_3 <- varying_factory(n_cells = 103L, n_trt = 3L)
  varying_5 <- varying_factory(n_cells = 105L, n_trt = 5L)
  varying_4 <- varying_factory(n_cells = 104L, n_trt = 4L)
  expect_equal(hybrid_ranges, list(c(100L, 3L, 3L), c(100L, 3L, 5L)))
  expect_false(identical(varying_3, varying_5))
  expect_identical(varying_5, varying_4)
})
