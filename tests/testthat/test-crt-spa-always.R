.info_always_full_statistic <- function(zero_based_indices, a, w, D) {
  indices <- zero_based_indices + 1L
  if (!length(indices)) return(NaN)
  projection <- rowSums(D[, indices, drop = FALSE])
  variance <- sum(w[indices]) - sum(projection^2)
  if (!is.finite(variance) || variance <= 0) return(NaN)
  sum(a[indices]) / sqrt(variance)
}

.info_always_d_free_statistic <- function(
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

.info_always_tail_geometry <- function(a, w, Z, propensity) {
  information_inverse <- solve(crossprod(Z, Z * w))
  features <- cbind(a, w * Z)
  tilted_variance <- propensity * (1 - propensity)
  moment <- colSums(features * propensity)
  nuisance_moment <- moment[-1L]
  beta <- drop(information_inverse %*% nuisance_moment)
  variance <- nuisance_moment[[1L]] - drop(crossprod(
    nuisance_moment,
    beta
  ))
  variance_gradient <- c(
    0,
    c(1, rep(0, ncol(Z) - 1L)) - 2 * beta
  )
  gradient <- c(1, rep(0, ncol(Z))) / sqrt(variance) -
    0.5 * moment[[1L]] * variance_gradient / variance^(3 / 2)
  feature_covariance <- crossprod(
    features,
    features * tilted_variance
  )
  list(
    center = moment[[1L]] / sqrt(variance),
    standard_deviation = sqrt(drop(crossprod(
      gradient,
      feature_covariance %*% gradient
    )))
  )
}

.info_always_fixture <- local({
  fixture <- NULL

  function() {
    if (!is.null(fixture)) return(fixture)

    set.seed(20260908)
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
    geometry <- .info_always_tail_geometry(
      pieces$a, pieces$w, Z, propensity
    )

    candidate_bank <- crt_index_sampler_fast(propensity, 2048L)
    candidate_list <- synth_idx_list_to_r_list(candidate_bank)
    candidate_statistics <- vapply(
      candidate_list,
      .info_always_full_statistic,
      numeric(1),
      a = pieces$a,
      w = pieces$w,
      D = pieces$D
    )
    desired <- geometry$center + 2 * geometry$standard_deviation
    usable <- which(
      is.finite(candidate_statistics) &
        candidate_statistics > max(
          0,
          geometry$center + 1.25 * geometry$standard_deviation
        ) &
        candidate_statistics <
          geometry$center + 3 * geometry$standard_deviation
    )
    if (!length(usable)) {
      stop("The deterministic information always-SPA fixture has no usable tail draw.")
    }
    selected <- usable[
      which.min(abs(candidate_statistics[usable] - desired))
    ]
    trt_idxs <- candidate_list[[selected]] + 1L

    fixture <<- list(
      y = y,
      mu = pieces$mu,
      a = pieces$a,
      w = pieces$w,
      D = pieces$D,
      Z = Z,
      propensity = propensity,
      trt_idxs = trt_idxs,
      z_orig = candidate_statistics[[selected]]
    )
    fixture
  }
})

.run_info_always_spa <- function(fixture, ...) {
  arguments <- list(
    y = fixture$y,
    mu = fixture$mu,
    a = fixture$a,
    w = fixture$w,
    Z = fixture$Z,
    fitted_probabilities = fixture$propensity,
    trt_idxs = fixture$trt_idxs,
    n_trt = length(fixture$trt_idxs),
    side_code = 1L,
    max_iterations = 50L
  )
  do.call(
    run_low_level_test_full_crt_spa_always_v1,
    utils::modifyList(arguments, list(...))
  )
}

.info_always_mock_response_matrix <- function(response_ids, n_cells = 4L) {
  dense <- matrix(
    seq_len(length(response_ids) * n_cells),
    nrow = length(response_ids),
    dimnames = list(response_ids, NULL)
  )
  methods::as(Matrix::Matrix(dense, sparse = TRUE), "RsparseMatrix")
}

.info_always_mock_result <- function(needs_empirical_fallback) {
  list(
    p = if (needs_empirical_fallback) NA_real_ else 0.014,
    z_orig = 1.6,
    lfc = 0.2,
    stage = 2L,
    sn_params = rep(NA_real_, 3L),
    p_value_source = if (needs_empirical_fallback) {
      "B2_empirical_pending"
    } else {
      "crt_spa_always"
    },
    spa_converged = !needs_empirical_fallback,
    spa_reason = if (needs_empirical_fallback) {
      "forced_test_failure"
    } else {
      "ok"
    },
    needs_empirical_fallback = needs_empirical_fallback,
    resampling_dist = numeric()
  )
}

test_that("crt_spa_always is CRT-only with its own zero-B1 banks", {
  set.seed(20260909)
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

  always <- set_analysis_parameters(
    imported,
    discovery_pairs = empty_pairs,
    resampling_mechanism = "crt",
    resampling_approximation = "crt_spa_always"
  )
  expect_identical(always@resampling_approximation, "crt_spa_always")
  expect_identical(always@B1, 0L)
  expect_identical(always@B2, 4999L)
  expect_identical(always@B3, 0L)

  hybrid <- set_analysis_parameters(
    imported,
    discovery_pairs = empty_pairs,
    resampling_mechanism = "crt",
    resampling_approximation = "crt_spa"
  )
  expect_identical(hybrid@resampling_approximation, "crt_spa")
  expect_identical(hybrid@B1, 499L)
  expect_identical(hybrid@B2, 4999L)
  expect_identical(hybrid@B3, 0L)

  expect_error(
    set_analysis_parameters(
      imported,
      discovery_pairs = empty_pairs,
      resampling_mechanism = "permutations",
      resampling_approximation = "crt_spa_always"
    ),
    "available only.*resampling_mechanism = 'crt'"
  )
})

test_that("information always-SPA succeeds without returning an empirical bank", {
  fixture <- .info_always_fixture()
  result <- .run_info_always_spa(fixture)
  zero_based_indices <- fixture$trt_idxs - 1L
  d_free_statistic <- .info_always_d_free_statistic(
    zero_based_indices,
    fixture$a,
    fixture$w,
    fixture$Z
  )
  legacy_statistic <- .info_always_full_statistic(
    zero_based_indices,
    fixture$a,
    fixture$w,
    fixture$D
  )

  expect_identical(result$stage, 2L)
  expect_identical(result$p_value_source, "crt_spa_always")
  expect_true(result$spa_attempted)
  expect_true(result$spa_converged)
  expect_identical(result$spa_reason, "ok")
  expect_false(result$needs_empirical_fallback)
  expect_length(result$resampling_dist, 0L)
  expect_equal(result$z_orig, fixture$z_orig, tolerance = 3e-13)
  expect_equal(result$z_orig, d_free_statistic, tolerance = 3e-12)
  expect_equal(d_free_statistic, legacy_statistic, tolerance = 3e-12)
  expect_true(is.finite(result$p))
  expect_gte(result$p, 0)
  expect_lte(result$p, 1)
  expect_identical(result$statistic_id, "information_studentized_crt_v1")
  expect_identical(result$equation_id, "crt_information_full_kkt_v1")
  expect_identical(
    result$spa_tail_geometry,
    "information_studentized_lugannani_rice"
  )
})

test_that("information always-SPA handles exact and near centers", {
  exact_fixture <- list(
    y = rep(2, 4L),
    mu = rep(1, 4L),
    a = c(1, -1, 1, -1),
    w = rep(1, 4L),
    D = matrix(0.5, nrow = 1L, ncol = 4L),
    Z = matrix(1, nrow = 4L, ncol = 1L),
    propensity = rep(0.5, 4L),
    trt_idxs = c(1L, 2L)
  )
  exact <- .run_info_always_spa(exact_fixture)

  expect_identical(exact$z_orig, 0)
  expect_identical(exact$stage, 2L)
  expect_true(exact$spa_attempted)
  expect_false(exact$spa_converged)
  expect_identical(
    exact$spa_reason,
    "central_target_requires_empirical_fallback"
  )
  expect_identical(exact$p_value_source, "B2_empirical_pending")
  expect_true(exact$needs_empirical_fallback)
  expect_true(is.na(exact$p))

  near_fixture <- exact_fixture
  near_fixture$a[[1L]] <- near_fixture$a[[1L]] + 2^-20
  near_fixture$a[[4L]] <- near_fixture$a[[4L]] - 2^-20
  near <- .run_info_always_spa(near_fixture)

  expect_gt(near$z_orig, 0)
  expect_lt(near$z_orig, 1e-6)
  expect_true(near$spa_converged)
  expect_identical(near$spa_reason, "ok")
  expect_false(near$needs_empirical_fallback)
  expect_true(is.finite(near$p))
  expect_lt(abs(near$p - 0.5), 1e-3)
})

test_that("information always-SPA preserves endpoint reflection diagnostics", {
  fixture <- .info_always_fixture()
  endpoint_propensity <- fixture$propensity
  endpoint_propensity[[1L]] <- 0
  forward <- .run_info_always_spa(
    fixture,
    fitted_probabilities = endpoint_propensity
  )
  reflected <- .run_info_always_spa(
    fixture,
    a = -fixture$a,
    fitted_probabilities = endpoint_propensity,
    side_code = -1L
  )

  for (result in list(forward, reflected)) {
    expect_identical(result$stage, 2L)
    expect_identical(result$spa_reason, "non_interior_propensity")
    expect_identical(result$p_value_source, "B2_empirical_pending")
    expect_true(result$needs_empirical_fallback)
    expect_false(result$spa_converged)
    expect_true(is.finite(result$z_orig))
    expect_true(is.na(result$p))
  }
  expect_equal(reflected$z_orig, -forward$z_orig, tolerance = 3e-12)
})

test_that("forced information SPA failure finalizes with the B2 bank", {
  fixture <- .info_always_fixture()
  attempt <- .run_info_always_spa(
    fixture,
    max_iterations = 0L,
    use_fast = TRUE
  )

  expect_identical(attempt$stage, 2L)
  expect_identical(attempt$p_value_source, "B2_empirical_pending")
  expect_true(attempt$needs_empirical_fallback)
  expect_false(attempt$spa_converged)
  expect_true(attempt$spa_fast)
  expect_identical(
    attempt$spa_reason,
    "solver_disabled_max_iterations_zero"
  )
  expect_length(attempt$resampling_dist, 0L)
  expect_identical(
    attempt$spa_diagnostics$initial_exact_count,
    sum(fixture$y > 0)
  )
  expect_identical(
    attempt$spa_diagnostics$initial_bulk_count,
    sum(fixture$y == 0)
  )

  set.seed(20260910)
  B2 <- 399L
  synthetic_idxs <- crt_index_sampler_fast(fixture$propensity, B2)
  synthetic_list <- synth_idx_list_to_r_list(synthetic_idxs)
  legacy_statistics <- vapply(
    synthetic_list,
    .info_always_full_statistic,
    numeric(1),
    a = fixture$a,
    w = fixture$w,
    D = fixture$D
  )
  information_inverse <- solve(crossprod(
    fixture$Z,
    fixture$Z * fixture$w
  ))
  expected_statistics <- vapply(
    synthetic_list,
    .info_always_d_free_statistic,
    numeric(1),
    a = fixture$a,
    w = fixture$w,
    Z = fixture$Z,
    information_inverse = information_inverse
  )
  expect_equal(expected_statistics, legacy_statistics, tolerance = 3e-12)
  result <- finalize_low_level_test_crt_spa_fallback_v1(
    a = fixture$a,
    w = fixture$w,
    Z = fixture$Z,
    synthetic_idxs = synthetic_idxs,
    B2 = B2,
    return_resampling_dist = TRUE,
    side_code = 1L,
    spa_attempt_result = attempt
  )

  expect_identical(result$stage, 3L)
  expect_identical(result$p_value_source, "B2_empirical")
  expect_false(result$needs_empirical_fallback)
  expect_false(result$spa_converged)
  expect_true(result$spa_fast)
  expect_identical(
    result$spa_reason,
    "solver_disabled_max_iterations_zero"
  )
  expect_identical(
    result$spa_diagnostics$initial_exact_count,
    attempt$spa_diagnostics$initial_exact_count
  )
  expect_identical(
    result$spa_diagnostics$initial_bulk_count,
    attempt$spa_diagnostics$initial_bulk_count
  )
  expect_equal(
    result$resampling_dist,
    expected_statistics,
    tolerance = 3e-13
  )
  expect_equal(
    result$p,
    compute_empirical_p_value(
      expected_statistics,
      attempt$z_orig,
      1L
    )
  )
})

test_that("no-intercept information geometry falls back without losing z", {
  set.seed(20260911)
  n <- 20L
  Z <- matrix(
    as.numeric(scale(seq_len(n), center = TRUE, scale = FALSE)),
    ncol = 1L
  )
  w <- rep(1, n)
  propensity <- rep(0.8, n)
  a <- as.numeric(scale(
    sin(seq_len(n)), center = TRUE, scale = FALSE
  ))
  trt_idxs <- seq_len(8L)
  attempt <- run_low_level_test_full_crt_spa_always_v1(
    y = rep(2, n),
    mu = rep(1, n),
    a = a,
    w = w,
    Z = Z,
    fitted_probabilities = propensity,
    trt_idxs = trt_idxs,
    n_trt = length(trt_idxs),
    side_code = 1L,
    max_iterations = 50L
  )
  expected_observed <- .info_always_d_free_statistic(
    trt_idxs - 1L, a, w, Z
  )

  expect_equal(attempt$z_orig, expected_observed, tolerance = 3e-13)
  expect_identical(
    attempt$spa_reason,
    "unsupported_spa_geometry_no_intercept"
  )
  expect_identical(attempt$p_value_source, "B2_empirical_pending")
  expect_true(attempt$needs_empirical_fallback)

  B2 <- 99L
  synthetic_idxs <- crt_index_sampler_fast(propensity, B2)
  synthetic_list <- synth_idx_list_to_r_list(synthetic_idxs)
  expected_statistics <- vapply(
    synthetic_list,
    .info_always_d_free_statistic,
    numeric(1),
    a = a,
    w = w,
    Z = Z
  )
  expect_true(all(is.finite(expected_statistics)))
  result <- finalize_low_level_test_crt_spa_fallback_v1(
    a = a,
    w = w,
    Z = Z,
    synthetic_idxs = synthetic_idxs,
    B2 = B2,
    return_resampling_dist = TRUE,
    side_code = 1L,
    spa_attempt_result = attempt
  )

  expect_identical(result$stage, 3L)
  expect_identical(result$p_value_source, "B2_empirical")
  expect_false(result$needs_empirical_fallback)
  expect_equal(
    result$resampling_dist,
    expected_statistics,
    tolerance = 3e-13
  )
})

test_that("singular information geometry and malformed indices are safe", {
  Z <- matrix(1, nrow = 4L, ncol = 2L)
  a <- c(1, -1, 1, -1)
  w <- rep(1, 4L)
  propensity <- rep(0.5, 4L)
  arguments <- list(
    y = rep(2, 4L),
    mu = rep(1, 4L),
    a = a,
    w = w,
    Z = Z,
    fitted_probabilities = propensity,
    trt_idxs = c(1L, 2L),
    n_trt = 2L,
    side_code = 1L,
    max_iterations = 50L
  )
  attempt <- do.call(
    run_low_level_test_full_crt_spa_always_v1,
    arguments
  )

  expect_identical(attempt$spa_reason, "singular_information_geometry")
  expect_identical(attempt$p_value_source, "B2_empirical_pending")
  expect_true(attempt$needs_empirical_fallback)
  expect_true(is.na(attempt$p))

  set.seed(20260912)
  B2 <- 20L
  synthetic_idxs <- crt_index_sampler_fast(propensity, B2)
  result <- finalize_low_level_test_crt_spa_fallback_v1(
    a = a,
    w = w,
    Z = Z,
    synthetic_idxs = synthetic_idxs,
    B2 = B2,
    return_resampling_dist = TRUE,
    side_code = 1L,
    spa_attempt_result = attempt
  )
  expect_identical(result$stage, 3L)
  expect_identical(result$p_value_source, "invalid_B2_studentizer")
  expect_false(result$needs_empirical_fallback)
  expect_true(is.na(result$p))
  expect_true(all(is.na(result$resampling_dist)))

  expect_error(
    do.call(
      run_low_level_test_full_crt_spa_always_v1,
      utils::modifyList(arguments, list(trt_idxs = c(0L, 2L)))
    ),
    "valid one-based indices"
  )
  expect_error(
    do.call(
      run_low_level_test_full_crt_spa_always_v1,
      utils::modifyList(arguments, list(trt_idxs = c(1L, 1L)))
    ),
    "cannot contain duplicates"
  )
})

test_that("all-success information groups skip sampling and null columns", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  sampler_calls <- 0L
  finalizer_calls <- 0L
  attempt_calls <- 0L
  attempt_weights <- list()
  full_test_stat_values <- logical()
  expected_weights <- c(0.6, 0.7, 0.8, 0.9)
  testthat::local_mocked_bindings(
    perform_grna_precomputation = function(...) rep(0.35, 4L),
    load_csr_row = function(...) c(2, 3, 4, 5),
    compute_precomputation_pieces = function(..., full_test_stat) {
      full_test_stat_values <<- c(
        full_test_stat_values,
        full_test_stat
      )
      list(
        mu = rep(2, 4L),
        a = c(-1, -0.2, 0.3, 1.1),
        b = expected_weights
      )
    },
    run_low_level_test_full_crt_spa_always_v1 = function(...) {
      attempt_calls <<- attempt_calls + 1L
      attempt_weights[[attempt_calls]] <<- list(...)$w
      .info_always_mock_result(FALSE)
    },
    crt_index_sampler_fast = function(...) {
      sampler_calls <<- sampler_calls + 1L
      stop("The all-success information path sampled unexpectedly.")
    },
    finalize_low_level_test_crt_spa_fallback_v1 = function(...) {
      finalizer_calls <<- finalizer_calls + 1L
      stop("The all-success information path finalized unexpectedly.")
    },
    .package = "sceptre"
  )

  response_ids <- c("response_1", "response_2")
  response_precomputations <- stats::setNames(
    lapply(response_ids, function(...) {
      list(fitted_coefs = 0, theta = 1)
    }),
    response_ids
  )
  result <- sceptre:::crt_glm_factored_out(
    B1 = 0L,
    B2 = 23L,
    B3 = 0L,
    fit_parametric_curve = FALSE,
    use_crt_spa = FALSE,
    use_crt_spa_always = TRUE,
    use_crt_spa_empirical = FALSE,
    use_crt_spa_empirical_always = FALSE,
    output_amount = 3L,
    response_ids = response_ids,
    response_precomputations = response_precomputations,
    covariate_matrix = matrix(1, nrow = 4L, ncol = 1L),
    get_idx_f = function(...) list(trt_idxs = 1L, n_trt = 1L),
    curr_grna_group = "target_1",
    subset_to_nt_cells = FALSE,
    all_nt_idxs = integer(),
    response_matrix = .info_always_mock_response_matrix(response_ids),
    side_code = 1L,
    cells_in_use = seq_len(4L)
  )
  result_df <- sceptre:::construct_data_frame_v2(
    data.frame(response_id = response_ids),
    result,
    output_amount = 3L
  )

  expect_identical(attempt_calls, length(response_ids))
  expect_identical(sampler_calls, 0L)
  expect_identical(finalizer_calls, 0L)
  expect_identical(full_test_stat_values, rep(FALSE, length(response_ids)))
  expect_true(all(vapply(
    attempt_weights,
    identical,
    logical(1),
    expected_weights
  )))
  expect_false(any(startsWith(names(result_df), "z_null_")))
})

test_that("failed information pairs share B2 and pad mixed null rows", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  attempt_calls <- 0L
  sampler_calls <- 0L
  finalizer_calls <- 0L
  sampled_B <- integer()
  bank <- new.env(parent = emptyenv())
  finalized_banks <- list()
  finalized_weights <- list()
  expected_weights <- c(0.6, 0.7, 0.8, 0.9)
  testthat::local_mocked_bindings(
    perform_grna_precomputation = function(...) rep(0.35, 4L),
    perform_response_precomputation = function(...) {
      list(fitted_coefs = 0, theta = 1)
    },
    load_csr_row = function(...) c(2, 3, 4, 5),
    compute_precomputation_pieces = function(...) {
      list(
        mu = rep(2, 4L),
        a = c(-1, -0.2, 0.3, 1.1),
        b = expected_weights
      )
    },
    run_low_level_test_full_crt_spa_always_v1 = function(...) {
      attempt_calls <<- attempt_calls + 1L
      .info_always_mock_result(attempt_calls > 1L)
    },
    crt_index_sampler_fast = function(fitted_probabilities, B) {
      sampler_calls <<- sampler_calls + 1L
      sampled_B <<- c(sampled_B, B)
      bank
    },
    finalize_low_level_test_crt_spa_fallback_v1 = function(
        ..., synthetic_idxs, spa_attempt_result) {
      finalizer_calls <<- finalizer_calls + 1L
      finalized_banks[[finalizer_calls]] <<- synthetic_idxs
      finalized_weights[[finalizer_calls]] <<- list(...)$w
      utils::modifyList(
        spa_attempt_result,
        list(
          p = 0.4,
          stage = 3L,
          p_value_source = "B2_empirical",
          needs_empirical_fallback = FALSE,
          resampling_dist = c(-1, 0, 1)
        )
      )
    },
    .package = "sceptre"
  )

  response_ids <- c("response_1", "response_2", "response_3")
  result <- sceptre:::discovery_ntcells_crt(
    B1 = 0L,
    B2 = 29L,
    B3 = 0L,
    fit_parametric_curve = FALSE,
    use_crt_spa = FALSE,
    use_crt_spa_always = TRUE,
    use_crt_spa_empirical = FALSE,
    use_crt_spa_empirical_always = FALSE,
    output_amount = 3L,
    get_idx_f = function(...) list(trt_idxs = 1L, n_trt = 1L),
    response_ids = response_ids,
    covariate_matrix = matrix(1, nrow = 4L, ncol = 1L),
    curr_grna_group = "target_1",
    all_nt_idxs = 2:4,
    response_matrix = .info_always_mock_response_matrix(response_ids),
    side_code = 1L,
    cells_in_use = seq_len(4L)
  )
  result_df <- sceptre:::construct_data_frame_v2(
    data.frame(response_id = response_ids),
    result,
    output_amount = 3L
  )
  null_columns <- paste0("z_null_", seq_len(3L))

  expect_identical(sampler_calls, 1L)
  expect_identical(sampled_B, 29L)
  expect_identical(finalizer_calls, 2L)
  expect_true(all(vapply(
    finalized_banks,
    identical,
    logical(1),
    bank
  )))
  expect_true(all(vapply(
    finalized_weights,
    identical,
    logical(1),
    expected_weights
  )))
  expect_true(all(is.na(result_df[1L, null_columns])))
  expect_equal(
    unlist(result_df[2:3, null_columns], use.names = FALSE),
    rep(c(-1, 0, 1), each = 2L)
  )
})

test_that("the existing information hybrid remains eager and unchanged", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  sampled_B <- integer()
  hybrid_calls <- 0L
  always_calls <- 0L
  bank <- new.env(parent = emptyenv())
  hybrid_arguments <- NULL
  hybrid_full_test_stat <- logical()
  testthat::local_mocked_bindings(
    perform_grna_precomputation = function(...) rep(0.35, 4L),
    load_csr_row = function(...) c(2, 3, 4, 5),
    compute_precomputation_pieces = function(..., full_test_stat) {
      hybrid_full_test_stat <<- c(
        hybrid_full_test_stat,
        full_test_stat
      )
      list(
        mu = rep(2, 4L),
        a = c(-1, -0.2, 0.3, 1.1),
        w = rep(1, 4L),
        D = matrix(0.5, nrow = 1L, ncol = 4L)
      )
    },
    crt_index_sampler_fast = function(fitted_probabilities, B) {
      sampled_B <<- c(sampled_B, B)
      bank
    },
    run_low_level_test_full_crt_spa_v1 = function(...) {
      hybrid_calls <<- hybrid_calls + 1L
      hybrid_arguments <<- list(...)
      .info_always_mock_result(FALSE)
    },
    run_low_level_test_full_crt_spa_always_v1 = function(...) {
      always_calls <<- always_calls + 1L
      stop("The hybrid entered the always-SPA path.")
    },
    .package = "sceptre"
  )

  response_ids <- "response_1"
  response_precomputations <- list(
    response_1 = list(fitted_coefs = 0, theta = 1)
  )
  result <- sceptre:::crt_glm_factored_out(
    B1 = 7L,
    B2 = 11L,
    B3 = 0L,
    fit_parametric_curve = FALSE,
    use_crt_spa = TRUE,
    use_crt_spa_always = FALSE,
    use_crt_spa_empirical = FALSE,
    use_crt_spa_empirical_always = FALSE,
    output_amount = 2L,
    response_ids = response_ids,
    response_precomputations = response_precomputations,
    covariate_matrix = matrix(1, nrow = 4L, ncol = 1L),
    get_idx_f = function(...) list(trt_idxs = 1L, n_trt = 1L),
    curr_grna_group = "target_1",
    subset_to_nt_cells = FALSE,
    all_nt_idxs = integer(),
    response_matrix = .info_always_mock_response_matrix(response_ids),
    side_code = 1L,
    cells_in_use = seq_len(4L)
  )

  expect_length(result, 1L)
  expect_identical(sampled_B, 18L)
  expect_identical(hybrid_calls, 1L)
  expect_identical(always_calls, 0L)
  expect_identical(hybrid_full_test_stat, TRUE)
  expect_identical(hybrid_arguments$B1, 7L)
  expect_identical(hybrid_arguments$B2, 11L)
  expect_identical(hybrid_arguments$synthetic_idxs, bank)
})
