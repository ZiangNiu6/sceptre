.always_empirical_crt_components <- function(a, propensity, trt_idxs) {
  exposure <- numeric(length(a))
  exposure[trt_idxs] <- 1
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

.always_empirical_crt_fixture <- local({
  fixture <- NULL

  function() {
    if (!is.null(fixture)) return(fixture)

    set.seed(20260905)
    n <- 127L
    z <- as.numeric(scale(seq_len(n) + rnorm(n, sd = 7)))
    propensity <- plogis(-0.85 + 0.55 * z)
    a <- 0.45 * z + sin(seq_len(n) / 5) + rnorm(n, sd = 0.55)
    mu <- exp(0.25 + 0.12 * z)
    y <- mu + 0.5 + abs(a)

    candidate_bank <- crt_index_sampler_fast(propensity, 1024L)
    candidate_list <- synth_idx_list_to_r_list(candidate_bank)
    candidate_statistics <- vapply(
      candidate_list,
      function(zero_based_indices) {
        .always_empirical_crt_components(
          a, propensity, zero_based_indices + 1L
        )$statistic
      },
      numeric(1)
    )
    usable <- which(
      is.finite(candidate_statistics) &
        candidate_statistics > 0.75 &
        candidate_statistics < 2.5
    )
    if (!length(usable)) {
      stop("The deterministic always-SPA fixture has no usable tail draw.")
    }
    selected <- usable[
      which.min(abs(candidate_statistics[usable] - 1.5))
    ]
    trt_idxs <- candidate_list[[selected]] + 1L

    fixture <<- list(
      y = y,
      mu = mu,
      a = a,
      propensity = propensity,
      trt_idxs = trt_idxs,
      z_orig = candidate_statistics[[selected]]
    )
    fixture
  }
})

.run_always_empirical_spa <- function(fixture, ...) {
  arguments <- list(
    y = fixture$y,
    mu = fixture$mu,
    a = fixture$a,
    fitted_probabilities = fixture$propensity,
    trt_idxs = fixture$trt_idxs,
    n_trt = length(fixture$trt_idxs),
    side_code = 1L,
    max_iterations = 60L
  )
  do.call(
    run_low_level_test_full_crt_spa_empirical_always_v1,
    utils::modifyList(arguments, list(...))
  )
}

.always_mock_response_matrix <- function(response_ids, n_cells = 4L) {
  dense <- matrix(
    seq_len(length(response_ids) * n_cells),
    nrow = length(response_ids),
    dimnames = list(response_ids, NULL)
  )
  methods::as(Matrix::Matrix(dense, sparse = TRUE), "RsparseMatrix")
}

.always_mock_result <- function(needs_empirical_fallback) {
  list(
    p = if (needs_empirical_fallback) NA_real_ else 0.012,
    z_orig = 1.7,
    lfc = 0.2,
    stage = 2L,
    sn_params = rep(NA_real_, 3L),
    p_value_source = if (needs_empirical_fallback) {
      "B2_empirical_pending"
    } else {
      "crt_spa_empirical_always_directional"
    },
    spa_reason = if (needs_empirical_fallback) {
      "forced_test_failure"
    } else {
      "ok"
    },
    needs_empirical_fallback = needs_empirical_fallback,
    resampling_dist = numeric()
  )
}

test_that("crt_spa_empirical_always configures only its own resampling banks", {
  expect_identical(
    formals(set_analysis_parameters)$resampling_approximation,
    "skew_normal"
  )

  set.seed(20260906)
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
    resampling_approximation = "crt_spa_empirical_always"
  )
  expect_identical(
    always@resampling_approximation,
    "crt_spa_empirical_always"
  )
  expect_identical(always@B1, 0L)
  expect_identical(always@B2, 4999L)
  expect_identical(always@B3, 0L)

  hybrid <- set_analysis_parameters(
    imported,
    discovery_pairs = empty_pairs,
    resampling_mechanism = "crt",
    resampling_approximation = "crt_spa_empirical"
  )
  expect_identical(hybrid@resampling_approximation, "crt_spa_empirical")
  expect_identical(hybrid@B1, 499L)
  expect_identical(hybrid@B2, 4999L)
  expect_identical(hybrid@B3, 0L)

  expect_error(
    set_analysis_parameters(
      imported,
      discovery_pairs = empty_pairs,
      resampling_mechanism = "permutations",
      resampling_approximation = "crt_spa_empirical_always"
    ),
    "available only.*resampling_mechanism = 'crt'"
  )
})

test_that("always-SPA succeeds without returning an empirical bank", {
  fixture <- .always_empirical_crt_fixture()
  result <- .run_always_empirical_spa(fixture)

  expect_identical(result$stage, 2L)
  expect_identical(
    result$p_value_source,
    "crt_spa_empirical_always_directional"
  )
  expect_true(result$spa_attempted)
  expect_true(result$spa_converged)
  expect_identical(result$spa_reason, "ok")
  expect_false(result$needs_empirical_fallback)
  expect_length(result$resampling_dist, 0L)
  expect_equal(result$z_orig, fixture$z_orig, tolerance = 3e-13)
  expect_true(is.finite(result$p))
  expect_gte(result$p, 0)
  expect_lte(result$p, 1)
  expect_identical(result$statistic_id, "empirical_studentized_crt_v1")
  expect_identical(result$equation_id, "crt_studentized_reduced_root_v1")
  expect_identical(result$outer_dimension, 2L)
  expect_identical(
    result$spa_tail_geometry,
    "directional_tangent_halfspace_lugannani_rice"
  )
  expect_true(result$spa_experimental)
})

test_that("always-SPA treats exact and near-zero targets stably", {
  exact_fixture <- list(
    y = rep(2, 4L),
    mu = rep(1, 4L),
    a = c(1, 2, -1, -2),
    propensity = rep(0.5, 4L),
    trt_idxs = c(1L, 3L)
  )
  exact <- .run_always_empirical_spa(exact_fixture)

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
  near_fixture$a[[1L]] <- near_fixture$a[[1L]] + 2^-12
  near <- .run_always_empirical_spa(near_fixture)

  expect_gt(near$z_orig, 0)
  expect_lt(near$z_orig, 1e-3)
  expect_identical(near$stage, 2L)
  expect_true(near$spa_attempted)
  expect_true(near$spa_converged)
  expect_identical(near$spa_reason, "ok")
  expect_false(near$needs_empirical_fallback)
  expect_true(is.finite(near$p))
  expect_gte(near$p, 0)
  expect_lte(near$p, 1)

  very_near_fixture <- exact_fixture
  very_near_fixture$a[[1L]] <- very_near_fixture$a[[1L]] + 2^-20
  very_near <- .run_always_empirical_spa(very_near_fixture)

  expect_gt(very_near$z_orig, 0)
  expect_lt(very_near$z_orig, 1e-6)
  expect_true(very_near$spa_converged)
  expect_identical(very_near$spa_reason, "ok")
  expect_false(very_near$needs_empirical_fallback)
  expect_true(is.finite(very_near$p))
  expect_lt(abs(very_near$p - 0.5), 1e-4)
})

test_that("forced always-SPA failure finalizes against an empirical B2 bank", {
  fixture <- .always_empirical_crt_fixture()
  attempt <- .run_always_empirical_spa(
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

  set.seed(20260907)
  B2 <- 399L
  synthetic_idxs <- crt_index_sampler_fast(fixture$propensity, B2)
  expected_statistics <- compute_null_empirical_crt_statistics_v1(
    fixture$a,
    fixture$propensity,
    0L,
    B2,
    synthetic_idxs
  )
  result <- finalize_low_level_test_empirical_crt_fallback_v1(
    a = fixture$a,
    fitted_probabilities = fixture$propensity,
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

test_that("all-success always-SPA groups never call the CRT sampler", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  sampler_calls <- 0L
  finalizer_calls <- 0L
  attempt_calls <- 0L
  full_test_stat_values <- logical()
  testthat::local_mocked_bindings(
    perform_grna_precomputation = function(...) rep(0.35, 4L),
    load_csr_row = function(...) c(2, 3, 4, 5),
    compute_precomputation_pieces = function(..., full_test_stat) {
      full_test_stat_values <<- c(
        full_test_stat_values,
        full_test_stat
      )
      list(mu = rep(2, 4L), a = c(-1, -0.2, 0.3, 1.1))
    },
    run_low_level_test_full_crt_spa_empirical_always_v1 = function(...) {
      attempt_calls <<- attempt_calls + 1L
      .always_mock_result(FALSE)
    },
    crt_index_sampler_fast = function(...) {
      sampler_calls <<- sampler_calls + 1L
      stop("The all-success always-SPA path sampled unexpectedly.")
    },
    finalize_low_level_test_empirical_crt_fallback_v1 = function(...) {
      finalizer_calls <<- finalizer_calls + 1L
      stop("The all-success always-SPA path finalized unexpectedly.")
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
    use_crt_spa_always = FALSE,
    use_crt_spa_empirical = FALSE,
    use_crt_spa_empirical_always = TRUE,
    output_amount = 3L,
    response_ids = response_ids,
    response_precomputations = response_precomputations,
    covariate_matrix = matrix(1, nrow = 4L, ncol = 1L),
    get_idx_f = function(...) list(trt_idxs = 1L, n_trt = 1L),
    curr_grna_group = "target_1",
    subset_to_nt_cells = FALSE,
    all_nt_idxs = integer(),
    response_matrix = .always_mock_response_matrix(response_ids),
    side_code = 1L,
    cells_in_use = seq_len(4L)
  )

  expect_length(result, length(response_ids))
  expect_identical(attempt_calls, length(response_ids))
  expect_identical(sampler_calls, 0L)
  expect_identical(finalizer_calls, 0L)
  expect_identical(full_test_stat_values, rep(FALSE, length(response_ids)))
})

test_that("failed always-SPA pairs share one lazy B2 bank per group", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  sampler_calls <- 0L
  finalizer_calls <- 0L
  sampled_B <- integer()
  bank <- new.env(parent = emptyenv())
  finalized_banks <- list()
  testthat::local_mocked_bindings(
    perform_grna_precomputation = function(...) rep(0.35, 4L),
    perform_response_precomputation = function(...) {
      list(fitted_coefs = 0, theta = 1)
    },
    load_csr_row = function(...) c(2, 3, 4, 5),
    compute_precomputation_pieces = function(...) {
      list(mu = rep(2, 4L), a = c(-1, -0.2, 0.3, 1.1))
    },
    run_low_level_test_full_crt_spa_empirical_always_v1 = function(...) {
      .always_mock_result(TRUE)
    },
    crt_index_sampler_fast = function(fitted_probabilities, B) {
      sampler_calls <<- sampler_calls + 1L
      sampled_B <<- c(sampled_B, B)
      bank
    },
    finalize_low_level_test_empirical_crt_fallback_v1 = function(
        ..., synthetic_idxs, spa_attempt_result) {
      finalizer_calls <<- finalizer_calls + 1L
      finalized_banks[[finalizer_calls]] <<- synthetic_idxs
      utils::modifyList(
        spa_attempt_result,
        list(
          p = 0.4,
          stage = 3L,
          p_value_source = "B2_empirical",
          needs_empirical_fallback = FALSE
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
    use_crt_spa_always = FALSE,
    use_crt_spa_empirical = FALSE,
    use_crt_spa_empirical_always = TRUE,
    output_amount = 2L,
    get_idx_f = function(...) list(trt_idxs = 1L, n_trt = 1L),
    response_ids = response_ids,
    covariate_matrix = matrix(1, nrow = 4L, ncol = 1L),
    curr_grna_group = "target_1",
    all_nt_idxs = 2:4,
    response_matrix = .always_mock_response_matrix(response_ids),
    side_code = 1L,
    cells_in_use = seq_len(4L)
  )

  expect_length(result, length(response_ids))
  expect_identical(sampler_calls, 1L)
  expect_identical(sampled_B, 29L)
  expect_identical(finalizer_calls, length(response_ids))
  expect_true(all(vapply(
    finalized_banks,
    identical,
    logical(1),
    bank
  )))
  expect_true(all(vapply(
    result,
    function(value) identical(value$p_value_source, "B2_empirical"),
    logical(1)
  )))
})

test_that("the existing hybrid path still allocates and uses both banks", {
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
  testthat::local_mocked_bindings(
    perform_grna_precomputation = function(...) rep(0.35, 4L),
    load_csr_row = function(...) c(2, 3, 4, 5),
    compute_precomputation_pieces = function(...) {
      list(mu = rep(2, 4L), a = c(-1, -0.2, 0.3, 1.1))
    },
    crt_index_sampler_fast = function(fitted_probabilities, B) {
      sampled_B <<- c(sampled_B, B)
      bank
    },
    run_low_level_test_full_crt_spa_empirical_v1 = function(...) {
      hybrid_calls <<- hybrid_calls + 1L
      hybrid_arguments <<- list(...)
      .always_mock_result(FALSE)
    },
    run_low_level_test_full_crt_spa_empirical_always_v1 = function(...) {
      always_calls <<- always_calls + 1L
      stop("The hybrid mode entered the always-SPA path.")
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
    use_crt_spa = FALSE,
    use_crt_spa_always = FALSE,
    use_crt_spa_empirical = TRUE,
    use_crt_spa_empirical_always = FALSE,
    output_amount = 2L,
    response_ids = response_ids,
    response_precomputations = response_precomputations,
    covariate_matrix = matrix(1, nrow = 4L, ncol = 1L),
    get_idx_f = function(...) list(trt_idxs = 1L, n_trt = 1L),
    curr_grna_group = "target_1",
    subset_to_nt_cells = FALSE,
    all_nt_idxs = integer(),
    response_matrix = .always_mock_response_matrix(response_ids),
    side_code = 1L,
    cells_in_use = seq_len(4L)
  )

  expect_length(result, 1L)
  expect_identical(sampled_B, 18L)
  expect_identical(hybrid_calls, 1L)
  expect_identical(always_calls, 0L)
  expect_identical(hybrid_arguments$B1, 7L)
  expect_identical(hybrid_arguments$B2, 11L)
  expect_identical(hybrid_arguments$synthetic_idxs, bank)
})
