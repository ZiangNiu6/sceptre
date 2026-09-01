make_response_fit_fixture <- function(n_responses = 5L, n_cells = 120L) {
  set.seed(2187)
  z <- stats::rnorm(n_cells)
  covariate_matrix <- cbind(`(Intercept)` = 1, z = z)
  intercepts <- seq(log(1.5), log(3), length.out = n_responses)
  slopes <- seq(-0.25, 0.25, length.out = n_responses)
  dispersions <- seq(2, 10, length.out = n_responses)
  response_matrix <- vapply(seq_len(n_responses), function(i) {
    stats::rnbinom(
      n_cells,
      mu = exp(intercepts[[i]] + slopes[[i]] * z),
      size = dispersions[[i]]
    )
  }, numeric(n_cells)) |>
    t()
  rownames(response_matrix) <- paste0("response_", seq_len(n_responses))
  list(response_matrix = response_matrix, covariate_matrix = covariate_matrix)
}


test_that("the default response fit is the existing SCEPTRE fit", {
  fixture <- make_response_fit_fixture()
  expressions <- fixture$response_matrix[1L, ]

  default_fit <- perform_response_precomputation(
    expressions = expressions,
    covariate_matrix = fixture$covariate_matrix
  )
  explicit_fit <- perform_response_precomputation(
    expressions = expressions,
    covariate_matrix = fixture$covariate_matrix,
    response_fit_method = "sceptre"
  )
  poisson_fit <- stats::glm.fit(
    y = expressions,
    x = fixture$covariate_matrix,
    family = stats::poisson()
  )
  estimated_theta <- estimate_theta(
    y = expressions,
    mu = poisson_fit$fitted.values,
    dfr = poisson_fit$df.residual,
    limit = 50,
    eps = (.Machine$double.eps)^(1 / 4)
  )[[1L]]
  legacy_fit <- list(
    fitted_coefs = poisson_fit$coefficients,
    theta = max(min(estimated_theta, 1000), 0.01)
  )

  expect_identical(default_fit, explicit_fit)
  expect_identical(default_fit, legacy_fit)
  expect_named(default_fit, c("fitted_coefs", "theta"))
  expect_length(default_fit$fitted_coefs, ncol(fixture$covariate_matrix))
  expect_true(is.finite(default_fit$theta))
  expect_gte(default_fit$theta, 0.01)
  expect_lte(default_fit$theta, 1000)

  batch_fit <- perform_response_precomputations_from_matrix(
    response_matrix = fixture$response_matrix,
    response_ids = rownames(fixture$response_matrix),
    cell_indices = seq_len(ncol(fixture$response_matrix)),
    covariate_matrix = fixture$covariate_matrix,
    response_fit_method = "sceptre",
    chunk_size = 2L
  )
  scalar_fit <- lapply(seq_len(nrow(fixture$response_matrix)), function(i) {
    perform_response_precomputation(
      expressions = fixture$response_matrix[i, ],
      covariate_matrix = fixture$covariate_matrix,
      response_fit_method = "sceptre"
    )
  })
  names(scalar_fit) <- rownames(fixture$response_matrix)
  expect_identical(batch_fit, scalar_fit)
})


test_that("glmGamPoi returns chunk-invariant response caches", {
  skip_if_not_installed("glmGamPoi", minimum_version = "1.16.0")
  fixture <- make_response_fit_fixture()
  response_ids <- rownames(fixture$response_matrix)
  cell_indices <- seq_len(ncol(fixture$response_matrix))

  one_at_a_time <- perform_response_precomputations_from_matrix(
    response_matrix = fixture$response_matrix,
    response_ids = response_ids,
    cell_indices = cell_indices,
    covariate_matrix = fixture$covariate_matrix,
    response_fit_method = "glmGamPoi",
    chunk_size = 1L
  )
  chunked <- perform_response_precomputations_from_matrix(
    response_matrix = fixture$response_matrix,
    response_ids = response_ids,
    cell_indices = cell_indices,
    covariate_matrix = fixture$covariate_matrix,
    response_fit_method = "glmGamPoi",
    chunk_size = 3L
  )
  scalar <- lapply(seq_len(nrow(fixture$response_matrix)), function(i) {
    perform_response_precomputation(
      expressions = fixture$response_matrix[i, ],
      covariate_matrix = fixture$covariate_matrix,
      response_fit_method = "glmGamPoi"
    )
  })
  names(scalar) <- response_ids
  direct_fit <- glmGamPoi::glm_gp(
    data = fixture$response_matrix,
    design = fixture$covariate_matrix,
    offset = 0,
    size_factors = FALSE,
    overdispersion = TRUE,
    overdispersion_shrinkage = FALSE,
    ridge_penalty = NULL,
    do_cox_reid_adjustment = FALSE,
    subsample = FALSE,
    on_disk = FALSE,
    verbose = FALSE
  )
  direct_theta <- pmax(pmin(1 / direct_fit$overdispersions, 1000), 0.01)
  reconstructed_mu <- exp(
    direct_fit$Beta %*% t(fixture$covariate_matrix)
  )

  expect_equal(chunked, one_at_a_time, tolerance = 1e-10)
  expect_equal(one_at_a_time, scalar, tolerance = 1e-10)
  expect_equal(direct_fit$Mu, reconstructed_mu, tolerance = 1e-10)
  expect_named(chunked, response_ids)
  for (response_fit in chunked) {
    expect_named(response_fit, c("fitted_coefs", "theta"))
    expect_named(response_fit$fitted_coefs, colnames(fixture$covariate_matrix))
    expect_length(response_fit$fitted_coefs, ncol(fixture$covariate_matrix))
    expect_true(all(is.finite(response_fit$fitted_coefs)))
    expect_length(response_fit$theta, 1L)
    expect_true(is.finite(response_fit$theta))
    expect_gte(response_fit$theta, 0.01)
    expect_lte(response_fit$theta, 1000)
  }
  for (i in seq_along(response_ids)) {
    expected_coefficients <- as.numeric(direct_fit$Beta[i, ])
    names(expected_coefficients) <- colnames(fixture$covariate_matrix)
    expect_equal(chunked[[i]]$fitted_coefs, expected_coefficients, tolerance = 1e-10)
    expect_equal(chunked[[i]]$theta, direct_theta[[i]], tolerance = 1e-10)
  }
})


test_that("glmGamPoi response fits reject zero fitted means", {
  skip_if_not_installed("glmGamPoi", minimum_version = "1.16.0")
  expression_matrix <- rbind(response_bad = c(0, 0, 1, 1))
  covariate_matrix <- cbind(
    `(Intercept)` = 1,
    z = c(-1, -1, 1, 1)
  )
  raw_fit <- glmGamPoi::glm_gp(
    data = expression_matrix,
    design = covariate_matrix,
    offset = 0,
    size_factors = FALSE,
    overdispersion = TRUE,
    overdispersion_shrinkage = FALSE,
    ridge_penalty = NULL,
    do_cox_reid_adjustment = FALSE,
    subsample = FALSE,
    on_disk = FALSE,
    verbose = FALSE
  )

  expect_true(all(is.finite(raw_fit$Beta)))
  expect_true(all(is.finite(raw_fit$overdispersions)))
  expect_true(all(is.finite(raw_fit$deviances)))
  expect_true(any(raw_fit$Mu <= 0))
  expect_error(
    perform_glmgampoi_response_precomputations(
      expression_matrix = expression_matrix,
      covariate_matrix = covariate_matrix,
      response_ids = "response_bad",
      check_dependency = FALSE
    ),
    "response_bad",
    fixed = TRUE
  )
})


test_that("response fit matrix inputs are validated", {
  fixture <- make_response_fit_fixture(n_responses = 3L, n_cells = 40L)
  response_ids <- rownames(fixture$response_matrix)
  cell_indices <- seq_len(ncol(fixture$response_matrix))
  call_fit <- function(...) {
    perform_response_precomputations_from_matrix(
      response_matrix = fixture$response_matrix,
      response_ids = response_ids,
      cell_indices = cell_indices,
      covariate_matrix = fixture$covariate_matrix,
      ...
    )
  }

  expect_error(call_fit(response_fit_method = "other"), "should be one of")
  expect_error(call_fit(chunk_size = 0L), "positive integer")
  expect_error(
    perform_response_precomputations_from_matrix(
      fixture$response_matrix, c(response_ids[[1L]], response_ids[[1L]]),
      cell_indices, fixture$covariate_matrix
    ),
    "unique"
  )
  expect_error(
    perform_response_precomputations_from_matrix(
      fixture$response_matrix, "absent_response", cell_indices,
      fixture$covariate_matrix
    ),
    "absent_response"
  )
  expect_error(
    perform_response_precomputations_from_matrix(
      fixture$response_matrix, response_ids, c(1L, 1L),
      fixture$covariate_matrix[1:2, , drop = FALSE]
    ),
    "unique valid column indices"
  )
  expect_error(
    perform_response_precomputations_from_matrix(
      fixture$response_matrix, response_ids, cell_indices,
      fixture$covariate_matrix[-1L, , drop = FALSE]
    ),
    "align"
  )

  skip_if_not_installed("glmGamPoi", minimum_version = "1.16.0")
  invalid_counts <- fixture$response_matrix
  invalid_counts[2L, 1L] <- -1
  expect_error(
    perform_response_precomputations_from_matrix(
      invalid_counts, response_ids, cell_indices, fixture$covariate_matrix,
      response_fit_method = "glmGamPoi"
    ),
    response_ids[[2L]], fixed = TRUE
  )
})


test_that("glmGamPoi response-fit plans preserve order and balance bounded chunks", {
  response_ids <- sprintf("response_%02d", seq_len(40L))
  set.seed(727)
  rng_state <- .Random.seed

  plan <- plan_glmgampoi_response_fits(
    response_ids = response_ids,
    response_precomputations = list(),
    chunk_size = 16L,
    parallel = TRUE,
    n_processors = 6L
  )

  expect_identical(.Random.seed, rng_state)
  expect_identical(plan$requested_ids, response_ids)
  expect_identical(plan$cached_ids, character())
  expect_identical(plan$missing_ids, response_ids)
  expect_identical(
    plan$chunks,
    list(
      chunk_001 = response_ids[1:7],
      chunk_002 = response_ids[8:14],
      chunk_003 = response_ids[15:21],
      chunk_004 = response_ids[22:28],
      chunk_005 = response_ids[29:34],
      chunk_006 = response_ids[35:40]
    )
  )
  expect_identical(
    unname(lengths(plan$chunks)), c(7L, 7L, 7L, 7L, 6L, 6L)
  )
  expect_identical(plan$n_chunks, 6L)
  expect_identical(plan$n_workers, 6L)
  expect_identical(plan$chunk_size, 16L)
})


test_that("glmGamPoi response-fit plans skip cached and empty work without RNG", {
  response_ids <- paste0("response_", seq_len(6L))
  cached_fit_2 <- list(fitted_coefs = c(1, 2), theta = 2)
  cached_fit_5 <- list(fitted_coefs = c(3, 4), theta = 5)
  response_precomputations <- list(
    response_2 = cached_fit_2,
    response_5 = cached_fit_5
  )
  set.seed(818)
  rng_state <- .Random.seed

  partial_plan <- plan_glmgampoi_response_fits(
    response_ids = response_ids,
    response_precomputations = response_precomputations,
    chunk_size = 2L,
    parallel = TRUE,
    n_processors = 8L
  )
  all_cached_plan <- plan_glmgampoi_response_fits(
    response_ids = c("response_5", "response_2"),
    response_precomputations = response_precomputations,
    chunk_size = 16L,
    parallel = TRUE,
    n_processors = 8L
  )
  empty_plan <- plan_glmgampoi_response_fits(
    response_ids = character(),
    response_precomputations = list(),
    chunk_size = 16L,
    parallel = TRUE,
    n_processors = 8L
  )
  small_plan <- plan_glmgampoi_response_fits(
    response_ids = response_ids[1:4],
    response_precomputations = list(),
    chunk_size = 16L,
    parallel = TRUE,
    n_processors = 6L
  )

  expect_identical(.Random.seed, rng_state)
  expect_identical(partial_plan$cached_ids, c("response_2", "response_5"))
  expect_identical(
    partial_plan$missing_ids,
    c("response_1", "response_3", "response_4", "response_6")
  )
  expect_identical(
    partial_plan$chunks,
    list(
      chunk_001 = c("response_1", "response_3"),
      chunk_002 = c("response_4", "response_6")
    )
  )
  expect_identical(partial_plan$n_workers, 2L)

  expect_identical(all_cached_plan$cached_ids, c("response_5", "response_2"))
  expect_identical(all_cached_plan$missing_ids, character())
  expect_identical(
    all_cached_plan$chunks,
    stats::setNames(list(), character())
  )
  expect_identical(all_cached_plan$n_chunks, 0L)
  expect_identical(all_cached_plan$n_workers, 0L)

  expect_identical(empty_plan$requested_ids, character())
  expect_identical(empty_plan$missing_ids, character())
  expect_identical(empty_plan$chunks, stats::setNames(list(), character()))
  expect_identical(empty_plan$n_chunks, 0L)
  expect_identical(empty_plan$n_workers, 0L)
  expect_identical(small_plan$chunks, list(chunk_001 = response_ids[1:4]))
  expect_identical(small_plan$n_workers, 1L)
})


test_that("glmGamPoi fit plans execute one atomic call per chunk", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  response_ids <- paste0("response_", seq_len(5L))
  response_matrix <- matrix(
    1, nrow = length(response_ids), ncol = 8L,
    dimnames = list(response_ids, NULL)
  )
  covariate_matrix <- cbind(`(Intercept)` = 1, z = seq_len(8L))
  plan <- plan_glmgampoi_response_fits(
    response_ids = response_ids,
    response_precomputations = list(),
    chunk_size = 2L,
    parallel = FALSE,
    n_processors = 1L
  )
  calls <- list()
  dependency_checks <- 0L
  fail_on_response <- NULL
  testthat::local_mocked_bindings(
    check_glmgampoi_response_fit_dependency = function() {
      dependency_checks <<- dependency_checks + 1L
      invisible(TRUE)
    },
    perform_response_precomputations_from_matrix = function(...) {
      args <- list(...)
      calls[[length(calls) + 1L]] <<- args
      if (!is.null(fail_on_response) && fail_on_response %in% args$response_ids) {
        stop("injected fit failure")
      }
      stats::setNames(lapply(args$response_ids, function(response_id) {
        list(fitted_coefs = c(0, 0), theta = nchar(response_id))
      }), args$response_ids)
    },
    .package = "sceptre"
  )

  fitted <- execute_glmgampoi_response_fit_plan(
    plan = plan,
    response_matrix = response_matrix,
    cell_indices = seq_len(ncol(response_matrix)),
    covariate_matrix = covariate_matrix
  )

  expect_identical(names(fitted), response_ids)
  expect_identical(lapply(calls, `[[`, "response_ids"), unname(plan$chunks))
  expect_true(all(vapply(calls, function(args) {
    identical(args$check_dependency, FALSE) &&
      identical(args$check_covariates, FALSE)
  }, logical(1L))))
  expect_identical(dependency_checks, 1L)

  calls <- list()
  fail_on_response <- "response_3"
  expect_error(
    execute_glmgampoi_response_fit_plan(
      plan = plan,
      response_matrix = response_matrix,
      cell_indices = seq_len(ncol(response_matrix)),
      covariate_matrix = covariate_matrix
    ),
    "chunk_002 [response_3, response_4]: injected fit failure",
    fixed = TRUE
  )
  expect_identical(dependency_checks, 2L)
})


test_that("batched response extraction preserves order and matrix contract", {
  dense_matrix <- matrix(
    seq_len(24L),
    nrow = 4L,
    dimnames = list(
      paste0("response_", seq_len(4L)),
      paste0("cell_", seq_len(6L))
    )
  )
  rsparse_matrix <- methods::as(
    Matrix::Matrix(dense_matrix, sparse = TRUE),
    "RsparseMatrix"
  )
  response_ids <- c("response_4", "response_2", "response_1")
  cell_indices <- c(6L, 2L, 5L)
  expected <- dense_matrix[response_ids, cell_indices, drop = FALSE]
  storage.mode(expected) <- "double"
  dimnames(expected) <- list(response_ids, NULL)

  for (response_matrix in list(dense_matrix, rsparse_matrix)) {
    extracted <- load_response_expression_chunk(
      response_matrix = response_matrix,
      response_ids = response_ids,
      cell_indices = cell_indices
    )

    expect_identical(extracted, expected)
    expect_type(extracted, "double")
    expect_identical(rownames(extracted), response_ids)
    expect_null(colnames(extracted))
  }
})


test_that("response-fit cache merging is deterministic and preserves entries", {
  old_response_1 <- list(fitted_coefs = c(1, 1), theta = 1)
  carried_forward <- list(fitted_coefs = c(9, 9), theta = 9)
  existing <- list(
    response_1 = old_response_1,
    carried_forward = carried_forward
  )
  new_response_2 <- list(fitted_coefs = c(3, 3), theta = 3)
  new_response_3 <- list(fitted_coefs = c(4, 4), theta = 4)
  newly_fitted <- list(
    response_3 = new_response_3,
    response_2 = new_response_2
  )

  merged <- merge_response_fit_cache(
    response_precomputations = existing,
    new_precomputations = newly_fitted,
    requested_ids = c("response_2", "response_1", "response_3")
  )

  expect_named(
    merged,
    c("response_1", "carried_forward", "response_2", "response_3")
  )
  expect_identical(merged$response_1, old_response_1)
  expect_identical(merged$response_2, new_response_2)
  expect_identical(merged$response_3, new_response_3)
  expect_identical(merged$carried_forward, carried_forward)
  expect_identical(existing$response_1, old_response_1)
  expect_error(
    merge_response_fit_cache(
      response_precomputations = existing,
      new_precomputations = c(
        newly_fitted,
        list(response_1 = list(fitted_coefs = c(2, 2), theta = 2))
      ),
      requested_ids = c("response_2", "response_1", "response_3")
    ),
    "unexpected `response_1`",
    fixed = TRUE
  )
})


test_that("analysis phase timings have a stable schema and object attribute", {
  phase_timing <- make_analysis_phase_timings(
    runner = "run_crt_in_memory_v2",
    analysis_type = "discovery_analysis",
    response_fit_method = "glmGamPoi",
    response_fitting_scope = "global_precomputation",
    runner_elapsed_seconds = 12,
    response_fitting_elapsed_seconds = 3,
    association_elapsed_seconds = 7.5,
    n_response_ids = 40L,
    n_response_fits_requested = 40L,
    n_response_fits_created = 40L,
    response_fit_chunk_sizes = c(16L, 16L, 8L),
    response_fit_n_workers = 3L,
    association_n_workers = 6L,
    parallel = TRUE,
    n_processors = 6L
  )

  expect_s3_class(phase_timing, "data.frame")
  expect_identical(nrow(phase_timing), 1L)
  expect_identical(
    names(phase_timing),
    c(
      "timing_schema_version", "runner", "analysis_type",
      "response_fit_method", "response_fitting_scope",
      "runner_elapsed_seconds", "response_fitting_elapsed_seconds",
      "association_testing_elapsed_seconds", "runner_other_elapsed_seconds",
      "n_response_ids", "n_response_fits_requested",
      "n_response_fits_created", "response_fit_chunk_sizes",
      "response_fit_n_workers", "association_n_workers", "parallel",
      "n_processors_requested", "n_processors_resolved"
    )
  )
  expect_identical(phase_timing$timing_schema_version[[1L]], 1L)
  expect_identical(phase_timing$response_fit_chunk_sizes[[1L]], "16,16,8")
  expect_equal(phase_timing$runner_other_elapsed_seconds[[1L]], 1.5)
  expect_identical(phase_timing$response_fit_n_workers[[1L]], 3L)
  expect_identical(phase_timing$association_n_workers[[1L]], 6L)
  expect_identical(phase_timing$n_processors_requested[[1L]], "6")
  expect_identical(phase_timing$n_processors_resolved[[1L]], 6L)

  sceptre_object <- methods::new("sceptre_object")
  expect_identical(get_analysis_phase_timings(sceptre_object), list())
  updated_object <- set_analysis_phase_timing(
    sceptre_object,
    analysis_type = "discovery_analysis",
    phase_timing = phase_timing
  )
  stored_timings <- get_analysis_phase_timings(updated_object)

  expect_named(stored_timings, "discovery_analysis")
  expect_identical(stored_timings$discovery_analysis, phase_timing)
  expect_identical(
    attr(updated_object, "sceptre.analysis_phase_timings", exact = TRUE),
    stored_timings
  )
  expect_identical(
    set_analysis_phase_timing(updated_object, "power_check", NULL),
    updated_object
  )
})


test_that("response fit method metadata is backward compatible", {
  legacy_object <- methods::new("sceptre_object")
  expect_identical(get_response_fit_method(legacy_object), "sceptre")

  glmgampoi_object <- set_response_fit_method(legacy_object, "glmGamPoi")
  object_path <- tempfile(fileext = ".rds")
  on.exit(unlink(object_path), add = TRUE)
  saveRDS(glmgampoi_object, object_path)
  restored_object <- readRDS(object_path)

  expect_identical(get_response_fit_method(restored_object), "glmGamPoi")
})


test_that("changing the response fit method invalidates cached fits", {
  skip_if_not_installed("glmGamPoi", minimum_version = "1.16.0")
  set.seed(611)
  n_cells <- 50L
  grna_target_data_frame <- make_mock_grna_target_data(
    num_guides_per_target = c(1, 2),
    chr_distances = 1,
    chr_starts = 1,
    num_nt_guides = 3
  )
  response_matrix <- make_mock_response_matrices(
    num_responses = 4, num_cells = n_cells, patterns = "column"
  )
  grna_matrix <- make_mock_grna_matrices(
    grna_target_data_frame, n_cells,
    non_nt_patterns = "column", nt_patterns = "row"
  )
  empty_pairs <- data.frame(
    grna_target = character(0), response_id = character(0)
  )
  sceptre_object <- import_data(
    response_matrix = response_matrix,
    grna_matrix = grna_matrix,
    grna_target_data_frame = grna_target_data_frame,
    moi = "high"
  ) |>
    set_analysis_parameters(discovery_pairs = empty_pairs)
  cached_fit <- list(
    fitted_coefs = rep(0, ncol(sceptre_object@covariate_matrix)),
    theta = 1
  )
  sceptre_object@response_precomputations <- list(response_1 = cached_fit)

  unchanged <- set_analysis_parameters(
    sceptre_object,
    discovery_pairs = empty_pairs,
    response_fit_method = "sceptre"
  )
  changed <- set_analysis_parameters(
    sceptre_object,
    discovery_pairs = empty_pairs,
    response_fit_method = "glmGamPoi"
  )
  invalid_call <- function(response_fit_method) {
    set_analysis_parameters(
      sceptre_object,
      discovery_pairs = empty_pairs,
      response_fit_method = response_fit_method
    )
  }

  expect_identical(unchanged@response_precomputations, list(response_1 = cached_fit))
  expect_identical(changed@response_precomputations, list())
  expect_identical(get_response_fit_method(changed), "glmGamPoi")
  expect_error(invalid_call("glm.nb"), "either 'sceptre' or 'glmGamPoi'", fixed = TRUE)
  expect_error(invalid_call(c("sceptre", "glmGamPoi")), "single string", fixed = TRUE)
})


test_that("changing the control group invalidates cached response fits", {
  set.seed(612)
  n_cells <- 50L
  grna_target_data_frame <- make_mock_grna_target_data(
    num_guides_per_target = c(1, 2),
    chr_distances = 1,
    chr_starts = 1,
    num_nt_guides = 3
  )
  response_matrix <- make_mock_response_matrices(
    num_responses = 4, num_cells = n_cells, patterns = "column"
  )
  grna_matrix <- make_mock_grna_matrices(
    grna_target_data_frame, n_cells,
    non_nt_patterns = "column", nt_patterns = "row"
  )
  empty_pairs <- data.frame(
    grna_target = character(0), response_id = character(0)
  )
  sceptre_object <- import_data(
    response_matrix = response_matrix,
    grna_matrix = grna_matrix,
    grna_target_data_frame = grna_target_data_frame,
    moi = "low"
  ) |>
    set_analysis_parameters(
      discovery_pairs = empty_pairs,
      control_group = "complement"
    )
  sceptre_object@response_precomputations <- list(
    response_1 = list(
      fitted_coefs = rep(0, ncol(sceptre_object@covariate_matrix)),
      theta = 1
    )
  )

  unchanged <- set_analysis_parameters(
    sceptre_object,
    discovery_pairs = empty_pairs,
    control_group = "complement"
  )
  changed <- set_analysis_parameters(
    sceptre_object,
    discovery_pairs = empty_pairs,
    control_group = "nt_cells"
  )

  expect_length(unchanged@response_precomputations, 1L)
  expect_identical(changed@response_precomputations, list())
})


test_that("high-level analysis propagates glmGamPoi to the CRT runner", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  captured_args <- NULL
  testthat::local_mocked_bindings(
    run_crt_in_memory_v2 = function(...) {
      captured_args <<- list(...)
      list(
        result = data.frame(response_id = "response_1", p_value = 0.5),
        response_precomputations = list()
      )
    },
    run_perm_test_in_memory = function(...) {
      stop("The permutation runner was selected unexpectedly.")
    },
    .package = "sceptre"
  )

  sceptre_object <- methods::new("sceptre_object")
  sceptre_object@response_matrix <- list(matrix(
    1, nrow = 1L, ncol = 3L,
    dimnames = list("response_1", NULL)
  ))
  sceptre_object@grna_assignments <- list()
  sceptre_object@covariate_matrix <- matrix(1, nrow = 3L, ncol = 1L)
  sceptre_object@resampling_approximation <- "no_approximation"
  sceptre_object@B1 <- 1L
  sceptre_object@B2 <- 0L
  sceptre_object@B3 <- 0L
  sceptre_object@control_group_complement <- TRUE
  sceptre_object@n_nonzero_trt_thresh <- 0L
  sceptre_object@n_nonzero_cntrl_thresh <- 0L
  sceptre_object@side_code <- 0L
  sceptre_object@low_moi <- FALSE
  sceptre_object@response_precomputations <- list()
  sceptre_object@cells_in_use <- seq_len(3L)
  sceptre_object@run_permutations <- FALSE
  sceptre_object <- set_response_fit_method(sceptre_object, "glmGamPoi")

  pairs <- data.frame(
    response_id = "response_1",
    grna_group = "target_1",
    pass_qc = TRUE
  )
  run_sceptre_analysis_high_level(
    sceptre_object = sceptre_object,
    response_grna_group_pairs = pairs,
    calibration_check = FALSE,
    analysis_type = "discovery_analysis",
    output_amount = 1L,
    print_progress = FALSE,
    parallel = FALSE,
    n_processors = 1L,
    log_dir = tempdir()
  )

  expect_identical(captured_args$response_fit_method, "glmGamPoi")
  expect_identical(captured_args$response_fit_chunk_size, 16L)
})


test_that("target-specific CRT batches glmGamPoi fits on the group cell subset", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  response_ids <- c("response_1", "response_2")
  dense <- matrix(
    seq_len(12L), nrow = 2L,
    dimnames = list(response_ids, NULL)
  )
  response_matrix <- methods::as(
    Matrix::Matrix(dense, sparse = TRUE), "RsparseMatrix"
  )
  cells_in_use <- c(2L, 4L, 5L, 6L)
  covariate_matrix <- cbind(
    `(Intercept)` = 1,
    cell_marker = cells_in_use
  )
  captured_batch <- NULL
  fitted_expressions <- list()

  testthat::local_mocked_bindings(
    perform_response_precomputations_from_matrix = function(...) {
      captured_batch <<- list(...)
      stats::setNames(lapply(response_ids, function(...) {
        list(fitted_coefs = c(0, 0), theta = 1)
      }), response_ids)
    },
    perform_response_precomputation = function(...) {
      stop("The scalar response fitter was called unexpectedly.")
    },
    perform_grna_precomputation = function(...) rep(0.3, 3L),
    load_csr_row = function(..., row_idx, n_cells) {
      seq_len(n_cells) + 10 * row_idx
    },
    compute_precomputation_pieces = function(expression_vector, ...) {
      fitted_expressions[[length(fitted_expressions) + 1L]] <<- expression_vector
      list(
        mu = rep(2, length(expression_vector)),
        a = rep(0, length(expression_vector)),
        b = rep(1, length(expression_vector))
      )
    },
    run_low_level_test_full_crt_spa_always_v1 = function(...) {
      list(
        p = 0.5, z_orig = 0, lfc = 0, stage = 2L,
        p_value_source = "SPA", needs_empirical_fallback = FALSE,
        resampling_dist = numeric()
      )
    },
    crt_index_sampler_fast = function(...) {
      stop("SPA-always should not sample before a fallback.")
    },
    .package = "sceptre"
  )

  result <- discovery_ntcells_crt(
    B1 = 0L,
    B2 = 5L,
    B3 = 0L,
    fit_parametric_curve = FALSE,
    use_crt_spa = FALSE,
    use_crt_spa_always = TRUE,
    use_crt_spa_empirical = FALSE,
    use_crt_spa_empirical_always = FALSE,
    output_amount = 1L,
    get_idx_f = function(...) list(trt_idxs = 1L, n_trt = 1L),
    response_ids = response_ids,
    covariate_matrix = covariate_matrix,
    curr_grna_group = "target_1",
    all_nt_idxs = 3:4,
    response_matrix = response_matrix,
    side_code = 1L,
    cells_in_use = cells_in_use,
    response_fit_method = "glmGamPoi",
    response_fit_chunk_size = 7L
  )

  expect_length(result, length(response_ids))
  expect_identical(captured_batch$response_ids, response_ids)
  expect_identical(captured_batch$cell_indices, c(2L, 5L, 6L))
  expect_identical(captured_batch$covariate_matrix, covariate_matrix[c(1L, 3L, 4L), ])
  expect_identical(captured_batch$response_fit_method, "glmGamPoi")
  expect_identical(captured_batch$chunk_size, 7L)
  expect_identical(fitted_expressions, list(c(12, 15, 16), c(22, 25, 26)))
})


test_that("a small glmGamPoi discovery analysis completes end to end", {
  skip_if_not_installed("glmGamPoi", minimum_version = "1.16.0")

  set.seed(1814)
  n_cells <- 60L
  grna_target_data_frame <- data.frame(
    grna_id = c("g1", "g2", "nt1", "nt2"),
    grna_target = c("target_1", "target_2", "non-targeting", "non-targeting"),
    chr = c("chr1", "chr1", NA, NA),
    start = c(1, 2, NA, NA),
    end = c(2, 3, NA, NA)
  )
  grna_matrix <- matrix(
    0, nrow = nrow(grna_target_data_frame), ncol = n_cells,
    dimnames = list(grna_target_data_frame$grna_id, NULL)
  )
  grna_matrix["g1", 1:18] <- 20
  grna_matrix["g2", 19:36] <- 20
  grna_matrix["nt1", 37:48] <- 20
  grna_matrix["nt2", 49:60] <- 20
  response_matrix <- rbind(
    response_1 = stats::rnbinom(n_cells, mu = 3, size = 4),
    response_2 = stats::rnbinom(n_cells, mu = 2, size = 6)
  )
  discovery_pairs <- data.frame(
    grna_target = "target_1",
    response_id = "response_1"
  )

  sceptre_object <- import_data(
    response_matrix = response_matrix,
    grna_matrix = grna_matrix,
    grna_target_data_frame = grna_target_data_frame,
    moi = "high"
  ) |>
    set_analysis_parameters(
      discovery_pairs = discovery_pairs,
      formula_object = ~ 1,
      resampling_mechanism = "crt",
      resampling_approximation = "no_approximation",
      response_fit_method = "glmGamPoi"
    ) |>
    assign_grnas(method = "thresholding", threshold = 10) |>
    run_qc(
      response_n_umis_range = c(0, 1),
      response_n_nonzero_range = c(0, 1),
      n_nonzero_trt_thresh = 0,
      n_nonzero_cntrl_thresh = 0
    )
  sceptre_object@B1 <- 9L
  sceptre_object@B2 <- 0L
  sceptre_object@B3 <- 0L

  result <- run_discovery_analysis(
    sceptre_object,
    print_progress = FALSE,
    parallel = FALSE
  )

  expect_identical(get_response_fit_method(result), "glmGamPoi")
  expect_equal(nrow(result@discovery_result), 1L)
  expect_true(is.finite(result@discovery_result$p_value[[1L]]))
  expect_named(result@response_precomputations, "response_1")
  expect_true(is.finite(result@response_precomputations$response_1$theta))
  expect_true(all(is.finite(
    result@response_precomputations$response_1$fitted_coefs
  )))
  phase_timings <- get_analysis_phase_timings(result)
  expect_named(phase_timings, "discovery_analysis")
  expect_identical(
    phase_timings$discovery_analysis$timing_schema_version[[1L]],
    1L
  )
  expect_identical(
    phase_timings$discovery_analysis$response_fit_method[[1L]],
    "glmGamPoi"
  )
})
