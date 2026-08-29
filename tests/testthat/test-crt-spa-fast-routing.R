.fast_spa_mock_import <- function() {
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
  import_data(
    response_matrix = response_matrix,
    grna_matrix = grna_matrix,
    grna_target_data_frame = grna_target_data_frame,
    moi = "high"
  )
}

test_that("fast SPA options mirror the banks of the exact variants", {
  set.seed(20260911)
  imported <- .fast_spa_mock_import()
  empty_pairs <- data.frame(
    grna_target = character(),
    response_id = character()
  )
  expected <- list(
    crt_spa_fast = c(B1 = 499L, B2 = 4999L, B3 = 0L),
    crt_spa_always_fast = c(B1 = 0L, B2 = 4999L, B3 = 0L),
    crt_spa_empirical_fast = c(B1 = 499L, B2 = 4999L, B3 = 0L),
    crt_spa_empirical_always_fast = c(B1 = 0L, B2 = 4999L, B3 = 0L)
  )

  for (method in names(expected)) {
    configured <- set_analysis_parameters(
      imported,
      discovery_pairs = empty_pairs,
      resampling_mechanism = "crt",
      resampling_approximation = method
    )
    expect_identical(configured@resampling_approximation, method)
    expect_identical(
      c(B1 = configured@B1, B2 = configured@B2, B3 = configured@B3),
      expected[[method]],
      info = method
    )
    expect_error(
      set_analysis_parameters(
        imported,
        discovery_pairs = empty_pairs,
        resampling_mechanism = "permutations",
        resampling_approximation = method
      ),
      "available only.*resampling_mechanism = 'crt'",
      info = method
    )
  }
})

test_that("fast SPA routing reaches the matching low-level wrapper", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  response_id <- "response_1"
  dense <- matrix(1:4, nrow = 1L, dimnames = list(response_id, NULL))
  response_matrix <- methods::as(
    Matrix::Matrix(dense, sparse = TRUE), "RsparseMatrix"
  )
  calls <- character()
  fast_flags <- logical()
  sampler_calls <- 0L
  full_stat_flags <- logical()
  success <- function(source) {
    list(
      p = 0.01,
      z_orig = 2,
      lfc = 0.1,
      stage = 2L,
      sn_params = rep(NA_real_, 3L),
      p_value_source = source,
      needs_empirical_fallback = FALSE,
      resampling_dist = numeric()
    )
  }
  record <- function(source, ...) {
    calls <<- c(calls, source)
    fast_flags <<- c(fast_flags, isTRUE(list(...)$use_fast))
    success(source)
  }

  testthat::local_mocked_bindings(
    perform_grna_precomputation = function(...) rep(0.35, 4L),
    crt_index_sampler_fast = function(...) {
      sampler_calls <<- sampler_calls + 1L
      structure(list(), class = "mock_bank")
    },
    load_csr_row = function(...) c(2, 3, 4, 5),
    compute_precomputation_pieces = function(..., full_test_stat) {
      full_stat_flags <<- c(full_stat_flags, full_test_stat)
      list(
        mu = rep(2, 4L),
        a = c(-1, -0.2, 0.3, 1.1),
        w = c(0.6, 0.7, 0.8, 0.9),
        b = c(0.6, 0.7, 0.8, 0.9),
        D = diag(4L)
      )
    },
    run_low_level_test_full_crt_spa_v1 = function(...) {
      record("crt_spa_fast", ...)
    },
    run_low_level_test_full_crt_spa_always_v1 = function(...) {
      record("crt_spa_always_fast", ...)
    },
    run_low_level_test_full_crt_spa_empirical_v1 = function(...) {
      record("crt_spa_empirical_fast", ...)
    },
    run_low_level_test_full_crt_spa_empirical_always_v1 = function(...) {
      record("crt_spa_empirical_always_fast", ...)
    },
    .package = "sceptre"
  )

  modes <- list(
    crt_spa_fast = c(TRUE, FALSE, FALSE, FALSE),
    crt_spa_always_fast = c(FALSE, TRUE, FALSE, FALSE),
    crt_spa_empirical_fast = c(FALSE, FALSE, TRUE, FALSE),
    crt_spa_empirical_always_fast = c(FALSE, FALSE, FALSE, TRUE)
  )
  for (mode in modes) {
    sceptre:::crt_glm_factored_out(
      B1 = if (mode[2L] || mode[4L]) 0L else 499L,
      B2 = 4999L,
      B3 = 0L,
      fit_parametric_curve = FALSE,
      use_crt_spa = mode[1L],
      use_crt_spa_always = mode[2L],
      use_crt_spa_empirical = mode[3L],
      use_crt_spa_empirical_always = mode[4L],
      output_amount = 2L,
      response_ids = response_id,
      response_precomputations = list(
        response_1 = list(fitted_coefs = 0, theta = 1)
      ),
      covariate_matrix = matrix(1, nrow = 4L, ncol = 1L),
      get_idx_f = function(...) list(trt_idxs = 1L, n_trt = 1L),
      curr_grna_group = "target_1",
      subset_to_nt_cells = FALSE,
      all_nt_idxs = integer(),
      response_matrix = response_matrix,
      side_code = 1L,
      cells_in_use = seq_len(4L),
      use_fast = TRUE
    )
  }

  expect_identical(calls, names(modes))
  expect_true(all(fast_flags))
  expect_identical(sampler_calls, 2L)
  expect_identical(full_stat_flags, c(TRUE, FALSE, FALSE, FALSE))
})

test_that("fast SPA routing also reaches every nt-cells branch", {
  skip_if_not(exists(
    "local_mocked_bindings",
    envir = asNamespace("testthat"),
    inherits = FALSE
  ))

  response_id <- "response_1"
  response_matrix <- methods::as(
    Matrix::Matrix(
      matrix(1:4, nrow = 1L, dimnames = list(response_id, NULL)),
      sparse = TRUE
    ),
    "RsparseMatrix"
  )
  calls <- character()
  fast_flags <- logical()
  sampler_calls <- 0L
  full_stat_flags <- logical()
  record <- function(source, ...) {
    calls <<- c(calls, source)
    fast_flags <<- c(fast_flags, isTRUE(list(...)$use_fast))
    list(
      p = 0.01, z_orig = 2, lfc = 0.1, stage = 2L,
      sn_params = rep(NA_real_, 3L), p_value_source = source,
      needs_empirical_fallback = FALSE, resampling_dist = numeric()
    )
  }

  testthat::local_mocked_bindings(
    perform_grna_precomputation = function(...) rep(0.35, 4L),
    crt_index_sampler_fast = function(...) {
      sampler_calls <<- sampler_calls + 1L
      structure(list(), class = "mock_bank")
    },
    load_csr_row = function(...) c(2, 3, 4, 5),
    perform_response_precomputation = function(...) {
      list(fitted_coefs = 0, theta = 1)
    },
    compute_precomputation_pieces = function(..., full_test_stat) {
      full_stat_flags <<- c(full_stat_flags, full_test_stat)
      list(
        mu = rep(2, 4L), a = c(-1, -0.2, 0.3, 1.1),
        w = c(0.6, 0.7, 0.8, 0.9), b = c(0.6, 0.7, 0.8, 0.9),
        D = diag(4L)
      )
    },
    run_low_level_test_full_crt_spa_v1 = function(...) {
      record("crt_spa_fast", ...)
    },
    run_low_level_test_full_crt_spa_always_v1 = function(...) {
      record("crt_spa_always_fast", ...)
    },
    run_low_level_test_full_crt_spa_empirical_v1 = function(...) {
      record("crt_spa_empirical_fast", ...)
    },
    run_low_level_test_full_crt_spa_empirical_always_v1 = function(...) {
      record("crt_spa_empirical_always_fast", ...)
    },
    .package = "sceptre"
  )

  modes <- list(
    crt_spa_fast = c(TRUE, FALSE, FALSE, FALSE),
    crt_spa_always_fast = c(FALSE, TRUE, FALSE, FALSE),
    crt_spa_empirical_fast = c(FALSE, FALSE, TRUE, FALSE),
    crt_spa_empirical_always_fast = c(FALSE, FALSE, FALSE, TRUE)
  )
  for (mode in modes) {
    sceptre:::discovery_ntcells_crt(
      B1 = if (mode[2L] || mode[4L]) 0L else 499L,
      B2 = 4999L,
      B3 = 0L,
      fit_parametric_curve = FALSE,
      use_crt_spa = mode[1L],
      use_crt_spa_always = mode[2L],
      use_crt_spa_empirical = mode[3L],
      use_crt_spa_empirical_always = mode[4L],
      output_amount = 2L,
      get_idx_f = function(...) list(trt_idxs = 1L, n_trt = 1L),
      response_ids = response_id,
      covariate_matrix = matrix(1, nrow = 4L, ncol = 1L),
      curr_grna_group = "target_1",
      all_nt_idxs = 2:4,
      response_matrix = response_matrix,
      side_code = 1L,
      cells_in_use = seq_len(4L),
      use_fast = TRUE
    )
  }

  expect_identical(calls, names(modes))
  expect_true(all(fast_flags))
  expect_identical(sampler_calls, 2L)
  expect_identical(full_stat_flags, c(TRUE, FALSE, FALSE, FALSE))
})
