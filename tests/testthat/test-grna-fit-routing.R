make_grna_runner_args <- function() {
  set.seed(7231)
  n <- 200L
  response_matrix <- methods::as(Matrix::Matrix(
    matrix(stats::rnbinom(2L * n, mu = 3, size = 4), nrow = 2L,
           dimnames = list(c("gene_1", "gene_2"), NULL)),
    sparse = TRUE
  ), "RsparseMatrix")
  list(
    response_matrix = response_matrix,
    grna_assignments = list(
      grna_group_idxs = list(t1 = 1:10, t2 = 11:25, t3 = 26:45, t4 = 46:60),
      indiv_nt_grna_idxs = list(nt1 = 61:80, nt2 = 81:100),
      all_nt_idxs = 61:100
    ),
    covariate_matrix = cbind(`(Intercept)` = 1, z = stats::rnorm(n)),
    response_grna_group_pairs = expand.grid(
      response_id = c("gene_1", "gene_2"),
      grna_group = paste0("t", 1:4), stringsAsFactors = FALSE
    ),
    output_amount = 2L, resampling_approximation = "no_approximation",
    B1 = 19L, B2 = 0L, B3 = 0L,
    calibration_check = FALSE, control_group_complement = TRUE,
    n_nonzero_trt_thresh = 0, n_nonzero_cntrl_thresh = 0,
    side_code = -1L, low_moi = TRUE, response_precomputations = list(),
    cells_in_use = seq.int(2L, n, by = 2L),
    print_progress = FALSE, parallel = FALSE, n_processors = 2L,
    log_dir = tempdir(), analysis_type = "discovery_analysis",
    grna_fit_method = "fast_logistic"
  )
}


test_that("CRT prepares the shared QC-filtered design once and fits each target once", {
  args <- make_grna_runner_args()
  original_prepare <- prepare_grna_logistic_design
  prepared_matrices <- list()
  testthat::local_mocked_bindings(
    prepare_grna_logistic_design = function(covariate_matrix) {
      prepared_matrices[[length(prepared_matrices) + 1L]] <<- covariate_matrix
      original_prepare(covariate_matrix)
    },
    .package = "sceptre"
  )
  out <- do.call(run_crt_in_memory_v2, args)
  timing <- out$phase_timings
  diagnostics <- attr(timing, "grna_fit_diagnostics")
  expect_length(prepared_matrices, 1L)
  expect_identical(prepared_matrices[[1L]], args$covariate_matrix[args$cells_in_use, ])
  expect_equal(nrow(out$result), 8L)
  expect_true(all(is.finite(out$result$p_value)))
  expect_equal(timing$n_grna_fits, 4L)
  expect_equal(timing$n_fast_grna_fits, 4L)
  expect_equal(timing$n_grna_fit_fallbacks, 0L)
  expect_identical(timing$grna_preparation_scope, "shared_design")
  expect_setequal(diagnostics$grna_group, paste0("t", 1:4))
  expect_equal(diagnostics$n_cells, rep(100L, 4L))
  expect_equal(diagnostics$preparation_cpu_seconds, rep(0, 4L))
  expect_equal(timing$grna_fitting_cpu_seconds, sum(diagnostics$fitting_cpu_seconds))
  expect_gte(timing$grna_total_cpu_seconds, timing$grna_fitting_cpu_seconds)
})


test_that("target-specific CRT designs are prepared on their own treatment and NT cells", {
  args <- make_grna_runner_args()
  args$control_group_complement <- FALSE
  original_prepare <- prepare_grna_logistic_design
  prepared_matrices <- list()
  testthat::local_mocked_bindings(
    prepare_grna_logistic_design = function(covariate_matrix) {
      prepared_matrices[[length(prepared_matrices) + 1L]] <<- covariate_matrix
      original_prepare(covariate_matrix)
    },
    .package = "sceptre"
  )
  out <- do.call(run_crt_in_memory_v2, args)
  diagnostics <- attr(out$phase_timings, "grna_fit_diagnostics")
  expect_length(prepared_matrices, 4L)
  z <- args$covariate_matrix[args$cells_in_use, ]
  for (i in seq_len(4L)) {
    expected_rows <- c(args$grna_assignments$grna_group_idxs[[i]], 61:100)
    expect_identical(prepared_matrices[[i]], z[expected_rows, ])
  }
  expect_identical(out$phase_timings$grna_preparation_scope, "per_target")
  expect_equal(diagnostics$n_cells, c(50L, 55L, 60L, 55L))
  expect_equal(out$phase_timings$n_grna_fits, 4L)
  expect_equal(out$phase_timings$n_fast_grna_fits, 4L)
  expect_true(all(is.finite(out$result$p_value)))
})


test_that("NT calibration prepares the subset design and remaps treatment indices", {
  args <- make_grna_runner_args()
  nt_assignments <- update_indiv_grna_assignments_for_nt_cells(
    args$grna_assignments$indiv_nt_grna_idxs
  )
  args$grna_assignments$indiv_nt_grna_idxs <- nt_assignments$indiv_nt_grna_idxs
  args$calibration_check <- TRUE
  args$control_group_complement <- FALSE
  args$analysis_type <- "calibration_check"
  args$response_grna_group_pairs <- expand.grid(
    response_id = c("gene_1", "gene_2"),
    grna_group = c("nt1", "nt2"), stringsAsFactors = FALSE
  )
  out <- do.call(run_crt_in_memory_v2, args)
  diagnostics <- attr(out$phase_timings, "grna_fit_diagnostics")
  expect_equal(out$phase_timings$n_grna_fits, 2L)
  expect_equal(out$phase_timings$n_fast_grna_fits, 2L)
  expect_equal(diagnostics$n_cells, c(40L, 40L))
  expect_equal(diagnostics$n_treated, c(20L, 20L))
  expect_identical(out$phase_timings$grna_preparation_scope, "shared_design")
  expect_true(all(is.finite(out$result$p_value)))
})


test_that("parallel CRT reuses the prepared design and reports worker fit diagnostics", {
  skip_on_os("windows")
  skip_on_cran()
  args <- make_grna_runner_args()
  args$resampling_approximation <- "crt_spa_always"
  args$B1 <- 0L
  args$B2 <- 19L
  serial <- do.call(run_crt_in_memory_v2, args)
  args$parallel <- TRUE
  parallel_result <- do.call(run_crt_in_memory_v2, args)
  diagnostics <- attr(parallel_result$phase_timings, "grna_fit_diagnostics")
  serial_diagnostics <- attr(serial$phase_timings, "grna_fit_diagnostics")
  expect_equal(parallel_result$phase_timings$association_n_workers, 2L)
  expect_equal(parallel_result$phase_timings$n_fast_grna_fits, 4L)
  expect_equal(diagnostics$preparation_cpu_seconds, rep(0, 4L))
  order_key <- function(df) order(df$grna_group)
  columns <- c("grna_group", "method_used", "converged", "iterations", "n_cells", "n_treated")
  expect_equal(
    unname(as.matrix(diagnostics[order_key(diagnostics), columns])),
    unname(as.matrix(serial_diagnostics[order_key(serial_diagnostics), columns]))
  )
  expect_true(all(is.finite(parallel_result$result$p_value)))
})


test_that("native CRT skips design preparation and RPT never enters the XZ fitter", {
  args <- make_grna_runner_args()
  args$grna_fit_method <- "glm.fit"
  testthat::local_mocked_bindings(
    prepare_grna_logistic_design = function(...) stop("unexpected design preparation"),
    .package = "sceptre"
  )
  native <- do.call(run_crt_in_memory_v2, args)
  expect_equal(native$phase_timings$n_fast_grna_fits, 0L)
  expect_equal(native$phase_timings$n_grna_fits, 4L)
  expect_identical(native$phase_timings$grna_preparation_scope, "not_required")
  testthat::local_mocked_bindings(
    perform_grna_precomputation = function(...) stop("unexpected XZ fit"),
    .package = "sceptre"
  )
  args$grna_fit_method <- NULL
  args$resampling_approximation <- "rpt_spa_always"
  args$B1 <- 0L
  args$B2 <- 19L
  args["synthetic_idxs"] <- list(NULL)
  rpt <- do.call(run_perm_test_in_memory, args)
  expect_null(attr(rpt$phase_timings, "grna_fit_diagnostics"))
  expect_identical(rpt$phase_timings$timing_schema_version, 1L)
  expect_true(all(is.finite(rpt$result$p_value)))
})


test_that("a failed target is counted once and other targets keep the fast backend", {
  args <- make_grna_runner_args()
  original_fit <- fit_grna_logistic_prepared_cpp
  testthat::local_mocked_bindings(
    fit_grna_logistic_prepared_cpp = function(prepared, treated_indices, ...) {
      if (identical(treated_indices, 1:10)) {
        return(list(converged = FALSE, reason = "maximum_iterations_reached"))
      }
      original_fit(prepared, treated_indices, ...)
    },
    .package = "sceptre"
  )
  out <- do.call(run_crt_in_memory_v2, args)
  diagnostics <- attr(out$phase_timings, "grna_fit_diagnostics")
  expect_equal(out$phase_timings$n_grna_fits, 4L)
  expect_equal(out$phase_timings$n_fast_grna_fits, 3L)
  expect_equal(out$phase_timings$n_grna_fit_fallbacks, 1L)
  expect_identical(diagnostics$method_used[diagnostics$grna_group == "t1"], "glm.fit")
  expect_identical(
    diagnostics$fallback_reason[diagnostics$grna_group == "t1"],
    "maximum_iterations_reached"
  )
  expect_true(all(is.finite(out$result$p_value)))
})
