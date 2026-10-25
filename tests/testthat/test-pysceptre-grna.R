make_pysceptre_grna_fixture <- function(n_cells = 960L, structured = TRUE) {
  set.seed(71421)
  covariates <- data.frame(z = stats::rnorm(n_cells), u = stats::runif(n_cells, -1, 1))
  if (structured) {
    covariates$batch <- factor(rep(seq_len(12L), length.out = n_cells))
    covariates$replicate <- factor(rep(
      rep(seq_len(4L), each = 12L), length.out = n_cells
    ))
    covariates$binary_1 <- stats::rbinom(n_cells, 1L, 0.4)
    covariates$binary_2 <- stats::rbinom(n_cells, 1L, 0.6)
  }
  design <- stats::model.matrix(~ ., data = covariates)
  beta <- seq(-0.2, 0.2, length.out = ncol(design))
  beta[[1L]] <- stats::qlogis(0.2)
  indicator <- stats::rbinom(n_cells, 1L, stats::plogis(drop(design %*% beta)))
  list(covariate_matrix = design, indicator = indicator, trt_idxs = which(indicator == 1L))
}


make_pysceptre_grna_runner_args <- function() {
  set.seed(37216)
  n_cells <- 240L
  response_matrix <- rbind(
    gene_1 = stats::rnbinom(n_cells, mu = 3, size = 4),
    gene_2 = stats::rnbinom(n_cells, mu = 2, size = 6)
  )
  list(
    response_matrix = methods::as(
      Matrix::Matrix(response_matrix, sparse = TRUE), "RsparseMatrix"
    ),
    grna_assignments = list(
      grna_group_idxs = list(t1 = 1:15, t2 = 16:35),
      indiv_nt_grna_idxs = list(nt1 = 51:75, nt2 = 76:100),
      all_nt_idxs = 51:100
    ),
    covariate_matrix = cbind(`(Intercept)` = 1, z = stats::rnorm(n_cells)),
    response_grna_group_pairs = expand.grid(
      response_id = c("gene_1", "gene_2"), grna_group = c("t1", "t2"),
      stringsAsFactors = FALSE
    ),
    output_amount = 2L, resampling_approximation = "no_approximation",
    B1 = 19L, B2 = 0L, B3 = 0L, calibration_check = FALSE,
    control_group_complement = TRUE, n_nonzero_trt_thresh = 0L,
    n_nonzero_cntrl_thresh = 0L, side_code = -1L, low_moi = TRUE,
    response_precomputations = list(), cells_in_use = seq.int(2L, n_cells, 2L),
    print_progress = FALSE, parallel = FALSE, n_processors = 2L,
    log_dir = tempdir(), analysis_type = "discovery_analysis",
    response_fit_method = "sceptre", grna_fit_method = "pysceptre"
  )
}


test_that("pysceptre gRNA fits optimize the same binomial objective as glm.fit", {
  for (structured in c(FALSE, TRUE)) {
    fixture <- make_pysceptre_grna_fixture(structured = structured)
    design <- fixture$covariate_matrix
    reference <- stats::glm.fit(
      x = design, y = fixture$indicator, family = stats::binomial(),
      control = stats::glm.control(epsilon = 1e-12, maxit = 100L)
    )
    prepared <- prepare_pysceptre_glm_design(design)
    details <- perform_grna_precomputation(
      trt_idxs = fixture$trt_idxs, covariate_matrix = design,
      return_fitted_values = TRUE, grna_fit_method = "pysceptre",
      prepared_design = prepared, return_details = TRUE
    )
    probabilities <- perform_grna_precomputation(
      trt_idxs = rev(fixture$trt_idxs), covariate_matrix = design,
      return_fitted_values = TRUE, grna_fit_method = "pysceptre"
    )
    coefficients <- perform_grna_precomputation(
      trt_idxs = fixture$trt_idxs, covariate_matrix = design,
      return_fitted_values = FALSE, grna_fit_method = "pysceptre",
      prepared_design = prepared
    )
    expect_named(details, c(
      "value", "method_requested", "method_used", "fallback_reason",
      "converged", "iterations"
    ))
    expect_identical(details$method_requested, "pysceptre")
    expect_identical(details$method_used, "pysceptre")
    expect_true(details$converged)
    expect_true(is.na(details$fallback_reason))
    expect_gt(details$iterations, 0L)
    expect_identical(probabilities, details$value)
    expect_identical(attributes(probabilities), attributes(reference$fitted.values))
    expect_identical(attributes(coefficients), attributes(reference$coefficients))
    expect_lt(max(abs(probabilities - reference$fitted.values)), 1e-7)
    expect_lt(max(abs(coefficients - reference$coefficients)), 1e-6)
    expect_equal(
      probabilities, stats::plogis(drop(design %*% coefficients)),
      tolerance = 1e-10, ignore_attr = TRUE
    )
    log_likelihood <- function(p) sum(stats::dbinom(fixture$indicator, 1L, p, log = TRUE))
    expect_equal(
      log_likelihood(probabilities), log_likelihood(reference$fitted.values),
      tolerance = 1e-10
    )
  }
})


test_that("unsafe pysceptre gRNA fits fall back exactly to native binomial fits", {
  fixture <- make_pysceptre_grna_fixture(n_cells = 240L, structured = FALSE)
  design <- fixture$covariate_matrix
  cases <- list(
    all_zero = list(x = design, y = rep(0, nrow(design))),
    all_one = list(x = design, y = rep(1, nrow(design))),
    rank_deficient = list(x = cbind(design, duplicate = design[, 2L]), y = fixture$indicator),
    no_intercept = list(x = design[, -1L, drop = FALSE], y = fixture$indicator),
    separated = list(x = design, y = as.integer(design[, 2L] > 0))
  )
  for (case_name in names(cases)) {
    case <- cases[[case_name]]
    reference <- suppressWarnings(stats::glm.fit(
      x = case$x, y = case$y, family = stats::binomial()
    ))
    details <- suppressWarnings(perform_grna_precomputation(
      trt_idxs = which(case$y == 1L), covariate_matrix = case$x,
      return_fitted_values = TRUE, grna_fit_method = "pysceptre",
      return_details = TRUE
    ))
    coefficients <- suppressWarnings(perform_grna_precomputation(
      trt_idxs = which(case$y == 1L), covariate_matrix = case$x,
      return_fitted_values = FALSE, grna_fit_method = "pysceptre"
    ))
    expect_identical(details$method_requested, "pysceptre", info = case_name)
    expect_identical(details$method_used, "glm.fit", info = case_name)
    expect_true(
      is.character(details$fallback_reason) && length(details$fallback_reason) == 1L &&
        !is.na(details$fallback_reason) && nzchar(details$fallback_reason),
      info = case_name
    )
    expect_identical(details$value, reference$fitted.values, info = case_name)
    expect_identical(coefficients, reference$coefficients, info = case_name)
    expect_identical(details$converged, isTRUE(reference$converged), info = case_name)
    expect_equal(details$iterations, reference$iter, info = case_name)
  }
})


test_that("pysceptre binomial iteration exhaustion and invalid designs are explicit", {
  fixture <- make_pysceptre_grna_fixture(n_cells = 240L, structured = FALSE)
  reference <- stats::glm.fit(
    x = fixture$covariate_matrix, y = fixture$indicator, family = stats::binomial()
  )
  fit <- fit_pysceptre_glm(
    expressions = fixture$indicator, covariate_matrix = fixture$covariate_matrix,
    family = "binomial", max_iterations = 0L
  )
  expect_identical(fit$method_used, "glm.fit")
  expect_true(nzchar(fit$fallback_reason))
  expect_identical(fit$fitted.values, reference$fitted.values)
  expect_identical(fit$coefficients, reference$coefficients)
  prepared <- prepare_pysceptre_glm_design(fixture$covariate_matrix)
  changed <- fixture$covariate_matrix
  changed[1L, 2L] <- changed[1L, 2L] + 0.1
  expect_error(
    perform_grna_precomputation(
      fixture$trt_idxs, changed, return_fitted_values = TRUE,
      grna_fit_method = "pysceptre", prepared_design = prepared
    ),
    "prepared_design|covariate matrix"
  )
  expect_error(
    perform_grna_precomputation(
      c(1L, 1L), fixture$covariate_matrix, return_fitted_values = TRUE,
      grna_fit_method = "pysceptre"
    ),
    "duplicates"
  )
  expect_error(
    perform_grna_precomputation(
      0L, fixture$covariate_matrix, return_fitted_values = TRUE,
      grna_fit_method = "pysceptre"
    ),
    "out-of-range"
  )
})


test_that("pysceptre CRT gRNA preparation is shared over the QC-filtered design", {
  args <- make_pysceptre_grna_runner_args()
  original_prepare <- prepare_pysceptre_glm_design
  captured <- list()
  testthat::local_mocked_bindings(
    prepare_pysceptre_glm_design = function(covariate_matrix) {
      captured[[length(captured) + 1L]] <<- covariate_matrix
      original_prepare(covariate_matrix)
    },
    .package = "sceptre"
  )
  result <- do.call(run_crt_in_memory_v2, args)
  expect_length(captured, 1L)
  expect_identical(captured[[1L]], args$covariate_matrix[args$cells_in_use, , drop = FALSE])
  expect_true(all(is.finite(result$result$p_value)))
  timing <- result$phase_timings
  diagnostics <- attr(timing, "grna_fit_diagnostics")
  expect_identical(timing$grna_preparation_scope, "shared_design")
  expect_equal(timing$n_grna_fits, 2L)
  expect_equal(timing$n_pysceptre_grna_fits, 2L)
  expect_equal(timing$n_optimized_grna_fits, 2L)
  expect_equal(timing$n_fast_grna_fits, 0L)
  expect_equal(timing$n_grna_fit_fallbacks, 0L)
  expect_identical(diagnostics$method_requested, rep("pysceptre", 2L))
  expect_identical(diagnostics$method_used, rep("pysceptre", 2L))
  expect_equal(diagnostics$n_cells, rep(120L, 2L))
  expect_equal(diagnostics$n_treated, c(15L, 20L))
})


test_that("pysceptre NT gRNA models use each target's subset and treatment indices", {
  args <- make_pysceptre_grna_runner_args()
  args$control_group_complement <- FALSE
  original_prepare <- prepare_pysceptre_glm_design
  original_fit <- fit_pysceptre_glm
  prepared_matrices <- list()
  fitted_responses <- list()
  testthat::local_mocked_bindings(
    prepare_pysceptre_glm_design = function(covariate_matrix) {
      prepared_matrices[[length(prepared_matrices) + 1L]] <<- covariate_matrix
      original_prepare(covariate_matrix)
    },
    fit_pysceptre_glm = function(expressions, covariate_matrix,
                               family = "poisson", prepared_design = NULL) {
      fitted_responses[[length(fitted_responses) + 1L]] <<- list(
        y = expressions, x = covariate_matrix, family = family
      )
      original_fit(expressions, covariate_matrix, family, prepared_design)
    },
    .package = "sceptre"
  )
  result <- do.call(run_crt_in_memory_v2, args)
  expect_length(prepared_matrices, 2L)
  expect_length(fitted_responses, 2L)
  for (i in seq_len(2L)) {
    treatment <- args$grna_assignments$grna_group_idxs[[i]]
    rows <- args$cells_in_use[c(treatment, 51:100)]
    expected_design <- args$covariate_matrix[rows, , drop = FALSE]
    expect_identical(prepared_matrices[[i]], expected_design)
    expect_identical(fitted_responses[[i]]$x, expected_design)
    expect_equal(fitted_responses[[i]]$y, c(rep(1, length(treatment)), rep(0, 50L)))
    expect_identical(fitted_responses[[i]]$family, "binomial")
  }
  expect_true(all(is.finite(result$result$p_value)))
  expect_identical(result$phase_timings$grna_preparation_scope, "per_target")
  expect_equal(result$phase_timings$n_pysceptre_grna_fits, 2L)
  expect_equal(attr(result$phase_timings, "grna_fit_diagnostics")$n_cells, c(65L, 70L))
})


test_that("pysceptre NT calibration prepares only the NT pool with remapped indices", {
  args <- make_pysceptre_grna_runner_args()
  nt <- update_indiv_grna_assignments_for_nt_cells(args$grna_assignments$indiv_nt_grna_idxs)
  args$grna_assignments$indiv_nt_grna_idxs <- nt$indiv_nt_grna_idxs
  args$control_group_complement <- FALSE
  args$calibration_check <- TRUE
  args$analysis_type <- "calibration_check"
  args$response_grna_group_pairs$grna_group <- rep(c("nt1", "nt2"), each = 2L)
  original_prepare <- prepare_pysceptre_glm_design
  original_fit <- fit_pysceptre_glm
  prepared_matrices <- list()
  fitted_responses <- list()
  testthat::local_mocked_bindings(
    prepare_pysceptre_glm_design = function(covariate_matrix) {
      prepared_matrices[[length(prepared_matrices) + 1L]] <<- covariate_matrix
      original_prepare(covariate_matrix)
    },
    fit_pysceptre_glm = function(expressions, covariate_matrix,
                               family = "poisson", prepared_design = NULL) {
      fitted_responses[[length(fitted_responses) + 1L]] <<- expressions
      original_fit(expressions, covariate_matrix, family, prepared_design)
    },
    .package = "sceptre"
  )
  result <- do.call(run_crt_in_memory_v2, args)
  expect_length(prepared_matrices, 1L)
  expect_identical(
    prepared_matrices[[1L]], args$covariate_matrix[args$cells_in_use[51:100], , drop = FALSE]
  )
  expect_equal(fitted_responses, list(c(rep(1, 25L), rep(0, 25L)), c(rep(0, 25L), rep(1, 25L))))
  expect_true(all(is.finite(result$result$p_value)))
  expect_identical(result$phase_timings$grna_preparation_scope, "shared_design")
  expect_equal(result$phase_timings$n_pysceptre_grna_fits, 2L)
  expect_equal(attr(result$phase_timings, "grna_fit_diagnostics")$n_cells, c(50L, 50L))
})


test_that("pysceptre gRNA fallback counts distinguish requested and completed methods", {
  args <- make_pysceptre_grna_runner_args()
  original_fit <- fit_pysceptre_glm
  testthat::local_mocked_bindings(
    fit_pysceptre_glm = function(expressions, covariate_matrix,
                               family = "poisson", prepared_design = NULL) {
      original_fit(
        expressions, covariate_matrix, family, prepared_design,
        max_iterations = if (sum(expressions) == 20L) 0L else 25L
      )
    },
    .package = "sceptre"
  )
  result <- do.call(run_crt_in_memory_v2, args)
  diagnostics <- attr(result$phase_timings, "grna_fit_diagnostics")
  expect_identical(diagnostics$method_requested, rep("pysceptre", 2L))
  expect_identical(diagnostics$method_used, c("pysceptre", "glm.fit"))
  expect_true(!is.na(diagnostics$fallback_reason[[2L]]))
  expect_equal(result$phase_timings$n_grna_fits, 2L)
  expect_equal(result$phase_timings$n_pysceptre_grna_fits, 1L)
  expect_equal(result$phase_timings$n_optimized_grna_fits, 1L)
  expect_equal(result$phase_timings$n_fast_grna_fits, 0L)
  expect_equal(result$phase_timings$n_grna_fit_fallbacks, 1L)
  expect_true(all(is.finite(result$result$p_value)))
})


test_that("pysceptre gRNA selection is independent of response fitting and cache lifetime", {
  set.seed(8812)
  targets <- make_mock_grna_target_data(c(1, 2), 1, 1, 3)
  empty_pairs <- data.frame(grna_target = character(), response_id = character())
  object <- import_data(
    response_matrix = make_mock_response_matrices(4, 50, patterns = "column"),
    grna_matrix = make_mock_grna_matrices(
      targets, 50, non_nt_patterns = "column", nt_patterns = "row"
    ),
    grna_target_data_frame = targets, moi = "high"
  ) |>
    set_analysis_parameters(discovery_pairs = empty_pairs, response_fit_method = "pysceptre")
  cache <- list(response_1 = list(
    fitted_coefs = rep(0, ncol(object@covariate_matrix)), theta = 4
  ))
  object@response_precomputations <- cache
  selected <- set_analysis_parameters(
    object, discovery_pairs = empty_pairs,
    response_fit_method = "pysceptre", grna_fit_method = "pysceptre"
  )
  expect_identical(get_grna_fit_method(selected), "pysceptre")
  expect_identical(get_response_fit_method(selected), "pysceptre")
  expect_identical(selected@response_precomputations, cache)
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(selected, path)
  expect_identical(get_grna_fit_method(readRDS(path)), "pysceptre")
  expect_error(
    set_analysis_parameters(
      object, discovery_pairs = empty_pairs, grna_fit_method = c("glm.fit", "pysceptre")
    ),
    "grna_fit_method"
  )
})


test_that("CRT response and gRNA backends can be chosen independently", {
  for (response_method in c("sceptre", "pysceptre")) {
    for (grna_method in c("glm.fit", "fast_logistic", "pysceptre")) {
      args <- make_pysceptre_grna_runner_args()
      args$response_fit_method <- response_method
      args$grna_fit_method <- grna_method
      result <- do.call(run_crt_in_memory_v2, args)
      expect_true(all(is.finite(result$result$p_value)))
      expect_identical(result$phase_timings$response_fit_method, response_method)
      expect_identical(
        attr(result$phase_timings, "grna_fit_diagnostics")$method_requested,
        rep(grna_method, 2L)
      )
      expect_equal(result$phase_timings$n_pysceptre_grna_fits, if (grna_method == "pysceptre") 2L else 0L)
      expect_equal(result$phase_timings$n_fast_grna_fits, if (grna_method == "fast_logistic") 2L else 0L)
      expect_equal(result$phase_timings$n_optimized_grna_fits, if (grna_method == "glm.fit") 0L else 2L)
    }
  }
})


test_that("parallel CRT keeps pysceptre gRNA results and diagnostics deterministic", {
  skip_on_os("windows")
  skip_on_cran()
  args <- make_pysceptre_grna_runner_args()
  serial <- do.call(run_crt_in_memory_v2, args)
  args$parallel <- TRUE
  concurrent <- do.call(run_crt_in_memory_v2, args)
  expect_true(all(is.finite(concurrent$result$p_value)))
  expect_equal(concurrent$phase_timings$n_pysceptre_grna_fits, 2L)
  expect_equal(concurrent$phase_timings$n_optimized_grna_fits, 2L)
  diagnostic_columns <- c(
    "grna_group", "method_requested", "method_used", "fallback_reason",
    "converged", "iterations", "n_cells", "n_treated"
  )
  diagnostic_key <- function(result) {
    diagnostics <- attr(result$phase_timings, "grna_fit_diagnostics")
    diagnostics[order(diagnostics$grna_group), diagnostic_columns]
  }
  expect_equal(diagnostic_key(concurrent), diagnostic_key(serial), ignore_attr = TRUE)
  result_key <- function(result) {
    result$result[order(result$result$grna_group, result$result$response_id), ]
  }
  expect_equal(result_key(concurrent), result_key(serial), ignore_attr = TRUE)
})


test_that("permutation analysis skips pysceptre gRNA fitting while using its response backend", {
  args <- make_pysceptre_grna_runner_args()
  original_fit <- fit_pysceptre_glm
  testthat::local_mocked_bindings(
    prepare_grna_design_timed = function(...) stop("unexpected gRNA design"),
    perform_grna_precomputation = function(...) stop("unexpected gRNA model"),
    fit_pysceptre_glm = function(expressions, covariate_matrix,
                               family = "poisson", prepared_design = NULL) {
      expect_identical(family, "poisson")
      original_fit(expressions, covariate_matrix, family, prepared_design)
    },
    .package = "sceptre"
  )
  object <- methods::new("sceptre_object")
  object@response_matrix <- list(args$response_matrix)
  object@grna_assignments <- args$grna_assignments
  object@covariate_matrix <- args$covariate_matrix
  object@resampling_approximation <- args$resampling_approximation
  object@B1 <- args$B1
  object@B2 <- args$B2
  object@B3 <- args$B3
  object@control_group_complement <- TRUE
  object@n_nonzero_trt_thresh <- 0L
  object@n_nonzero_cntrl_thresh <- 0L
  object@side_code <- -1L
  object@low_moi <- TRUE
  object@response_precomputations <- list()
  object@cells_in_use <- args$cells_in_use
  object@run_permutations <- TRUE
  object <- set_grna_fit_method(set_response_fit_method(object, "pysceptre"), "pysceptre")
  pairs <- args$response_grna_group_pairs
  pairs$pass_qc <- TRUE
  result <- run_sceptre_analysis_high_level(
    object, pairs, calibration_check = FALSE, analysis_type = "discovery_analysis",
    output_amount = 2L, print_progress = FALSE, parallel = FALSE,
    n_processors = 1L, log_dir = tempdir()
  )
  expect_equal(nrow(result$result), 4L)
  expect_true(all(is.finite(result$result$p_value)))
  expect_identical(result$phase_timings$response_fit_method, "pysceptre")
  expect_null(attr(result$phase_timings, "grna_fit_diagnostics"))
})
