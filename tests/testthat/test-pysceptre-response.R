make_pysceptre_response_fixture <- function(n_cells = 480L, structured = TRUE) {
  set.seed(62413)
  z <- stats::rnorm(n_cells)
  u <- stats::runif(n_cells, -1, 1)
  if (structured) {
    covariates <- data.frame(
      z = z, u = u,
      batch = factor(rep(seq_len(12L), length.out = n_cells)),
      replicate = factor(rep(rep(seq_len(4L), each = 12L), length.out = n_cells)),
      binary_1 = stats::rbinom(n_cells, 1L, 0.4),
      binary_2 = stats::rbinom(n_cells, 1L, 0.6)
    )
    design <- stats::model.matrix(~ ., data = covariates)
  } else {
    design <- cbind(`(Intercept)` = 1, z = z, u = u)
  }
  beta <- seq(-0.15, 0.15, length.out = ncol(design))
  beta[[1L]] <- log(3)
  mean <- exp(drop(design %*% beta))
  responses <- rbind(
    gene_1 = stats::rnbinom(n_cells, mu = mean, size = 5),
    gene_2 = stats::rnbinom(n_cells, mu = mean * 0.7, size = 2),
    gene_3 = stats::rnbinom(n_cells, mu = mean * 1.5, size = 10)
  )
  list(response_matrix = responses, covariate_matrix = design)
}


make_pysceptre_response_runner_args <- function() {
  fixture <- make_pysceptre_response_fixture(n_cells = 240L, structured = FALSE)
  list(
    response_matrix = methods::as(
      Matrix::Matrix(fixture$response_matrix[1:2, ], sparse = TRUE),
      "RsparseMatrix"
    ),
    grna_assignments = list(
      grna_group_idxs = list(t1 = 1:15, t2 = 16:35),
      indiv_nt_grna_idxs = list(nt1 = 51:75, nt2 = 76:100),
      all_nt_idxs = 51:100
    ),
    covariate_matrix = fixture$covariate_matrix,
    response_grna_group_pairs = expand.grid(
      response_id = c("gene_1", "gene_2"), grna_group = c("t1", "t2"),
      stringsAsFactors = FALSE
    ),
    output_amount = 2L, resampling_approximation = "no_approximation",
    B1 = 19L, B2 = 0L, B3 = 0L, calibration_check = FALSE,
    control_group_complement = TRUE, n_nonzero_trt_thresh = 0L,
    n_nonzero_cntrl_thresh = 0L, side_code = -1L, low_moi = TRUE,
    response_precomputations = list(), cells_in_use = seq.int(2L, 240L, 2L),
    print_progress = FALSE, parallel = FALSE, n_processors = 1L,
    log_dir = tempdir(), analysis_type = "discovery_analysis",
    response_fit_method = "pysceptre", response_fit_chunk_size = 2L
  )
}


test_that("pysceptre response IRLS fits the same unpenalized Poisson objective", {
  for (structured in c(FALSE, TRUE)) {
    fixture <- make_pysceptre_response_fixture(structured = structured)
    design <- fixture$covariate_matrix
    prepared <- prepare_pysceptre_glm_design(design)
    fits <- lapply(seq_len(nrow(fixture$response_matrix)), function(i) {
      y <- fixture$response_matrix[i, ]
      reference <- stats::glm.fit(
        x = design, y = y, family = stats::poisson(),
        control = stats::glm.control(epsilon = 1e-12, maxit = 100L)
      )
      fit <- fit_pysceptre_glm(
        expressions = y, covariate_matrix = design, family = "poisson",
        prepared_design = prepared
      )
      expect_true(fit$converged)
      expect_identical(fit$method_used, "pysceptre")
      expect_true(is.na(fit$fallback_reason))
      expect_true(is.numeric(fit$iterations) && length(fit$iterations) == 1L)
      expect_gt(fit$iterations, 0L)
      expect_named(fit$coefficients, colnames(design))
      expect_lt(max(abs(log(fit$fitted.values / reference$fitted.values))), 1e-7)
      expect_lt(max(abs(fit$coefficients - reference$coefficients)), 1e-6)
      expect_equal(
        fit$fitted.values, exp(drop(design %*% fit$coefficients)),
        tolerance = 1e-10, ignore_attr = TRUE
      )
      log_likelihood <- function(mu) sum(stats::dpois(y, mu, log = TRUE))
      expect_equal(
        log_likelihood(fit$fitted.values),
        log_likelihood(reference$fitted.values), tolerance = 1e-10
      )
      fit
    })
    # One prepared design is safe to reuse across responses and fit order.
    repeat_fit <- fit_pysceptre_glm(
      expressions = fixture$response_matrix[1L, ], covariate_matrix = design,
      family = "poisson", prepared_design = prepared
    )
    expect_identical(repeat_fit, fits[[1L]])
    unprepared <- fit_pysceptre_glm(
      expressions = fixture$response_matrix[2L, ], covariate_matrix = design,
      family = "poisson"
    )
    expect_identical(unprepared, fits[[2L]])
  }
})


test_that("pysceptre response caches retain the existing theta estimator and bounds", {
  fixture <- make_pysceptre_response_fixture()
  y <- fixture$response_matrix[1L, ]
  fit <- fit_pysceptre_glm(y, fixture$covariate_matrix, family = "poisson")
  expected_theta <- estimate_theta(
    y = y, mu = fit$fitted.values,
    dfr = length(y) - ncol(fixture$covariate_matrix), limit = 50,
    eps = .Machine$double.eps^(1 / 4)
  )[[1L]]
  cache <- perform_response_precomputation(
    y, fixture$covariate_matrix, response_fit_method = "pysceptre"
  )
  expect_named(cache, c("fitted_coefs", "theta"))
  expect_identical(cache$fitted_coefs, fit$coefficients)
  expect_equal(cache$theta, max(min(expected_theta, 1000), 0.01), tolerance = 1e-10)
  diagnostics <- attr(cache, "fit_diagnostics", exact = TRUE)
  expect_identical(diagnostics$method_used, "pysceptre")
  expect_true(isTRUE(diagnostics$converged))
  expect_true(is.na(diagnostics$fallback_reason))
  expect_equal(diagnostics$iterations, fit$iterations)

  forced_theta <- 5000
  testthat::local_mocked_bindings(
    estimate_theta = function(y, mu, dfr, limit, eps) {
      expect_equal(mu, fit$fitted.values, tolerance = 1e-10)
      expect_identical(limit, 50)
      expect_equal(eps, .Machine$double.eps^(1 / 4))
      list(forced_theta, 1L)
    },
    .package = "sceptre"
  )
  expect_identical(perform_response_precomputation(
    y, fixture$covariate_matrix, response_fit_method = "pysceptre"
  )$theta, 1000)
  forced_theta <- 0.0001
  expect_identical(perform_response_precomputation(
    y, fixture$covariate_matrix, response_fit_method = "pysceptre"
  )$theta, 0.01)
})


test_that("pysceptre response chunks preserve requested order and exact fit results", {
  fixture <- make_pysceptre_response_fixture()
  cell_indices <- setdiff(
    seq_len(ncol(fixture$response_matrix)),
    seq.int(7L, ncol(fixture$response_matrix), by = 7L)
  )
  response_ids <- c("gene_3", "gene_1", "gene_2")
  design <- fixture$covariate_matrix[cell_indices, , drop = FALSE]
  fit_chunks <- function(chunk_size, response_matrix = fixture$response_matrix) {
    perform_response_precomputations_from_matrix(
      response_matrix = response_matrix, response_ids = response_ids,
      cell_indices = cell_indices, covariate_matrix = design,
      response_fit_method = "pysceptre", chunk_size = chunk_size
    )
  }
  scalar <- stats::setNames(lapply(response_ids, function(id) {
    perform_response_precomputation(
      expressions = fixture$response_matrix[id, cell_indices],
      covariate_matrix = design, response_fit_method = "pysceptre"
    )
  }), response_ids)
  one <- fit_chunks(1L)
  expect_named(one, response_ids)
  expect_identical(one, scalar)
  expect_identical(fit_chunks(2L), one)
  expect_identical(fit_chunks(16L), one)
  sparse <- methods::as(
    Matrix::Matrix(fixture$response_matrix, sparse = TRUE), "RsparseMatrix"
  )
  expect_identical(fit_chunks(2L, sparse), one)
})


test_that("unsafe pysceptre response fits use native glm.fit fallback diagnostics", {
  fixture <- make_pysceptre_response_fixture(n_cells = 120L, structured = FALSE)
  cases <- list(
    zero_gene = list(y = rep(0, 120L), x = fixture$covariate_matrix),
    rank_deficient = list(
      y = fixture$response_matrix[1L, ],
      x = cbind(fixture$covariate_matrix, duplicate = fixture$covariate_matrix[, 2L])
    )
  )
  for (case_name in names(cases)) {
    case <- cases[[case_name]]
    reference <- suppressWarnings(stats::glm.fit(
      y = case$y, x = case$x, family = stats::poisson()
    ))
    fit <- suppressWarnings(fit_pysceptre_glm(
      expressions = case$y, covariate_matrix = case$x, family = "poisson"
    ))
    expect_identical(fit$method_used, "glm.fit", info = case_name)
    expect_true(
      is.character(fit$fallback_reason) && length(fit$fallback_reason) == 1L &&
        !is.na(fit$fallback_reason) && nzchar(fit$fallback_reason),
      info = case_name
    )
    expect_identical(fit$coefficients, reference$coefficients, info = case_name)
    expect_identical(fit$fitted.values, reference$fitted.values, info = case_name)
    expect_identical(fit$converged, isTRUE(reference$converged), info = case_name)
    expect_equal(fit$iterations, reference$iter, info = case_name)
  }
  expect_error(
    perform_response_precomputation(
      cases$rank_deficient$y, cases$rank_deficient$x,
      response_fit_method = "pysceptre"
    ),
    "duplicate.*cannot be estimated|cannot be estimated.*duplicate"
  )
  zero_cache <- suppressWarnings(perform_response_precomputation(
    cases$zero_gene$y, cases$zero_gene$x, response_fit_method = "pysceptre"
  ))
  expect_named(zero_cache, c("fitted_coefs", "theta"))
  expect_identical(attr(zero_cache, "fit_diagnostics")$method_used, "glm.fit")
  expect_gte(zero_cache$theta, 0.01)
  expect_lte(zero_cache$theta, 1000)
})


test_that("pysceptre response failure controls and prepared designs are checked", {
  fixture <- make_pysceptre_response_fixture(n_cells = 120L, structured = FALSE)
  y <- fixture$response_matrix[1L, ]
  design <- fixture$covariate_matrix
  prepared <- prepare_pysceptre_glm_design(design)
  reference <- stats::glm.fit(y = y, x = design, family = stats::poisson())
  nonconverged <- fit_pysceptre_glm(
    expressions = y, covariate_matrix = design, family = "poisson",
    prepared_design = prepared, max_iterations = 0L
  )
  expect_identical(nonconverged$method_used, "glm.fit")
  expect_true(nzchar(nonconverged$fallback_reason))
  expect_identical(nonconverged$coefficients, reference$coefficients)
  expect_identical(nonconverged$fitted.values, reference$fitted.values)

  altered <- design
  altered[1L, 2L] <- altered[1L, 2L] + 0.5
  expect_error(
    fit_pysceptre_glm(y, altered, prepared_design = prepared),
    "prepared_design|covariate matrix"
  )
  expect_error(
    fit_pysceptre_glm(y[-1L], design[-1L, , drop = FALSE], prepared_design = prepared),
    "prepared_design|covariate matrix"
  )
})


test_that("pysceptre response metadata invalidates caches only when the method changes", {
  set.seed(614)
  target_data <- make_mock_grna_target_data(c(1, 2), 1, 1, 3)
  empty_pairs <- data.frame(grna_target = character(), response_id = character())
  object <- import_data(
    response_matrix = make_mock_response_matrices(4, 50, patterns = "column"),
    grna_matrix = make_mock_grna_matrices(
      target_data, 50, non_nt_patterns = "column", nt_patterns = "row"
    ),
    grna_target_data_frame = target_data, moi = "high"
  ) |>
    set_analysis_parameters(discovery_pairs = empty_pairs)
  cached <- list(fitted_coefs = rep(0, ncol(object@covariate_matrix)), theta = 4)
  object@response_precomputations <- list(response_1 = cached)
  selected <- set_analysis_parameters(
    object, discovery_pairs = empty_pairs, response_fit_method = "pysceptre"
  )
  expect_identical(get_response_fit_method(selected), "pysceptre")
  expect_identical(selected@response_precomputations, list())
  selected@response_precomputations <- list(response_1 = cached)
  unchanged <- set_analysis_parameters(
    selected, discovery_pairs = empty_pairs, response_fit_method = "pysceptre"
  )
  expect_identical(unchanged@response_precomputations, list(response_1 = cached))
  reset <- set_analysis_parameters(
    selected, discovery_pairs = empty_pairs, response_fit_method = "sceptre"
  )
  expect_identical(reset@response_precomputations, list())
  object_path <- tempfile(fileext = ".rds")
  on.exit(unlink(object_path), add = TRUE)
  saveRDS(selected, object_path)
  expect_identical(get_response_fit_method(readRDS(object_path)), "pysceptre")
})


test_that("high-level CRT and permutation analyses propagate the pysceptre backend", {
  captured <- list()
  capture_runner <- function(...) {
    captured[[length(captured) + 1L]] <<- list(...)
    list(
      result = data.frame(response_id = "gene_1", p_value = 0.5),
      response_precomputations = list()
    )
  }
  testthat::local_mocked_bindings(
    run_crt_in_memory_v2 = capture_runner,
    run_perm_test_in_memory = capture_runner,
    get_synthetic_permutation_idxs = function(...) matrix(1L, 1L, 1L),
    .package = "sceptre"
  )
  object <- methods::new("sceptre_object")
  object@response_matrix <- list(matrix(
    1, nrow = 1L, ncol = 3L, dimnames = list("gene_1", NULL)
  ))
  object@grna_assignments <- list()
  object@covariate_matrix <- matrix(1, nrow = 3L, ncol = 1L)
  object@resampling_approximation <- "no_approximation"
  object@B1 <- 1L
  object@B2 <- 0L
  object@B3 <- 0L
  object@control_group_complement <- TRUE
  object@n_nonzero_trt_thresh <- 0L
  object@n_nonzero_cntrl_thresh <- 0L
  object@side_code <- 0L
  object@low_moi <- FALSE
  object@response_precomputations <- list()
  object@cells_in_use <- seq_len(3L)
  object <- set_response_fit_method(object, "pysceptre")
  pairs <- data.frame(response_id = "gene_1", grna_group = "t1", pass_qc = TRUE)
  for (use_permutations in c(FALSE, TRUE)) {
    object@run_permutations <- use_permutations
    run_sceptre_analysis_high_level(
      sceptre_object = object, response_grna_group_pairs = pairs,
      calibration_check = FALSE, analysis_type = "discovery_analysis",
      output_amount = 1L, print_progress = FALSE, parallel = FALSE,
      n_processors = 1L, log_dir = tempdir()
    )
  }
  expect_length(captured, 2L)
  expect_identical(lapply(captured, `[[`, "response_fit_method"), rep(list("pysceptre"), 2L))
  expect_identical(lapply(captured, `[[`, "response_fit_chunk_size"), rep(list(16L), 2L))
})


test_that("pysceptre response models run in CRT and permutation global cache paths", {
  for (use_permutations in c(FALSE, TRUE)) {
    args <- make_pysceptre_response_runner_args()
    runner <- run_crt_in_memory_v2
    if (use_permutations) {
      args$synthetic_idxs <- fisher_yates_samlper(120L, 20L, args$B1)
      runner <- run_perm_test_in_memory
    }
    result <- do.call(runner, args)
    expect_equal(nrow(result$result), 4L)
    expect_true(all(is.finite(result$result$p_value)))
    expect_named(result$response_precomputations, c("gene_1", "gene_2"))
    expect_identical(result$phase_timings$response_fit_method, "pysceptre")
    for (response_id in c("gene_1", "gene_2")) {
      expected <- perform_response_precomputation(
        as.numeric(args$response_matrix[response_id, args$cells_in_use]),
        args$covariate_matrix[args$cells_in_use, , drop = FALSE],
        response_fit_method = "pysceptre"
      )
      expect_identical(result$response_precomputations[[response_id]], expected)
    }
    args$response_precomputations <- result$response_precomputations
    repeated <- do.call(runner, args)
    expect_identical(repeated$response_precomputations, result$response_precomputations)
    expect_equal(repeated$phase_timings$n_response_fits_created, 0L)
  }
})


test_that("pysceptre fits use each target's treatment and NT subset in both runners", {
  original_fit <- fit_pysceptre_glm
  captured <- list()
  testthat::local_mocked_bindings(
    fit_pysceptre_glm = function(expressions, covariate_matrix,
                               family = "poisson", prepared_design = NULL) {
      captured[[length(captured) + 1L]] <<- list(
        y = expressions, x = covariate_matrix
      )
      original_fit(expressions, covariate_matrix, family, prepared_design)
    },
    .package = "sceptre"
  )
  for (use_permutations in c(FALSE, TRUE)) {
    args <- make_pysceptre_response_runner_args()
    args$control_group_complement <- FALSE
    captured <- list()
    runner <- run_crt_in_memory_v2
    if (use_permutations) {
      args$synthetic_idxs <- hybrid_fisher_iwor_sampler(
        N = 50L, m = 15L, M = 20L, B = args$B1
      )
      runner <- run_perm_test_in_memory
    }
    result <- do.call(runner, args)
    expect_equal(nrow(result$result), 4L)
    expect_true(all(is.finite(result$result$p_value)))
    expect_length(result$response_precomputations, 0L)
    expect_length(captured, 4L)
    expected <- unlist(lapply(c("t1", "t2"), function(target) {
      rows <- c(args$grna_assignments$grna_group_idxs[[target]], 51:100)
      lapply(c("gene_1", "gene_2"), function(gene) {
        list(
          y = as.numeric(args$response_matrix[gene, args$cells_in_use[rows]]),
          x = args$covariate_matrix[args$cells_in_use[rows], , drop = FALSE]
        )
      })
    }), recursive = FALSE)
    # CRT loops over targets, permutations over responses: compare the actual
    # cell/expression mappings independently of loop order.
    for (expected_fit in expected) {
      matches <- vapply(captured, function(actual_fit) {
        isTRUE(all.equal(actual_fit$y, expected_fit$y, check.attributes = FALSE)) &&
          identical(actual_fit$x, expected_fit$x)
      }, logical(1L))
      expect_equal(sum(matches), 1L)
    }
  }
})


test_that("existing response backends never enter the pysceptre engine", {
  fixture <- make_pysceptre_response_fixture(n_cells = 120L, structured = FALSE)
  testthat::local_mocked_bindings(
    fit_pysceptre_glm = function(...) stop("unexpected pysceptre fit"),
    prepare_pysceptre_glm_design = function(...) stop("unexpected pysceptre design"),
    .package = "sceptre"
  )
  default <- perform_response_precomputation(
    fixture$response_matrix[1L, ], fixture$covariate_matrix
  )
  expect_identical(default, perform_sceptre_response_precomputation(
    fixture$response_matrix[1L, ], fixture$covariate_matrix
  ))
  expect_null(attr(default, "fit_diagnostics", exact = TRUE))
})


test_that("glmGamPoi remains independently selectable beside pysceptre", {
  skip_if_not_installed("glmGamPoi", minimum_version = "1.16.0")
  fixture <- make_pysceptre_response_fixture(n_cells = 120L, structured = FALSE)
  testthat::local_mocked_bindings(
    fit_pysceptre_glm = function(...) stop("unexpected pysceptre fit"),
    prepare_pysceptre_glm_design = function(...) stop("unexpected pysceptre design"),
    .package = "sceptre"
  )
  fit <- perform_response_precomputation(
    fixture$response_matrix[1L, ], fixture$covariate_matrix,
    response_fit_method = "glmGamPoi"
  )
  expect_named(fit, c("fitted_coefs", "theta"))
  expect_true(all(is.finite(fit$fitted_coefs)))
  expect_true(is.finite(fit$theta))
  expect_null(attr(fit, "fit_diagnostics", exact = TRUE))
})
