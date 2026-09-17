make_grna_fit_fixture <- function(n_cells = 500L, p = 6L, seed = 721L) {
  stopifnot(p >= 1L)
  set.seed(seed)
  if (p == 1L) {
    covariate_matrix <- matrix(
      1,
      nrow = n_cells,
      ncol = 1L,
      dimnames = list(NULL, "(Intercept)")
    )
    linear_predictor <- rep(stats::qlogis(0.2), n_cells)
  } else {
    predictors <- matrix(
      stats::rnorm(n_cells * (p - 1L)),
      nrow = n_cells,
      ncol = p - 1L
    )
    colnames(predictors) <- paste0("z", seq_len(p - 1L))
    covariate_matrix <- cbind(`(Intercept)` = 1, predictors)
    slopes <- seq(-0.35, 0.35, length.out = p - 1L)
    linear_predictor <- drop(
      stats::qlogis(0.2) + predictors %*% slopes
    )
  }
  indicator <- stats::rbinom(
    n = n_cells,
    size = 1L,
    prob = stats::plogis(linear_predictor)
  )
  stopifnot(any(indicator == 0L), any(indicator == 1L))
  list(
    covariate_matrix = covariate_matrix,
    indicator = indicator,
    trt_idxs = which(indicator == 1L)
  )
}


fit_grna_with_glm <- function(fixture, epsilon = NULL) {
  control <- if (is.null(epsilon)) {
    stats::glm.control()
  } else {
    stats::glm.control(epsilon = epsilon, maxit = 100L)
  }
  stats::glm.fit(
    y = fixture$indicator,
    x = fixture$covariate_matrix,
    family = stats::binomial(),
    control = control
  )
}


test_that("the default gRNA fit remains the bare glm.fit result", {
  fixture <- make_grna_fit_fixture()
  reference <- fit_grna_with_glm(fixture)

  default_probabilities <- perform_grna_precomputation(
    trt_idxs = fixture$trt_idxs,
    covariate_matrix = fixture$covariate_matrix,
    return_fitted_values = TRUE
  )
  explicit_probabilities <- perform_grna_precomputation(
    trt_idxs = fixture$trt_idxs,
    covariate_matrix = fixture$covariate_matrix,
    return_fitted_values = TRUE,
    grna_fit_method = "glm.fit"
  )
  default_coefficients <- perform_grna_precomputation(
    trt_idxs = fixture$trt_idxs,
    covariate_matrix = fixture$covariate_matrix,
    return_fitted_values = FALSE
  )

  expect_identical(default_probabilities, reference$fitted.values)
  expect_identical(explicit_probabilities, reference$fitted.values)
  expect_identical(default_coefficients, reference$coefficients)
  expect_identical(
    attributes(default_probabilities),
    attributes(reference$fitted.values)
  )
  expect_identical(
    attributes(default_coefficients),
    attributes(reference$coefficients)
  )

  details <- perform_grna_precomputation(
    trt_idxs = fixture$trt_idxs,
    covariate_matrix = fixture$covariate_matrix,
    return_fitted_values = TRUE,
    return_details = TRUE
  )
  expect_named(
    details,
    c(
      "value", "method_requested", "method_used", "fallback_reason",
      "converged", "iterations"
    )
  )
  expect_identical(details$value, reference$fitted.values)
  expect_identical(details$method_requested, "glm.fit")
  expect_identical(details$method_used, "glm.fit")
  expect_true(is.na(details$fallback_reason))
  expect_identical(details$converged, isTRUE(reference$converged))
  expect_identical(details$iterations, reference$iter)
})


test_that("fast logistic agrees with a tight glm.fit reference", {
  fixtures <- list(
    p6 = make_grna_fit_fixture(p = 6L, seed = 1006L),
    p19 = make_grna_fit_fixture(n_cells = 700L, p = 19L, seed = 1019L)
  )
  prepared_by_p <- lapply(fixtures, function(fixture) {
    prepare_grna_logistic_design(fixture$covariate_matrix)
  })

  for (fixture_name in names(fixtures)) {
    fixture <- fixtures[[fixture_name]]
    reference <- fit_grna_with_glm(fixture, epsilon = 1e-12)
    prepared_details <- perform_grna_precomputation(
      trt_idxs = fixture$trt_idxs,
      covariate_matrix = fixture$covariate_matrix,
      return_fitted_values = TRUE,
      grna_fit_method = "fast_logistic",
      prepared_design = prepared_by_p[[fixture_name]],
      return_details = TRUE
    )
    direct_probabilities <- perform_grna_precomputation(
      trt_idxs = fixture$trt_idxs,
      covariate_matrix = fixture$covariate_matrix,
      return_fitted_values = TRUE,
      grna_fit_method = "fast_logistic"
    )
    fast_coefficients <- perform_grna_precomputation(
      trt_idxs = fixture$trt_idxs,
      covariate_matrix = fixture$covariate_matrix,
      return_fitted_values = FALSE,
      grna_fit_method = "fast_logistic",
      prepared_design = prepared_by_p[[fixture_name]]
    )

    expect_identical(prepared_details$method_requested, "fast_logistic")
    expect_identical(prepared_details$method_used, "fast_logistic")
    expect_true(is.na(prepared_details$fallback_reason))
    expect_true(prepared_details$converged)
    expect_true(is.numeric(prepared_details$iterations))
    expect_length(prepared_details$iterations, 1L)
    expect_lt(
      max(abs(prepared_details$value - reference$fitted.values)),
      1e-7
    )
    expect_lt(
      max(abs(direct_probabilities - reference$fitted.values)),
      1e-7
    )
    expect_lt(
      max(abs(fast_coefficients - reference$coefficients)),
      1e-6
    )
    expect_identical(
      attributes(direct_probabilities),
      attributes(reference$fitted.values)
    )
    expect_identical(
      attributes(fast_coefficients),
      attributes(reference$coefficients)
    )
  }
})


test_that("prepared gRNA designs must match the covariate dimensions", {
  fixture <- make_grna_fit_fixture(p = 6L, seed = 141L)
  prepared <- prepare_grna_logistic_design(fixture$covariate_matrix)
  wider_fixture <- make_grna_fit_fixture(p = 7L, seed = 142L)

  expect_no_error(perform_grna_precomputation(
    trt_idxs = fixture$trt_idxs,
    covariate_matrix = fixture$covariate_matrix,
    return_fitted_values = TRUE,
    grna_fit_method = "fast_logistic",
    prepared_design = prepared
  ))
  expect_error(
    perform_grna_precomputation(
      trt_idxs = wider_fixture$trt_idxs,
      covariate_matrix = wider_fixture$covariate_matrix,
      return_fitted_values = TRUE,
      grna_fit_method = "fast_logistic",
      prepared_design = prepared
    ),
    "[Pp]repared|[Dd]esign"
  )
})


test_that("unsafe fast logistic fits fall back exactly to glm.fit", {
  fixture <- make_grna_fit_fixture(n_cells = 300L, p = 6L, seed = 1902L)
  z <- fixture$covariate_matrix
  cases <- list(
    single_cell = list(
      covariate_matrix = matrix(1, 1L, 1L),
      indicator = 1L
    ),
    empty_design = list(
      covariate_matrix = matrix(numeric(), 5L, 0L),
      indicator = c(1L, 0L, 0L, 1L, 0L)
    ),
    rank_deficient = list(
      covariate_matrix = cbind(z, duplicate = z[, 2L]),
      indicator = fixture$indicator
    ),
    no_intercept = list(
      covariate_matrix = z[, -1L, drop = FALSE],
      indicator = fixture$indicator
    ),
    all_zero = list(
      covariate_matrix = z,
      indicator = integer(nrow(z))
    ),
    all_one = list(
      covariate_matrix = z,
      indicator = rep.int(1L, nrow(z))
    ),
    separated = list(
      covariate_matrix = z,
      indicator = as.integer(z[, 2L] > 0)
    )
  )

  for (case_name in names(cases)) {
    case <- cases[[case_name]]
    reference <- suppressWarnings(stats::glm.fit(
      y = case$indicator,
      x = case$covariate_matrix,
      family = stats::binomial()
    ))
    details <- suppressWarnings(perform_grna_precomputation(
      trt_idxs = which(case$indicator == 1L),
      covariate_matrix = case$covariate_matrix,
      return_fitted_values = TRUE,
      grna_fit_method = "fast_logistic",
      return_details = TRUE
    ))

    expect_identical(details$method_requested, "fast_logistic")
    expect_identical(details$method_used, "glm.fit")
    expect_true(
      is.character(details$fallback_reason) &&
        length(details$fallback_reason) == 1L &&
        !is.na(details$fallback_reason) &&
        nzchar(details$fallback_reason),
      info = case_name
    )
    expect_identical(details$value, reference$fitted.values, info = case_name)
    expect_identical(
      details$converged,
      isTRUE(reference$converged),
      info = case_name
    )
    expect_identical(details$iterations, reference$iter, info = case_name)
  }
})


test_that("fast solver nonconvergence falls back exactly to glm.fit", {
  fixture <- make_grna_fit_fixture(n_cells = 250L, p = 6L, seed = 1903L)
  reference <- fit_grna_with_glm(fixture)
  prepared <- prepare_grna_logistic_design(fixture$covariate_matrix)
  testthat::local_mocked_bindings(
    fit_grna_logistic_prepared_cpp = function(...) {
      list(
        coefficients = rep(NA_real_, ncol(fixture$covariate_matrix)),
        fitted.values = numeric(0),
        converged = FALSE,
        iterations = 25L,
        reason = "maximum_iterations_reached"
      )
    },
    .package = "sceptre"
  )

  details <- perform_grna_precomputation(
    trt_idxs = fixture$trt_idxs,
    covariate_matrix = fixture$covariate_matrix,
    return_fitted_values = TRUE,
    grna_fit_method = "fast_logistic",
    prepared_design = prepared,
    return_details = TRUE
  )

  expect_identical(details$value, reference$fitted.values)
  expect_identical(details$method_requested, "fast_logistic")
  expect_identical(details$method_used, "glm.fit")
  expect_identical(details$fallback_reason, "maximum_iterations_reached")
  expect_identical(details$converged, isTRUE(reference$converged))
  expect_identical(details$iterations, reference$iter)
})


test_that("invalid fast probabilities fall back exactly to glm.fit", {
  fixture <- make_grna_fit_fixture(n_cells = 250L, p = 6L, seed = 1905L)
  reference <- fit_grna_with_glm(fixture)
  prepared <- prepare_grna_logistic_design(fixture$covariate_matrix)
  testthat::local_mocked_bindings(
    fit_grna_logistic_prepared_cpp = function(...) {
      list(
        coefficients = rep(0, ncol(fixture$covariate_matrix)),
        fitted.values = c(0, rep(0.2, nrow(fixture$covariate_matrix) - 1L)),
        converged = TRUE,
        iterations = 2L,
        reason = "ok"
      )
    },
    .package = "sceptre"
  )

  details <- perform_grna_precomputation(
    trt_idxs = fixture$trt_idxs,
    covariate_matrix = fixture$covariate_matrix,
    return_fitted_values = FALSE,
    grna_fit_method = "fast_logistic",
    prepared_design = prepared,
    return_details = TRUE
  )

  expect_identical(details$value, reference$coefficients)
  expect_identical(details$method_requested, "fast_logistic")
  expect_identical(details$method_used, "glm.fit")
  expect_identical(details$fallback_reason, "invalid_fitted_probabilities")
  expect_identical(details$converged, isTRUE(reference$converged))
  expect_identical(details$iterations, reference$iter)
})


test_that("fast fallback preserves native glm.fit errors", {
  fixture <- make_grna_fit_fixture(n_cells = 100L, p = 6L, seed = 1906L)
  fixture$covariate_matrix[1L, 2L] <- Inf
  capture_error <- function(expr) {
    tryCatch(
      {
        force(expr)
        NA_character_
      },
      error = conditionMessage
    )
  }

  native_error <- capture_error(stats::glm.fit(
    y = fixture$indicator,
    x = fixture$covariate_matrix,
    family = stats::binomial()
  ))
  fast_error <- capture_error(perform_grna_precomputation(
    trt_idxs = fixture$trt_idxs,
    covariate_matrix = fixture$covariate_matrix,
    return_fitted_values = TRUE,
    grna_fit_method = "fast_logistic"
  ))

  expect_false(is.na(native_error))
  expect_identical(fast_error, native_error)
})


test_that("fast logistic validates treated indices and prepared designs", {
  fixture <- make_grna_fit_fixture(n_cells = 100L, p = 6L, seed = 1904L)
  prepared <- prepare_grna_logistic_design(fixture$covariate_matrix)
  fit_with_indices <- function(trt_idxs) {
    perform_grna_precomputation(
      trt_idxs = trt_idxs,
      covariate_matrix = fixture$covariate_matrix,
      return_fitted_values = TRUE,
      grna_fit_method = "fast_logistic",
      prepared_design = prepared
    )
  }

  expect_error(fit_with_indices(0L), "out-of-range")
  expect_error(fit_with_indices(nrow(fixture$covariate_matrix) + 1L), "out-of-range")
  expect_error(fit_with_indices(NA_integer_), "finite, integer-valued")
  expect_error(fit_with_indices(c(1L, 1L)), "duplicates")
  expect_error(fit_with_indices(1.5), "finite, integer-valued")
  expect_error(
    perform_grna_precomputation(
      trt_idxs = fixture$trt_idxs,
      covariate_matrix = fixture$covariate_matrix,
      return_fitted_values = TRUE,
      grna_fit_method = "fast_logistic",
      prepared_design = list()
    ),
    "prepare_grna_logistic_design"
  )
  expect_error(
    perform_grna_precomputation(
      trt_idxs = fixture$trt_idxs,
      covariate_matrix = fixture$covariate_matrix,
      return_fitted_values = TRUE,
      grna_fit_method = "fast_logistic",
      return_details = NA
    ),
    "`return_details`"
  )
})


test_that("gRNA fit method metadata is backward compatible", {
  legacy_object <- methods::new("sceptre_object")
  expect_identical(get_grna_fit_method(legacy_object), "glm.fit")

  fast_object <- set_grna_fit_method(legacy_object, "fast_logistic")
  object_path <- tempfile(fileext = ".rds")
  on.exit(unlink(object_path), add = TRUE)
  saveRDS(fast_object, object_path)
  restored_object <- readRDS(object_path)

  expect_identical(get_grna_fit_method(restored_object), "fast_logistic")
})


test_that("gRNA fit method validation is strict and preserves response caches", {
  set.seed(216)
  n_cells <- 50L
  grna_target_data_frame <- make_mock_grna_target_data(
    num_guides_per_target = c(1, 2),
    chr_distances = 1,
    chr_starts = 1,
    num_nt_guides = 3
  )
  response_matrix <- make_mock_response_matrices(
    num_responses = 4,
    num_cells = n_cells,
    patterns = "column"
  )
  grna_matrix <- make_mock_grna_matrices(
    grna_target_data_frame,
    n_cells,
    non_nt_patterns = "column",
    nt_patterns = "row"
  )
  empty_pairs <- data.frame(
    grna_target = character(0),
    response_id = character(0)
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
    grna_fit_method = "glm.fit"
  )
  changed <- set_analysis_parameters(
    sceptre_object,
    discovery_pairs = empty_pairs,
    grna_fit_method = "fast_logistic"
  )
  invalid_call <- function(grna_fit_method) {
    set_analysis_parameters(
      sceptre_object,
      discovery_pairs = empty_pairs,
      grna_fit_method = grna_fit_method
    )
  }

  expect_identical(unchanged@response_precomputations, list(response_1 = cached_fit))
  expect_identical(changed@response_precomputations, list(response_1 = cached_fit))
  expect_identical(get_grna_fit_method(unchanged), "glm.fit")
  expect_identical(get_grna_fit_method(changed), "fast_logistic")
  expect_error(invalid_call("fast"), "`grna_fit_method`")
  expect_error(
    invalid_call(c("glm.fit", "fast_logistic")),
    "`grna_fit_method`"
  )
  expect_error(invalid_call(NA_character_), "`grna_fit_method`")
  expect_error(invalid_call(1), "`grna_fit_method`")
})
