.rpt_moment_fixture <- local({
  fixture <- NULL
  function() {
    if (!is.null(fixture)) return(fixture)
    set.seed(93)
    n <- 2000L
    Z <- cbind(1, matrix(rnorm(n * 5L), n, 5L))
    mu <- exp(-1.4 + 0.1 * Z[, 2L] - 0.2 * Z[, 3L])
    y <- rnbinom(n, mu = mu, size = 2)
    w <- mu / (1 + mu / 2)
    a <- (y - mu) / (1 + mu / 2)
    fixture <<- list(n = n, Z = Z, mu = mu, y = y, a = a, w = w)
    fixture
  }
})

.rpt_moment_prepare <- function(fixture, ...) {
  do.call(prepare_rpt_spa_response_cpp, utils::modifyList(list(
    a = fixture$a, w = fixture$w, Z = fixture$Z,
    zero_mask = fixture$y == 0, minimum_zero_fraction = 0
  ), list(...)))
}

.rpt_moment_statistic <- function(indices, fixture) {
  B <- colSums(fixture$w[indices] * fixture$Z[indices, , drop = FALSE])
  variance <- sum(fixture$w[indices]) - drop(crossprod(
    B, solve(crossprod(fixture$Z, fixture$w * fixture$Z), B)
  ))
  sum(fixture$a[indices]) / sqrt(variance)
}

.rpt_moment_center <- function(fixture, m) {
  u <- m / fixture$n
  B <- u * colSums(fixture$w * fixture$Z)
  variance <- u * sum(fixture$w) - drop(crossprod(
    B, solve(crossprod(fixture$Z, fixture$w * fixture$Z), B)
  ))
  u * sum(fixture$a) / sqrt(variance)
}

.rpt_moment_always <- function(fixture, ...) {
  do.call(run_low_level_test_full_rpt_spa_always_v1,
    utils::modifyList(list(
      y = fixture$y, mu = fixture$mu, a = fixture$a, w = fixture$w,
      Z = fixture$Z, trt_idxs = 1:30, n_trt = 30L, side_code = 1L
    ), list(...)))
}

.rpt_moment_mock_success <- function() {
  list(p = 0.1, needs_empirical_fallback = FALSE, resampling_dist = numeric())
}

.rpt_moment_mock_workhorse <- function(...) {
  do.call(sceptre:::perm_test_glm_factored_out, utils::modifyList(list(
    synthetic_idxs = NULL, B1 = 0L, B2 = 19L, B3 = 0L,
    fit_parametric_curve = FALSE, output_amount = 3L,
    grna_groups = c("one", "two"), expression_vector = c(0, 1, 0, 2),
    pieces_precomp = list(mu = rep(1, 4L), a = c(-1, 0, -1, 1),
                         b = rep(0.5, 4L), w = rep(0.5, 4L)),
    get_idx_f = function(...) list(trt_idxs = 1:2, n_trt = 2L),
    side_code = 1L, covariate_matrix = matrix(1, 4L, 1L),
    use_rpt_spa_always = TRUE, use_rpt_spa_fast = TRUE
  ), list(...)))
}

test_that("28 and 84 moment fits retain exact probabilities through guarded fallback", {
  f <- .rpt_moment_fixture()
  for (degree in 2:3) {
    context <- .rpt_moment_prepare(f, maximum_moment_degree = degree)
    on.exit(release_rpt_spa_response_cpp(context), add = TRUE)
    for (treated in list(1:30, 101:140)) {
      result <- rpt_spa_moment_outward_prepared_cpp(
        context, treated, delta_limit = 2
      )
      reference <- rpt_spa_full_cpp(
        f$a, f$w, f$Z, length(treated),
        result$outward_score_sign * result$observed_target,
        score_sign = result$outward_score_sign, tolerance = 1e-9
      )
      expect_true(result$converged)
      expect_true(reference$converged)
      expect_true(result$moment_eligible)
      if (result$exact_audit_passed) {
        expect_match(result$path, "^moment_compressed_exact_(audit|polish)$")
      } else {
        expect_identical(result$path, "moment_full_exact_fallback")
        expect_identical(result$moment_fallback_reason,
                         "invalid_compressed_evaluation")
      }
      expect_equal(result$packed_moment_count, if (degree == 2L) 28L else 84L)
      expect_equal(result$maximum_moment_degree, degree)
      expect_equal(result$effective_exact_audit_tolerance, 1e-4)
      expect_lte(result$max_residual, 1e-4)
      expect_equal(result$observed_target, .rpt_moment_statistic(treated, f),
                   tolerance = 1e-12)
      expect_lt(abs(result$p_value - reference$p_value), 2e-4)
      strict <- rpt_spa_moment_outward_prepared_cpp(
        context, treated, tolerance = 1e-9, delta_limit = 2,
        exact_audit_tolerance = 1e-9
      )
      expect_true(strict$converged)
      expect_lte(strict$max_residual, 1e-9)
      expect_lt(abs(strict$p_value - reference$p_value), 1e-7)
    }
    # A modest outward displacement exercises successful compression at both
    # degrees; the larger observed displacements above also test its guards.
    target <- .rpt_moment_center(f, 30L) + 0.5
    audited <- rpt_spa_moment_prepared_cpp(context, 30L, target, delta_limit = 2)
    reference <- rpt_spa_full_cpp(f$a, f$w, f$Z, 30L, target, tolerance = 1e-9)
    expect_true(audited$converged)
    expect_true(reference$converged)
    expect_true(audited$exact_audit_passed)
    expect_match(audited$path, "^moment_compressed_exact_(audit|polish)$")
    expect_identical(audited$moment_fallback_reason, "none")
    expect_equal(audited$packed_moment_count, if (degree == 2L) 28L else 84L)
    expect_lte(audited$max_residual, 1e-4)
    expect_lt(abs(audited$p_value - reference$p_value), 2e-4)
    release_rpt_spa_response_cpp(context)
  }
})

test_that("prepared responses safely reuse geometry across counts and signs", {
  f <- .rpt_moment_fixture()
  context <- .rpt_moment_prepare(f)
  on.exit(release_rpt_spa_response_cpp(context))
  counts <- c(30L, 40L, 30L, 30L)
  signs <- c(1L, -1L, -1L, 1L)
  results <- vector("list", length(counts))
  for (i in seq_along(counts)) {
    fresh <- .rpt_moment_prepare(f)
    target <- signs[i] * .rpt_moment_center(f, counts[i]) + 1
    args <- list(m = counts[i], target = target, score_sign = signs[i],
                 tolerance = 1e-8, exact_audit_tolerance = 1e-8,
                 delta_limit = 2)
    result <- do.call(rpt_spa_moment_prepared_cpp,
                      c(list(prepared_context = context), args))
    reference <- do.call(rpt_spa_moment_prepared_cpp,
                         c(list(prepared_context = fresh), args))
    release_rpt_spa_response_cpp(fresh)
    expect_true(result$converged)
    # Profiling diagnostics, if enabled later, need not have equal timings.
    results[[i]] <- result[!grepl("_ms$", names(result))]
    expect_equal(results[[i]], reference[!grepl("_ms$", names(reference))],
                 tolerance = 1e-12)
    expect_equal(result$treated_count, counts[i])
  }
  expect_equal(results[[1L]], results[[4L]], tolerance = 1e-12)
})

test_that("dense and locality guards fall back to the exact conditional solver", {
  f <- .rpt_moment_fixture()
  dense <- .rpt_moment_prepare(f, zero_mask = rep(FALSE, f$n),
                              minimum_zero_fraction = 0.8)
  sparse <- .rpt_moment_prepare(f)
  on.exit(release_rpt_spa_response_cpp(dense), add = TRUE)
  on.exit(release_rpt_spa_response_cpp(sparse), add = TRUE)
  target <- .rpt_moment_center(f, 30L) + 1
  reference <- rpt_spa_full_cpp(f$a, f$w, f$Z, 30L, target, tolerance = 1e-8)
  dense_result <- rpt_spa_moment_prepared_cpp(
    dense, 30L, target, tolerance = 1e-8
  )
  locality_result <- rpt_spa_moment_prepared_cpp(
    sparse, 30L, target, tolerance = 1e-8, delta_limit = 1e-12
  )
  expect_true(reference$converged)
  expect_false(dense_result$moment_eligible)
  expect_equal(dense_result$packed_moment_count, 0L)
  expect_identical(dense_result$moment_fallback_reason, "insufficient_zero_fraction")
  expect_match(locality_result$moment_fallback_reason, "locality")
  for (result in list(dense_result, locality_result)) {
    expect_true(result$converged)
    expect_identical(result$path, "moment_full_exact_fallback")
    expect_false(result$exact_audit_passed)
    expect_equal(result$p_value, reference$p_value, tolerance = 1e-7)
  }
  for (context in list(dense, sparse)) {
    disabled <- rpt_spa_moment_prepared_cpp(context, 30L, target, max_iterations = 0L)
    expect_false(disabled$converged)
    expect_identical(disabled$moment_fallback_reason,
                     "solver_disabled_max_iterations_zero")
    expect_identical(disabled$path, "moment_solver_disabled_empirical_fallback")
    expect_identical(disabled$final_result_source, "empirical_fallback_required")
    expect_equal(disabled$exact_evaluations, 0L)
    expect_equal(disabled$compressed_evaluations, 0L)
  }
})

test_that("prepared response pointers validate identity and release idempotently", {
  f <- .rpt_moment_fixture()
  context <- .rpt_moment_prepare(f)
  expect_s3_class(context, "sceptre_rpt_spa_response_v1")
  expect_equal(attr(context, "n_cells"), f$n)
  expect_equal(attr(context, "feature_dimension"), ncol(f$Z))
  wrong_tag <- methods::new("externalptr")
  class(wrong_tag) <- class(context)
  for (invalid in list(NULL, list(), wrong_tag)) {
    expect_error(rpt_spa_moment_prepared_cpp(invalid, 30L, 1),
                 "expected an RPT prepared response context")
    expect_error(release_rpt_spa_response_cpp(invalid),
                 "expected an RPT prepared response context")
  }
  class(context) <- "unrelated_context"
  expect_error(rpt_spa_moment_prepared_cpp(context, 30L, 1), "expected an RPT")
  class(context) <- "sceptre_rpt_spa_response_v1"
  expect_silent(release_rpt_spa_response_cpp(context))
  expect_silent(release_rpt_spa_response_cpp(context))
  expect_error(rpt_spa_moment_prepared_cpp(context, 30L, 1), "released")
  expect_error(rpt_spa_moment_outward_prepared_cpp(context, 1:30), "released")
})

test_that("moment preparation and solve reject malformed inputs and controls", {
  f <- .rpt_moment_fixture()
  context <- .rpt_moment_prepare(f)
  on.exit(release_rpt_spa_response_cpp(context))
  expect_error(.rpt_moment_prepare(f, zero_mask = FALSE), "one value per cell")
  mask <- f$y == 0
  mask[1L] <- NA
  expect_error(.rpt_moment_prepare(f, zero_mask = mask), "missing values")
  expect_error(.rpt_moment_prepare(f, zero_mask = rep(TRUE, f$n)),
               "NB zero-row identity")
  expect_error(.rpt_moment_prepare(f, minimum_zero_fraction = -0.1),
               "minimum_zero_fraction")
  expect_error(.rpt_moment_prepare(f, maximum_moment_degree = 1L),
               "maximum_moment_degree")
  expect_error(.rpt_moment_prepare(f, maximum_moment_degree = 5L),
               "maximum_moment_degree")
  for (m in c(0L, f$n)) {
    expect_error(rpt_spa_moment_prepared_cpp(context, m, 1), "m must be")
  }
  expect_error(rpt_spa_moment_prepared_cpp(context, 30L, 1, score_sign = 0L),
               "score_sign")
  expect_error(rpt_spa_moment_prepared_cpp(context, 30L, Inf), "target")
  for (control in c("tolerance", "compressed_tolerance", "exact_audit_tolerance",
                    "delta_limit")) {
    for (value in c(0, NA_real_, Inf)) {
      args <- c(list(prepared_context = context, m = 30L, target = 1),
                stats::setNames(list(value), control))
      expect_error(do.call(rpt_spa_moment_prepared_cpp, args), control)
    }
  }
  expect_error(rpt_spa_moment_prepared_cpp(context, 30L, 1, max_iterations = -1L),
               "max_iterations")
  expect_error(rpt_spa_moment_prepared_cpp(context, 30L, 1,
                                        maximum_polish_updates = -1L),
               "maximum_polish_updates")
  for (indices in list(integer(), 0:2, c(1L, NA_integer_), c(1L, f$n + 1L),
                       c(1L, 1L), seq_len(f$n))) {
    expect_error(rpt_spa_moment_outward_prepared_cpp(context, indices),
                 "treated_indices")
  }
})

test_that("fast always mode preserves side mappings and exact preparation backup", {
  f <- .rpt_moment_fixture()
  original <- .rpt_moment_always(f)
  expect_identical(original, .rpt_moment_always(f, use_moment = FALSE))
  context <- .rpt_moment_prepare(f)
  on.exit(release_rpt_spa_response_cpp(context))
  results <- lapply(c(-1L, 0L, 1L), function(side) {
    .rpt_moment_always(f, side_code = side, use_moment = TRUE,
                       prepared_context = context)
  })
  for (result in results) {
    expect_true(result$spa_converged)
    expect_identical(result$p_value_source, "rpt_spa_always_fast")
    expect_false(result$needs_empirical_fallback)
    expect_equal(result$spa_solver_tolerance, 1e-4)
    expect_equal(result$spa_moment_degree, 2L)
    expect_equal(result$spa_packed_moment_count, 28L)
  }
  p <- vapply(results, `[[`, numeric(1), "p")
  expect_equal(p[1L] + p[3L], 1, tolerance = 1e-12)
  expect_equal(p[2L], 2 * min(p[c(1L, 3L)]), tolerance = 1e-12)
  expect_lt(abs(results[[3L]]$p - original$p), 2e-4)
  for (provider in list(function() NULL, function() stop("forced preparation failure"))) {
    backup <- .rpt_moment_always(f, use_moment = TRUE,
                               prepared_context = provider)
    expect_true(backup$spa_converged)
    expect_false(backup$needs_empirical_fallback)
    expect_identical(backup$spa_cgf_mode, "exact")
    expect_identical(backup$spa_acceleration_path,
                     "original_exact_preparation_fallback")
    expect_lt(abs(backup$p - original$p), 2e-4)
  }
  failed <- .rpt_moment_always(f, use_moment = TRUE,
                             prepared_context = context, max_iterations = 0L)
  expect_false(failed$spa_converged)
  expect_true(failed$needs_empirical_fallback)
  expect_identical(failed$p_value_source, "B2_empirical_pending")
  expect_length(failed$resampling_dist, 0L)
  bank <- fisher_yates_samlper(n_tot = f$n, M = 30L, B = 29L)
  finished <- finalize_low_level_test_rpt_spa_fallback_v1(
    f$a, f$w, f$Z, 30L, bank, 29L, TRUE, 1L, failed
  )
  expect_false(finished$needs_empirical_fallback)
  expect_identical(finished$p_value_source, "B2_empirical")
  expect_length(finished$resampling_dist, 29L)
  expect_true(all(is.finite(finished$resampling_dist)))
  expect_equal(finished$spa_solver_tolerance, 1e-4)
})

test_that("screened moment fits prepare lazily and retain the exact backup", {
  f <- .rpt_moment_fixture()
  set.seed(902)
  bank <- fisher_yates_samlper(n_tot = f$n, M = 30L, B = 128L)
  indices <- synth_idx_list_to_r_list(bank)[seq_len(99L)]
  statistics <- vapply(indices, function(idx) .rpt_moment_statistic(idx + 1L, f),
                        numeric(1))
  treated <- indices[[which.min(abs(statistics - median(statistics)))]] + 1L
  args <- list(
    y = f$y, mu = f$mu, a = f$a, w = f$w,
    D = compute_D_matrix(crossprod(f$Z, f$w * f$Z), f$w * f$Z),
    Z = f$Z, trt_idxs = treated, n_trt = 30L, synthetic_idxs = bank,
    B1 = 99L, B2 = 29L, side_code = 0L, use_moment = TRUE,
    use_all_cells = FALSE, return_resampling_dist = TRUE,
    prepared_context = function() stop("A central screen prepared moments.")
  )
  result <- do.call(run_low_level_test_full_rpt_spa_v1, args)
  expect_identical(result$p_value_source, "B1_empirical")
  expect_false(result$spa_attempted)
  expect_gt(result$p, 0.02)
  args$trt_idxs <- indices[[which.max(statistics)]] + 1L
  args$side_code <- 1L
  args$prepared_context <- function() stop("forced preparation failure")
  backup <- do.call(run_low_level_test_full_rpt_spa_v1, args)
  expect_true(backup$spa_attempted)
  expect_true(backup$spa_converged)
  expect_identical(backup$p_value_source, "rpt_spa_fast")
  expect_identical(backup$spa_acceleration_path,
                   "original_exact_preparation_fallback")
  expect_match(backup$spa_acceleration_fallback_reason, "forced preparation failure")
  args$use_moment <- FALSE
  original <- do.call(run_low_level_test_full_rpt_spa_v1, args)
  expect_lt(abs(backup$p - original$p), 2e-4)

  interrupt <- function() stop(structure(
    list(message = "mock interrupt", call = NULL), class = c("interrupt", "condition")
  ))
  args$use_moment <- TRUE
  args$prepared_context <- interrupt
  expect_identical(tryCatch(
    do.call(run_low_level_test_full_rpt_spa_v1, args),
    interrupt = function(condition) "interrupt", error = function(condition) "error"
  ), "interrupt")
  expect_identical(tryCatch(
    .rpt_moment_always(f, use_moment = TRUE, prepared_context = interrupt),
    interrupt = function(condition) "interrupt", error = function(condition) "error"
  ), "interrupt")
})

test_that("response context factories are lazy, cache errors, and release once", {
  skip_if_not(exists("local_mocked_bindings", asNamespace("testthat")))
  preparations <- 0L
  releases <- 0L
  fail <- FALSE
  context <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    prepare_rpt_spa_response_cpp = function(...) {
      preparations <<- preparations + 1L
      if (fail) stop("mock preparation failure")
      context
    },
    release_rpt_spa_response_cpp = function(prepared_context) {
      expect_identical(prepared_context, context)
      releases <<- releases + 1L
    }, .package = "sceptre"
  )
  factory <- sceptre:::make_rpt_spa_response_context_factory(1, 1, matrix(1), 0)
  expect_identical(preparations, 0L)
  factory$release()
  expect_identical(releases, 0L)
  expect_identical(factory$get(), context)
  expect_identical(factory$get(), context)
  expect_identical(preparations, 1L)
  factory$release()
  factory$release()
  expect_identical(releases, 1L)
  fail <- TRUE
  failed <- sceptre:::make_rpt_spa_response_context_factory(1, 1, matrix(1), 0)
  expect_error(failed$get(), "mock preparation failure")
  expect_error(failed$get(), "mock preparation failure")
  expect_identical(preparations, 2L)
  expect_silent(failed$release())
  expect_identical(releases, 1L)
})

test_that("shared-response routing reuses moments and cleans up on errors", {
  skip_if_not(exists("local_mocked_bindings", asNamespace("testthat")))
  preparations <- 0L
  releases <- 0L
  seen <- list()
  fail <- FALSE
  testthat::local_mocked_bindings(
    prepare_rpt_spa_response_cpp = function(...) {
      preparations <<- preparations + 1L
      expect_equal(list(...)$zero_mask, c(TRUE, FALSE, TRUE, FALSE))
      new.env(parent = emptyenv())
    },
    release_rpt_spa_response_cpp = function(prepared_context) {
      releases <<- releases + 1L
    },
    run_low_level_test_full_rpt_spa_always_v1 = function(..., prepared_context) {
      expect_true(list(...)$use_moment)
      seen[[length(seen) + 1L]] <<- prepared_context()
      if (fail) stop("mock association failure")
      .rpt_moment_mock_success()
    },
    fisher_yates_samlper = function(...) stop("Successful SPA requested a bank."),
    .package = "sceptre"
  )
  result <- .rpt_moment_mock_workhorse()
  expect_length(result, 2L)
  expect_identical(preparations, 1L)
  expect_identical(releases, 1L)
  expect_identical(seen[[1L]], seen[[2L]])
  fail <- TRUE
  expect_error(.rpt_moment_mock_workhorse(), "mock association failure")
  expect_identical(preparations, 2L)
  expect_identical(releases, 2L)
})

test_that("pair-specific response refits use isolated moment contexts", {
  skip_if_not(exists("local_mocked_bindings", asNamespace("testthat")))
  prepared <- list()
  released <- list()
  testthat::local_mocked_bindings(
    get_idx_vector_discovery_analysis = function(curr_grna_group, ...) {
      idx <- if (curr_grna_group == "one") 1:2 else 3:5
      list(trt_idxs = idx, n_trt = length(idx))
    },
    perform_response_precomputation = function(...) {
      list(fitted_coefs = 0, theta = 2)
    },
    compute_precomputation_pieces = function(expression_vector, ...) {
      list(mu = rep(1, length(expression_vector)), a = expression_vector - 1,
           b = rep(0.5, length(expression_vector)))
    },
    prepare_rpt_spa_response_cpp = function(a, w, Z, zero_mask, ...) {
      pointer <- new.env(parent = emptyenv())
      prepared[[length(prepared) + 1L]] <<- list(
        pointer = pointer, a = a, Z = Z, zero_mask = zero_mask
      )
      pointer
    },
    release_rpt_spa_response_cpp = function(prepared_context) {
      released[[length(released) + 1L]] <<- prepared_context
    },
    run_low_level_test_full_rpt_spa_always_v1 = function(..., prepared_context) {
      prepared_context()
      .rpt_moment_mock_success()
    }, .package = "sceptre"
  )
  Z <- cbind(1, seq_len(8L))
  y <- c(0, 1, 2, 0, 1, 0, 2, 1)
  result <- sceptre:::discovery_ntcells_perm_test(
    synthetic_idxs = NULL, B1 = 0L, B2 = 19L, B3 = 0L,
    fit_parametric_curve = FALSE, output_amount = 3L, covariate_matrix = Z,
    all_nt_idxs = 6:8, grna_group_idxs = list(), grna_groups = c("one", "two"),
    expression_vector = y, side_code = 1L,
    use_rpt_spa_always = TRUE, use_rpt_spa_fast = TRUE
  )
  expect_length(result, 2L)
  expect_length(prepared, 2L)
  expect_length(released, 2L)
  expect_false(identical(prepared[[1L]]$pointer, prepared[[2L]]$pointer))
  for (i in 1:2) {
    indices <- c(if (i == 1L) 1:2 else 3:5, 6:8)
    expect_identical(released[[i]], prepared[[i]]$pointer)
    expect_equal(prepared[[i]]$Z, Z[indices, ])
    expect_equal(prepared[[i]]$a, y[indices] - 1)
    expect_equal(prepared[[i]]$zero_mask, y[indices] == 0)
  }
})

test_that("fast RPT public options retain the permutation-only bank plans", {
  set.seed(20260903)
  n <- 35L
  targets <- make_mock_grna_target_data(
    num_guides_per_target = 1, chr_distances = 1, chr_starts = 1, num_nt_guides = 2
  )
  imported <- import_data(
    response_matrix = matrix(rpois(3L * n, 1), nrow = 3L),
    grna_matrix = matrix(rpois(nrow(targets) * n, 1), nrow = nrow(targets),
                         dimnames = list(targets$grna_id, NULL)),
    grna_target_data_frame = targets, moi = "high"
  )
  pairs <- data.frame(grna_target = character(), response_id = character())
  for (method in c("rpt_spa_fast", "rpt_spa_always_fast")) {
    object <- set_analysis_parameters(imported, discovery_pairs = pairs,
      resampling_mechanism = "permutations", resampling_approximation = method)
    expect_identical(object@resampling_approximation, method)
    expect_identical(object@B1, if (method == "rpt_spa_fast") 499L else 0L)
    expect_identical(object@B2, 4999L)
    expect_identical(object@B3, 0L)
    expect_true(object@run_permutations)
    expect_error(set_analysis_parameters(imported, discovery_pairs = pairs,
      resampling_mechanism = "crt", resampling_approximation = method),
      "available only.*resampling_mechanism = 'permutations'")
  }
})

test_that("detailed output retains moment diagnostics without changing minimal output", {
  base <- list(p = 0.1, lfc = 0.2, stage = 2L, z_orig = 1.3,
               sn_params = rep(NA_real_, 3L), resampling_dist = numeric())
  diagnostics <- list(
    p_value_source = "rpt_spa_always_fast", spa_fast = TRUE,
    spa_cgf_mode = "moment_initialized_exact_audit",
    spa_cache_strategy = "prepared_response", spa_solver_tolerance = 1e-4,
    spa_compressed_tolerance = 1e-4, spa_exact_audit_tolerance = 1e-4,
    spa_moment_degree = 2L, spa_packed_moment_count = 28L,
    spa_acceleration_path = "moment_compressed_exact_polish",
    spa_acceleration_fallback_reason = "none", spa_moment_eligible = TRUE,
    spa_exact_audit_passed = TRUE, spa_experimental = TRUE
  )
  pairs <- data.frame(response_id = c("gene", "gene"),
                      grna_target = c("first", "second"))
  results <- list(c(base, diagnostics), base)
  detailed <- sceptre:::construct_data_frame_v2(pairs, results, output_amount = 3L)
  for (name in names(diagnostics)) {
    expect_identical(detailed[[name]][1L], diagnostics[[name]])
    expect_true(is.na(detailed[[name]][2L]))
  }
  expect_false(any(grepl("^z_null_", names(detailed))))
  minimal <- sceptre:::construct_data_frame_v2(pairs, results, output_amount = 1L)
  expect_false(any(names(diagnostics) %in% names(minimal)))
  expect_equal(minimal$p_value, c(0.1, 0.1))
})
