make_shared_crt_spa_runner_args <- function() {
  set.seed(572906)
  n <- 240L
  ids <- paste0("gene_", seq_len(4L))
  design <- cbind(`(Intercept)` = 1, z = stats::rnorm(n), batch = rep(0:1, length.out = n))
  counts <- matrix(stats::rnbinom(length(ids) * n, mu = 2, size = 3),
                   nrow = length(ids), dimnames = list(ids, NULL))
  fits <- stats::setNames(lapply(seq_along(ids), function(i) {
    list(fitted_coefs = stats::setNames(c(log(1.7 + i / 5), .04, -.03), colnames(design)),
         theta = 3 + i / 4)
  }), ids)
  list(response_matrix = methods::as(Matrix::Matrix(counts, sparse = TRUE), "RsparseMatrix"),
       grna_assignments = list(grna_group_idxs = list(t1 = 1:20, t2 = 21:40,
         t3 = 41:60, t4 = 61:80), indiv_nt_grna_idxs = list(nt1 = 101:140), all_nt_idxs = 101:220),
       covariate_matrix = design,
       response_grna_group_pairs = expand.grid(response_id = ids,
         grna_group = paste0("t", seq_len(4L)), stringsAsFactors = FALSE),
       output_amount = 2L, resampling_approximation = "crt_spa_always_fast",
       B1 = 0L, B2 = 31L, B3 = 0L, calibration_check = FALSE,
       control_group_complement = TRUE, n_nonzero_trt_thresh = 0L,
       n_nonzero_cntrl_thresh = 0L, side_code = -1L, low_moi = FALSE,
       response_precomputations = fits, cells_in_use = seq_len(n), print_progress = FALSE,
       parallel = FALSE, n_processors = 2L, log_dir = tempdir(),
       analysis_type = "discovery_analysis", response_fit_method = "pysceptre",
       grna_fit_method = "pysceptre")
}

run_shared_crt_spa_fixture <- function(args, enabled) {
  withr::with_options(list(sceptre.crt_spa_shared_preparation = enabled),
    do.call(sceptre:::run_crt_in_memory_v2, args))
}

expect_shared_crt_spa_math_identical <- function(current, reference) {
  key <- function(frame) paste(frame$response_id, frame$grna_group, sep = "\r")
  expect_setequal(key(current$result), key(reference$result))
  current_result <- as.data.frame(current$result)
  reference_result <- as.data.frame(reference$result)
  current_result <- current_result[match(key(reference_result), key(current_result)), , drop = FALSE]
  fields <- intersect(c("p_value", "z_orig", "log_2_fold_change", "stage",
    "p_value_source", "spa_converged", "spa_reason", "spa_iterations", "spa_max_residual",
    "spa_rate", "spa_r_lr", "spa_q_lr", "statistic_id", "equation_id"),
    names(reference_result))
  for (field in fields) expect_identical(current_result[[field]], reference_result[[field]], info = field)
  expect_identical(current$response_precomputations, reference$response_precomputations)
}

test_that("shared CRT SPA preparation is opt-in and reuses each gene and target", {
  args <- make_shared_crt_spa_runner_args()
  frozen <- serialize(args, NULL)
  baseline <- run_shared_crt_spa_fixture(args, FALSE)
  shared <- run_shared_crt_spa_fixture(args, TRUE)
  default <- run_shared_crt_spa_fixture(args, NULL)
  expect_shared_crt_spa_math_identical(shared, baseline)
  expect_shared_crt_spa_math_identical(default, baseline)
  expect_identical(serialize(args, NULL), frozen)
  expect_identical(shared$response_precomputations, args$response_precomputations)
  expect_false(baseline$phase_timings$crt_spa_shared_preparation_enabled)
  expect_false(default$phase_timings$crt_spa_shared_preparation_enabled)
  expect_true(shared$phase_timings$crt_spa_shared_preparation_enabled)
  expect_equal(shared$phase_timings$crt_spa_gene_geometry_n_prepared, 4)
  expect_equal(shared$phase_timings$crt_spa_target_n_prepared, 4)
  expect_equal(shared$phase_timings$crt_spa_shared_preparation_pair_count, 16)
  expect_equal(shared$phase_timings$n_response_fits_created, 0)
  expect_gt(shared$phase_timings$crt_spa_gene_geometry_bytes, 0)
  expect_gte(shared$phase_timings$association_testing_elapsed_seconds,
             shared$phase_timings$crt_spa_gene_preparation_elapsed_seconds)
  expect_gte(shared$phase_timings$crt_spa_gene_preparation_cpu_seconds, 0)
  expect_gte(shared$phase_timings$crt_spa_target_preparation_cpu_seconds, 0)
  expect_true(all(shared$result$spa_cache_strategy == "shared_preparation"))
  expect_true(all(baseline$result$spa_cache_strategy == "cached_exact"))
})

test_that("shared CRT geometry and target preparation occur once, not once per pair", {
  args <- make_shared_crt_spa_runner_args()
  gene_calls <- target_calls <- 0L
  original_gene <- sceptre:::prepare_crt_spa_gene_geometry_cpp
  original_target <- sceptre:::prepare_crt_spa_target_cpp
  testthat::local_mocked_bindings(
    prepare_crt_spa_gene_geometry_cpp = function(...) {
      gene_calls <<- gene_calls + 1L
      original_gene(...)
    },
    prepare_crt_spa_target_cpp = function(...) {
      target_calls <<- target_calls + 1L
      original_target(...)
    }, .package = "sceptre")
  first <- run_shared_crt_spa_fixture(args, TRUE)
  expect_equal(gene_calls, 4L)
  expect_equal(target_calls, 4L)
  second <- run_shared_crt_spa_fixture(args, TRUE)
  expect_equal(gene_calls, 8L)
  expect_equal(target_calls, 8L)
  expect_shared_crt_spa_math_identical(second, first)
})

test_that("shared CRT preparation stays outside other modes and embedded response fits", {
  args <- make_shared_crt_spa_runner_args()
  testthat::local_mocked_bindings(prepare_crt_spa_gene_geometries = function(...) {
    stop("The shared geometry stage must not run in this branch.")
  }, .package = "sceptre")
  for (mode in c("crt_spa_always", "crt_spa_empirical_always_fast", "no_approximation")) {
    args$resampling_approximation <- mode
    args$B1 <- if (mode == "no_approximation") 7L else 0L
    current <- run_shared_crt_spa_fixture(args, TRUE)
    reference <- run_shared_crt_spa_fixture(args, FALSE)
    expect_shared_crt_spa_math_identical(current, reference)
    expect_false(current$phase_timings$crt_spa_shared_preparation_enabled)
    expect_equal(current$phase_timings$crt_spa_gene_geometry_n_prepared, 0)
    expect_equal(current$phase_timings$crt_spa_target_n_prepared, 0)
  }
  args$resampling_approximation <- "crt_spa_always_fast"
  args$B1 <- 0L
  args$control_group_complement <- FALSE
  current <- run_shared_crt_spa_fixture(args, TRUE)
  reference <- run_shared_crt_spa_fixture(args, FALSE)
  expect_shared_crt_spa_math_identical(current, reference)
  expect_false(current$phase_timings$crt_spa_shared_preparation_enabled)
})

test_that("shared CRT fallback pairs still share one empirical bank per target", {
  args <- make_shared_crt_spa_runner_args()
  original_baseline <- sceptre:::run_low_level_test_full_crt_spa_always_v1
  original_shared <- sceptre:::run_low_level_test_full_crt_spa_always_prepared_v1
  original_sampler <- sceptre:::crt_index_sampler_fast
  sampler_calls <- 0L
  force_failure <- function(original, ...) {
    values <- list(...)
    values$max_iterations <- 0L
    do.call(original, values)
  }
  testthat::local_mocked_bindings(
    run_low_level_test_full_crt_spa_always_v1 = function(...) force_failure(original_baseline, ...),
    run_low_level_test_full_crt_spa_always_prepared_v1 = function(...) force_failure(original_shared, ...),
    crt_index_sampler_fast = function(...) {
      sampler_calls <<- sampler_calls + 1L
      original_sampler(...)
    }, .package = "sceptre")
  baseline <- run_shared_crt_spa_fixture(args, FALSE)
  expect_equal(sampler_calls, 4L)
  shared <- run_shared_crt_spa_fixture(args, TRUE)
  expect_equal(sampler_calls, 8L)
  expect_true(all(shared$result$p_value_source == "B2_empirical"))
  expect_shared_crt_spa_math_identical(shared, baseline)
})

test_that("shared CRT geometry preparation is parallel, immutable, and cell-order specific", {
  args <- make_shared_crt_spa_runner_args()
  frozen <- serialize(args$response_precomputations, NULL)
  geometry <- sceptre:::prepare_crt_spa_gene_geometries(
    names(args$response_precomputations), args$response_precomputations,
    args$covariate_matrix, parallel = FALSE)
  expect_named(geometry$geometry, names(args$response_precomputations))
  expect_equal(geometry$n_workers, 1L)
  expect_identical(serialize(args$response_precomputations, NULL), frozen)
  if (.Platform$OS.type != "windows") {
    parallel_geometry <- sceptre:::prepare_crt_spa_gene_geometries(
      names(args$response_precomputations), args$response_precomputations,
      args$covariate_matrix, parallel = TRUE, n_processors = 2L)
    expect_equal(parallel_geometry$n_workers, 2L)
    expect_identical(parallel_geometry$geometry, geometry$geometry)
    args$parallel <- TRUE
    shared <- run_shared_crt_spa_fixture(args, TRUE)
    args$parallel <- FALSE
    reference <- run_shared_crt_spa_fixture(args, FALSE)
    expect_shared_crt_spa_math_identical(shared, reference)
  }
  args$cells_in_use <- rev(seq.int(2L, ncol(args$response_matrix), 2L))
  # Assignment indices refer to positions in the selected cells.
  args$grna_assignments$grna_group_idxs <- list(t1 = 1:10, t2 = 11:20,
                                              t3 = 21:30, t4 = 31:40)
  shared <- run_shared_crt_spa_fixture(args, TRUE)
  reference <- run_shared_crt_spa_fixture(args, FALSE)
  expect_shared_crt_spa_math_identical(shared, reference)
  invalid <- args$response_precomputations
  invalid[[1L]] <- NULL
  expect_error(sceptre:::prepare_crt_spa_gene_geometries(
    names(args$response_precomputations), invalid, args$covariate_matrix),
    "Invalid response precomputation")
})
