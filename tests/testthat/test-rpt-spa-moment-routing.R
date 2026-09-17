.rpt_moment_routing_object <- function(method, control_group) {
  set.seed(82913)
  n <- 480L
  z <- rnorm(n)
  targets <- data.frame(
    grna_id = c("g1", "g2", "nt1", "nt2"),
    grna_target = c("target_1", "target_2", "non-targeting", "non-targeting"),
    chr = c("chr1", "chr1", NA, NA), start = c(1, 2, NA, NA),
    end = c(2, 3, NA, NA)
  )
  grna <- matrix(0, 4L, n, dimnames = list(targets$grna_id, NULL))
  grna["g1", 1:40] <- 20
  grna["g2", 41:100] <- 20
  grna["nt1", 101:290] <- 20
  grna["nt2", 291:480] <- 20
  mu <- 0.15 * exp(0.1 * z)
  response <- rbind(
    gene_1 = rnbinom(n, mu = mu * ifelse(seq_len(n) <= 40L, 6, 1), size = 2),
    gene_2 = rnbinom(n, mu = mu * ifelse(seq_len(n) %in% 41:100, 6, 1), size = 2),
    gene_3 = rnbinom(n, mu = mu * ifelse(seq_len(n) <= 40L, 4, 1), size = 3),
    gene_4 = rnbinom(n, mu = mu * ifelse(seq_len(n) %in% 41:100, 4, 1), size = 3),
    background = rep(1, n)
  )
  pairs <- expand.grid(response_id = paste0("gene_", 1:4),
                       grna_target = c("target_1", "target_2"),
                       stringsAsFactors = FALSE)
  object <- import_data(
    response_matrix = response, grna_matrix = grna,
    grna_target_data_frame = targets, moi = "low", extra_covariates = data.frame(z = z)
  ) |>
    set_analysis_parameters(
      discovery_pairs = pairs, formula_object = ~ z, side = "right",
      resampling_mechanism = "permutations", resampling_approximation = method,
      control_group = control_group
    ) |>
    assign_grnas(method = "thresholding", threshold = 10) |>
    run_qc(response_n_umis_range = c(0, 1), response_n_nonzero_range = c(0, 1),
           n_nonzero_trt_thresh = 0, n_nonzero_cntrl_thresh = 0)
  # Retain the real 499-draw screen while bounding an unexpected empirical fallback.
  object@B2 <- 499L
  object
}

.rpt_moment_routing_results <- function(object, parallel) {
  set.seed(52899)
  result <- run_discovery_analysis(object, output_amount = 2L,
    print_progress = FALSE, parallel = parallel, n_processors = 2L)
  table <- as.data.frame(result@discovery_result)
  table <- table[order(table$response_id, table$grna_target), , drop = FALSE]
  rownames(table) <- NULL
  list(table = table, object = result)
}

test_that("public fast RPT modes route nonempty discovery pairs serially and in workers", {
  skip_on_os("windows")
  skip_on_cran()
  for (control_group in c("complement", "nt_cells")) {
    for (method in c("rpt_spa_fast", "rpt_spa_always_fast")) {
      object <- .rpt_moment_routing_object(method, control_group)
      expect_equal(object@n_ok_discovery_pairs, 8L)
      serial <- .rpt_moment_routing_results(object, parallel = FALSE)
      workers <- .rpt_moment_routing_results(object, parallel = TRUE)
      for (run in list(serial, workers)) {
        table <- run$table
        expect_equal(nrow(table), 8L)
        expect_identical(table$response_id, rep(paste0("gene_", 1:4), each = 2L))
        expect_identical(table$grna_target, rep(c("target_1", "target_2"), 4L))
        expect_true(all(is.finite(table$p_value)))
        expect_true(all(table$p_value >= 0 & table$p_value <= 1))
        expect_true(all(table$spa_fast))
        expect_equal(table$spa_solver_tolerance, rep(1e-4, 8L))
        expect_equal(table$spa_compressed_tolerance, rep(1e-4, 8L))
        expect_equal(table$spa_exact_audit_tolerance, rep(1e-4, 8L))
        attempted <- table$spa_acceleration_path != "not_attempted"
        expect_true(any(attempted))
        expect_true(all(table$spa_converged[attempted]))
        expect_true(all(table$p_value_source[attempted] == method))
        expect_equal(table$spa_moment_degree[attempted], rep(2L, sum(attempted)))
        if (method == "rpt_spa_always_fast") expect_true(all(attempted))
      }
      columns <- c("response_id", "grna_target", "p_value", "z_orig", "stage",
                   "p_value_source", "spa_fast", "spa_converged", "spa_cgf_mode",
                   "spa_cache_strategy", "spa_moment_degree", "spa_packed_moment_count",
                   "spa_acceleration_path", "spa_acceleration_fallback_reason",
                   "spa_moment_eligible", "spa_exact_audit_passed")
      expect_equal(workers$table[, columns], serial$table[, columns], tolerance = 1e-10)
      timing <- get_analysis_phase_timings(workers$object)$discovery_analysis
      expect_equal(timing$association_n_workers, 2L)
      if (control_group == "complement") {
        expect_setequal(names(workers$object@response_precomputations), paste0("gene_", 1:4))
      } else {
        expect_length(workers$object@response_precomputations, 0L)
      }
    }
  }
})
