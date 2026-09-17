prepare_grna_design_timed <- function(covariate_matrix) {
  started <- proc.time()
  design <- prepare_grna_logistic_design(covariate_matrix)
  duration <- proc.time() - started
  list(
    design = design,
    cpu_seconds = unname(duration[["user.self"]] + duration[["sys.self"]]),
    elapsed_seconds = unname(duration[["elapsed"]])
  )
}


fit_grna_with_diagnostics <- function(
    trt_idxs, covariate_matrix, grna_group, grna_fit_method,
    prepared_design = NULL) {
  preparation_cpu <- preparation_elapsed <- 0
  if (identical(grna_fit_method, "fast_logistic") && is.null(prepared_design)) {
    preparation <- prepare_grna_design_timed(covariate_matrix)
    prepared_design <- preparation$design
    preparation_cpu <- preparation$cpu_seconds
    preparation_elapsed <- preparation$elapsed_seconds
  }
  started <- proc.time()
  fit <- perform_grna_precomputation(
    trt_idxs = trt_idxs, covariate_matrix = covariate_matrix,
    return_fitted_values = TRUE, grna_fit_method = grna_fit_method,
    prepared_design = prepared_design, return_details = TRUE
  )
  duration <- proc.time() - started
  list(
    fitted_probabilities = fit$value,
    diagnostics = data.frame(
      grna_group = as.character(grna_group),
      method_requested = fit$method_requested,
      method_used = fit$method_used,
      fallback_reason = fit$fallback_reason,
      converged = fit$converged,
      iterations = as.integer(fit$iterations),
      n_cells = nrow(covariate_matrix), n_treated = length(trt_idxs),
      preparation_cpu_seconds = preparation_cpu,
      preparation_elapsed_seconds = preparation_elapsed,
      fitting_cpu_seconds = unname(duration[["user.self"]] + duration[["sys.self"]]),
      fitting_elapsed_seconds = unname(duration[["elapsed"]]),
      stringsAsFactors = FALSE
    )
  )
}


add_grna_phase_timings <- function(
    phase_timings, diagnostics, grna_fit_method, shared_preparation,
    shared_design) {
  phase_timings$timing_schema_version <- 2L
  phase_timings$grna_fit_method <- grna_fit_method
  phase_timings$n_grna_fits <- nrow(diagnostics)
  phase_timings$n_fast_grna_fits <- sum(diagnostics$method_used == "fast_logistic")
  phase_timings$n_grna_fit_fallbacks <- sum(
    diagnostics$method_requested == "fast_logistic" & diagnostics$method_used == "glm.fit"
  )
  phase_timings$grna_preparation_scope <- if (grna_fit_method == "glm.fit") {
    "not_required"
  } else if (shared_design) {
    "shared_design"
  } else {
    "per_target"
  }
  phase_timings$grna_preparation_cpu_seconds <- shared_preparation$cpu_seconds +
    sum(diagnostics$preparation_cpu_seconds)
  phase_timings$grna_preparation_elapsed_sum_seconds <- shared_preparation$elapsed_seconds +
    sum(diagnostics$preparation_elapsed_seconds)
  phase_timings$grna_fitting_cpu_seconds <- sum(diagnostics$fitting_cpu_seconds)
  phase_timings$grna_fitting_elapsed_sum_seconds <- sum(diagnostics$fitting_elapsed_seconds)
  phase_timings$grna_total_cpu_seconds <- phase_timings$grna_preparation_cpu_seconds +
    phase_timings$grna_fitting_cpu_seconds
  attr(phase_timings, "grna_fit_diagnostics") <- as.data.frame(diagnostics)
  phase_timings
}


#' Get gRNA assignment-model fitting diagnostics
#'
#' Returns a named list of per-target diagnostic tables for completed CRT
#' analyses. Each target is recorded once, regardless of how many genes were
#' paired with it. The tables report the requested and used fitting backends,
#' fallback reason, convergence, iterations, and preparation/fitting CPU and
#' elapsed times. Fitting diagnostics do not describe gRNA-to-cell assignment
#' in [assign_grnas()].
#'
#' Elapsed times in these tables are individual-call times; their sum is not
#' parallel wall-clock analysis time. Shared design preparation is reported
#' separately in the object's analysis phase timings, rather than repeated on
#' each target. Analyses using permutations do not fit an assignment model and
#' have no diagnostic table. Older objects without these diagnostics return
#' an empty list.
#' Tables from analyses invalidated by a later parameter, assignment, or QC
#' update are omitted until that analysis is rerun.
#'
#' @param sceptre_object a `sceptre_object`
#' @return A named list of data frames, one for each recorded CRT analysis.
#' @export
get_grna_fit_diagnostics <- function(sceptre_object) {
  timings <- get_analysis_phase_timings(sceptre_object)
  current <- vapply(names(timings), function(analysis) {
    isTRUE(sceptre_object@functs_called[paste0("run_", analysis)])
  }, logical(1L))
  diagnostics <- lapply(timings[current], function(timing) {
    attr(timing, "grna_fit_diagnostics", exact = TRUE)
  })
  diagnostics[!vapply(diagnostics, is.null, logical(1L))]
}
