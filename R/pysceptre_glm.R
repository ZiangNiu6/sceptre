# Native structured GLM assembly adapted from broadinstitute/pysceptre,
# commit 7adaa9f5fda12e0cf87df972c8fe73b73ef13e7d (GPL-3.0-only).
# These helpers accelerate nuisance fitting; they do not change theta
# estimation, association statistics, or resampling.

prepare_pysceptre_glm_design <- function(covariate_matrix) {
  if (!is.matrix(covariate_matrix) || !is.numeric(covariate_matrix) ||
      nrow(covariate_matrix) < 1L || ncol(covariate_matrix) < 1L) {
    stop("`covariate_matrix` must be a nonempty numeric matrix.", call. = FALSE)
  }
  if (typeof(covariate_matrix) != "double") storage.mode(covariate_matrix) <- "double"
  prepared <- prepare_pysceptre_design_cpp(covariate_matrix)
  class(prepared) <- c("sceptre_pysceptre_design", "list")
  prepared
}


validate_pysceptre_glm_prepared_design <- function(prepared_design, covariate_matrix) {
  if (!inherits(prepared_design, "sceptre_pysceptre_design") ||
      !is.list(prepared_design) ||
      !identical(dim(prepared_design$raw_matrix), dim(covariate_matrix)) ||
      !identical(prepared_design$raw_matrix, covariate_matrix)) {
    stop("`prepared_design` was not made from this covariate matrix.", call. = FALSE)
  }
  invisible(TRUE)
}


fit_pysceptre_glm <- function(
    expressions, covariate_matrix, family = "poisson", prepared_design = NULL,
    tolerance = 1e-8, max_iterations = 25L) {
  family <- match.arg(family, c("poisson", "binomial"))
  if (!is.matrix(covariate_matrix) || !is.numeric(covariate_matrix)) {
    stop("`covariate_matrix` must be a numeric matrix.", call. = FALSE)
  }
  if (typeof(covariate_matrix) != "double") storage.mode(covariate_matrix) <- "double"
  if (!is.numeric(expressions) || length(expressions) != nrow(covariate_matrix) ||
      any(!is.finite(expressions)) || any(expressions < 0)) {
    stop("`expressions` must be finite nonnegative values, one per cell.", call. = FALSE)
  }
  if (identical(family, "binomial") && any(expressions > 1)) {
    stop("Binomial responses must lie between zero and one.", call. = FALSE)
  }
  if (length(tolerance) != 1L || !is.finite(tolerance) || tolerance <= 0 ||
      length(max_iterations) != 1L || !is.finite(max_iterations) ||
      max_iterations < 0 || max_iterations != as.integer(max_iterations)) {
    stop("Invalid structured GLM solver controls.", call. = FALSE)
  }
  if (is.null(prepared_design)) {
    prepared_design <- prepare_pysceptre_glm_design(covariate_matrix)
  } else {
    validate_pysceptre_glm_prepared_design(prepared_design, covariate_matrix)
  }
  fast <- fit_pysceptre_glm_cpp(
    prepared_design, as.numeric(expressions), family, tolerance,
    as.integer(max_iterations)
  )
  valid <- isTRUE(fast$converged) &&
    length(fast$coefficients) == ncol(covariate_matrix) &&
    all(is.finite(fast$coefficients)) &&
    length(fast$fitted.values) == nrow(covariate_matrix) &&
    all(is.finite(fast$fitted.values)) && all(fast$fitted.values > 0) &&
    (family != "binomial" || all(fast$fitted.values < 1))
  reconstruction_mismatch <- FALSE
  if (valid) {
    # Caches retain original-unit coefficients. Large covariate centers can
    # destroy precision when that inverse transform cancels at the intercept,
    # even when the conditioned optimizer itself converged. Require the means
    # reconstructed by downstream code to agree at a fixed relative 1e-8.
    reconstructed_eta <- drop(covariate_matrix %*% fast$coefficients)
    native_eta <- fast$linear.predictors
    reconstruction_ok <- length(native_eta) == length(reconstructed_eta) &&
      all(is.finite(native_eta)) && all(is.finite(reconstructed_eta))
    if (reconstruction_ok) {
      delta <- reconstructed_eta - native_eta
      reconstruction_ok <- if (family == "poisson") {
        all(is.finite(expm1(delta))) && max(abs(expm1(delta))) <= 1e-8
      } else {
        max(abs(delta)) <= 1e-8 &&
          max(abs(stats::plogis(reconstructed_eta) - fast$fitted.values)) <= 1e-8
      }
    }
    reconstruction_mismatch <- !reconstruction_ok
    valid <- reconstruction_ok
  }
  if (valid) {
    names(fast$coefficients) <- colnames(covariate_matrix)
    fast$method_used <- "pysceptre"
    fast$fallback_reason <- NA_character_
    return(fast)
  }
  fallback <- stats::glm.fit(
    x = covariate_matrix, y = expressions,
    family = if (family == "poisson") stats::poisson() else stats::binomial()
  )
  fallback$iterations <- fallback$iter
  fallback$method_used <- "glm.fit"
  fallback$fallback_reason <- if (reconstruction_mismatch) {
    "coefficient_reconstruction_mismatch"
  } else if (isTRUE(fast$converged)) {
    "invalid_fitted_values"
  } else {
    fast$reason
  }
  fallback$structured_diagnostics <- fast
  fallback
}
