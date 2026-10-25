# Run from the package root after building the native code:
# Rscript inst/benchmarks/pysceptre-glm.R [n_cells] [repetitions] [library]
# Timings include preparation and fallback work. They are machine- and
# design-specific, not guarantees about complete discovery-analysis speed.
args <- commandArgs(trailingOnly = TRUE)
n_cells <- if (length(args)) as.integer(args[[1L]]) else 20000L
repetitions <- if (length(args) > 1L) as.integer(args[[2L]]) else 3L
stopifnot(is.finite(n_cells), n_cells >= 1000L,
          is.finite(repetitions), repetitions >= 1L)
if (length(args) > 2L) {
  library(sceptre, lib.loc = args[[3L]])
} else {
  pkgload::load_all(quiet = TRUE)
}
set.seed(20261021)

make_case <- function(indicators) {
  z <- cbind(`(Intercept)` = 1,
             matrix(rnorm(n_cells * 4L), ncol = 4L))
  if (indicators) {
    batch <- factor(rep(seq_len(30L), length.out = n_cells))
    replicate <- factor(rep(seq_len(3L), each = 31L, length.out = n_cells))
    z <- cbind(z, model.matrix(~ batch + replicate)[, -1L, drop = FALSE])
  }
  beta <- c(log(3), rep(0.05, ncol(z) - 1L))
  y <- rnbinom(n_cells, mu = exp(drop(z %*% beta)), size = 5)
  probability <- plogis(-3 + 0.2 * z[, 2L])
  treatment <- which(rbinom(n_cells, 1L, probability) == 1L)
  list(z = z, y = y, treatment = treatment)
}

rows <- list()
for (case_name in c("dense", "batch_indicators")) {
  fixture <- make_case(case_name == "batch_indicators")
  for (model in c("response", "grna")) {
    methods <- if (model == "response") c("sceptre", "pysceptre", "glmGamPoi") else
      c("glm.fit", "fast_logistic", "pysceptre")
    for (method in methods) {
      if (method == "glmGamPoi" && !requireNamespace("glmGamPoi", quietly = TRUE)) next
      run_fit <- if (model == "response") {
        function() sceptre:::perform_response_precomputation(
          fixture$y, fixture$z, response_fit_method = method)
      } else {
        function() sceptre:::perform_grna_precomputation(
          fixture$treatment, fixture$z, return_fitted_values = TRUE,
          grna_fit_method = method, return_details = TRUE)
      }
      invisible(run_fit()) # Warm the code and libraries, then measure fresh fits.
      durations <- numeric(repetitions)
      for (i in seq_len(repetitions)) {
        timing <- system.time(fit <- run_fit())
        durations[[i]] <- timing[["elapsed"]]
      }
      diagnostics <- if (model == "response") attr(fit, "fit_diagnostics") else fit
      used <- if (is.null(diagnostics)) method else diagnostics$method_used
      reference <- stats::glm.fit(
        fixture$z,
        if (model == "response") fixture$y else
          as.integer(seq_len(n_cells) %in% fixture$treatment),
        family = if (model == "response") poisson() else binomial(),
        control = stats::glm.control(epsilon = 1e-12, maxit = 100L))
      fitted <- if (model == "response") exp(drop(fixture$z %*% fit$fitted_coefs)) else fit$value
      rows[[length(rows) + 1L]] <- data.frame(
        design = case_name, n_cells = n_cells, n_covariates = ncol(fixture$z),
        model = model, requested = method, used = used,
        median_elapsed_seconds = median(durations),
        max_abs_fitted_difference_from_glm = max(abs(fitted - reference$fitted.values)))
    }
  }
}
print(do.call(rbind, rows), row.names = FALSE)
