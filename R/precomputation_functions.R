#' Run response precomputation
#'
#' Fits the response nuisance model for a given response vector.
#'
#' The `"sceptre"` method fits a Poisson regression and estimates the
#' negative-binomial size parameter conditional on its fitted means. The
#' `"glmGamPoi"` method uses `glmGamPoi::glm_gp()` to fit the regression
#' coefficients and dispersion. Both methods return the same cached nuisance
#' parameter interface used by the downstream SCEPTRE association tests; they
#' do not change the CRT, SPA, or other resampling procedures.
#'
#' @param expressions the numeric vector of response expressions
#' @param covariate_matrix the covariate matrix on which to regress the expressions (NOTE: the matrix should contain an intercept term)
#' @param response_fit_method the response nuisance-fitting method, either
#' `"sceptre"` or `"glmGamPoi"`
#'
#' @return a list containing the following elements: (i) "fitted_coefs": a vector of fitted coefficients; (ii) "theta": the fitted theta.
#' @noRd
perform_response_precomputation <- function(
    expressions, covariate_matrix, response_fit_method = "sceptre") {
  response_fit_method <- match.arg(
    response_fit_method, choices = c("sceptre", "glmGamPoi")
  )
  if (identical(response_fit_method, "glmGamPoi")) {
    expressions <- matrix(expressions, nrow = 1L)
    result <- perform_glmgampoi_response_precomputations(
      expression_matrix = expressions,
      covariate_matrix = covariate_matrix,
      response_ids = "response"
    )
    return(result[[1L]])
  }

  perform_sceptre_response_precomputation(expressions, covariate_matrix)
}


perform_sceptre_response_precomputation <- function(expressions, covariate_matrix) {
  pois_fit <- stats::glm.fit(y = expressions, x = covariate_matrix, family = stats::poisson())
  response_theta_list <- estimate_theta(
    y = expressions, mu = pois_fit$fitted.values, dfr = pois_fit$df.residual,
    limit = 50, eps = (.Machine$double.eps)^(1 / 4)
  )
  # check that NAs are absent from the fitted coefficient vector
  if (any(is.na(pois_fit$coefficients))) {
    problem_covariates <- paste0(names(which(is.na(pois_fit$coefficients))))
    stop("The coefficients corresponding to the following covariates cannot be estimated in the regression model: ",
         paste0(problem_covariates, collapse = ", "), ". Consider removing these covariates from the model (by updating `formula_object` in `set_analysis_parameters()`).")
  }
  theta <- max(min(response_theta_list[[1]], 1000), 0.01)
  result <- list(fitted_coefs = pois_fit$coefficients, theta = theta)
  return(result)
}


check_glmgampoi_response_fit_dependency <- function() {
  minimum_version <- base::package_version("1.16.0")
  installation_message <- paste0(
    "Install it with `BiocManager::install(\"glmGamPoi\")`."
  )
  if (!requireNamespace("glmGamPoi", quietly = TRUE)) {
    stop(
      "`response_fit_method = \"glmGamPoi\"` requires glmGamPoi >= ",
      minimum_version, ". ", installation_message,
      call. = FALSE
    )
  }
  installed_version <- utils::packageVersion("glmGamPoi")
  if (installed_version < minimum_version) {
    stop(
      "`response_fit_method = \"glmGamPoi\"` requires glmGamPoi >= ",
      minimum_version, ", but version ", installed_version,
      " is installed. ", installation_message,
      call. = FALSE
    )
  }
  invisible(TRUE)
}


validate_response_fit_chunk_size <- function(chunk_size) {
  if (
    length(chunk_size) != 1L || is.na(chunk_size) ||
    !is.numeric(chunk_size) || !is.finite(chunk_size) ||
    chunk_size < 1 || chunk_size != as.integer(chunk_size)
  ) {
    stop("`chunk_size` must be a positive integer.", call. = FALSE)
  }
  as.integer(chunk_size)
}


validate_response_ids <- function(response_ids) {
  response_ids <- as.character(response_ids)
  if (
    anyNA(response_ids) || any(response_ids == "") ||
    anyDuplicated(response_ids)
  ) {
    stop("`response_ids` must be unique, non-missing identifiers.", call. = FALSE)
  }
  response_ids
}


plan_glmgampoi_response_fits <- function(
    response_ids, response_precomputations, chunk_size = 16L,
    parallel = FALSE, n_processors = "auto") {
  response_ids <- validate_response_ids(response_ids)
  chunk_size <- validate_response_fit_chunk_size(chunk_size)
  if (!is.list(response_precomputations)) {
    stop("`response_precomputations` must be a list.", call. = FALSE)
  }
  cache_names <- names(response_precomputations)
  if (length(response_precomputations) > 0L &&
      (is.null(cache_names) || anyNA(cache_names) || any(cache_names == "") ||
       anyDuplicated(cache_names))) {
    stop(
      "`response_precomputations` must have unique, non-missing response names.",
      call. = FALSE
    )
  }

  cached <- vapply(
    response_ids,
    function(response_id) !is.null(response_precomputations[[response_id]]),
    logical(1L)
  )
  missing_ids <- response_ids[!cached]
  if (length(missing_ids) == 0L) {
    chunks <- stats::setNames(list(), character(0L))
    n_workers <- 0L
  } else {
    resolved_processors <- if (parallel) {
      .resolve_n_processors(n_processors)
    } else {
      1L
    }
    required_chunks <- ceiling(length(missing_ids) / chunk_size)
    parallel_chunks <- if (
      parallel && length(missing_ids) >= 2L * resolved_processors
    ) {
      resolved_processors
    } else {
      1L
    }
    n_chunks <- min(
      length(missing_ids), max(required_chunks, parallel_chunks)
    )
    base_chunk_size <- length(missing_ids) %/% n_chunks
    chunk_sizes <- rep.int(base_chunk_size, n_chunks)
    remainder <- length(missing_ids) %% n_chunks
    if (remainder > 0L) {
      chunk_sizes[seq_len(remainder)] <-
        chunk_sizes[seq_len(remainder)] + 1L
    }
    chunk_ends <- cumsum(chunk_sizes)
    chunk_starts <- c(1L, utils::head(chunk_ends, -1L) + 1L)
    chunks <- Map(
      function(start, end) missing_ids[seq.int(start, end)],
      chunk_starts, chunk_ends
    )
    names(chunks) <- sprintf(
      paste0("chunk_%0", max(3L, nchar(length(chunks))), "d"),
      seq_along(chunks)
    )
    n_workers <- min(as.integer(resolved_processors), length(chunks))
  }

  list(
    requested_ids = response_ids,
    cached_ids = response_ids[cached],
    missing_ids = missing_ids,
    chunks = chunks,
    n_chunks = length(chunks),
    n_workers = as.integer(n_workers),
    chunk_size = chunk_size
  )
}


merge_response_fit_cache <- function(
    response_precomputations, new_precomputations, requested_ids) {
  requested_ids <- validate_response_ids(requested_ids)
  if (!is.list(response_precomputations) || !is.list(new_precomputations)) {
    stop("Response precomputation caches must be lists.", call. = FALSE)
  }
  existing_names <- names(response_precomputations)
  new_names <- names(new_precomputations)
  if (length(response_precomputations) > 0L &&
      (is.null(existing_names) || anyNA(existing_names) ||
       any(existing_names == "") || anyDuplicated(existing_names))) {
    stop("The existing response cache has invalid names.", call. = FALSE)
  }
  if (length(new_precomputations) > 0L &&
      (is.null(new_names) || anyNA(new_names) || any(new_names == "") ||
       anyDuplicated(new_names))) {
    stop("The new response cache has invalid names.", call. = FALSE)
  }
  expected_new_names <- requested_ids[vapply(
    requested_ids,
    function(response_id) is.null(response_precomputations[[response_id]]),
    logical(1L)
  )]
  unexpected <- setdiff(new_names, expected_new_names)
  missing <- setdiff(expected_new_names, new_names)
  if (length(unexpected) > 0L || length(missing) > 0L) {
    details <- character(0L)
    if (length(unexpected) > 0L) {
      details <- c(
        details,
        paste0("unexpected ", format_response_ids_for_error(unexpected))
      )
    }
    if (length(missing) > 0L) {
      details <- c(
        details,
        paste0("missing ", format_response_ids_for_error(missing))
      )
    }
    stop(
      "The new response cache does not match the requested cache misses (",
      paste(details, collapse = "; "), ").",
      call. = FALSE
    )
  }

  updated <- response_precomputations
  for (response_id in expected_new_names) {
    updated[[response_id]] <- new_precomputations[[response_id]]
  }
  updated
}


format_response_ids_for_error <- function(response_ids) {
  paste0("`", response_ids, "`", collapse = ", ")
}


perform_glmgampoi_response_precomputations <- function(
    expression_matrix, covariate_matrix, response_ids,
    check_dependency = TRUE, check_covariates = TRUE,
    validate_full_result = FALSE) {
  if (check_dependency) check_glmgampoi_response_fit_dependency()

  n_responses <- nrow(expression_matrix)
  n_cells <- ncol(expression_matrix)
  n_covariates <- ncol(covariate_matrix)
  if (
    is.null(n_responses) || is.null(n_cells) || n_responses < 1L ||
    n_cells < 1L || n_responses != length(response_ids) ||
    nrow(covariate_matrix) != n_cells || n_covariates < 1L
  ) {
    stop(
      "The glmGamPoi expression and covariate matrices have incompatible dimensions.",
      call. = FALSE
    )
  }

  if (check_covariates && any(!is.finite(covariate_matrix))) {
    stop("The covariate matrix contains non-finite values.", call. = FALSE)
  }

  fit <- tryCatch(
    glmGamPoi::glm_gp(
      data = expression_matrix,
      design = covariate_matrix,
      offset = 0,
      size_factors = FALSE,
      overdispersion = TRUE,
      overdispersion_shrinkage = FALSE,
      ridge_penalty = NULL,
      do_cox_reid_adjustment = FALSE,
      subsample = FALSE,
      on_disk = FALSE,
      verbose = FALSE
    ),
    error = function(e) {
      invalid_count_rows <- apply(
        expression_matrix, 1L,
        function(x) any(!is.finite(x)) || any(x < 0)
      )
      if (any(invalid_count_rows)) {
        stop(
          "Responses ",
          format_response_ids_for_error(response_ids[invalid_count_rows]),
          " contain non-finite or negative expression counts.",
          call. = FALSE
        )
      }
      stop(
        "glmGamPoi response fitting failed for ",
        format_response_ids_for_error(response_ids), ": ",
        conditionMessage(e),
        call. = FALSE
      )
    }
  )

  expected_beta_dim <- c(n_responses, n_covariates)
  expected_mu_dim <- c(n_responses, n_cells)
  if (
    !identical(dim(fit$Beta), expected_beta_dim) ||
    !identical(dim(fit$Mu), expected_mu_dim) ||
    length(fit$overdispersions) != n_responses ||
    length(fit$deviances) != n_responses
  ) {
    stop(
      "glmGamPoi returned an unexpected result shape for responses ",
      format_response_ids_for_error(response_ids), ".",
      call. = FALSE
    )
  }

  invalid_beta <- apply(fit$Beta, 1L, function(x) any(!is.finite(x)))
  invalid_dispersion <- !is.finite(fit$overdispersions) |
    fit$overdispersions < 0
  invalid_deviance <- !is.finite(fit$deviances)
  invalid_fit <- invalid_beta | invalid_dispersion | invalid_deviance
  mu_range <- range(fit$Mu)
  if (any(!is.finite(mu_range)) || mu_range[[1L]] <= 0) {
    invalid_mu <- apply(
      fit$Mu, 1L,
      function(x) any(!is.finite(x)) || any(x <= 0)
    )
    invalid_fit <- invalid_fit | invalid_mu
  }
  if (any(invalid_fit)) {
    stop(
      "glmGamPoi returned invalid fitted values for responses ",
      format_response_ids_for_error(response_ids[invalid_fit]), ".",
      call. = FALSE
    )
  }

  invalid_contract <- !is.null(fit$overdispersion_shrinkage_list) ||
    !is.null(fit$ridge_penalty)
  if (validate_full_result) {
    invalid_contract <- invalid_contract || any(fit$size_factors != 1) ||
      any(fit$Offset != 0)
  }
  if (invalid_contract) {
    stop(
      "glmGamPoi did not honor the requested response-fitting settings for ",
      format_response_ids_for_error(response_ids), ".",
      call. = FALSE
    )
  }

  theta <- 1 / fit$overdispersions
  theta <- pmax(pmin(theta, 1000), 0.01)
  coefficient_names <- colnames(covariate_matrix)
  result <- lapply(seq_len(n_responses), function(i) {
    fitted_coefs <- as.numeric(fit$Beta[i, ])
    names(fitted_coefs) <- coefficient_names
    list(fitted_coefs = fitted_coefs, theta = theta[[i]])
  })
  names(result) <- response_ids
  result
}


load_response_expression_chunk <- function(
    response_matrix, response_ids, cell_indices) {
  response_ids <- validate_response_ids(response_ids)
  response_indices <- match(response_ids, rownames(response_matrix))
  if (anyNA(response_indices)) {
    stop("Some requested responses are absent from `response_matrix`.", call. = FALSE)
  }
  if (methods::is(response_matrix, "odm")) {
    expression_rows <- lapply(response_indices, function(response_index) {
      as.numeric(load_row(response_matrix, response_index))[cell_indices]
    })
    expression_matrix <- do.call(rbind, expression_rows)
  } else {
    expression_matrix <- as.matrix(
      response_matrix[response_indices, cell_indices, drop = FALSE]
    )
  }
  storage.mode(expression_matrix) <- "double"
  dimnames(expression_matrix) <- list(response_ids, NULL)
  expression_matrix
}


execute_glmgampoi_response_fit_plan <- function(
    plan, response_matrix, cell_indices, covariate_matrix,
    print_progress = FALSE) {
  if (!is.list(plan) || is.null(plan$chunks) || is.null(plan$n_workers) ||
      is.null(plan$n_chunks) || is.null(plan$missing_ids) ||
      is.null(plan$chunk_size)) {
    stop("Invalid glmGamPoi response-fit plan.", call. = FALSE)
  }
  if (plan$n_chunks == 0L) {
    return(stats::setNames(list(), character(0L)))
  }
  check_glmgampoi_response_fit_dependency()
  if (nrow(covariate_matrix) != length(cell_indices)) {
    stop(
      "The rows of `covariate_matrix` must align with `cell_indices`.",
      call. = FALSE
    )
  }
  if (any(!is.finite(covariate_matrix))) {
    stop("The covariate matrix contains non-finite values.", call. = FALSE)
  }

  fit_chunk <- function(chunk_index) {
    chunk_name <- names(plan$chunks)[[chunk_index]]
    chunk_ids <- plan$chunks[[chunk_index]]
    if (print_progress) {
      message(
        "Running glmGamPoi response-fit ", chunk_name, " on ",
        length(chunk_ids), " response", if (length(chunk_ids) == 1L) "" else "s", "."
      )
    }
    tryCatch(
      list(
        ok = TRUE,
        value = perform_response_precomputations_from_matrix(
          response_matrix = response_matrix,
          response_ids = chunk_ids,
          cell_indices = cell_indices,
          covariate_matrix = covariate_matrix,
          response_fit_method = "glmGamPoi",
          chunk_size = plan$chunk_size,
          check_dependency = FALSE,
          check_covariates = FALSE
        ),
        chunk_name = chunk_name,
        response_ids = chunk_ids,
        error = NA_character_
      ),
      error = function(e) {
        list(
          ok = FALSE, value = NULL, chunk_name = chunk_name,
          response_ids = chunk_ids, error = conditionMessage(e)
        )
      }
    )
  }

  chunk_indices <- seq_along(plan$chunks)
  outcomes <- if (plan$n_workers <= 1L) {
    lapply(chunk_indices, fit_chunk)
  } else {
    parallel::mclapply(
      chunk_indices, fit_chunk, mc.cores = plan$n_workers,
      mc.preschedule = TRUE
    )
  }
  failed <- !vapply(outcomes, function(outcome) isTRUE(outcome$ok), logical(1L))
  if (any(failed)) {
    details <- vapply(outcomes[failed], function(outcome) {
      paste0(
        outcome$chunk_name, " [",
        paste(outcome$response_ids, collapse = ", "), "]: ", outcome$error
      )
    }, character(1L))
    stop(
      "glmGamPoi response-fit chunk failure(s): ",
      paste(details, collapse = "; "),
      call. = FALSE
    )
  }

  fitted <- unlist(lapply(outcomes, `[[`, "value"), recursive = FALSE)
  fitted_names <- names(fitted)
  if (is.null(fitted_names) || anyDuplicated(fitted_names) ||
      !identical(fitted_names, plan$missing_ids)) {
    stop(
      "glmGamPoi response-fit chunks returned an invalid response set.",
      call. = FALSE
    )
  }
  fitted
}


perform_response_precomputations_from_matrix <- function(
    response_matrix, response_ids, cell_indices, covariate_matrix,
    response_fit_method = "sceptre", chunk_size = 16L,
    check_dependency = TRUE, check_covariates = TRUE) {
  response_fit_method <- match.arg(
    response_fit_method, choices = c("sceptre", "glmGamPoi")
  )
  chunk_size <- validate_response_fit_chunk_size(chunk_size)
  response_ids <- validate_response_ids(response_ids)
  if (length(response_ids) == 0L) {
    return(stats::setNames(list(), character(0)))
  }
  available_response_ids <- rownames(response_matrix)
  if (is.null(available_response_ids)) {
    stop("`response_matrix` must have response IDs as row names.", call. = FALSE)
  }
  missing_response_ids <- setdiff(response_ids, available_response_ids)
  if (length(missing_response_ids) > 0L) {
    stop(
      "The following responses are absent from `response_matrix`: ",
      format_response_ids_for_error(missing_response_ids), ".",
      call. = FALSE
    )
  }
  if (
    !is.numeric(cell_indices) || length(cell_indices) < 1L ||
    anyNA(cell_indices) || any(!is.finite(cell_indices)) ||
    any(cell_indices != as.integer(cell_indices)) ||
    any(cell_indices < 1L) || any(cell_indices > ncol(response_matrix)) ||
    anyDuplicated(cell_indices)
  ) {
    stop(
      "`cell_indices` must contain unique valid column indices for `response_matrix`.",
      call. = FALSE
    )
  }
  cell_indices <- as.integer(cell_indices)
  if (nrow(covariate_matrix) != length(cell_indices)) {
    stop(
      "The rows of `covariate_matrix` must align with `cell_indices`.",
      call. = FALSE
    )
  }
  if (check_covariates && any(!is.finite(covariate_matrix))) {
    stop("The covariate matrix contains non-finite values.", call. = FALSE)
  }
  if (identical(response_fit_method, "glmGamPoi")) {
    if (check_dependency) check_glmgampoi_response_fit_dependency()
  }

  chunk_starts <- seq.int(1L, length(response_ids), by = chunk_size)
  result <- vector("list", length(response_ids))
  names(result) <- response_ids
  for (chunk_start in chunk_starts) {
    chunk_end <- min(chunk_start + chunk_size - 1L, length(response_ids))
    chunk_indices <- seq.int(chunk_start, chunk_end)
    chunk_response_ids <- response_ids[chunk_indices]
    expression_matrix <- load_response_expression_chunk(
      response_matrix = response_matrix,
      response_ids = chunk_response_ids,
      cell_indices = cell_indices
    )
    if (identical(response_fit_method, "glmGamPoi")) {
      chunk_result <- perform_glmgampoi_response_precomputations(
        expression_matrix = expression_matrix,
        covariate_matrix = covariate_matrix,
        response_ids = chunk_response_ids,
        check_dependency = FALSE,
        check_covariates = FALSE
      )
    } else {
      chunk_result <- lapply(seq_along(chunk_response_ids), function(i) {
        perform_sceptre_response_precomputation(
          expressions = expression_matrix[i, ],
          covariate_matrix = covariate_matrix
        )
      })
      names(chunk_result) <- chunk_response_ids
    }
    result[chunk_indices] <- chunk_result
  }
  result
}


perform_grna_precomputation <- function(trt_idxs, covariate_matrix, return_fitted_values) {
  indicator <- integer(length = nrow(covariate_matrix))
  indicator[trt_idxs] <- 1L
  logistic_fit <- stats::glm.fit(y = indicator, x = covariate_matrix, family = stats::binomial())
  if (return_fitted_values) {
    out <- logistic_fit$fitted.values
  } else {
    out <- logistic_fit$coefficients
  }
  return(out)
}


compute_D_matrix <- function(Zt_wZ, wZ) {
  P_decomp <- eigen(Zt_wZ, symmetric = TRUE)
  U <- P_decomp$vectors
  Lambda_minus_half <- 1 / sqrt(P_decomp$values)
  D <- (Lambda_minus_half * t(U)) %*% t(wZ)
  return(D)
}


compute_precomputation_pieces <- function(expression_vector, covariate_matrix, fitted_coefs, theta, full_test_stat) {
  mu <- exp(as.numeric(covariate_matrix %*% fitted_coefs))
  if (full_test_stat) {
    denom <- 1 + mu / theta
    w <- mu / denom
    a <- (expression_vector - mu) / denom
    wZ <- w * covariate_matrix
    Zt_wZ <- t(covariate_matrix) %*% wZ
    D <- compute_D_matrix(Zt_wZ, wZ)
    out <- list(mu = mu, w = w, a = a, D = D)
  } else {
    a <- expression_vector - (expression_vector * mu + theta * mu) / (theta + mu)
    b <- (theta * mu) / (theta + mu)
    out <- list(mu = mu, a = a, b = b)
  }
  return(out)
}
