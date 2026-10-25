# Run-local shared preparation for exact information-studentized CRT SPA.
# These R-owned lists are discarded at the end of an analysis; no cell-sized
# vectors or external pointers are retained in the object's response-fit cache.
prepare_crt_spa_gene_geometries <- function(
    response_ids, response_precomputations, covariate_matrix,
    parallel = FALSE, n_processors = "auto", chunk_size = 16L) {
  started <- proc.time()[["elapsed"]]
  plan <- plan_response_fits(
    response_ids = response_ids, response_precomputations = list(),
    chunk_size = chunk_size, parallel = parallel, n_processors = n_processors
  )
  if (!plan$n_chunks) {
    return(list(geometry = stats::setNames(list(), character()),
                elapsed_seconds = proc.time()[["elapsed"]] - started,
                cpu_seconds = 0, n_workers = 0L))
  }
  prepare_chunk <- function(chunk_index) {
    chunk_started <- proc.time()
    ids <- plan$chunks[[chunk_index]]
    tryCatch({
      geometry <- stats::setNames(lapply(ids, function(response_id) {
        fit <- response_precomputations[[response_id]]
        if (is.null(fit) || length(fit$fitted_coefs) != ncol(covariate_matrix) ||
            any(!is.finite(fit$fitted_coefs)) || length(fit$theta) != 1L ||
            !is.finite(fit$theta) || fit$theta <= 0) {
          stop("Invalid response precomputation for `", response_id, "`.")
        }
        # Match compute_precomputation_pieces exactly. In particular, the
        # conditioned GLM optimizer's fitted means must not replace these
        # downstream means reconstructed from original-unit coefficients.
        mu <- exp(as.numeric(covariate_matrix %*% fit$fitted_coefs))
        w <- (fit$theta * mu) / (fit$theta + mu)
        prepare_crt_spa_gene_geometry_cpp(w = w, Z = covariate_matrix)
      }), ids)
      duration <- proc.time() - chunk_started
      list(ok = TRUE, geometry = geometry,
           cpu_seconds = unname(duration[["user.self"]] + duration[["sys.self"]]))
    }, error = function(error) {
      list(ok = FALSE, error = conditionMessage(error), response_ids = ids)
    })
  }
  outcomes <- if (plan$n_workers > 1L) {
    parallel::mclapply(seq_along(plan$chunks), prepare_chunk,
                      mc.cores = plan$n_workers, mc.preschedule = TRUE)
  } else {
    lapply(seq_along(plan$chunks), prepare_chunk)
  }
  failed <- !vapply(outcomes, function(outcome) {
    is.list(outcome) && isTRUE(outcome$ok)
  }, logical(1L))
  if (any(failed)) {
    details <- vapply(outcomes[failed], function(outcome) {
      if (!is.list(outcome)) return(as.character(outcome)[1L])
      paste0(paste(outcome$response_ids, collapse = ", "), ": ", outcome$error)
    }, character(1L))
    stop("CRT SPA gene preparation failed: ", paste(details, collapse = "; "),
         call. = FALSE)
  }
  geometry <- unlist(lapply(outcomes, `[[`, "geometry"), recursive = FALSE)
  if (!identical(names(geometry), plan$requested_ids)) {
    stop("CRT SPA gene preparation returned an unexpected response set.", call. = FALSE)
  }
  list(geometry = geometry,
       elapsed_seconds = proc.time()[["elapsed"]] - started,
       # Sum direct chunk-process CPU rather than a potentially lagging
       # parent proc.time() child snapshot. Wall time includes the whole stage.
       cpu_seconds = sum(vapply(outcomes, `[[`, numeric(1L), "cpu_seconds")),
       n_workers = plan$n_workers)
}
