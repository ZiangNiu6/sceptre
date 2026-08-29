construct_data_frame_v2 <- function(curr_df, curr_response_result, output_amount) {
  curr_df$p_value <- vapply(X = curr_response_result, FUN = function(l) l$p, FUN.VALUE = numeric(1))
  curr_df$log_2_fold_change <- vapply(curr_response_result, FUN = function(l) l$lfc, FUN.VALUE = numeric(1))
  if (output_amount >= 2L) {
    curr_df$stage <- vapply(curr_response_result, FUN = function(l) l$stage, FUN.VALUE = integer(1))
    curr_df$z_orig <- vapply(curr_response_result, FUN = function(l) l$z_orig, FUN.VALUE = numeric(1))
    curr_df$xi <- vapply(curr_response_result, FUN = function(l) l$sn_params[1L], FUN.VALUE = numeric(1))
    curr_df$omega <- vapply(curr_response_result, FUN = function(l) l$sn_params[2L], FUN.VALUE = numeric(1))
    curr_df$alpha <- vapply(curr_response_result, FUN = function(l) l$sn_params[3L], FUN.VALUE = numeric(1))

    optional_diagnostic_types <- list(
      p_value_source = NA_character_,
      spa_fast = NA,
      spa_converged = NA,
      spa_reason = NA_character_,
      spa_iterations = NA_integer_,
      spa_max_residual = NA_real_,
      spa_rate = NA_real_,
      spa_r_lr = NA_real_,
      spa_q_lr = NA_real_,
      statistic_id = NA_character_,
      equation_id = NA_character_,
      outer_dimension = NA_integer_,
      spa_tail_geometry = NA_character_,
      spa_experimental = NA
    )
    for (diagnostic_name in names(optional_diagnostic_types)) {
      diagnostic_present <- vapply(
        curr_response_result,
        FUN = function(l) !is.null(l[[diagnostic_name]]) && length(l[[diagnostic_name]]) >= 1L,
        FUN.VALUE = logical(1)
      )
      if (any(diagnostic_present)) {
        default_value <- optional_diagnostic_types[[diagnostic_name]]
        curr_df[[diagnostic_name]] <- vapply(
          curr_response_result,
          FUN = function(l) {
            value <- l[[diagnostic_name]]
            if (is.null(value) || length(value) == 0L) return(default_value)
            value <- value[[1L]]
            if (is.character(default_value)) return(as.character(value))
            if (is.integer(default_value)) return(as.integer(value))
            if (is.logical(default_value)) return(as.logical(value))
            as.numeric(value)
          },
          FUN.VALUE = default_value
        )
      }
    }
  }
  if (output_amount >= 3L) {
    resampling_lengths <- vapply(
      curr_response_result,
      FUN = function(l) length(l$resampling_dist),
      FUN.VALUE = integer(1)
    )
    max_resampling_length <- max(c(0L, resampling_lengths))
    if (max_resampling_length > 0L) {
      resampling_matrix <- matrix(
        NA_real_,
        nrow = length(curr_response_result),
        ncol = max_resampling_length
      )
      for (i in seq_along(curr_response_result)) {
        if (resampling_lengths[i] > 0L) {
          resampling_matrix[i, seq_len(resampling_lengths[i])] <-
            curr_response_result[[i]]$resampling_dist
        }
      }
      colnames(resampling_matrix) <- paste0(
        "z_null_", seq_len(max_resampling_length)
      )
      curr_df <- cbind(curr_df, data.table::as.data.table(resampling_matrix))
    }
  }
  return(curr_df)
}


auto_construct_formula_object <- function(cell_covariates, include_grna_covariates) {
  MAX_N_LEVELS_ALLOWED <- 15L
  cell_covariate_names <- colnames(cell_covariates)
  cell_covariate_names <- cell_covariate_names[cell_covariate_names != "response_p_mito"]
  if (!include_grna_covariates) { # by default, do not use grna count-based covariates in low moi
    cell_covariate_names <- cell_covariate_names[!(cell_covariate_names %in% c("grna_n_umis", "grna_n_nonzero"))]
  }
  form_str <- vapply(cell_covariate_names, function(curr_name) {
    count_based_covariate <- grepl(pattern = "n_umis|n_nonzero", x = curr_name)
    if (count_based_covariate) {
      if (any(cell_covariates[[curr_name]] == 0)) {
        out <- paste0("log(", curr_name, "+1)")
      } else {
        out <- paste0("log(", curr_name, ")")
      }
    } else {
      if (length(unique(cell_covariates[[curr_name]])) >= MAX_N_LEVELS_ALLOWED) {
        out <- NA_character_
      } else {
        out <- curr_name
      }
    }
    return(out)
  }, FUN.VALUE = character(1)) |>
    stats::na.omit() |>
    paste0(collapse = " + ")
  form <- paste0("~ ", form_str) |> stats::as.formula()
  return(form)
}


auto_compute_cell_covariates <- function(response_matrix, grna_matrix, extra_covariates, response_names) {
  # compute the response covariates
  covariate_df <- compute_cell_covariates(
    matrix_in = response_matrix,
    feature_names = response_names,
    compute_p_mito = TRUE,
    compute_max_feature = FALSE
  )
  colnames(covariate_df) <- paste0("response_", colnames(covariate_df))

  # compute the grna covariates
  grna_covariate_df <- compute_cell_covariates(
    matrix_in = grna_matrix,
    feature_names = character(0),
    compute_p_mito = TRUE,
    compute_max_feature = TRUE
  )
  colnames(grna_covariate_df) <- paste0("grna_", colnames(grna_covariate_df))
  covariate_df <- cbind(covariate_df, grna_covariate_df)

  # if extra covariates have been provided, add those as well
  if (nrow(extra_covariates) >= 1L) {
    covariate_df <- cbind(covariate_df, extra_covariates)
  }
  return(covariate_df)
}


.resolve_n_processors <- function(n_processors, os_type = .Platform$OS.type,
                                  detected_cores = parallel::detectCores(logical = FALSE)) {
  if (identical(os_type, "windows")) return(1L)
  if (!identical(n_processors, "auto")) return(n_processors)
  if (length(detected_cores) != 1L || is.na(detected_cores) ||
      !is.finite(detected_cores) || detected_cores < 1L) {
    return(1L)
  }
  max(1L, as.integer(floor(detected_cores / 2)))
}


partition_response_ids <- function(response_ids, parallel, n_processors) {
  groups_set <- FALSE
  if (parallel) {
    n_processors <- .resolve_n_processors(n_processors)
    if (length(response_ids) >= 2 * n_processors) {
      set.seed(4)
      s <- sample(response_ids)
      out <- split(s, cut(seq_along(s), n_processors, labels = paste0("group_", seq(1, n_processors))))
      groups_set <- TRUE
    }
  }
  if (!groups_set) out <- list(group_1 = response_ids)
  return(out)
}


get_log_dir <- function(log_dir) {
  log_dir <- paste0(log_dir, "/sceptre_logs/")
  if (!dir.exists(log_dir)) dir.create(log_dir, recursive = TRUE)
  return(log_dir)
}
