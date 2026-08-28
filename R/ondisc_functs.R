##########################
# READ AND WRITE FUNCTIONS
##########################
#' Write or read an `ondisc`-backed `sceptre_object`
#'
#' `write_ondisc_backed_sceptre_object()` and `read_ondisc_backed_sceptre_object()` enable the writing and reading of `ondisc`-backed `sceptre_object`s, respectively. First, `write_ondisc_backed_sceptre_object()` writes an `ondisc`-backed `sceptre_object` to disk, creating a file `sceptre_object.rds` in the specified directory. Next, `read_ondisc_backed_sceptre_object()` reads and initializes a `sceptre_object` from a `sceptre_object.rds` file, `response.odm` file, and `grna.odm` file stored on disk.
#'
#' @param sceptre_object a `sceptre_object`
#' @param directory_to_write the directory in which to write the `sceptre_object.rds` file
#' @param sceptre_object_fp file path to a `sceptre_object.rds` file
#' @param response_odm_file_fp file path to the backing `.odm` file for the response modality
#' @param grna_odm_file_fp file path to the backing `.odm` file for the gRNA modality
#'
#' @return `write_ondisc_backed_sceptre_object()` returns NULL, and `read_ondisc_backed_sceptre_object()` returns an `ondisc`-backed `sceptre_object`
#' @export
#' @examples
#' if (requireNamespace("ondisc", quietly = TRUE) &&
#'     utils::packageVersion("ondisc") >= package_version("1.2.0")) {
#'   data(lowmoi_example_data)
#'   # 1. create ondisc-backed sceptre_object
#'   sceptre_object <- import_data(
#'     response_matrix = lowmoi_example_data$response_matrix,
#'     grna_matrix = lowmoi_example_data$grna_matrix,
#'     grna_target_data_frame = lowmoi_example_data$grna_target_data_frame,
#'     extra_covariates = lowmoi_example_data$extra_covariates,
#'     moi = "low",
#'     use_ondisc = TRUE,
#'     directory_to_write = tempdir()
#'   )
#'
#'   # 2. write
#'   write_ondisc_backed_sceptre_object(
#'     sceptre_object = sceptre_object,
#'     directory_to_write = tempdir()
#'   )
#'
#'   # 3. read
#'   rm(sceptre_object)
#'   sceptre_object <- read_ondisc_backed_sceptre_object(
#'     sceptre_object_fp = paste0(tempdir(), "/sceptre_object.rds"),
#'     response_odm_file_fp = paste0(tempdir(), "/response.odm"),
#'     grna_odm_file_fp = paste0(tempdir(), "/grna.odm")
#'   )
#' }
read_ondisc_backed_sceptre_object <- function(sceptre_object_fp, response_odm_file_fp, grna_odm_file_fp) {
  # read in objects
  sceptre_object <- readRDS(sceptre_object_fp)
  initialize_odm <- .get_ondisc_function("initialize_odm_from_backing_file")
  response_odm <- initialize_odm(response_odm_file_fp)
  grna_odm <- initialize_odm(grna_odm_file_fp)
  # check for concordance of integer ids
  if (sceptre_object@integer_id != response_odm@integer_id) {
    stop("The `sceptre_object` and `response_odm` have distinct IDs. The `sceptre_object` likely is associated with a different backing .odm file.")
  }
  if (sceptre_object@integer_id != grna_odm@integer_id) {
    stop("The `sceptre_object` and `grna_odm` have distinct IDs. The `sceptre_object` likely is associated with a different backing .odm file.")
  }
  # update response_odm and grna_odm and return
  sceptre_object <- set_response_matrix(sceptre_object, response_odm)
  sceptre_object <- set_grna_matrix(sceptre_object, grna_odm)
  return(sceptre_object)
}


#' @rdname read_ondisc_backed_sceptre_object
#' @export
write_ondisc_backed_sceptre_object <- function(sceptre_object, directory_to_write) {
  sceptre_object <- set_grna_matrix(sceptre_object, list())
  sceptre_object <- set_response_matrix(sceptre_object, list())
  # check that dir exists
  if (!dir.exists(directory_to_write)) dir.create(directory_to_write, recursive = TRUE)
  saveRDS(sceptre_object, paste0(directory_to_write, "/sceptre_object.rds"))
}


###########################
# PIPELINE HELPER FUNCTIONS
###########################
.get_ondisc_function <- function(function_name, namespace = NULL) {
  if (!is.character(function_name) || length(function_name) != 1L ||
      is.na(function_name) || !nzchar(function_name)) {
    stop("`function_name` must be a single, nonempty string.", call. = FALSE)
  }

  installed_version <- NULL
  if (is.null(namespace)) {
    if (!requireNamespace("ondisc", quietly = TRUE)) {
      stop(
        "This operation requires the optional package `ondisc` (>= 1.2.0).",
        call. = FALSE
      )
    }

    installed_version <- utils::packageVersion("ondisc")
    if (installed_version < package_version("1.2.0")) {
      stop(
        paste0(
          "This operation requires `ondisc` >= 1.2.0, but version ",
          installed_version, " is installed."
        ),
        call. = FALSE
      )
    }
    namespace <- asNamespace("ondisc")
  }

  ondisc_function <- get0(
    function_name,
    envir = namespace,
    mode = "function",
    inherits = FALSE
  )
  if (is.null(ondisc_function)) {
    version_suffix <- if (is.null(installed_version)) {
      ""
    } else {
      paste0(" ", installed_version)
    }
    stop(
      paste0(
        "The installed `ondisc` package", version_suffix,
        " does not provide the required function `", function_name,
        "()`. Install a compatible `ondisc` release (>= 1.2.0)."
      ),
      call. = FALSE
    )
  }

  ondisc_function
}

get_id_vect <- function(v, pod_size) {
  n_elements <- length(v)
  breaks <- round(n_elements / pod_size)
  if (breaks >= 2) {
    as.integer(cut(seq(1, n_elements), breaks))
  } else {
    rep(1L, n_elements)
  }
}

write_vector <- function(vector, file_name) {
  file_con <- file(file_name)
  writeLines(as.character(vector), file_con)
  close(file_con)
}
