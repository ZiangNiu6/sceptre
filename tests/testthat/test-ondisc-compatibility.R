test_that("ondisc functions are resolved from a namespace", {
  fake_namespace <- new.env(parent = emptyenv())
  expected_function <- function(x) x
  fake_namespace$expected_function <- expected_function

  expect_identical(
    .get_ondisc_function("expected_function", namespace = fake_namespace),
    expected_function
  )
  expect_error(
    .get_ondisc_function("missing_function", namespace = fake_namespace),
    "does not provide the required function `missing_function\\(\\)`"
  )
})

test_that("ondisc function names are validated", {
  fake_namespace <- new.env(parent = emptyenv())

  expect_error(
    .get_ondisc_function(character(), namespace = fake_namespace),
    "single, nonempty string"
  )
  expect_error(
    .get_ondisc_function(NA_character_, namespace = fake_namespace),
    "single, nonempty string"
  )
})

test_that("the supported ondisc API provides every required function", {
  skip_if_not_installed("ondisc", minimum_version = "1.2.0")

  required_functions <- c(
    "initialize_odm_from_backing_file",
    "threshold_count_matrix_cpp",
    "compute_n_trt_cells_matrix_ondisc",
    "compute_n_ok_pairs_ondisc",
    "compute_nt_nonzero_matrix_and_n_ok_pairs_ondisc",
    "create_odm_from_r_matrix_internal"
  )
  resolved_functions <- lapply(required_functions, .get_ondisc_function)

  expect_true(all(vapply(resolved_functions, is.function, logical(1))))
})
