test_that("automatic processor selection is always positive", {
  expect_identical(
    .resolve_n_processors("auto", os_type = "unix", detected_cores = 1L),
    1L
  )
  expect_identical(
    .resolve_n_processors("auto", os_type = "unix", detected_cores = NA_integer_),
    1L
  )
  expect_identical(
    .resolve_n_processors("auto", os_type = "unix", detected_cores = 8L),
    4L
  )
})

test_that("processor selection is serial on Windows", {
  expect_identical(
    .resolve_n_processors("auto", os_type = "windows", detected_cores = 8L),
    1L
  )
  expect_identical(
    .resolve_n_processors(8L, os_type = "windows", detected_cores = 8L),
    1L
  )
})
