test_that("cached CGF diagnostics survive mixed result tables", {
  result <- list(
    p = 0.02, lfc = -1, stage = 2L, z_orig = -2,
    sn_params = rep(NA_real_, 3),
    spa_cgf_mode = "exact_bernoulli", spa_cache_strategy = "cached_exact"
  )
  native <- result
  native$spa_cgf_mode <- NULL
  native$spa_cache_strategy <- NULL
  output <- construct_data_frame_v2(
    data.frame(response_id = c("cached", "native")),
    list(result, native), output_amount = 2L
  )
  expect_identical(output$spa_cgf_mode, c("exact_bernoulli", NA_character_))
  expect_identical(output$spa_cache_strategy, c("cached_exact", NA_character_))
  minimal <- construct_data_frame_v2(
    data.frame(response_id = "cached"), list(result), output_amount = 1L
  )
  expect_false("spa_cgf_mode" %in% names(minimal))
  expect_false("spa_cache_strategy" %in% names(minimal))
})
