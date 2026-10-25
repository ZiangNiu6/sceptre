.retained_tilt_fixture <- function(p) {
  set.seed(90431L + p)
  n <- 43L
  Z <- cbind(1, matrix(rnorm(n * (p - 1L)), nrow = n))
  list(a = rnorm(n), w = runif(n, 0.1, 2), Z = Z,
       propensity = plogis(rnorm(n, -1, 0.8)))
}

test_that("final-rate reuse requires a complete matching seven-moment state", {
  for (p in c(2L, 3L, 6L, 9L)) {
    fixture <- .retained_tilt_fixture(p)
    theta <- seq(-0.04, 0.03, length.out = p + 1L)
    for (sign in c(-1L, 1L)) {
      result <- do.call(crt_spa_evaluate_kernels_cpp,
        c(fixture, list(theta = theta, score_sign = sign)))
      expect_identical(result$retained$rate, result$optimized$rate)
      expect_identical(result$retained$cell_rate, result$optimized$cell_rate)
      expect_identical(result$retained$used_retained_rate, p == 6L)
      expect_equal(result$retained$n_tilts_stored, if (p == 6L) length(fixture$a) else 0L)
      expect_equal(result$retained$n_tilts_reused, if (p == 6L) length(fixture$a) else 0L)
    }
  }
})

test_that("a different final theta forces safe rate recomputation", {
  fixture <- .retained_tilt_fixture(6L)
  theta <- seq(-0.04, 0.03, length.out = 7L)
  changed <- theta
  changed[[3L]] <- changed[[3L]] + 1e-8
  mismatch <- do.call(crt_spa_evaluate_kernels_cpp,
    c(fixture, list(theta = theta, rate_theta = changed)))
  expected <- do.call(crt_spa_evaluate_kernels_cpp,
    c(fixture, list(theta = changed)))
  expect_false(mismatch$retained$used_retained_rate)
  expect_equal(mismatch$retained$n_tilts_stored, length(fixture$a))
  expect_equal(mismatch$retained$n_tilts_reused, 0L)
  expect_identical(mismatch$retained$rate, expected$optimized$rate)
  expect_identical(mismatch$retained$cell_rate, expected$optimized$cell_rate)
})

test_that("retained rates preserve endpoints and extreme tilts without reading partial buffers", {
  fixture <- .retained_tilt_fixture(6L)
  fixture$a[] <- 1
  fixture$propensity[c(1L, 2L)] <- c(0, 1)
  for (tilt in c(-1000, -700, -1e-12, 0, 1e-12, 699.999999, 700, 1000)) {
    result <- do.call(crt_spa_evaluate_kernels_cpp,
      c(fixture, list(theta = c(tilt, rep(0, 6L)))))
    expect_true(result$retained$used_retained_rate)
    expect_identical(result$retained$rate, result$optimized$rate)
    expect_identical(result$retained$cell_rate, result$optimized$cell_rate)
  }
  fixture$a[] <- 2
  overflow <- do.call(crt_spa_evaluate_kernels_cpp,
    c(fixture, list(theta = c(1e308, rep(0, 6L)))))
  expect_false(overflow$retained$used_retained_rate)
  expect_equal(overflow$retained$n_tilts_stored, 0L)
  expect_equal(overflow$retained$n_tilts_reused, 0L)
  expect_identical(overflow$retained$rate, overflow$optimized$rate)
  expect_identical(overflow$retained$cell_rate, overflow$optimized$cell_rate)
})
