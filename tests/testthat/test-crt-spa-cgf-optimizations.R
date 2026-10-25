.cgf_kernel_fixture <- function(p) {
  set.seed(90431L + p)
  n <- 43L
  Z <- cbind(1, matrix(rnorm(n * (p - 1L)), nrow = n))
  list(a = rnorm(n), w = runif(n, 0.1, 2), Z = Z,
       propensity = plogis(rnorm(n, -1, 0.8)))
}

test_that("prepared CGF kernels reproduce all reference moment and rate fields", {
  for (p in c(1L, 2L, 3L, 6L, 9L)) {
    fixture <- .cgf_kernel_fixture(p)
    theta <- seq(-0.035, 0.045, length.out = p + 1L)
    for (state in list(rep(0, p + 1L), theta, -theta, theta * 30)) {
      for (sign in c(-1L, 1L)) {
        result <- do.call(crt_spa_evaluate_kernels_cpp, c(fixture,
          list(theta = state, lambda = 0.3, target = 0.2, score_sign = sign)))
        expect_identical(result$optimized, result$reference,
          info = paste("p", p, "sign", sign, "theta", paste(state, collapse = ",")))
        expect_identical(result$optimized_d7, p == 6L)
      }
    }
  }
})

test_that("optimized stable KL retains tiny, endpoint, and extreme-tilt behavior", {
  probabilities <- c(0, 1, .Machine$double.xmin, 1e-14, 1e-8,
                     0.001, 0.2, 0.5, 0.9, 1 - 1e-14)
  fixture <- list(a = rep(1, length(probabilities)),
    w = rep(1, length(probabilities)), Z = matrix(1, length(probabilities), 1L),
    propensity = probabilities)
  tilts <- c(-1000, -700, -699.999999, -1, -0.001000001,
             -0.000999999, -1e-12, 0, 1e-12, 0.000999999,
             0.001000001, 1, 699.999999, 700, 1000)
  for (tilt in tilts) {
    result <- do.call(crt_spa_evaluate_kernels_cpp,
      c(fixture, list(theta = c(tilt, 0))))
    expect_identical(result$optimized, result$reference,
      info = paste("tilt", tilt))
    expect_length(result$optimized$cell_rate, length(probabilities))
    expect_identical(result$optimized$cell_rate[1:2], c(0, 0))
  }
})
