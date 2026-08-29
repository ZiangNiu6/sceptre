.make_fast_information_fixture <- function(n = 3000L) {
  set.seed(20260912)
  z <- runif(n, -1, 1)
  Z <- cbind(`(Intercept)` = 1, z = z)
  w <- exp(-1.7 + 0.25 * z)
  a <- rnorm(n, sd = sqrt(w))
  propensity <- plogis(-1.5 + 0.4 * z)

  C_inv <- solve(crossprod(Z, Z * w))
  G <- cbind(a, w * Z)
  moment <- colSums(G * propensity)
  B <- moment[-1L]
  beta <- drop(C_inv %*% B)
  V <- B[[1L]] - drop(crossprod(B, beta))
  grad_V <- c(0, 1 - 2 * beta[[1L]], -2 * beta[[2L]])
  gradient <- c(1, 0, 0) / sqrt(V) -
    0.5 * moment[[1L]] * grad_V / V^(3 / 2)
  Sgg <- crossprod(G, G * (propensity * (1 - propensity)))
  center <- moment[[1L]] / sqrt(V)
  center_sd <- sqrt(drop(crossprod(gradient, Sgg %*% gradient)))

  list(
    a = a,
    w = w,
    Z = Z,
    propensity = propensity,
    target = center + 2.5 * center_sd
  )
}

test_that("fast native solvers default to a 1e-6 root tolerance", {
  expect_identical(formals(crt_spa_full_fast_cpp)$tolerance, 1e-6)
  expect_identical(
    formals(crt_spa_full_outward_fast_cpp)$tolerance,
    1e-6
  )
  expect_identical(
    formals(crt_empirical_spa_full_fast_cpp)$tolerance,
    1e-6
  )
})

test_that("information partial-normal SPA tracks the exact solver", {
  fixture <- .make_fast_information_fixture()
  exact <- crt_spa_full_cpp(
    fixture$a, fixture$w, fixture$Z, fixture$propensity,
    fixture$target, tolerance = 1e-9
  )
  fast <- crt_spa_full_fast_cpp(
    fixture$a, fixture$w, fixture$Z, fixture$propensity,
    fixture$target
  )

  expect_true(exact$converged)
  expect_true(fast$converged)
  expect_true(fast$approximation_safe)
  expect_gte(fast$bulk_fraction, 0.5)
  expect_identical(fast$exact_count + fast$bulk_count, length(fixture$a))
  expect_lte(fast$berry_esseen_bound, fast$berry_esseen_threshold)
  expect_lte(fast$max_bulk_tilt, fast$max_bulk_tilt_threshold)
  expect_lt(abs(log(fast$p_value / exact$p_value)), 0.05)
  expect_equal(fast$center, exact$center, tolerance = 1e-10)

  set.seed(20260913)
  permutation <- sample.int(length(fixture$a))
  permuted <- crt_spa_full_fast_cpp(
    fixture$a[permutation], fixture$w[permutation],
    fixture$Z[permutation, , drop = FALSE],
    fixture$propensity[permutation], fixture$target
  )
  expect_identical(permuted$exact_count, fast$exact_count)
  expect_equal(permuted$p_value, fast$p_value, tolerance = 1e-8)
})

test_that("empirical partial-normal SPA tracks the exact solver", {
  set.seed(20260914)
  n <- 3000L
  z <- rnorm(n)
  propensity <- plogis(-1.1 + 0.35 * z)
  a <- 0.4 * z + sin(seq_len(n) / 17) + rnorm(n, sd = 0.7)
  target <- 2.5

  exact <- crt_empirical_spa_full_cpp(
    a, propensity, target, tolerance = 1e-9
  )
  fast <- crt_empirical_spa_full_fast_cpp(a, propensity, target)

  expect_true(exact$converged)
  expect_true(fast$converged)
  expect_true(fast$approximation_safe)
  expect_gte(fast$bulk_fraction, 0.5)
  expect_identical(fast$exact_count + fast$bulk_count, n)
  expect_lte(
    fast$directional_berry_esseen,
    fast$directional_berry_esseen_threshold
  )
  expect_lte(fast$max_bulk_tilt, fast$max_bulk_tilt_threshold)
  expect_lt(abs(log(fast$p_value / exact$p_value)), 0.05)

  set.seed(20260915)
  permutation <- sample.int(n)
  permuted <- crt_empirical_spa_full_fast_cpp(
    a[permutation], propensity[permutation], target
  )
  scaled <- crt_empirical_spa_full_fast_cpp(
    7 * a, propensity, target
  )
  expect_identical(permuted$exact_count, fast$exact_count)
  expect_equal(permuted$p_value, fast$p_value, tolerance = 1e-8)
  expect_equal(scaled$p_value, fast$p_value, tolerance = 1e-10)
})

test_that("target-aware promotion refines E and remains deterministic", {
  set.seed(1)
  n <- 1200L
  z1 <- runif(n, -2, 2)
  z2 <- rnorm(n)
  Z <- cbind(`(Intercept)` = 1, z1 = z1, z2 = z2)
  w <- exp(-1 + 0.4 * z1 - 0.2 * z2)
  a <- 0.7 * z1 * z2 + rnorm(n, sd = sqrt(w))
  propensity <- plogis(-1.5 + 0.5 * z1 - 0.3 * z2)

  C_inv <- solve(crossprod(Z, Z * w))
  G <- cbind(a, w * Z)
  moment <- colSums(G * propensity)
  B <- moment[-1L]
  beta <- drop(C_inv %*% B)
  V <- B[[1L]] - sum(B * beta)
  grad_V <- c(0, 1 - 2 * beta[[1L]], -2 * beta[-1L])
  gradient <- c(1, rep(0, 3)) / sqrt(V) -
    0.5 * moment[[1L]] * grad_V / V^(3 / 2)
  Sgg <- crossprod(G, G * (propensity * (1 - propensity)))
  target <- moment[[1L]] / sqrt(V) +
    3 * sqrt(drop(crossprod(gradient, Sgg %*% gradient)))

  information_exact <- crt_spa_full_cpp(
    a, w, Z, propensity, target, tolerance = 1e-9
  )
  information_fast <- crt_spa_full_fast_cpp(
    a, w, Z, propensity, target
  )
  empirical_exact <- crt_empirical_spa_full_cpp(
    a, propensity, target = 3, tolerance = 1e-9
  )
  empirical_fast <- crt_empirical_spa_full_fast_cpp(
    a, propensity, target = 3
  )

  for (result in list(information_fast, empirical_fast)) {
    expect_true(result$converged)
    expect_true(result$approximation_safe)
    expect_gt(result$promotion_rounds, 0L)
    expect_gt(result$promoted_count, 0L)
    expect_identical(result$promotion_stop_reason, "safe_after_promotion")
    expect_lte(result$exact_fraction, 0.5)
    expect_lte(result$max_bulk_tilt, result$max_bulk_tilt_threshold)
  }
  expect_lt(
    abs(log(information_fast$p_value / information_exact$p_value)), 0.01
  )
  expect_lt(
    abs(log(empirical_fast$p_value / empirical_exact$p_value)), 0.01
  )

  set.seed(99)
  permutation <- sample.int(n)
  information_permuted <- crt_spa_full_fast_cpp(
    a[permutation], w[permutation], Z[permutation, , drop = FALSE],
    propensity[permutation], target
  )
  empirical_permuted <- crt_empirical_spa_full_fast_cpp(
    a[permutation], propensity[permutation], target = 3
  )
  expect_identical(
    information_permuted$promoted_count, information_fast$promoted_count
  )
  expect_identical(
    empirical_permuted$promoted_count, empirical_fast$promoted_count
  )
  expect_equal(
    information_permuted$p_value, information_fast$p_value,
    tolerance = 1e-10
  )
  expect_equal(
    empirical_permuted$p_value, empirical_fast$p_value,
    tolerance = 1e-10
  )
})

test_that("fast solvers reject unsafe Gaussian-block tilts", {
  set.seed(20260916)
  n <- 1000L
  a <- rnorm(n)
  w <- rep(1, n)
  Z <- matrix(1, nrow = n, ncol = 1L)
  propensity <- rep(0.1, n)

  information <- crt_spa_full_fast_cpp(
    a, w, Z, propensity, target = 8
  )
  empirical <- crt_empirical_spa_full_fast_cpp(
    a, propensity, target = 8
  )

  expect_false(information$converged)
  expect_false(empirical$converged)
  expect_match(information$reason, "unsafe|promotion")
  expect_match(empirical$reason, "unsafe|promotion")
  expect_true(
    information$max_bulk_tilt > information$max_bulk_tilt_threshold ||
      information$berry_esseen_bound > information$berry_esseen_threshold
  )
  expect_true(
    empirical$max_bulk_tilt > empirical$max_bulk_tilt_threshold ||
      empirical$directional_berry_esseen >
        empirical$directional_berry_esseen_threshold
  )
  expect_true(is.na(information$p_value))
  expect_true(is.na(empirical$p_value))
})

test_that("leverage-boundary ties fall back independently of row order", {
  n <- 1000L
  a <- rep(1, n)
  w <- rep(1, n)
  Z <- matrix(1, nrow = n, ncol = 1L)
  propensity <- rep(0.1, n)

  information <- crt_spa_full_fast_cpp(
    a, w, Z, propensity, target = 2.5
  )
  empirical <- crt_empirical_spa_full_fast_cpp(
    a, propensity, target = 2.5
  )
  permutation <- rev(seq_len(n))
  information_permuted <- crt_spa_full_fast_cpp(
    a[permutation], w[permutation], Z[permutation, , drop = FALSE],
    propensity[permutation], target = 2.5
  )
  empirical_permuted <- crt_empirical_spa_full_fast_cpp(
    a[permutation], propensity[permutation], target = 2.5
  )

  for (result in list(
    information, empirical, information_permuted, empirical_permuted
  )) {
    expect_false(result$converged)
    expect_identical(result$reason, "invalid_partial_normal_partition")
    expect_false(result$approximation_safe)
    expect_true(is.na(result$p_value))
  }
  expect_identical(information_permuted$exact_count, information$exact_count)
  expect_identical(empirical_permuted$exact_count, empirical$exact_count)
})

test_that("empirical fast safety requires a regular converged root", {
  set.seed(20260917)
  n <- 1000L
  a <- rnorm(n)
  propensity <- plogis(rnorm(n, -1.5, 0.2))

  disabled <- crt_empirical_spa_full_fast_cpp(
    a, propensity, target = 2.5, max_iterations = 0L
  )

  expect_false(disabled$converged)
  expect_false(disabled$approximation_safe)
  expect_true(is.na(disabled$p_value))
})
