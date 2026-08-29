.make_fast_information_fixture <- function(n = 3000L) {
  set.seed(20260912)
  z <- runif(n, -1, 1)
  Z <- cbind(`(Intercept)` = 1, z = z)
  mu <- exp(-2.5 + 0.25 * z)
  dispersion <- 0.6
  y <- rpois(n, mu)
  w <- mu / (1 + dispersion * mu)
  a <- (y - mu) / (1 + dispersion * mu)
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
    y = y,
    Z = Z,
    propensity = propensity,
    target = center + 2.5 * center_sd
  )
}

test_that("native SPA solvers default to a unified 1e-5 root tolerance", {
  expect_identical(formals(crt_spa_full_cpp)$tolerance, 1e-5)
  expect_identical(formals(crt_empirical_spa_full_cpp)$tolerance, 1e-5)
  expect_identical(formals(crt_spa_full_fast_cpp)$tolerance, 1e-5)
  expect_identical(
    formals(crt_spa_full_outward_fast_cpp)$tolerance,
    1e-5
  )
  expect_identical(
    formals(crt_empirical_spa_full_fast_cpp)$tolerance,
    1e-5
  )
})

test_that("information partial-normal SPA tracks the exact solver", {
  fixture <- .make_fast_information_fixture()
  exact <- crt_spa_full_cpp(
    fixture$a, fixture$w, fixture$Z, fixture$propensity,
    fixture$target, tolerance = 1e-9
  )
  fast <- crt_spa_full_fast_cpp(
    fixture$a, fixture$w, fixture$y, fixture$Z, fixture$propensity,
    fixture$target
  )

  expect_true(exact$converged)
  expect_true(fast$converged)
  expect_true(fast$approximation_safe)
  expect_identical(fast$initial_partition_rule, "E={Y>0};B={Y=0}")
  expect_identical(fast$initial_exact_count, sum(fixture$y > 0))
  expect_identical(fast$initial_bulk_count, sum(fixture$y == 0))
  expect_identical(
    fast$exact_count,
    fast$initial_exact_count + fast$promoted_count
  )
  expect_identical(fast$exact_count + fast$bulk_count, length(fixture$a))
  expect_lte(fast$berry_esseen_bound, fast$berry_esseen_threshold)
  expect_lte(fast$max_bulk_tilt, fast$max_bulk_tilt_threshold)
  expect_lt(abs(log(fast$p_value / exact$p_value)), 0.05)
  expect_equal(fast$center, exact$center, tolerance = 1e-10)

  set.seed(20260913)
  permutation <- sample.int(length(fixture$a))
  permuted <- crt_spa_full_fast_cpp(
    fixture$a[permutation], fixture$w[permutation],
    fixture$y[permutation],
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
  y <- rpois(n, lambda = 0.1)
  target <- 2.5

  exact <- crt_empirical_spa_full_cpp(
    a, propensity, target, tolerance = 1e-9
  )
  fast <- crt_empirical_spa_full_fast_cpp(a, propensity, y, target)

  expect_true(exact$converged)
  expect_true(fast$converged)
  expect_true(fast$approximation_safe)
  expect_identical(fast$initial_partition_rule, "E={Y>0};B={Y=0}")
  expect_identical(fast$initial_exact_count, sum(y > 0))
  expect_identical(fast$initial_bulk_count, sum(y == 0))
  expect_identical(
    fast$exact_count,
    fast$initial_exact_count + fast$promoted_count
  )
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
    a[permutation], propensity[permutation], y[permutation], target
  )
  scaled <- crt_empirical_spa_full_fast_cpp(
    7 * a, propensity, y, target
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
  y <- as.numeric(seq_len(n) %% 12L == 0L)
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
    a, w, y, Z, propensity, target
  )
  empirical_exact <- crt_empirical_spa_full_cpp(
    a, propensity, target = 3, tolerance = 1e-9
  )
  empirical_fast <- crt_empirical_spa_full_fast_cpp(
    a, propensity, y, target = 3
  )

  for (result in list(information_fast, empirical_fast)) {
    expect_true(result$converged)
    expect_true(result$approximation_safe)
    expect_gt(result$promotion_rounds, 0L)
    expect_gt(result$promoted_count, 0L)
    expect_identical(result$promotion_stop_reason, "safe_after_promotion")
    expect_lte(result$max_bulk_tilt, result$max_bulk_tilt_threshold)
  }
  expect_identical(information_fast$initial_exact_count, sum(y > 0))
  expect_identical(information_fast$initial_bulk_count, sum(y == 0))
  expect_identical(
    information_fast$exact_count,
    information_fast$initial_exact_count + information_fast$promoted_count
  )
  expect_identical(empirical_fast$initial_exact_count, sum(y > 0))
  expect_identical(empirical_fast$initial_bulk_count, sum(y == 0))
  expect_identical(
    empirical_fast$exact_count,
    empirical_fast$initial_exact_count + empirical_fast$promoted_count
  )
  expect_lt(
    abs(log(information_fast$p_value / information_exact$p_value)), 0.01
  )
  expect_lt(
    abs(log(empirical_fast$p_value / empirical_exact$p_value)), 0.01
  )

  set.seed(99)
  permutation <- sample.int(n)
  information_permuted <- crt_spa_full_fast_cpp(
    a[permutation], w[permutation], y[permutation],
    Z[permutation, , drop = FALSE],
    propensity[permutation], target
  )
  empirical_permuted <- crt_empirical_spa_full_fast_cpp(
    a[permutation], propensity[permutation], y[permutation], target = 3
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

test_that("empirical fast solver rejects an unsafe Gaussian-block bound", {
  set.seed(20260916)
  n <- 1000L
  a <- rnorm(n)
  propensity <- rep(0.1, n)
  y <- rep(1, n)
  y[[1L]] <- 0

  empirical <- crt_empirical_spa_full_fast_cpp(
    a, propensity, y, target = 2.5
  )

  expect_false(empirical$converged)
  expect_identical(empirical$reason, "fast_normal_bulk_berry_esseen_unsafe")
  expect_identical(empirical$initial_exact_count, n - 1L)
  expect_identical(empirical$initial_bulk_count, 1L)
  expect_gt(
    empirical$directional_berry_esseen,
    empirical$directional_berry_esseen_threshold
  )
  expect_true(is.na(empirical$p_value))
})

test_that("information outcome partition permits an empty Gaussian bulk", {
  fixture <- .make_fast_information_fixture(1200L)
  y <- rep(1, length(fixture$a))
  exact <- crt_spa_full_cpp(
    fixture$a, fixture$w, fixture$Z, fixture$propensity,
    fixture$target, tolerance = 1e-9
  )
  fast <- crt_spa_full_fast_cpp(
    fixture$a, fixture$w, y, fixture$Z, fixture$propensity,
    fixture$target, tolerance = 1e-9
  )

  expect_true(exact$converged)
  expect_true(fast$converged)
  expect_true(fast$approximation_safe)
  expect_identical(fast$initial_exact_count, length(y))
  expect_identical(fast$initial_bulk_count, 0L)
  expect_identical(fast$exact_count, length(y))
  expect_identical(fast$bulk_count, 0L)
  expect_equal(fast$exact_fraction, 1)
  expect_equal(fast$berry_esseen_bound, 0)
  expect_equal(fast$max_bulk_tilt, 0)
  expect_equal(fast$p_value, exact$p_value, tolerance = 1e-8)
})

test_that("information outcome partition permits an empty exception set", {
  fixture <- .make_fast_information_fixture(1200L)
  y <- numeric(length(fixture$a))
  fast <- crt_spa_full_fast_cpp(
    fixture$a, fixture$w, y, fixture$Z, fixture$propensity,
    fixture$target
  )

  expect_identical(fast$initial_exact_count, 0L)
  expect_identical(fast$initial_bulk_count, length(y))
  expect_false(identical(fast$reason, "invalid_partial_normal_partition"))
  expect_identical(
    fast$exact_count,
    fast$initial_exact_count + fast$promoted_count
  )
})

test_that("information fast solver validates the explicit outcome vector", {
  fixture <- .make_fast_information_fixture(100L)
  call_with_y <- function(y) {
    crt_spa_full_fast_cpp(
      fixture$a, fixture$w, y, fixture$Z, fixture$propensity,
      fixture$target
    )
  }

  expect_error(call_with_y(fixture$y[-1L]), "input dimensions do not match")
  for (bad_value in c(NA_real_, Inf, -1)) {
    bad_y <- fixture$y
    bad_y[[1L]] <- bad_value
    expect_error(call_with_y(bad_y), "y must be finite and nonnegative")
  }
})

test_that("empirical outcome partition permits an empty Gaussian bulk", {
  set.seed(20260918)
  n <- 1200L
  a <- rnorm(n)
  propensity <- plogis(rnorm(n, -1.2, 0.2))
  y <- rep(1, n)

  exact <- crt_empirical_spa_full_cpp(
    a, propensity, target = 2.5, tolerance = 1e-9
  )
  fast <- crt_empirical_spa_full_fast_cpp(
    a, propensity, y, target = 2.5, tolerance = 1e-9
  )

  expect_true(exact$converged)
  expect_true(fast$converged)
  expect_true(fast$approximation_safe)
  expect_identical(fast$initial_partition_rule, "E={Y>0};B={Y=0}")
  expect_identical(fast$initial_exact_count, n)
  expect_identical(fast$initial_bulk_count, 0L)
  expect_identical(fast$exact_count, n)
  expect_identical(fast$bulk_count, 0L)
  expect_equal(fast$directional_berry_esseen, 0)
  expect_equal(fast$max_bulk_tilt, 0)
  expect_equal(fast$p_value, exact$p_value, tolerance = 1e-8)
  expect_equal(fast$K, exact$K, tolerance = 1e-8)
  expect_equal(fast$rate, exact$rate, tolerance = 1e-8)
})

test_that("empirical outcome partition permits an empty exception set", {
  set.seed(20260919)
  n <- 1200L
  a <- rnorm(n)
  propensity <- plogis(rnorm(n, -1.2, 0.2))
  y <- numeric(n)
  fast <- crt_empirical_spa_full_fast_cpp(
    a, propensity, y, target = 2.5
  )

  expect_identical(fast$initial_exact_count, 0L)
  expect_identical(fast$initial_bulk_count, n)
  expect_false(identical(fast$reason, "invalid_partial_normal_partition"))
  expect_identical(
    fast$exact_count,
    fast$initial_exact_count + fast$promoted_count
  )
})

test_that("empirical fast solver validates the explicit outcome vector", {
  set.seed(20260920)
  n <- 100L
  a <- rnorm(n)
  propensity <- plogis(rnorm(n, -1.2, 0.2))
  y <- rpois(n, lambda = 0.1)
  call_with_y <- function(y) {
    crt_empirical_spa_full_fast_cpp(
      a, propensity, y, target = 2.5
    )
  }

  expect_error(call_with_y(y[-1L]), "must have the same length")
  for (bad_value in c(NA_real_, Inf, -1)) {
    bad_y <- y
    bad_y[[1L]] <- bad_value
    expect_error(call_with_y(bad_y), "y must be finite and nonnegative")
  }
})

test_that("empirical fast safety requires a regular converged root", {
  set.seed(20260917)
  n <- 1000L
  a <- rnorm(n)
  propensity <- plogis(rnorm(n, -1.5, 0.2))
  y <- rpois(n, lambda = 0.1)

  disabled <- crt_empirical_spa_full_fast_cpp(
    a, propensity, y, target = 2.5, max_iterations = 0L
  )

  expect_false(disabled$converged)
  expect_false(disabled$approximation_safe)
  expect_true(is.na(disabled$p_value))
})
