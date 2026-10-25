.shared_spa_fixture <- function(p = 3L) {
  set.seed(7153 + p)
  n <- 600L
  Z <- cbind(1, matrix(rnorm(n * (p - 1L)), nrow = n))
  beta <- c(0.4, rep(0.12, p - 1L))
  mu <- exp(drop(Z %*% beta))
  theta <- 1.3
  y <- rnbinom(n, mu = mu, size = theta)
  pieces <- compute_precomputation_pieces(y, Z, beta, theta, FALSE)
  list(y = y, mu = pieces$mu, a = pieces$a, w = pieces$b, Z = Z,
       propensity = plogis(-2 + 0.15 * Z[, 2L]))
}

.shared_spa_strip_metadata <- function(x) {
  if (!is.list(x)) return(x)
  if (!is.null(names(x))) {
    x <- x[!names(x) %in% c("cache_strategy", "shared_preparation",
                             "spa_cache_strategy")]
  }
  lapply(x, .shared_spa_strip_metadata)
}

.shared_spa_compare <- function(fixture, trt, side = -1L,
                                max_iterations = 50L,
                                target = NULL, geometry = NULL) {
  if (is.null(target)) target <- prepare_crt_spa_target_cpp(fixture$propensity)
  if (is.null(geometry)) geometry <- prepare_crt_spa_gene_geometry_cpp(fixture$w, fixture$Z)
  args <- c(fixture[c("y", "mu", "a", "w", "Z")], list(
    fitted_probabilities = fixture$propensity, trt_idxs = as.integer(trt),
    n_trt = length(trt), side_code = side,
    max_iterations = max_iterations, use_fast = TRUE))
  reference <- do.call(run_low_level_test_full_crt_spa_always_v1, args)
  prepared <- do.call(run_low_level_test_full_crt_spa_always_prepared_v1,
    c(args, list(target_prepared = target, gene_geometry = geometry)))
  expect_identical(.shared_spa_strip_metadata(prepared),
                   .shared_spa_strip_metadata(reference))
  expect_identical(prepared$spa_cache_strategy, "shared_preparation")
  list(reference = reference, prepared = prepared)
}

test_that("shared preparation exactly reproduces both outward directions and sides", {
  for (p in c(2L, 6L)) {
    x <- .shared_spa_fixture(p)
    target <- prepare_crt_spa_target_cpp(x$propensity)
    geometry <- prepare_crt_spa_gene_geometry_cpp(x$w, x$Z)
    before <- serialize(list(target, geometry), NULL)
    treatments <- list(order(x$a)[1:60], tail(order(x$a), 60), seq_len(60))
    signs <- integer()
    for (trt in treatments) for (side in -1:1) {
      out <- .shared_spa_compare(x, trt, side, target = target, geometry = geometry)
      signs <- c(signs, out$prepared$spa_score_sign)
    }
    expect_true(all(c(-1L, 1L) %in% signs))
    expect_identical(serialize(list(target, geometry), NULL), before)
  }
})

test_that("shared preparation retains failures and endpoint propensity handling", {
  x <- .shared_spa_fixture()
  .shared_spa_compare(x, seq_len(70), max_iterations = 0L)
  for (probabilities in list(rep(0, 600), rep(1, 600),
                             replace(x$propensity, c(1, 2), c(0, 1)),
                             replace(x$propensity, c(1, 2), c(1e-14, 1 - 1e-14)))) {
    y <- x
    y$propensity <- probabilities
    .shared_spa_compare(y, seq_len(70))
  }
  singular <- x
  singular$Z[, 3L] <- singular$Z[, 2L]
  .shared_spa_compare(singular, seq_len(70))
  no_intercept <- x
  no_intercept$Z[, 1L] <- 2
  .shared_spa_compare(no_intercept, seq_len(70))
  .shared_spa_compare(x, integer(0))
})

test_that("prepared contexts reject malformed dimensions and values", {
  x <- .shared_spa_fixture()
  expect_error(prepare_crt_spa_target_cpp(replace(x$propensity, 1L, NA_real_)))
  expect_error(prepare_crt_spa_target_cpp(replace(x$propensity, 1L, -0.1)))
  expect_error(prepare_crt_spa_gene_geometry_cpp(replace(x$w, 1L, 0), x$Z))
  target <- prepare_crt_spa_target_cpp(x$propensity)
  geometry <- prepare_crt_spa_gene_geometry_cpp(x$w, x$Z)
  wrong_target <- target
  wrong_target$n <- 601L
  expect_error(.shared_spa_compare(x, 1:60, target = wrong_target, geometry = geometry))
  wrong_geometry <- geometry
  wrong_geometry$p <- 4L
  expect_error(.shared_spa_compare(x, 1:60, target = target, geometry = wrong_geometry))
})

test_that("R-owned prepared contexts survive owner scope exit, GC, and serialization", {
  # The vectors that prepare_target() initially receives are owned only by
  # the returned R context once this function exits. Native read-only views
  # must never rely on a previous call's temporary C++ or R stack objects.
  make_contexts <- function() {
    temporary <- .shared_spa_fixture(6L)
    list(target = prepare_crt_spa_target_cpp(temporary$propensity),
         geometry = prepare_crt_spa_gene_geometry_cpp(temporary$w, temporary$Z))
  }
  contexts <- make_contexts()
  invisible(gc())
  fixture <- .shared_spa_fixture(6L)
  treatment <- tail(order(fixture$a), 60L)
  initial <- .shared_spa_compare(fixture, treatment,
    target = contexts$target, geometry = contexts$geometry)$prepared
  serialized <- serialize(contexts, NULL)
  restored <- unserialize(serialized)
  rm(contexts)
  invisible(gc())
  for (iteration in seq_len(3L)) {
    current <- .shared_spa_compare(fixture, treatment,
      target = restored$target, geometry = restored$geometry)$prepared
    expect_identical(current, initial)
    invisible(gc())
  }
  expect_identical(serialize(restored, NULL), serialized)
})

test_that("borrowed preparation arrays remain immutable when calls switch contexts and signs", {
  fixture <- .shared_spa_fixture(6L)
  alternate <- fixture
  alternate$propensity <- rev(fixture$propensity)
  owners <- list(
    primary = prepare_crt_spa_target_cpp(fixture$propensity),
    alternate = prepare_crt_spa_target_cpp(alternate$propensity),
    geometry = prepare_crt_spa_gene_geometry_cpp(fixture$w, fixture$Z),
    fixture = fixture, alternate_fixture = alternate)
  aliases <- owners
  snapshot <- serialize(owners, NULL)
  positive <- tail(order(fixture$a), 60L)
  negative <- head(order(fixture$a), 60L)
  first <- .shared_spa_compare(fixture, positive,
    target = owners$primary, geometry = owners$geometry)$prepared
  second <- .shared_spa_compare(alternate, negative,
    target = owners$alternate, geometry = owners$geometry)$prepared
  expect_identical(first$spa_score_sign, 1L)
  expect_identical(second$spa_score_sign, -1L)
  invisible(gc())
  again <- .shared_spa_compare(fixture, positive,
    target = owners$primary, geometry = owners$geometry)$prepared
  expect_identical(again, first)
  expect_identical(serialize(owners, NULL), snapshot)
  expect_identical(serialize(aliases, NULL), snapshot)
})

test_that("native error unwinding and early returns leave borrowed contexts reusable", {
  fixture <- .shared_spa_fixture()
  target <- prepare_crt_spa_target_cpp(fixture$propensity)
  geometry <- prepare_crt_spa_gene_geometry_cpp(fixture$w, fixture$Z)
  owners <- list(target = target, geometry = geometry, weights = fixture$w,
                 propensity = fixture$propensity, design = fixture$Z)
  snapshot <- serialize(owners, NULL)
  treatment <- head(order(fixture$a), 60L)
  expected <- .shared_spa_compare(fixture, treatment,
    target = target, geometry = geometry)$prepared
  # This entry point builds its cache before checking treatment indices, so
  # the exception is raised after native views have attached their R owners.
  expect_error(crt_spa_full_outward_prepared_cpp(
    a = fixture$a, w = fixture$w, Z = fixture$Z,
    target_prepared = target, gene_geometry = geometry,
    treated_indices = c(1L, 1L)), "duplicates")
  invisible(gc())
  .shared_spa_compare(fixture, treatment, max_iterations = 0L,
    target = target, geometry = geometry)
  invisible(gc())
  recovered <- .shared_spa_compare(fixture, treatment,
    target = target, geometry = geometry)$prepared
  expect_identical(recovered, expected)
  expect_identical(serialize(owners, NULL), snapshot)
})
