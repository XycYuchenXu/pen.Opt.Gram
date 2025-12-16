#' Solve (weak) Lasso regression
#'
#' @param G The Gram matrix X'X / n
#' @param g The vector X'y / n
#' @param lambda The regularization parameter
#' @param beta0 The initial value of beta
#' @param weak True for weak Lasso, False for standard Lasso
#' @param max_iter The maximum number of iterations
#' @param tolerance The convergence tolerance
#'
#' @return beta The estimated coefficients
#' @import Rcpp
#' @import RcppArmadillo
#' @importFrom MASS ginv
#' @importFrom Rdpack reprompt
#' @export
#'
#' @examples
#' # Generate random data
#' set.seed(123)
#' n <- 100
#' p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' y <- rnorm(n)
#' # Compute Gram matrix and cross-product
#' G <- crossprod(X) / n
#' g <- crossprod(X, y) / n
#' # Set regularization parameter
#' lambda <- 0.1
#' # Estimate coefficients with weak Lasso penalty
#' beta_est <- wlasso_gram(G, g, lambda, weak = TRUE)
wlasso_gram = function(G, g, lambda, beta0 = NULL, weak = T, max_iter = 1000,
                       tolerance = 1e-6){
  if (is.null(beta0)) {
    p = nrow(G)
    beta0 = rep(0, p)
  }
  beta_est = lasso_row_cpp(G, g, lambda, beta0, weak, max_iter, tolerance)

  if (!weak) {
    # Debiasing step
    supp_ind = which(beta_est != 0)
    Si_l0 = length(supp_ind)
    if (Si_l0 > 0) {
      beta_est[supp_ind] = crossprod(ginv(G[supp_ind, supp_ind, drop = F], tol = 1e-6),
                                     g[supp_ind])
    }
  }
  return(beta_est)
}

#' Solve group sparse regression
#'
#' @param G The Gram matrix X'X / n
#' @param g The matrix X'y / n
#' @param Grp The group assignment matrix, indexes starting from 1
#' @param lambda The regularization parameter
#' @param beta0 The initial value of beta
#' @param max_iter The maximum number of iterations
#' @param tolerance The convergence tolerance
#'
#' @return C The estimated coefficients matrix
#' @import Rcpp
#' @import RcppArmadillo
#' @importFrom Rdpack reprompt
#' @importFrom RSpectra eigs_sym
#' @importFrom MASS ginv
#' @export
#'
#' @examples
#' # Generate random data
#' set.seed(123)
#' n <- 100
#' p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- matrix(rnorm(n * p), n, p)
#' # Compute Gram matrix and cross-product
#' G <- crossprod(X) / n
#' g <- crossprod(X, Y) / n
#' # Define group assignments
#' Grp <- matrix(sample(1:5, p * p, replace = TRUE), nrow = p)
#' # Set regularization parameter
#' lambda <- 0.1
#' # Estimate matrix with group Lasso penalty
#' C_est <- group_lasso_gram(G, g, Grp, lambda)
group_lasso_gram = function(G, g, Grp, lambda, beta0 = NULL,
                            max_iter = 1000, tolerance = 1e-6){
  p = nrow(Grp)
  unique_vals <- unique(as.vector(Grp))
  sorted_unique_vals <- sort(unique_vals)
  factor_indices <- match(Grp, sorted_unique_vals)
  Grp <- matrix(factor_indices, nrow = p)

  eta <- tryCatch({
    1 / eigs_sym(G, k = 1, which = "LM")$values[1]
  }, warning = function(w) {
    1 / max(eigen(G, symmetric = TRUE, only.values = TRUE)$values)
  }, error = function(e) {
    1 / max(eigen(G, symmetric = TRUE, only.values = TRUE)$values)
  })

  if (is.infinite(eta)) {return(matrix(0, p, p))}
  if (is.null(beta0)) {beta0 = matrix(0, p, p)}
  C = group_lasso_cpp(G, g, Grp, lambda, beta0, eta,
                      max_iter = max_iter, tolerance = tolerance)

  for (i in 1:p) {
    supp_ind = which(C[i,] != 0)
    Si_l0 = length(supp_ind)
    if (Si_l0 > 0) {
      C[i,supp_ind] = crossprod(ginv(G[supp_ind, supp_ind, drop = F], tol = 1e-6),
                                g[supp_ind,i])
    }
  }
  return(C)
}

#' Solve the matrix, wrapper.
#'
#' @param G The Gram matrix X'X / n
#' @param g The matrix X'y / n
#' @param lambda The regularization parameter
#' @param alpha The update weight
#' @param weak True for weak Lasso, False for standard Lasso
#' @param Grp The group assignment matrix, indexes starting from 1
#' @param C_init The initial value of C
#' @param method The method to use: "fista" or "rcpp"
#' @param max_iter The maximum number of iterations
#' @param tolerance The convergence tolerance
#' @param pb A progress bar function
#'
#' @return C The estimated coefficients matrix
#' @import Rcpp
#' @import RcppArmadillo
#' @importFrom Rdpack reprompt
#' @export
#'
#' @examples
#' # Generate random data
#' set.seed(123)
#' n <- 100
#' p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- matrix(rnorm(n * p), n, p)
#' # Compute Gram matrix and cross-product
#' G <- crossprod(X) / n
#' g <- crossprod(X, Y) / n
#' # Set regularization parameter
#' lambda <- 0.1
#' # Estimate matrix with Lasso penalty
#' C_est <- mat_lasso(G, g, lambda, alpha = 1, weak = FALSE)
mat_lasso = function(G, g, lambda, alpha = 1, weak = F, Grp = NULL, C_init = NULL,
                     method = c("fista", "rcpp"), max_iter = 200, tolerance = 1e-4,
                     pb = NULL){
  p = nrow(G)
  C_temp = matrix(0, p, p)
  lambda0 = lambda * alpha
  if (is.null(C_init)) {C_init = C_temp}

  method <- match.arg(method)
  if (is.null(Grp)) {
    if (method == "fista") {
      for (i in 1:p) {
        C_temp[i,] = fista_lasso(G, g[,i], C_init[i,], lambda0,
                                 rep(1, p), weak, max_iter = max_iter,
                                 tolerance = tolerance)
      }
    } else {
      for (i in 1:p) {
        C_temp[i,] = wlasso_gram(G, g[,i], lambda0, C_init[i,], weak, max_iter = max_iter,
                                 tolerance = tolerance)
      }
    }
    if (!is.null(pb)) {pb()}
  } else {
    C_temp = group_lasso_gram(G, g, Grp, lambda0, C_init,
                              max_iter = max_iter, tolerance = tolerance)
  }

  return(C_temp)
}

#' Solve the matrix with nuclear norm penalty
#'
#' @param G The Gram matrix X'X / n
#' @param g The matrix X'y / n
#' @param lambda The regularization parameter
#' @param X0 The initial value of X
#' @param max_iter The maximum number of iterations
#' @param tolerance The convergence tolerance
#' @param verbose Whether to print convergence information
#'
#' @return X The estimated matrix
#' @importFrom Rdpack reprompt
#' @importFrom RSpectra eigs_sym
#' @export
#'
#' @examples
#' # Generate random data
#' set.seed(123)
#' n <- 100
#' p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- matrix(rnorm(n * p), n, p)
#' # Compute Gram matrix and cross-product
#' G <- crossprod(X) / n
#' g <- crossprod(X, Y) / n
#' # Set regularization parameter
#' lambda <- 0.1
#' # Estimate matrix with nuclear norm penalty
#' X_est <- mat_nuclear(G, g, lambda)
mat_nuclear <- function(G, g, lambda, X0 = NULL, max_iter = 1000,
                        tolerance = 1e-6, verbose = FALSE) {
  p <- nrow(G)

  # Initialize X0
  if (is.null(X0)) {
    X0 <- matrix(0, p, ncol(g))
  }

  # Compute step size using Rspectra
  eta <- tryCatch({
    1 / eigs_sym(G, k = 1, which = "LM")$values[1]
  }, warning = function(w) {
    1 / max(eigen(G, symmetric = TRUE, only.values = TRUE)$values)
  }, error = function(e) {
    1 / max(eigen(G, symmetric = TRUE, only.values = TRUE)$values)
  })
  if (is.infinite(eta)) {
    return(matrix(0, p, ncol(g)))
  }

  X_curr <- X0
  X_prev <- X0

  # FISTA
  Z <- X0
  t_curr <- 1
  t_prev <- 1

  for (iter in 1:max_iter) {
    # Gradient step
    Grad <- tcrossprod(Z, G) - t(g)
    Y <- Z - eta * Grad

    # SVT
    svd_result <- svd(Y)
    s_thresh <- pmax(svd_result$d - eta * lambda, 0)
    X_curr <- svd_result$u %*% (s_thresh * t(svd_result$v))

    # FISTA momentum
    if (sum((Z - X_curr) * (X_curr - X_prev)) > 0) {
      # Restart
      Z <- X_curr
      t_curr <- 1
    } else {
      # Continue with momentum
      # FISTA momentum
      t_curr <- (1 + sqrt(1 + 4 * t_prev^2)) / 2
      beta <- (t_prev - 1) / t_curr
      Z <- X_curr + beta * (X_curr - X_prev)
    }

    # Convergence check
    norm_diff <- sqrt(sum((X_curr - X_prev)^2))
    norm_prev <- sqrt(sum(X_prev^2))

    if (ifelse(norm_prev < 1e-10, norm_diff < tolerance, norm_diff / norm_prev < tolerance)) {
      if (verbose) cat(sprintf("Converged in %d iterations\n", iter))
      break
    }

    X_prev <- X_curr
    t_prev <- t_curr
  }

  return(X_curr)
}
