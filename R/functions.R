#' Solve (weak) Lasso regression
#'
#' @param G The Gram matrix X'X / n
#' @param g The vector X'y / n
#' @param lambda The regularization parameter
#' @param beta0 The initial value of beta
#' @param weak True for weak Lasso, False for standard Lasso
#' @param refine Whether to perform debiasing on the support
#' @param max_iter The maximum number of iterations
#' @param tolerance The convergence tolerance
#'
#' @return beta The estimated coefficients
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
#' y <- rnorm(n)
#' # Compute Gram matrix and cross-product
#' G <- crossprod(X) / n
#' g <- crossprod(X, y) / n
#' # Set regularization parameter
#' lambda <- 0.1
#' # Estimate coefficients with weak Lasso penalty
#' beta_est <- wlasso_gram(G, g, lambda, weak = TRUE)
wlasso_gram = function(G, g, lambda, beta0 = NULL, weak = T, refine = T,
                       max_iter = 1000, tolerance = 1e-6){
  if (is.null(beta0)) {
    p = nrow(G)
    beta0 = rep(0, p)
  }
  beta_est = lasso_row_cpp(G, g, lambda, beta0, weak, max_iter, tolerance)

  if (!weak && refine) {
    # Debiasing step
    supp_ind = which(beta_est != 0)
    Si_l0 = length(supp_ind)
    if (Si_l0 > 0) {
      beta_est[supp_ind] = crossprod(ginv_robust(G[supp_ind, supp_ind, drop = F], tol = 1e-6),
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
#' @param refine Whether to perform debiasing on the support
#' @param max_iter The maximum number of iterations
#' @param tolerance The convergence tolerance
#'
#' @return C The estimated coefficients matrix
#' @import Rcpp
#' @import RcppArmadillo
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
#' # Define group assignments
#' Grp <- matrix(sample(1:5, p * p, replace = TRUE), nrow = p)
#' # Set regularization parameter
#' lambda <- 0.1
#' # Estimate matrix with group Lasso penalty
#' C_est <- group_lasso_gram(G, g, Grp, lambda)
group_lasso_gram = function(G, g, Grp, lambda, beta0 = NULL, refine = T,
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

  if (refine) {
    for (i in 1:p) {
      supp_ind = which(C[i,] != 0)
      Si_l0 = length(supp_ind)
      if (Si_l0 > 0) {
        C[i,supp_ind] = crossprod(ginv_robust(G[supp_ind, supp_ind, drop = F], tol = 1e-6),
                                  g[supp_ind,i])
      }
    }
  }

  return(C)
}

#' Solve the matrix, wrapper.
#'
#' Solve the coefficient C for the regularized problem with model Y = X C' + E,
#' Y of size n x q, X of size n x p, C of size q x p.
#'
#' The objective function is f(C) = 0.5 * trace(C G C') - trace(C g) + lambda * P(C),
#' where P(C) is the penalty function (Lasso, weak Lasso, or
#' group Lasso).
#'
#' @param G The Gram matrix X'X / n, of size p x p
#' @param g The matrix X'y / n, of size p x q
#' @param lambda The regularization parameter
#' @param alpha The update weight
#' @param weak True for weak Lasso, False for standard Lasso
#' @param Grp The group assignment matrix, indexes starting from 1
#' @param C_init The initial value of C
#' @param method The method to use: "fista" or "rcpp"
#' @param refine Whether to perform debiasing on the support
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
                     method = c("fista", "rcpp"), refine = T,
                     max_iter = 200, tolerance = 1e-4, pb = NULL){
  p = nrow(G); q = ncol(g)
  C_temp = matrix(0, q, p)
  lambda0 = lambda * alpha
  if (is.null(C_init)) {C_init = C_temp}

  method <- match.arg(method)
  if (is.null(Grp)) {
    if (method == "fista") {
      for (i in 1:q) {
        C_temp[i,] = fista_lasso(G, g[,i], C_init[i,], lambda0,
                                 rep(1, p), weak, refine, max_iter = max_iter,
                                 tolerance = tolerance)
      }
    } else {
      for (i in 1:q) {
        C_temp[i,] = wlasso_gram(G, g[,i], lambda0, C_init[i,], weak, refine, max_iter = max_iter,
                                 tolerance = tolerance)
      }
    }
    if (!is.null(pb)) {pb()}
  } else {
    C_temp = group_lasso_gram(G, g, Grp, lambda0, C_init, refine,
                              max_iter = max_iter, tolerance = tolerance)
  }

  return(C_temp)
}

#' @keywords internal
soft_threshold_nuclear <- function(M, thresh) {
  s <- svd(M)
  d_new <- pmax(s$d - thresh, 0)

  if (d_new[1] == 0) return(matrix(0, nrow(M), ncol(M)))

  # Optimization: Broadcast vector multiply instead of diag matrix
  pos <- d_new > 0
  s$u[, pos, drop=FALSE] %*% (d_new[pos] * t(s$v[, pos, drop=FALSE]))
}

#' @keywords internal
#' @importFrom MASS ginv
ginv_robust <- function(X, tol = 1e-6) {
  tryCatch({
    return(ginv(X, tol = tol))
  }, error = function(e) {
    X <- as.matrix(X)
    n <- nrow(X)
    p <- ncol(X)

    if (n >= p) {
      S <- eigen(crossprod(X), symmetric = TRUE)
      pos <- S$values > max(tol * S$values[1], 0)
      if (!any(pos)) return(matrix(0, p, n))

      inv_vals <- 1 / S$values[pos]
      U = S$vectors[, pos, drop=FALSE]
      return(U %*% (inv_vals * t(U)) %*% t(X))
    } else {
      # Wide matrix: Decompose XX'
      S <- eigen(tcrossprod(X), symmetric = TRUE)
      pos <- S$values > max(tol * S$values[1], 0)
      if (!any(pos)) return(matrix(0, p, n))

      # X^+ = X^T * U * Sigma^-2 * U^T
      inv_vals <- 1 / S$values[pos]
      U = S$vectors[, pos, drop=FALSE]
      return(t(X) %*% U %*% (inv_vals * t(U)))
    }
  })
}

#' Solve the matrix with nuclear norm penalty.
#'
#' Solve the coefficient B for the regularized problem with model Y = X B' + E,
#' Y of size n x q, X of size n x p, B of size q x p.
#'
#' The objective function is f(B) = 0.5 * trace(B G B') - trace(B g) + lambda * ||B||_*,
#' where ||B||_* is the nuclear norm of B.
#'
#' @param G The Gram matrix X'X / n, of size p x p
#' @param g The matrix X'y / n, of size p x q
#' @param lambda The regularization parameter
#' @param B0 The initial value of B
#' @param method The method to use: "fista" or "admm"
#' @param rho The ADMM penalty parameter (if method is "admm")
#' @param refine Whether to perform debiasing on the support
#' @param max_iter The maximum number of iterations
#' @param tolerance The convergence tolerance
#' @param verbose Whether to print convergence information
#'
#' @return B The estimated matrix
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
#' B_est <- mat_nuclear(G, g, lambda)
mat_nuclear <- function(G, g, lambda, B0 = NULL, method = c("fista", "admm"), rho = NULL,
                        refine = T, max_iter = 1000, tolerance = 1e-6, verbose = FALSE) {

  p <- nrow(G); q = ncol(g)
  # Initialize X0
  if (is.null(B0)) {
    B0 <- matrix(0, q, p)
  }

  B_prev <- B0

  method <- match.arg(method)

  # --- ALGORITHM-SPECIFIC SETUP ---
  if (method == "admm") {
    # ADMM: Pre-compute Cholesky for linear solve
    if (is.null(rho)) { rho <- sum(diag(G)) / p }
    M_chol <- tryCatch(chol(G + diag(rho, p)), error = function(e) NULL)
    Z <- matrix(0, q, p); U <- matrix(0, q, p)
  } else if (method == "fista") {
    # Compute step size using Rspectra
    eta <- tryCatch({
      1 / eigs_sym(G, k = 1, which = "LM")$values[1]
    }, warning = function(w) {
      1 / max(eigen(G, symmetric = TRUE, only.values = TRUE)$values)
    }, error = function(e) {
      1 / max(eigen(G, symmetric = TRUE, only.values = TRUE)$values)
    })
    if (is.infinite(eta)) {
      return(matrix(0, q, p))
    }

    # FISTA
    Z <- B0
    t_curr <- 1
    t_prev <- 1
  }

  for (iter in 1:max_iter) {
    if (method == "admm") {
      # ================= ADMM UPDATE =================
      # 1. Ridge Step: B(G + rho*I) = g' + rho(Z - U)
      RHS <- t(g) + rho * (Z - U)

      # Solve M * B' = RHS' (Using pre-computed Cholesky if possible)
      B_t <- if (!is.null(M_chol)) backsolve(M_chol, forwardsolve(t(M_chol), t(RHS))) else solve(G + diag(rho, p), t(RHS))
      B_curr <- t(B_t)

      # 2. SVT Step: Z = SVT(B + U)
      Z <- soft_threshold_nuclear(B_curr + U, lambda / rho)

      # 3. Dual Step
      U <- U + (B_curr - Z)
    } else {
      # ================= FISTA UPDATE =================
      # Gradient step
      Grad <- tcrossprod(Z, G) - t(g)

      # SVT
      B_curr <- soft_threshold_nuclear(Z - eta * Grad, eta * lambda)

      # FISTA momentum
      if (sum((Z - B_curr) * (B_curr - B_prev)) < 0) {
        # Restart
        Z <- B_curr
        t_curr <- 1
      } else {
        # Continue with momentum
        # FISTA momentum
        t_curr <- (1 + sqrt(1 + 4 * t_prev^2)) / 2
        Z <- B_curr + (t_prev - 1) / t_curr * (B_curr - B_prev)
      }

      t_prev <- t_curr
    }

    # Convergence check
    norm_diff <- sqrt(sum((B_curr - B_prev)^2))
    norm_prev <- sqrt(sum(B_prev^2))

    if (ifelse(norm_prev < 1e-10, norm_diff < tolerance, norm_diff / norm_prev < tolerance)) {
      if (verbose) cat(sprintf("Converged in %d iterations\n", iter))
      break
    }

    B_prev <- B_curr
  }

  # --- DEBIASING / REFITTING STEP (DIAGONAL S) ---
  if (refine) {
    # 1. Extract Support via SVD
    svd_B <- svd(B_curr)
    r <- length(svd_B$d > 1e-8)

    if (r > 0) {
      U_r <- svd_B$u[, 1:r, drop = FALSE] # q x r
      V_r <- svd_B$v[, 1:r, drop = FALSE] # p x r

      # 2. Compute numerators and denominators efficiently
      # Numerator: diag(V_r' * g * U_r) -> colSums(V_r * (g %*% U_r))
      # Denominator: diag(V_r' * G * V_r) -> colSums(V_r * (G %*% V_r))

      # g is p x q, U_r is q x r -> gU is p x r
      gV = crossprod(g, V_r)
      num = colSums(U_r * gV)

      # G is p x p, V_r is p x r -> GV is p x r
      GV <- crossprod(G, V_r)
      den <- colSums(V_r * GV)

      # 3. Solve for new singular values (element-wise division)
      # Handle potential division by zero if variance in that direction is 0
      s_new <- numeric(r)
      valid_idx <- abs(den) > 1e-10
      s_new[valid_idx] <- num[valid_idx] / den[valid_idx]

      # 4. Reconstruct B with new singular values
      # B = U * diag(s_new) * V'
      # Efficient multiplication: scale columns of U then multiply by V'
      B_curr = crossprod(t(U_r), s_new * t(V_r))

    } else {
      B_curr <- matrix(0, nrow(B_curr), ncol(B_curr))
    }
  }

  return(B_curr)
}
