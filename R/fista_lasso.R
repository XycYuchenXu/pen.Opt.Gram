#' @keywords internal
soft_threshold_lasso <- function(z, lambda_j, weak = FALSE) {
  if (weak) {
    # Weak sparsity: use half threshold for values in [-lambda, lambda]
    ifelse(z > lambda_j, z - lambda_j,
           ifelse(z < -lambda_j, z + lambda_j, lambda_j / 2))
  } else {
    # Standard lasso
    sign(z) * pmax(abs(z) - lambda_j, 0)
  }
}

#' FISTA algorithm for Lasso regression
#'
#' @param gram_matrix The Gram matrix X'X / n
#' @param xy The cross-product X'y / n
#' @param beta_init The initial value of beta
#' @param lambda The regularization parameter
#' @param penalty_factors The penalty factors for each coefficient
#' @param weak Logical, whether to use weak sparsity
#' @param refine Whether to perform debiasing on the support
#' @param max_iter The maximum number of iterations
#' @param tolerance The convergence tolerance
#'
#' @returns beta The estimated coefficients
#' @importFrom RSpectra eigs_sym
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
#' # Set regularization parameter and penalty factors
#' lambda <- 0.1
#' penalty_factors <- rep(1, p)
#' # Initial beta
#' beta_init <- rep(0, p)
#' # Estimate coefficients with FISTA
#' beta_est <- fista_lasso(G, g, beta_init, lambda, penalty_factors, weak = TRUE)
fista_lasso <- function(gram_matrix, xy, beta_init, lambda, penalty_factors,
                        weak = FALSE, refine = T, max_iter = 1000, tolerance = 1e-6) {

  p <- length(beta_init)
  beta <- beta_init
  beta_old <- beta_init
  y <- beta_init  # Extrapolation point
  t <- 1  # Momentum parameter

  # Compute Lipschitz constant (largest eigenvalue of Gram matrix)
  step_size <- tryCatch({
    1 / eigs_sym(gram_matrix, k = 1, which = "LM")$values[1]
  }, warning = function(w) {
    1 / max(eigen(gram_matrix, symmetric = TRUE, only.values = TRUE)$values)
  }, error = function(e) {
    1 / max(eigen(gram_matrix, symmetric = TRUE, only.values = TRUE)$values)
  })

  for (iter in 1:max_iter) {
    # Gradient step
    gradient <- tcrossprod(y, gram_matrix) - t(xy)
    z <- y - step_size * gradient

    # Proximal step (soft-thresholding)
    scaled_penalties <- lambda * penalty_factors * step_size
    beta_new <- soft_threshold_lasso(z, scaled_penalties, weak)

    # Check convergence
    max_change <- max(abs(beta_new - beta))
    if (max_change < tolerance) {
      return(as.vector(beta_new))
    }

    # Nesterov momentum update
    t_new <- (1 + sqrt(1 + 4 * t^2)) / 2
    y <- beta_new + ((t - 1) / t_new) * (beta_new - beta)

    # Update for next iteration
    beta <- beta_new
    t <- t_new
  }

  if (!weak && refine) {
    supp_ind = which(beta != 0)
    Si_l0 = length(supp_ind)
    if (Si_l0 > 0) {
      beta[supp_ind] = crossprod(ginv(gram_matrix[supp_ind, supp_ind, drop = F], tol = 1e-6),
                                 xy[supp_ind])
    }
  }

  return(as.vector(beta))
}
