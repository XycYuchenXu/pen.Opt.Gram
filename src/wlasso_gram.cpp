// src/wlasso_gram.cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

//' @importFrom Rcpp evalCpp
// [[Rcpp::export]]
arma::rowvec lasso_row_cpp(const arma::mat& xtx,
                           const arma::vec& xty_i,
                           double lambda,
                           const arma::rowvec& x0,
                           bool weak = false,
                           int max_iter = 1000,
                           double tolerance = 1e-6) {

  int p = xtx.n_rows;

  // Extract diagonal of xtx
  vec xtx_diag = xtx.diag();

  // Initialize
  rowvec x_curr = x0;
  rowvec x_prev = x_curr;

  // Coordinate descent for this row
  for (int iter = 0; iter < max_iter; iter++) {

    for (int j = 0; j < p; j++) {

      // Compute residual: xty_i[j] - xtx[j,:] * x_curr^T (excluding j-th component)
      double r_j = xty_i(j);
      for (int k = 0; k < p; k++) {
        r_j -= xtx(j, k) * x_curr(k);
      }
      r_j += xtx_diag(j) * x_curr(j);

      // Soft-thresholding operator
      double x_new;
      if (r_j > lambda) {
        x_new = (r_j - lambda) / xtx_diag(j);
      } else if (r_j < -lambda) {
        x_new = (r_j + lambda) / xtx_diag(j);
      } else {
        if (weak) {
          x_new = (r_j > 0) ? lambda / (2.0 * xtx_diag(j)) : -lambda / (2.0 * xtx_diag(j));
        } else {
          x_new = 0.0;
        }
      }

      // Update coordinate
      x_curr(j) = x_new;
    }

    // Check convergence
    double norm_diff = norm(x_curr - x_prev, 2);
    double norm_prev = norm(x_prev, 2);

    if (norm_prev < 1e-10 ? norm_diff < tolerance : norm_diff / norm_prev < tolerance) {
      break;
    }

    x_prev = x_curr;
  }

  // Debiasing step
  if (!weak) {
    uvec supp_ind = find(x_curr != 0);
    int Si_l0 = supp_ind.n_elem;

    if (Si_l0 > 0) {
      mat xtx_sub = xtx(supp_ind, supp_ind);
      vec xty_sub = xty_i(supp_ind);

      vec beta = pinv(xtx_sub) * xty_sub;

      for (int j = 0; j < Si_l0; j++) {
        x_curr(supp_ind(j)) = beta(j);
      }
    }
  }

  return x_curr;
}
