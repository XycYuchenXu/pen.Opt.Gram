// src/group_lasso_gram.cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

//' @importFrom Rcpp evalCpp
// [[Rcpp::export]]
arma::mat group_lasso_cpp(const arma::mat& xtx,
                          const arma::mat& xty,
                          const arma::mat Grp,
                          const double lambda,
                          const arma::mat& X0,
                          const double eta,
                          const int max_iter = 1000,
                          const double tolerance = 1e-6) {

  int p = Grp.n_rows;

  // Convert group matrix to factor indices
  vec grp_vec = vectorise(Grp);
  vec unique_vals = unique(grp_vec);
  int n_groups = unique_vals.size();

  std::vector<std::vector<uvec>> grps(n_groups);
  std::vector<double> weights(n_groups);
  std::vector<int> group_sizes(n_groups);

  for (int g = 0; g < n_groups; g++) {
    double val = unique_vals(g);
    uvec indices = find(grp_vec == val);
    int n_elem = indices.n_elem;

    group_sizes[g] = n_elem;
    weights[g] = std::sqrt((double)n_elem);

    // Convert linear indices to row, col pairs
    grps[g].reserve(n_elem);
    for (int j = 0; j < n_elem; j++) {
      uvec rc(2);
      rc(0) = indices(j) % p;  // row
      rc(1) = indices(j) / p;  // col
      grps[g].push_back(rc);
    }
  }

  // Initialize gradient
  mat Grad = X0 * xtx;
  Grad -= xty.t();

  mat X_curr = X0;
  mat X_prev = X0;
  // Main iteration loop
  for (int iter = 0; iter < max_iter; iter++) {
    for (uword i = 0; i < unique_vals.n_elem; i++) {
      const std::vector<uvec>& ind_pairs = grps[i];
      double weight = weights[i];
      int n_elem = group_sizes[i];

      // Extract x_k and grad_k
      vec x_k(n_elem);
      vec grad_k(n_elem);

      for (int j = 0; j < n_elem; j++) {
        int ri = ind_pairs[j](0);
        int ci = ind_pairs[j](1);
        x_k(j) = X_prev(ri, ci);
        grad_k(j) = Grad(ri, ci);
      }

      // Proximal gradient step
      vec x_temp = x_k - eta * grad_k;
      double x_temp_norm = norm(x_temp, 2);
      double shrink = eta * lambda * weight / x_temp_norm;

      if (shrink >= 1.0) {
        x_temp.zeros();
      } else {
        x_temp = (1.0 - shrink) * x_temp;
      }

      // Update gradient and X
      vec diff = x_temp - x_k;
      for (int j = 0; j < n_elem; j++) {
        int ri = ind_pairs[j](0);
        int ci = ind_pairs[j](1);
        Grad.row(ri) += diff(j) * xtx.row(ci);
        X_curr(ri, ci) = x_temp(j);
      }
    }

    // Check convergence
    if (norm(X_curr - X_prev, "inf") < tolerance) {
      break;
    }

    X_prev = X_curr;
  }
  return X_curr;
}
