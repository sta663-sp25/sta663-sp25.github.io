data {
  int<lower=1> N;
  array[N] real x;
  vector[N] y;
}
transformed data {
  array[N] real xn = to_array_1d(x);
  vector[N] zeros = rep_vector(0, N);
}
parameters {
  real<lower=0> l;
  real<lower=0> s;
  real<lower=0> nug;
}
model {
  // Covariance
  matrix[N, N] K = gp_exp_quad_cov(x, s, l);
  matrix[N, N] L = cholesky_decompose(add_diag(K, nug^2));
  // priors
  l ~ gamma(2, 1);
  s ~ cauchy(0, 5);
  nug ~ cauchy(0, 1);
  // model
  y ~ multi_normal_cholesky(rep_vector(0, N), L);
}
