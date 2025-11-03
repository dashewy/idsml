sigma_lda <- function (vector_1, vector_2, length, mu) {
  sigma = 0
  for (i in 1:length) {
    rh = t(t(c(vector_1[i] - mu[1], vector_2[i] - mu[2])))
    lh = t(c(vector_1[i] - mu[1], vector_2[i] - mu[2]))

    sigma = sigma + (rh %*% lh)
  }
  return(sigma)
}

delta_lda <- function (mu, sigma, pi_value, x) {
  sigma_inv = solve(sigma)
  eq_1 = t(c(x[1], x[2])) %*% sigma_inv %*% t(t(c(mu[1], mu[2])))
  eq_2 = (-1 / 2) * t(c(mu[1], mu[2])) %*% sigma_inv %*% t(t(c(mu[1], mu[2])))

  return(eq_1 + eq_2 + log(pi_value))

}

gradient_descent <- function (func, alpha, x_int, y_int, iter, inverse=T){
  partial_0 = D(func, "theta_0")
  partial_1 = D(func, "theta_1")

  theta_0 = x_int
  theta_1 = y_int

  for (i in 1:iter) {
    env = list2env(list(theta_0 = theta_0, theta_1 = theta_1))

    gradient_0 = eval(partial_0, envir = env)
    gradient_1 = eval(partial_1, envir = env)

    if (inverse){
      theta_0 = theta_0 + alpha * gradient_0
      theta_1 = theta_1 + alpha * gradient_1
    }
    else if (! inverse) {
      theta_0 = theta_0 - alpha * gradient_0
      theta_1 = theta_1 - alpha * gradient_1
    }
  }
  z = eval(func, envir = env)
  return(c(theta_0, theta_1, z))

}

# test_func <- expression(-3*log(1 + exp(-theta_0)) - 7*log(1 + exp(-(theta_0 + theta_1))) - 6*theta_0 - 5*theta_1)

# gradient_descent(test_func, 0.01, 2, 1, 2, inverse=F)