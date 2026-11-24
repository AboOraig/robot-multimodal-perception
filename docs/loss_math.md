# Heteroscedastic Gaussian NLL for the 4-D state y = [x, y, z, phi]  (phi = 4*yaw)

Simulation-only. Written before the M5 implementation, as required by the project brief.

## Model
For input x the network outputs a mean y_hat = [p_hat, phi_hat] and a lower-triangular factor L(x):
`l_ii = softplus(a_ii) + eps`, off-diagonals unconstrained, `Sigma = L L^T` (symmetric positive definite for eps > 0).
`Sigma` is a full 4x4 matrix: cross-covariances (position-yaw, x-z, ...) are learned; no independence is assumed.
Position outputs are affine in the training mean/std and the covariance is `S L L^T S` with `S = diag(sd_x, sd_y, sd_z, 1)`.

## Loss
Tangent error `e = [p - p_hat ; wrap(phi - phi_hat)]`, `z = L^-1 e` (triangular solve):

    NLL(x, y) = 1/2 z^T z + sum_i log l_ii + 2 log(2 pi)         (the constant is kept so values match metrics.py)

Gradient wrt the mean: `-Sigma^-1 e`. Large predicted variance therefore shuts off mean learning (it can "explain away" a bad fit).
Remedies: (i) MSE warm-up on the normalised tangent error, (ii) beta-NLL (Seitzer et al., 2022): weight each sample by
`stopgrad(det(Sigma)^(beta/4))`; beta = 0 is plain NLL. Default here: warm-up on, beta = 0.

## Properties and limits
* Population minimiser over a flexible Gaussian family = conditional mean and covariance of y | x, even when the true
  conditional is non-Gaussian. Mean/d2 may look fine while 68/95% coverage is off for multimodal or heavy-tailed errors.
* Calibrated model: E[d2] = 4 (chi-square, 4 dof).
* Wrapped residual: bounded, so no blow-up, but the Gaussian is a good model only for sigma_phi << pi. With unobservable yaw
  the optimum is sigma_phi^2 ~ pi^2/3 (uniform on the circle) and the Gaussian is crude.
* eps floor prevents log-det -> -inf on memorised frames. Overfitting still produces overconfidence: select on validation, report on test.
* Aleatoric only: unseen degradations (not in training) are not covered by the predicted covariance; this is measured, not assumed.
