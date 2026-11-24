# Methodology and statistical caveats

## 1. Synthetic data-generating process

For asset `i` on date `t`,

```text
r_it = beta_i' f_t + epsilon_it
```

The default world contains market, size, value and momentum factors. Annual factor premia are converted to daily means and annual factor volatilities to daily standard deviations. Betas are partly linked to observable synthetic characteristics so that characteristic sorts have a genuine signal to recover.

When `factor_ar = phi != 0`, factor returns follow a stationary AR(1):

```text
f_t - mu = phi (f_{t-1} - mu) + eta_t
```

with innovation volatility scaled by `sqrt(1-phi^2)` so that unconditional factor volatility remains unchanged.

## 2. Time-series beta estimation

For each asset, OLS estimates

```text
r_i,t = alpha_i + beta_i' f_t + e_i,t.
```

The synthetic benchmark can compare estimated betas with the known generating betas.

## 3. Fama-MacBeth two-pass premium estimation

At each date, the cross-section is regressed on the estimated betas:

```text
r_t = lambda_0,t + B lambda_t + u_t.
```

The annualised factor-premium estimate is the time-series mean of `lambda_t` times 252.

Two uncertainty estimates are reported:

- iid standard error of the mean;
- Newey-West HAC standard error using Bartlett weights.

The implementation deliberately distinguishes two issues:

1. HAC handles serial dependence/heteroskedasticity in the **premium time series**.
2. It does **not** correct for first-stage beta estimation uncertainty (for example, Shanken-style generated-regressor adjustments).

The latter is a stated limitation rather than hidden behind a confidence interval.

## 4. Why the robustness study matters

A single simulated sample can make an estimator look unusually good or bad. Repeated worlds therefore evaluate:

- mean/median absolute premium error;
- RMSE;
- empirical coverage of nominal 95% intervals;
- sensitivity to sample length;
- sensitivity to serial persistence.

The study uses the same known annual premia in every world. Only realised factors and residual noise change.

## 5. Newey-West stress test

When `phi=0`, the slope series are close to iid and iid/HAC intervals have similar coverage. As persistence increases, iid standard errors become increasingly too small. Newey-West restores much of the nominal coverage, though with `phi=0.6` the simple default lag rule still under-covers. This is why the report describes HAC as a tool with a bandwidth choice, not a magic correction.

## 6. Portfolio risk attribution

For portfolio factor exposure `b_p = B' w` and factor covariance `Sigma_f`, total factor variance is

```text
b_p' Sigma_f b_p.
```

The per-factor Euler contribution is

```text
b_p,j (Sigma_f b_p)_j.
```

With correlated factors an individual contribution can be negative, but the contributions sum to total factor variance. Residual portfolio variance uses the full residual covariance matrix:

```text
w' Sigma_e w.
```

This avoids silently imposing diagonal residual covariance.

## 7. Characteristic-sorted factor-mimicking portfolios

Assets are ranked by a synthetic characteristic and split into terciles. Size uses small-minus-big; value and momentum use high-minus-low. The market factor is the equal-weight universe return.

Correlation with the true factor tests whether the constructed portfolio tracks the **direction** of the latent risk. The sample mean of that mimicking return is a much noisier estimate of the long-run premium, so a Newey-West uncertainty interval is reported as well.

## 8. Factor count: Kaiser versus parallel analysis

The Kaiser rule retains correlation-matrix eigenvalues greater than one. In the default four-factor world it selects five components.

Horn parallel analysis simulates noise-only panels of the same shape and retains an observed eigenvalue only when it exceeds the chosen (here 95th) percentile of its noise counterpart. The default world selects four.

This is a model-selection demonstration, not a claim that parallel analysis is universally optimal.

## 9. Rotation-invariant latent-space recovery

Factor analysis does not identify a unique coordinate system: rotations and sign changes can represent the same latent subspace. Therefore the project uses canonical correlations between the estimated score space and true factor-return space.

After centring each score matrix, QR bases `Q_est` and `Q_true` are formed. The singular values of

```text
Q_est' Q_true
```

are the canonical correlations (cosines of the principal angles) between the spaces. Values close to one mean strong subspace alignment regardless of axis labelling.

## 10. What this project does not show

The repository does not demonstrate a profitable trading strategy, real-world factor persistence, live portfolio construction, or professional investment performance. It is a controlled statistical benchmark designed so that errors, uncertainty and model-selection decisions can be checked against a known data-generating process.
