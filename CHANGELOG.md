# Changelog

## 0.5.0 — defensibility and robustness pass

- Reframed single-seed premium estimates as illustrations rather than proof of recovery.
- Added 100-world sample-length study across 1, 3, 10 and 30 years.
- Added AR(1) factor persistence and empirical iid versus Newey-West coverage study.
- Added Horn parallel analysis; the default four-factor world now demonstrates Kaiser over-extraction (5) versus parallel-analysis recovery (4).
- Added rotation-invariant canonical correlations/principal angles for latent factor-space recovery.
- Vectorised Fama-MacBeth cross-sectional regressions and documented the absence of Shanken correction.
- Upgraded portfolio risk decomposition to retain full residual covariance.
- Added HAC uncertainty intervals to characteristic-sorted mimicking premia.
- Reworked five-factor wording so one seed is not treated as validation.
- Added real-panel CLI support for returns-only and returns-plus-factor CSV inputs.
- Expanded tests from 7 to 13 and added `docs/METHODOLOGY.md`.

## 0.4.0 — five factor extension

Added profitability and investment to the synthetic world behind a flag that leaves the four-factor default untouched.

## 0.3.0 — factor-mimicking portfolios

Added characteristic-sorted tercile spreads for market, size, value and momentum.

## 0.2.0 — HAC uncertainty

Added Newey-West standard errors alongside iid Fama-MacBeth errors.

## 0.1.0

Initial release: pre-specified and statistical factor-model routes on a synthetic panel with known truth.
