# Multifactor Risk Models

[![CI](https://github.com/AhmadEblagh205/multifactor-risk-models/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/multifactor-risk-models/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I built this as an independent, simulated benchmark to see what a factor model can identify and how uncertain factor premia are. All data is synthetic, generated from a process where the true factors are known, so I can score each estimator against the answer. It is a methodology exercise, not an investment strategy or a track record.

## What it does

The same synthetic equity panel is analysed in three ways.

- Pre-specified economic factors: time-series betas, then Fama-MacBeth premium estimation with iid and Newey-West standard errors.
- Characteristic-sorted mimicking portfolios for market, size, value and momentum, built without estimating betas.
- Maximum-likelihood factor analysis with varimax rotation, with the factor count chosen by Kaiser's rule or Horn's parallel analysis.
- Euler-style factor risk contributions for an equal-weight portfolio, with the full residual covariance kept.
- A switch that adds profitability and investment factors to the simulator.

## Results

All figures below come from simulated data. In the default three-year world the size and value sample premia can take the wrong sign, so I treat that single world as an illustration. A 100-world Monte Carlo study gives the following mean absolute premium error.

| History | Mean absolute premium error |
|---:|---:|
| 1 year | 0.108 |
| 3 years | 0.065 |
| 10 years | 0.038 |
| 30 years | 0.021 |

With persistent factor returns, 95% interval coverage over 100 three-year worlds is:

| Persistence | iid | Newey-West |
|---:|---:|---:|
| 0.0 | 0.950 | 0.943 |
| 0.2 | 0.922 | 0.948 |
| 0.4 | 0.885 | 0.955 |
| 0.6 | 0.743 | 0.907 |

The default world has four latent factors. Kaiser's rule selects five and parallel analysis selects four. The canonical correlations between the estimated and true factor spaces are 0.980, 0.926, 0.878 and 0.859. The mimicking portfolios correlate with the true factors at 0.977 (market), 0.832 (size), 0.827 (value) and 0.897 (momentum), although their three-year mean returns are noisy.

Figures are in `reports/figures/` and the numbers are in `reports/*.json`. Derivations are in `docs/METHODOLOGY.md`.

## How to run

```bash
pip install -r requirements.txt

# Synthetic benchmark
python src/run.py
python src/mimicking.py
python src/five_factor.py

# Repeated-simulation study (100 worlds per setting)
python src/robustness.py

# Optional real returns panel (statistical-factor route)
python src/run.py --returns-csv data/raw/returns.csv

# Returns plus factor returns (both routes where applicable)
python src/run.py \
  --returns-csv data/raw/returns.csv \
  --factors-csv data/raw/factors.csv

pytest
```

CSV inputs need a date index, numeric columns and no missing values. The returns and factor panels are aligned on their overlapping dates. CI runs the tests on every push and pull request.

## Notes

The Newey-West errors deal with serial dependence in the premium series. They do not include a Shanken correction for first-stage beta estimation. Latent factors are only identified up to rotation, so I compare subspaces rather than matching axes to named factors. The five-factor report covers a single simulated world, and I rely on the repeated-simulation study rather than that one run.

## Licence

MIT, see [LICENSE](LICENSE).
