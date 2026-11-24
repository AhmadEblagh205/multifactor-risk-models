"""Pre-specified factor-model route: time-series betas, Fama-MacBeth
cross-sectional premia, HAC uncertainty and portfolio risk attribution."""
import numpy as np
import pandas as pd

from world import TRADING_DAYS


def time_series_betas(returns: pd.DataFrame,
                      factors: pd.DataFrame) -> pd.DataFrame:
    if not returns.index.equals(factors.index):
        raise ValueError("returns and factors must have identical aligned index")
    names = list(factors.columns)
    X = np.column_stack([np.ones(len(factors)), factors.to_numpy()])
    coef, _, _, _ = np.linalg.lstsq(X, returns.to_numpy(), rcond=None)
    fitted = X @ coef
    resid = returns.to_numpy() - fitted
    total_var = returns.to_numpy().var(axis=0)
    r2 = 1 - resid.var(axis=0) / total_var
    out = pd.DataFrame(coef[1:].T, index=returns.columns, columns=names)
    out["alpha_ann"] = coef[0] * TRADING_DAYS
    out["r2"] = r2
    return out


def newey_west_se(series: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC standard error of a sample mean using Bartlett weights."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    T = len(x)
    if T < 2:
        raise ValueError("need at least two observations")
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    lags = min(max(int(lags), 0), T - 1)
    long_run_var = x @ x / T
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma = (x[lag:] @ x[:-lag]) / T
        long_run_var += 2 * weight * gamma
    return float(np.sqrt(max(long_run_var, 0.0) / T))


def fama_macbeth(returns: pd.DataFrame, betas: pd.DataFrame,
                  factor_names: list[str] | None = None) -> pd.DataFrame:
    """Two-pass Fama-MacBeth premium estimates.

    The same cross-sectional design matrix applies at every date, so all
    period-by-period regressions are solved in one least-squares call. Reported
    HAC errors quantify serial dependence in the premium time series; they do
    *not* implement a Shanken correction for first-stage beta estimation.
    """
    if factor_names is None:
        factor_names = [c for c in betas.columns
                        if c not in {"alpha_ann", "r2"}]
    B = np.column_stack([np.ones(len(betas)),
                         betas[factor_names].to_numpy()])
    # Solve B @ coefficients_t = cross-sectional returns_t for every t.
    G = np.linalg.lstsq(B, returns.to_numpy().T, rcond=None)[0].T
    names = ["intercept"] + list(factor_names)
    mean = G.mean(axis=0) * TRADING_DAYS
    se = G.std(axis=0, ddof=1) / np.sqrt(len(G)) * TRADING_DAYS
    se_nw = np.array([newey_west_se(G[:, j]) for j in range(G.shape[1])]) \
        * TRADING_DAYS
    return pd.DataFrame({
        "premium_ann": mean,
        "se_ann": se,
        "se_nw_ann": se_nw,
        "t_stat": np.divide(mean, se, out=np.full_like(mean, np.nan),
                            where=se > 0),
        "t_stat_nw": np.divide(mean, se_nw,
                               out=np.full_like(mean, np.nan),
                               where=se_nw > 0),
    }, index=names)


def risk_decomposition(weights: pd.Series, betas: pd.DataFrame,
                       factor_cov_ann: pd.DataFrame,
                       residual_cov_ann: pd.DataFrame) -> pd.Series:
    """Euler-style factor contributions plus residual portfolio variance.

    For correlated factors, an individual factor contribution may be negative;
    contributions still sum to total factor variance. Residual covariance is
    retained rather than assuming asset-specific residuals are independent.
    """
    names = list(factor_cov_ann.columns)
    w = weights.reindex(betas.index).to_numpy()
    b_p = betas[names].to_numpy().T @ w
    sigma_f = factor_cov_ann.loc[names, names].to_numpy()
    contrib = b_p * (sigma_f @ b_p)
    sigma_e = residual_cov_ann.loc[betas.index, betas.index].to_numpy()
    residual = float(w @ sigma_e @ w)
    parts = pd.Series(contrib, index=names, dtype=float)
    parts["residual"] = residual
    parts["total"] = float(parts[names].sum() + residual)
    return parts
