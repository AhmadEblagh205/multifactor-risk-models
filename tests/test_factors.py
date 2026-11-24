import numpy as np
import pandas as pd

from five_factor import evaluate as evaluate_five
from mimicking import evaluate as evaluate_mimicking
from prespecified import (fama_macbeth, newey_west_se, risk_decomposition,
                          time_series_betas)
from statistical import (canonical_correlations, fit_factors, kaiser_count,
                         parallel_analysis, scree)
from world import FACTORS, TRADING_DAYS, FactorWorld


def _world(seed=3):
    return FactorWorld(seed=seed)


def test_time_series_betas_recover_truth():
    w = _world()
    est = time_series_betas(w.returns, w.factor_returns)
    err = (est[FACTORS] - w.beta).abs().mean().mean()
    assert err < 0.15
    assert est.r2.mean() > 0.25


def test_fama_macbeth_truth_inside_broad_uncertainty_bands():
    w = _world()
    betas = time_series_betas(w.returns, w.factor_returns)
    fm = fama_macbeth(w.returns, betas, FACTORS)
    for f in FACTORS:
        assert abs(fm.loc[f, "premium_ann"] - w.premia[f]) \
            < 2.5 * fm.loc[f, "se_nw_ann"]


def test_fama_macbeth_vectorised_solution_matches_period_regressions():
    w = _world(seed=4)
    betas = time_series_betas(w.returns, w.factor_returns)
    fm = fama_macbeth(w.returns, betas, FACTORS)
    B = np.column_stack([np.ones(len(betas)), betas[FACTORS].to_numpy()])
    slopes = []
    for t in range(len(w.returns)):
        slopes.append(np.linalg.lstsq(B, w.returns.iloc[t].to_numpy(),
                                      rcond=None)[0])
    expected = np.asarray(slopes).mean(axis=0) * TRADING_DAYS
    assert np.allclose(fm["premium_ann"].to_numpy(), expected)


def test_newey_west_matches_iid_on_white_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4000)
    iid = x.std(ddof=1) / np.sqrt(len(x))
    assert abs(newey_west_se(x) - iid) / iid < 0.1


def test_newey_west_expands_se_for_persistent_series():
    rng = np.random.default_rng(1)
    x = np.zeros(5000)
    for t in range(1, len(x)):
        x[t] = 0.6 * x[t - 1] + rng.normal()
    iid = x.std(ddof=1) / np.sqrt(len(x))
    assert newey_west_se(x) > 1.35 * iid


def test_risk_decomposition_reconstructs_fitted_covariance():
    w = _world()
    betas = time_series_betas(w.returns, w.factor_returns)
    weights = pd.Series(1 / w.n_assets, index=w.returns.columns)
    fitted = (w.factor_returns @ betas[FACTORS].T
              + betas["alpha_ann"] / TRADING_DAYS)
    resid = w.returns - fitted
    decomp = risk_decomposition(weights, betas,
                                w.factor_returns.cov() * TRADING_DAYS,
                                resid.cov() * TRADING_DAYS)
    # OLS residuals are sample-orthogonal to the factor regressors, so the
    # covariance of fitted factor part + residual part reconstructs the sample
    # portfolio variance up to floating-point noise.
    port_var = float((w.returns @ weights).var() * TRADING_DAYS)
    assert abs(decomp["total"] - port_var) / port_var < 1e-10


def test_parallel_analysis_recovers_known_four_factor_dimension():
    w = FactorWorld(seed=0)
    eig = scree(w.returns)
    assert kaiser_count(eig) == 5  # deliberately over-extracts this seed
    pa = parallel_analysis(w.returns, n_iter=80, seed=123)
    assert pa["count"] == 4


def test_latent_space_recovery_is_rotation_invariant_and_strong():
    w = FactorWorld(seed=0)
    scores = fit_factors(w.returns, 4)[1]
    cc = canonical_correlations(scores, w.factor_returns)
    assert len(cc) == 4
    assert cc.min() > 0.80
    # Rotate latent scores: canonical correlations should not change.
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated = pd.DataFrame(scores.to_numpy() @ q,
                           index=scores.index, columns=scores.columns)
    assert np.allclose(cc, canonical_correlations(rotated, w.factor_returns))


def test_mimicking_portfolios_track_true_factors_and_report_uncertainty():
    res = evaluate_mimicking(_world(seed=0))
    corr = res["corr_with_true_factor"]
    assert corr["market"] > 0.95
    assert min(corr[f] for f in ["size", "value", "momentum"]) > 0.6
    assert set(res["premium_95ci_nw_ann"]) == set(FACTORS)


def test_five_factor_extension_reports_hac_intervals():
    res = evaluate_five(FactorWorld(five_factor=True, seed=2))
    assert list(res["factors"])[-2:] == ["profitability", "investment"]
    for row in res["factors"].values():
        assert row["se_nw"] > 0
        assert isinstance(row["truth_in_95ci"], bool)


def test_ar_world_preserves_mean_and_creates_persistence():
    w = FactorWorld(seed=8, n_days=10000, factor_ar=0.5)
    f = w.factor_returns["market"]
    expected_daily = w.premia["market"] / TRADING_DAYS
    assert abs(f.mean() - expected_daily) < 4 * f.std() / np.sqrt(len(f))
    assert 0.45 < f.autocorr(1) < 0.55
