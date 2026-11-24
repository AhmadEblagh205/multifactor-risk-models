"""Run the factor-model diagnostics.

Examples
--------
Synthetic benchmark with known truth:
    python src/run.py

Returns-only panel (statistical-factor route):
    python src/run.py --returns-csv data/raw/returns.csv

Returns plus pre-specified factor returns:
    python src/run.py --returns-csv data/raw/returns.csv \
                      --factors-csv data/raw/factors.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prespecified import fama_macbeth, risk_decomposition, time_series_betas
from statistical import (canonical_correlations, fit_factors, kaiser_count,
                         match_to_truth, parallel_analysis, scree)
from world import FACTORS, TRADING_DAYS, FactorWorld

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"{path} contains no data")
    if df.isna().any().any():
        raise ValueError(f"{path} contains missing values")
    return df.astype(float)


def _align(returns: pd.DataFrame, factors: pd.DataFrame):
    common = returns.index.intersection(factors.index)
    if len(common) < 20:
        raise ValueError("returns and factors have too few overlapping dates")
    return returns.loc[common], factors.loc[common]


def dot_range(ax, values, errors, labels, title, xlabel):
    y = range(len(values))
    ax.hlines(y, values - errors, values + errors, lw=2)
    ax.plot(values, y, "o")
    ax.axvline(0, lw=0.8)
    ax.set_yticks(list(y), labels)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(alpha=0.3, axis="x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns-csv", "--csv", dest="returns_csv", default=None,
                    help="wide CSV of asset returns with a date index")
    ap.add_argument("--factors-csv", default=None,
                    help="optional wide CSV of pre-specified factor returns")
    ap.add_argument("--parallel-iterations", type=int, default=200)
    args = ap.parse_args()
    if args.factors_csv and not args.returns_csv:
        ap.error("--factors-csv requires --returns-csv")

    FIGURES.mkdir(parents=True, exist_ok=True)
    world = FactorWorld()
    truth_available = args.returns_csv is None

    if truth_available:
        returns = world.returns
        factors = world.factor_returns
    else:
        returns = _load_csv(args.returns_csv)
        factors = _load_csv(args.factors_csv) if args.factors_csv else None
        if factors is not None:
            returns, factors = _align(returns, factors)

    summary = {"assets": int(returns.shape[1]), "days": int(len(returns))}

    if factors is not None:
        factor_names = list(factors.columns)
        betas = time_series_betas(returns, factors)
        fm = fama_macbeth(returns, betas, factor_names)
        summary["mean_time_series_r2"] = round(float(betas.r2.mean()), 3)
        summary["fama_macbeth"] = fm.round(4).to_dict("index")

        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        prem = fm.loc[factor_names]
        dot_range(ax, prem.premium_ann.to_numpy(),
                  1.96 * prem.se_nw_ann.to_numpy(), factor_names,
                  "Fama-MacBeth premia with Newey-West 95% intervals",
                  "Annualised premium")
        if truth_available:
            for i, f in enumerate(factor_names):
                ax.plot(world.premia[f], i, "x", ms=9)
            summary["true_premia"] = world.premia
        fig.tight_layout()
        fig.savefig(FIGURES / "fm_premia.png", dpi=150)
        plt.close(fig)

        weights = pd.Series(1 / returns.shape[1], index=returns.columns)
        residuals = returns - (factors @ betas[factor_names].T
                               + betas["alpha_ann"] / TRADING_DAYS)
        decomp = risk_decomposition(
            weights,
            betas,
            factors.cov() * TRADING_DAYS,
            residuals.cov() * TRADING_DAYS,
        )
        summary["ew_portfolio_variance_split"] = decomp.round(5).to_dict()

        if truth_available:
            # Compare the estimated pricing relation with the expected returns
            # implied by the known data-generating process, not with noisy
            # realised sample means.
            est_prem = fm.loc[factor_names, "premium_ann"]
            est_expected = (float(fm.loc["intercept", "premium_ann"])
                            + betas[factor_names] @ est_prem)
            true_expected = world.beta[factor_names] @ pd.Series(world.premia)
            rmse = float(np.sqrt(np.mean((est_expected - true_expected) ** 2)))
            corr = float(np.corrcoef(est_expected, true_expected)[0, 1])
            summary["pricing_expected_return_rmse_ann"] = round(rmse, 4)
            summary["pricing_expected_return_corr"] = round(corr, 3)

            fig2, ax2 = plt.subplots(figsize=(6, 5.2))
            ax2.scatter(true_expected, est_expected, s=22, alpha=0.7)
            lo = min(true_expected.min(), est_expected.min()) - 0.01
            hi = max(true_expected.max(), est_expected.max()) + 0.01
            ax2.plot([lo, hi], [lo, hi], "k--", lw=1, label="45 degree line")
            ax2.set(
                xlabel="True expected return (annualised)",
                ylabel="Estimated expected return (annualised)",
                title="Pricing relation against known synthetic truth",
            )
            ax2.legend()
            ax2.grid(alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(FIGURES / "sml.png", dpi=150)
            plt.close(fig2)

    # Statistical-factor route.
    eig = scree(returns)
    k_kaiser = kaiser_count(eig)
    pa = parallel_analysis(returns, n_iter=args.parallel_iterations)
    k_parallel = pa["count"]
    if k_parallel < 1:
        k_parallel = 1
    loadings, scores, comm = fit_factors(returns, k_parallel)
    summary["kaiser_factor_count"] = k_kaiser
    summary["parallel_analysis_factor_count"] = k_parallel
    summary["mean_communality"] = round(float(comm.mean()), 3)

    fig3, ax3 = plt.subplots(figsize=(7.5, 4.2))
    n_show = min(15, len(eig))
    x = np.arange(1, n_show + 1)
    ax3.plot(x, eig[:n_show], "o-", lw=1.6, label="Observed")
    ax3.plot(x, pa["threshold"][:n_show], "--", lw=1.4,
             label="95% parallel-analysis threshold")
    ax3.axhline(1.0, ls=":", lw=1, label="Kaiser = 1")
    ax3.set(xlabel="Component", ylabel="Eigenvalue",
            title=(f"Factor-count diagnostics: Kaiser {k_kaiser}, "
                   f"parallel analysis {k_parallel}"))
    ax3.legend()
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(FIGURES / "scree.png", dpi=150)
    plt.close(fig3)

    if truth_available:
        match = match_to_truth(scores, world.factor_returns)
        cc = canonical_correlations(scores, world.factor_returns)
        angles = np.degrees(np.arccos(np.clip(cc, -1, 1)))
        summary["true_factor_count"] = len(world.factors)
        summary["latent_space_recovery_R"] = {
            f: round(float(match[f]), 3) for f in world.factors}
        summary["latent_space_canonical_correlations"] = [
            round(float(v), 3) for v in cc]
        summary["latent_space_principal_angles_deg"] = [
            round(float(v), 2) for v in angles]

    out = ROOT / "reports" / "summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Figures -> {FIGURES}")


if __name__ == "__main__":
    main()
