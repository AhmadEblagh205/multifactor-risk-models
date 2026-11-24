"""Repeated-simulation diagnostics for premium estimation and HAC coverage.

The point is not to find a lucky seed. It is to ask whether the estimator's
headline conclusions survive repeated synthetic worlds where the truth is
known.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prespecified import fama_macbeth, time_series_betas
from world import FACTORS, FactorWorld

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def _one(seed: int, years: int, factor_ar: float = 0.0):
    world = FactorWorld(seed=seed, n_days=252 * years, factor_ar=factor_ar)
    betas = time_series_betas(world.returns, world.factor_returns)
    fm = fama_macbeth(world.returns, betas, FACTORS)
    est = fm.loc[FACTORS, "premium_ann"].to_numpy()
    se_iid = fm.loc[FACTORS, "se_ann"].to_numpy()
    se_nw = fm.loc[FACTORS, "se_nw_ann"].to_numpy()
    truth = np.array([world.premia[f] for f in FACTORS])
    return est, se_iid, se_nw, truth


def sample_length_study(n_worlds: int = 200, horizons=(1, 3, 10, 30),
                        seed: int = 20000) -> dict:
    rows = {}
    for years in horizons:
        estimates, se_nw, truths = [], [], []
        for i in range(n_worlds):
            est, _, nw, truth = _one(seed + years * 1000 + i, years)
            estimates.append(est)
            se_nw.append(nw)
            truths.append(truth)
        estimates = np.asarray(estimates)
        se_nw = np.asarray(se_nw)
        truths = np.asarray(truths)
        err = estimates - truths
        cover = np.abs(err) <= 1.96 * se_nw
        rows[str(years)] = {
            "mean_absolute_error_by_factor": {
                f: round(float(v), 4)
                for f, v in zip(FACTORS, np.abs(err).mean(axis=0))},
            "median_absolute_error_by_factor": {
                f: round(float(v), 4)
                for f, v in zip(FACTORS, np.median(np.abs(err), axis=0))},
            "rmse_by_factor": {
                f: round(float(v), 4)
                for f, v in zip(FACTORS, np.sqrt((err ** 2).mean(axis=0)))},
            "nw_95_coverage_by_factor": {
                f: round(float(v), 3)
                for f, v in zip(FACTORS, cover.mean(axis=0))},
            "mean_mae_across_factors": round(float(np.abs(err).mean()), 4),
            "mean_nw_95_coverage": round(float(cover.mean()), 3),
        }
    return {"n_worlds": n_worlds, "horizons_years": list(horizons),
            "results": rows}


def hac_coverage_study(n_worlds: int = 200,
                       phis=(0.0, 0.2, 0.4, 0.6), years: int = 3,
                       seed: int = 50000) -> dict:
    rows = {}
    for j, phi in enumerate(phis):
        iid_cover, nw_cover = [], []
        for i in range(n_worlds):
            est, iid, nw, truth = _one(seed + j * 1000 + i, years, phi)
            iid_cover.append(np.abs(est - truth) <= 1.96 * iid)
            nw_cover.append(np.abs(est - truth) <= 1.96 * nw)
        iid_cover = np.asarray(iid_cover)
        nw_cover = np.asarray(nw_cover)
        rows[str(phi)] = {
            "iid_95_coverage_by_factor": {
                f: round(float(v), 3)
                for f, v in zip(FACTORS, iid_cover.mean(axis=0))},
            "nw_95_coverage_by_factor": {
                f: round(float(v), 3)
                for f, v in zip(FACTORS, nw_cover.mean(axis=0))},
            "mean_iid_95_coverage": round(float(iid_cover.mean()), 3),
            "mean_nw_95_coverage": round(float(nw_cover.mean()), 3),
        }
    return {"n_worlds": n_worlds, "years": years,
            "factor_ar_values": list(phis), "results": rows}


def _plot_sample_length(study: dict):
    years = np.array(study["horizons_years"], dtype=float)
    mae = np.array([study["results"][str(int(y))]["mean_mae_across_factors"]
                    for y in years])
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(years, mae, "o-", lw=2)
    ax.set_xscale("log")
    ax.set_xticks(years, [str(int(y)) for y in years])
    ax.set(xlabel="Sample length (years, log scale)",
           ylabel="Mean absolute premium error",
           title="Factor-premium estimation error shrinks slowly with history")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "premium_horizon.png", dpi=160)
    plt.close(fig)


def _plot_hac(study: dict):
    phis = study["factor_ar_values"]
    iid = [study["results"][str(p)]["mean_iid_95_coverage"] for p in phis]
    nw = [study["results"][str(p)]["mean_nw_95_coverage"] for p in phis]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(phis, iid, "o-", lw=2, label="iid SE")
    ax.plot(phis, nw, "o-", lw=2, label="Newey-West SE")
    ax.axhline(0.95, ls="--", lw=1, label="Nominal 95%")
    ax.set(xlabel="AR(1) persistence in factor returns",
           ylabel="Empirical 95% interval coverage",
           ylim=(0.65, 1.0),
           title="HAC uncertainty matters when premium series are persistent")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "hac_coverage.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=100)
    args = ap.parse_args()
    if args.worlds < 50:
        raise SystemExit("Use at least 50 worlds for a meaningful robustness run")
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    length = sample_length_study(args.worlds)
    hac = hac_coverage_study(args.worlds)
    _plot_sample_length(length)
    _plot_hac(hac)
    out = {"sample_length": length, "hac_coverage": hac}
    (REPORTS / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
