"""Simulated equity cross-sections with known factor structure.

The default world is intentionally simple enough that every estimator can be
scored against truth. Daily returns follow

    r_it = beta_i' f_t + epsilon_it

with factor premia specified in annual units, characteristics tied to selected
betas, and idiosyncratic noise. ``factor_ar`` can add AR(1) persistence while
preserving each factor's unconditional mean and volatility; this is used to
stress-test iid versus HAC uncertainty estimates.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FACTORS = ["market", "size", "value", "momentum"]
FACTORS5 = FACTORS + ["profitability", "investment"]
TRADING_DAYS = 252


@dataclass
class FactorWorld:
    n_assets: int = 60
    n_days: int = 756
    seed: int = 0
    premia: dict = field(default_factory=lambda: {
        "market": 0.06, "size": 0.025, "value": 0.03, "momentum": 0.04})
    factor_vol: dict = field(default_factory=lambda: {
        "market": 0.16, "size": 0.10, "value": 0.09, "momentum": 0.12})
    idio_vol: float = 0.25
    five_factor: bool = False
    factor_ar: float = 0.0

    def __post_init__(self):
        if not -0.95 < self.factor_ar < 0.95:
            raise ValueError("factor_ar must lie between -0.95 and 0.95")

        rng = np.random.default_rng(self.seed)
        n, T = self.n_assets, self.n_days

        self.factors = list(FACTORS5 if self.five_factor else FACTORS)
        if self.five_factor:
            self.premia = {**self.premia,
                           "profitability": 0.03, "investment": 0.02}
            self.factor_vol = {**self.factor_vol,
                               "profitability": 0.08, "investment": 0.07}

        self.chars = pd.DataFrame({
            "log_size": rng.normal(0, 1, n),
            "book_to_market": rng.normal(0, 1, n),
            "past_return": rng.normal(0, 1, n),
            "profitability": rng.normal(0, 1, n),
            "asset_growth": rng.normal(0, 1, n),
        }, index=[f"A{i:02d}" for i in range(n)])

        self.beta = pd.DataFrame({
            "market": 1.0 + 0.25 * rng.normal(0, 1, n),
            "size": -0.6 * self.chars.log_size.to_numpy()
                    + 0.2 * rng.normal(0, 1, n),
            "value": 0.6 * self.chars.book_to_market.to_numpy()
                     + 0.2 * rng.normal(0, 1, n),
            "momentum": 0.6 * self.chars.past_return.to_numpy()
                        + 0.2 * rng.normal(0, 1, n),
        }, index=self.chars.index)
        if self.five_factor:
            self.beta["profitability"] = (
                0.5 * self.chars.profitability.to_numpy()
                + 0.2 * rng.normal(0, 1, n))
            self.beta["investment"] = (
                -0.5 * self.chars.asset_growth.to_numpy()
                + 0.2 * rng.normal(0, 1, n))

        mu = np.array([self.premia[f] for f in self.factors]) / TRADING_DAYS
        vol = (np.array([self.factor_vol[f] for f in self.factors])
               / np.sqrt(TRADING_DAYS))

        if self.factor_ar == 0.0:
            f = rng.normal(mu, vol, size=(T, len(self.factors)))
        else:
            # Innovations are scaled so the stationary unconditional variance
            # remains ``vol**2`` despite AR(1) persistence.
            phi = self.factor_ar
            innovation_vol = vol * np.sqrt(1.0 - phi ** 2)
            f = np.empty((T, len(self.factors)))
            f[0] = rng.normal(mu, vol)
            for t in range(1, T):
                f[t] = (mu + phi * (f[t - 1] - mu)
                        + rng.normal(0.0, innovation_vol))

        index = pd.bdate_range("2022-01-03", periods=T)
        self.factor_returns = pd.DataFrame(f, columns=self.factors, index=index)
        eps = rng.normal(0, self.idio_vol / np.sqrt(TRADING_DAYS), (T, n))
        self.returns = pd.DataFrame(
            f @ self.beta.to_numpy().T + eps,
            index=index, columns=self.beta.index)
