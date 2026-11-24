"""Characteristic-sorted factor-mimicking portfolios for the synthetic world."""
import json
from pathlib import Path

import pandas as pd

from prespecified import newey_west_se
from world import FACTORS, TRADING_DAYS, FactorWorld

ROOT = Path(__file__).resolve().parents[1]
SORTS = {"size": ("log_size", -1),
         "value": ("book_to_market", +1),
         "momentum": ("past_return", +1)}


def mimicking_returns(world: FactorWorld) -> pd.DataFrame:
    rets = world.returns
    out = {"market": rets.mean(axis=1)}
    for factor, (char, sign) in SORTS.items():
        ranked = world.chars[char].rank(pct=True)
        top = ranked >= 2 / 3
        bottom = ranked <= 1 / 3
        spread = (rets.loc[:, top.values].mean(axis=1)
                  - rets.loc[:, bottom.values].mean(axis=1))
        out[factor] = sign * spread
    return pd.DataFrame(out)


def evaluate(world: FactorWorld) -> dict:
    mim = mimicking_returns(world)
    corr = {f: round(float(mim[f].corr(world.factor_returns[f])), 3)
            for f in FACTORS}
    premia = {f: float(mim[f].mean() * TRADING_DAYS) for f in FACTORS}
    se = {f: float(newey_west_se(mim[f].to_numpy()) * TRADING_DAYS)
          for f in FACTORS}
    ci = {f: [round(premia[f] - 1.96 * se[f], 4),
              round(premia[f] + 1.96 * se[f], 4)] for f in FACTORS}
    return {
        "corr_with_true_factor": corr,
        "implied_premium_ann": {f: round(v, 4) for f, v in premia.items()},
        "premium_se_nw_ann": {f: round(v, 4) for f, v in se.items()},
        "premium_95ci_nw_ann": ci,
        "true_premium_ann": world.premia,
        "note": ("Sorted spreads recover factor directions strongly, but "
                 "three-year mean premia remain noisy; uncertainty bands are "
                 "reported rather than treating the sample means as truth."),
    }


def main():
    result = evaluate(FactorWorld())
    out = ROOT / "reports" / "mimicking_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
