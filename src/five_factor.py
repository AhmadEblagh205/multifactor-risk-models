"""Five-factor extension of the same synthetic two-pass benchmark."""
import json
from pathlib import Path

from prespecified import fama_macbeth, time_series_betas
from world import FactorWorld

ROOT = Path(__file__).resolve().parents[1]


def evaluate(world: FactorWorld) -> dict:
    betas = time_series_betas(world.returns, world.factor_returns)
    fm = fama_macbeth(world.returns, betas, world.factors)
    rows = {}
    for f in world.factors:
        est = float(fm.loc[f, "premium_ann"])
        se = float(fm.loc[f, "se_nw_ann"])
        rows[f] = {
            "estimate": round(est, 4),
            "se_nw": round(se, 4),
            "true": world.premia[f],
            "truth_in_95ci": bool(abs(est - world.premia[f]) <= 1.96 * se),
        }
    return {
        "factors": rows,
        "note": ("A single three-year world is deliberately noisy. The "
                 "robustness study evaluates repeated-sample error and "
                 "coverage rather than declaring success from one seed."),
    }


def main():
    out = evaluate(FactorWorld(five_factor=True, seed=2))
    (ROOT / "reports" / "five_factor.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
