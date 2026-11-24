"""Data-driven factor route: scree diagnostics, Horn parallel analysis,
maximum-likelihood factor analysis, and rotation-invariant subspace checks."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis


def _standardise(returns: pd.DataFrame) -> np.ndarray:
    z = (returns - returns.mean()) / returns.std(ddof=1)
    if z.isna().any().any():
        raise ValueError("returns contain constant or missing columns")
    return z.to_numpy()


def scree(returns: pd.DataFrame) -> np.ndarray:
    return PCA().fit(_standardise(returns)).explained_variance_


def kaiser_count(eigenvalues: np.ndarray) -> int:
    return int((np.asarray(eigenvalues) > 1.0).sum())


def parallel_analysis(returns: pd.DataFrame, n_iter: int = 200,
                      quantile: float = 0.95, seed: int = 123) -> dict:
    """Horn-style parallel analysis against iid Gaussian noise eigenvalues."""
    if n_iter < 20:
        raise ValueError("n_iter must be at least 20 for a stable diagnostic")
    z = _standardise(returns)
    observed = np.linalg.eigvalsh(np.cov(z, rowvar=False))[::-1]
    rng = np.random.default_rng(seed)
    null = np.empty((n_iter, z.shape[1]))
    for i in range(n_iter):
        x = rng.normal(size=z.shape)
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        null[i] = np.linalg.eigvalsh(np.cov(x, rowvar=False))[::-1]
    threshold = np.quantile(null, quantile, axis=0)
    count = int(np.sum(observed > threshold))
    return {"observed": observed, "threshold": threshold, "count": count,
            "quantile": quantile, "n_iter": n_iter}


def fit_factors(returns: pd.DataFrame, n_factors: int):
    z = _standardise(returns)
    fa = FactorAnalysis(n_components=n_factors, rotation="varimax",
                        random_state=0)
    scores = fa.fit_transform(z)
    loadings = pd.DataFrame(
        fa.components_.T,
        index=returns.columns,
        columns=[f"F{i+1}" for i in range(n_factors)])
    communalities = (loadings ** 2).sum(axis=1)
    return loadings, pd.DataFrame(
        scores, index=returns.index, columns=loadings.columns), communalities


def match_to_truth(scores: pd.DataFrame,
                   true_factors: pd.DataFrame) -> pd.Series:
    """Multiple correlation of each true factor with the whole latent space."""
    X = np.column_stack([np.ones(len(scores)), scores.to_numpy()])
    out = {}
    for tru in true_factors.columns:
        y = true_factors[tru].to_numpy()
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        out[tru] = float(np.sqrt(max(0.0, 1 - resid.var() / y.var())))
    return pd.Series(out)


def canonical_correlations(scores: pd.DataFrame,
                           true_factors: pd.DataFrame) -> np.ndarray:
    """Canonical correlations between two centred subspaces.

    These singular values are invariant to rotations/sign changes of either
    coordinate system, so they are a cleaner 'did I recover the factor space?'
    diagnostic than matching latent factor 1 to true factor 1.
    """
    # explicit copies: pandas can hand back read-only views, and I centre
    # in place below
    a = np.array(scores.to_numpy(dtype=float), copy=True)
    b = np.array(true_factors.loc[scores.index].to_numpy(dtype=float),
                 copy=True)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    return np.linalg.svd(qa.T @ qb, compute_uv=False)
