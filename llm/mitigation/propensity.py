"""Propensity models for counterfactual preference-data debiasing.

Estimates ψ(x, y) ≈ P(m = 1 | x, y) — the probability that a given
preference pair would enter the observed training set.

Variants:
- OraclePropensityModel: uses known obs_probs (oracle / cheating upper bound).
- ClassifierPropensityModel: logistic regression on observable features.

Propensities are clipped to [clip_eps, 1-clip_eps] to prevent blow-up under
inverse weighting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from llm.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_CLIP_EPS = 0.05


class OraclePropensityModel:
    """Oracle propensity: uses known obs_probs from data generation config.

    Cheating upper bound — use for ablation only. The main result uses
    ClassifierPropensityModel.
    """

    def __init__(self, obs_probs: dict[str, float], clip_eps: float = _DEFAULT_CLIP_EPS) -> None:
        self.obs_probs = obs_probs
        self.clip_eps = clip_eps

    def fit(self, train_df: pd.DataFrame, audit_df: pd.DataFrame | None = None) -> None:
        pass  # oracle requires no fitting

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        probs = df["demographic_signal"].map(self.obs_probs).fillna(0.5).to_numpy(dtype=float)
        return np.clip(probs, self.clip_eps, 1.0 - self.clip_eps)


class ClassifierPropensityModel:
    """Classifier-based propensity: logistic regression on observable features.

    Features: demographic_signal (binary), seniority (binary), domain (one-hot),
    quality_score, chosen response length.

    Target: whether the pair appears in the training set (observation indicator).
    Requires audit_df to construct negative examples (unobserved pairs). Pairs
    in audit_df whose prompt also appears in train_df are labelled observed=1;
    the remainder are observed=0.
    """

    def __init__(self, clip_eps: float = _DEFAULT_CLIP_EPS) -> None:
        self.clip_eps = clip_eps
        self._clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        self._constant_prob: float | None = None

    def _featurize(self, df: pd.DataFrame) -> np.ndarray:
        cols: list[np.ndarray] = []

        # Demographic signal (A = 1, B = 0)
        cols.append((df["demographic_signal"] == "A").to_numpy(dtype=float).reshape(-1, 1))

        # Seniority (senior = 1, junior = 0)
        if "seniority" in df.columns:
            cols.append((df["seniority"] == "senior").to_numpy(dtype=float).reshape(-1, 1))
        else:
            cols.append(np.zeros((len(df), 1)))

        # Domain one-hot
        for domain in ["frontend", "backend", "ml"]:
            if "domain" in df.columns:
                cols.append((df["domain"] == domain).to_numpy(dtype=float).reshape(-1, 1))
            else:
                cols.append(np.zeros((len(df), 1)))

        # Quality score
        if "quality_score" in df.columns:
            cols.append(df["quality_score"].fillna(0.5).to_numpy(dtype=float).reshape(-1, 1))
        else:
            cols.append(np.full((len(df), 1), 0.5))

        # Chosen response length (normalised)
        if "chosen" in df.columns:
            lens = df["chosen"].str.len().fillna(0).to_numpy(dtype=float)
            cols.append((lens / 1000.0).reshape(-1, 1))
        else:
            cols.append(np.zeros((len(df), 1)))

        return np.hstack(cols)

    def fit(self, train_df: pd.DataFrame, audit_df: pd.DataFrame | None = None) -> None:
        if audit_df is None:
            log.warning(
                "ClassifierPropensityModel.fit called without audit_df; "
                "all train samples treated as observed. Model will have low discriminability."
            )
            X = self._featurize(train_df)
            y = np.ones(len(train_df), dtype=int)
        else:
            # Label each audit row: 1 if its prompt also appears in train, 0 otherwise.
            train_prompts = set(train_df["prompt"].tolist())
            is_obs = audit_df["prompt"].isin(train_prompts).to_numpy(dtype=int)
            X = self._featurize(audit_df)
            y = is_obs

        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        log.info("Fitting classifier propensity", n_observed=n_pos, n_unobserved=n_neg)
        if len(np.unique(y)) < 2:
            # Only one class present — store a constant fallback and skip fitting.
            self._constant_prob: float | None = float(y.mean()) if len(y) > 0 else 0.5
            return
        self._constant_prob = None
        self._clf.fit(X, y)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if getattr(self, "_constant_prob", None) is not None:
            probs = np.full(len(df), self._constant_prob)
        else:
            X = self._featurize(df)
            probs = self._clf.predict_proba(X)[:, 1]
        return np.clip(probs, self.clip_eps, 1.0 - self.clip_eps)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

PropensityModel = OraclePropensityModel | ClassifierPropensityModel


def compute_ipw_weights(
    df: pd.DataFrame,
    model: OraclePropensityModel | ClassifierPropensityModel,
) -> np.ndarray:
    """Return IPW weights 1 / ψ(x, y) for each row in df."""
    propensities = model.predict_proba(df)
    return 1.0 / propensities


def fit_and_weight(
    train_df: pd.DataFrame,
    mitigation_cfg: dict,
    audit_df: pd.DataFrame | None = None,
) -> list[float]:
    """Build and fit a propensity model; return IPW weights for train_df.

    Args:
        train_df: biased training set (each row is an observed preference pair).
        mitigation_cfg: dict with keys ``propensity_variant``, ``obs_probs``,
            and optionally ``clip_eps``.
        audit_df: full audit dataset (needed for classifier variant).

    Returns:
        List of float weights aligned with train_df rows.
    """
    variant: str = str(mitigation_cfg.get("propensity_variant", "oracle"))
    obs_probs: dict[str, float] = dict(mitigation_cfg.get("obs_probs", {"A": 0.8, "B": 0.4}))
    clip_eps: float = float(mitigation_cfg.get("clip_eps", _DEFAULT_CLIP_EPS))

    if variant == "oracle":
        model: OraclePropensityModel | ClassifierPropensityModel = OraclePropensityModel(
            obs_probs=obs_probs, clip_eps=clip_eps
        )
        model.fit(train_df)
    elif variant == "classifier":
        model = ClassifierPropensityModel(clip_eps=clip_eps)
        model.fit(train_df, audit_df)
    else:
        raise ValueError(
            f"Unknown propensity variant: {variant!r}. Choose 'oracle' or 'classifier'."
        )

    weights = compute_ipw_weights(train_df, model)
    log.info(
        "IPW weights computed",
        variant=variant,
        n=len(weights),
        mean_weight=f"{weights.mean():.3f}",
        max_weight=f"{weights.max():.3f}",
    )
    return weights.tolist()
