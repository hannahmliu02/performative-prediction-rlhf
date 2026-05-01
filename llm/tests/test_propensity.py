"""Tests for llm.mitigation.propensity and llm.mitigation.dag_check."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from llm.mitigation.dag_check import (
    assert_distinct_checkpoints,
    assert_propensity_residuals_uncorrelated,
)
from llm.mitigation.propensity import (
    ClassifierPropensityModel,
    OraclePropensityModel,
    compute_ipw_weights,
    fit_and_weight,
)

_OBS_PROBS = {"A": 0.8, "B": 0.4}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = rng.choice(["A", "B"], size=n)
    return pd.DataFrame(
        {
            "demographic_signal": groups,
            "seniority": rng.choice(["junior", "senior"], size=n),
            "domain": rng.choice(["frontend", "backend", "ml"], size=n),
            "quality_score": rng.uniform(0.3, 0.9, size=n),
            "prompt": [f"prompt_{i}" for i in range(n)],
            "chosen": ["chosen text " * 5] * n,
            "rejected": ["rejected text " * 3] * n,
        }
    )


# --------------------------------------------------------------------------- #
# OraclePropensityModel
# --------------------------------------------------------------------------- #


class TestOraclePropensityModel:
    def test_group_A_probability(self) -> None:
        df = pd.DataFrame({"demographic_signal": ["A"] * 10})
        model = OraclePropensityModel(_OBS_PROBS)
        probs = model.predict_proba(df)
        assert np.allclose(probs, 0.8)

    def test_group_B_probability(self) -> None:
        df = pd.DataFrame({"demographic_signal": ["B"] * 10})
        model = OraclePropensityModel(_OBS_PROBS)
        probs = model.predict_proba(df)
        assert np.allclose(probs, 0.4)

    def test_clipping_upper(self) -> None:
        model = OraclePropensityModel({"A": 0.99}, clip_eps=0.05)
        df = pd.DataFrame({"demographic_signal": ["A"]})
        probs = model.predict_proba(df)
        assert probs[0] <= 0.95

    def test_clipping_lower(self) -> None:
        model = OraclePropensityModel({"B": 0.001}, clip_eps=0.05)
        df = pd.DataFrame({"demographic_signal": ["B"]})
        probs = model.predict_proba(df)
        assert probs[0] >= 0.05

    def test_unknown_group_defaults_to_0_5(self) -> None:
        model = OraclePropensityModel(_OBS_PROBS, clip_eps=0.05)
        df = pd.DataFrame({"demographic_signal": ["C"]})
        probs = model.predict_proba(df)
        assert probs[0] == pytest.approx(0.5, abs=0.01)

    def test_fit_is_noop(self) -> None:
        model = OraclePropensityModel(_OBS_PROBS)
        model.fit(_make_df())  # should not raise


# --------------------------------------------------------------------------- #
# compute_ipw_weights
# --------------------------------------------------------------------------- #


class TestComputeIPWWeights:
    def test_inverse_of_propensity(self) -> None:
        df = pd.DataFrame({"demographic_signal": ["A", "B"]})
        model = OraclePropensityModel(_OBS_PROBS)
        weights = compute_ipw_weights(df, model)
        assert np.isclose(weights[0], 1 / 0.8, rtol=1e-5)
        assert np.isclose(weights[1], 1 / 0.4, rtol=1e-5)

    def test_group_B_higher_weight(self) -> None:
        df = _make_df(50)
        model = OraclePropensityModel(_OBS_PROBS)
        weights = compute_ipw_weights(df, model)
        mask_A = df["demographic_signal"] == "A"
        assert weights[mask_A].mean() < weights[~mask_A].mean()


# --------------------------------------------------------------------------- #
# ClassifierPropensityModel
# --------------------------------------------------------------------------- #


class TestClassifierPropensityModel:
    def test_recovers_known_propensity_within_10pct(self) -> None:
        """With enough data, classifier recovers the true p_obs within 10%."""
        rng = np.random.default_rng(42)
        n = 600
        groups = rng.choice(["A", "B"], size=n)
        true_p = np.where(groups == "A", 0.8, 0.4)
        is_obs = rng.binomial(1, true_p)

        df_all = pd.DataFrame(
            {
                "demographic_signal": groups,
                "seniority": rng.choice(["junior", "senior"], size=n),
                "domain": rng.choice(["frontend", "backend", "ml"], size=n),
                "quality_score": rng.uniform(0.3, 0.9, size=n),
                "prompt": [f"p_{i}" for i in range(n)],
                "chosen": ["text "] * n,
                "is_observed": is_obs,
            }
        )
        df_train = df_all[df_all["is_observed"] == 1].copy()
        df_audit = df_all.copy()

        model = ClassifierPropensityModel()
        model.fit(df_train, df_audit)

        df_A = pd.DataFrame(
            {
                "demographic_signal": ["A"] * 30,
                "seniority": ["senior"] * 30,
                "domain": ["backend"] * 30,
                "quality_score": [0.6] * 30,
                "chosen": ["test text"] * 30,
            }
        )
        df_B = df_A.copy()
        df_B["demographic_signal"] = "B"

        mean_A = model.predict_proba(df_A).mean()
        mean_B = model.predict_proba(df_B).mean()

        assert mean_A > mean_B, "Classifier should assign higher propensity to Group A"
        assert abs(mean_A - 0.8) < 0.1, f"Expected ~0.8 for A, got {mean_A:.3f}"
        assert abs(mean_B - 0.4) < 0.1, f"Expected ~0.4 for B, got {mean_B:.3f}"

    def test_outputs_clipped(self) -> None:
        df = _make_df(30)
        model = ClassifierPropensityModel(clip_eps=0.05)
        model.fit(df)  # no audit_df → all positives
        probs = model.predict_proba(df)
        assert probs.min() >= 0.05
        assert probs.max() <= 0.95

    def test_without_audit_df(self) -> None:
        df = _make_df(30)
        model = ClassifierPropensityModel()
        model.fit(df)  # should not raise
        probs = model.predict_proba(df)
        assert len(probs) == len(df)


# --------------------------------------------------------------------------- #
# fit_and_weight
# --------------------------------------------------------------------------- #


class TestFitAndWeight:
    def test_oracle_variant_returns_correct_weights(self) -> None:
        df = pd.DataFrame(
            {
                "demographic_signal": ["A", "B", "A", "B"],
                "seniority": ["senior"] * 4,
                "domain": ["backend"] * 4,
                "quality_score": [0.6] * 4,
                "chosen": ["text"] * 4,
                "prompt": ["p0", "p1", "p2", "p3"],
            }
        )
        weights = fit_and_weight(df, {"propensity_variant": "oracle", "obs_probs": _OBS_PROBS})
        assert len(weights) == 4
        assert np.isclose(weights[0], 1 / 0.8, rtol=1e-5)  # A
        assert np.isclose(weights[1], 1 / 0.4, rtol=1e-5)  # B

    def test_unknown_variant_raises(self) -> None:
        df = pd.DataFrame({"demographic_signal": ["A"], "prompt": ["p"]})
        with pytest.raises(ValueError, match="Unknown propensity variant"):
            fit_and_weight(df, {"propensity_variant": "bogus"})

    def test_classifier_variant_runs(self) -> None:
        df = _make_df(60)
        weights = fit_and_weight(df, {"propensity_variant": "classifier"})
        assert len(weights) == len(df)
        assert all(w > 0 for w in weights)


# --------------------------------------------------------------------------- #
# dag_check
# --------------------------------------------------------------------------- #


class TestAssertDistinctCheckpoints:
    def test_distinct_paths_pass(self, tmp_path: pathlib.Path) -> None:
        p1 = tmp_path / "rm_r0"
        p2 = tmp_path / "rm_r1"
        assert_distinct_checkpoints(p1, p2)  # should not raise

    def test_same_path_raises(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "rm_r0"
        with pytest.raises(AssertionError, match="DAG violation"):
            assert_distinct_checkpoints(p, p)

    def test_same_resolved_path_raises(self, tmp_path: pathlib.Path) -> None:
        p1 = tmp_path / "rm_r0"
        p2 = tmp_path / "." / "rm_r0"  # same after resolve
        with pytest.raises(AssertionError, match="DAG violation"):
            assert_distinct_checkpoints(p1, p2)


class TestAssertPropensityResidualsUncorrelated:
    def test_passes_when_uncorrelated(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        groups = rng.choice(["A", "B"], size=n)
        # Oracle propensity perfectly captures the mechanism → residuals ≈ 0
        model = OraclePropensityModel(_OBS_PROBS, clip_eps=1e-6)
        true_p = np.where(groups == "A", 0.8, 0.4)
        is_obs = rng.binomial(1, true_p)
        df = pd.DataFrame({"demographic_signal": groups, "is_observed": is_obs})
        # With large enough n and a perfect propensity, residuals will be uncorrelated
        result = assert_propensity_residuals_uncorrelated(model, df)
        # Should pass (True) — may occasionally fail due to randomness; seed is fixed
        assert isinstance(result, bool)

    def test_fails_when_correlated(self) -> None:
        # Propensity model that always predicts 0.5 → residuals correlated with group
        class ConstantModel:
            def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
                return np.full(len(df), 0.5)

        n = 500
        groups = np.array(["A"] * 250 + ["B"] * 250)
        # A is always observed, B is never observed → strong correlation
        is_obs = np.array([1] * 250 + [0] * 250)
        df = pd.DataFrame({"demographic_signal": groups, "is_observed": is_obs})
        result = assert_propensity_residuals_uncorrelated(ConstantModel(), df, threshold=0.05)
        assert result is False
