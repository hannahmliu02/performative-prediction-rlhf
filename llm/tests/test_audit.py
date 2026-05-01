"""Unit tests for llm/audit/observability_audit.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = ["A", "B"] * (n // 2)
    return pd.DataFrame(
        {
            "prompt": [f"Candidate prompt {i}" for i in range(n)],
            "chosen": [f"Response word word word {i}" for i in range(n)],
            "demographic_signal": groups,
            "seniority": rng.choice(["junior", "senior"], size=n),
            "domain": rng.choice(["tech", "finance"], size=n),
            "quality_score": rng.uniform(0.3, 0.9, size=n),
        }
    )


def _make_margins_and_labels(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    margins = rng.normal(0.0, 1.0, size=n)
    demo_b = np.array([0, 1] * (n // 2), dtype=np.float64)
    chosen_is_good = (margins > 0).astype(int)
    return margins, demo_b, chosen_is_good


# --------------------------------------------------------------------------- #
# _mask_prompt
# --------------------------------------------------------------------------- #


def test_mask_prompt_k0_unchanged() -> None:
    from llm.audit.observability_audit import _mask_prompt

    prompt = "Name: Alice\nExperience: 5 years\nDomain: tech"
    assert _mask_prompt(prompt, 0) == prompt


def test_mask_prompt_k1_masks_name() -> None:
    from llm.audit.observability_audit import _mask_prompt

    prompt = "Name: Alice\nExperience: 5 years"
    result = _mask_prompt(prompt, 1)
    assert "Alice" not in result
    assert "Experience: 5 years" in result


def test_mask_prompt_k6_replaces_block() -> None:
    from llm.audit.observability_audit import _mask_prompt

    prompt = "preamble\nName: Alice\nExperience: 5 years"
    result = _mask_prompt(prompt, 6)
    assert "[candidate details masked]" in result
    assert "Alice" not in result


# --------------------------------------------------------------------------- #
# run_linear_probe
# --------------------------------------------------------------------------- #


def test_run_linear_probe_returns_keys() -> None:
    from llm.audit.observability_audit import run_linear_probe

    margins, demo_b, chosen_is_good = _make_margins_and_labels()
    result = run_linear_probe(margins, demo_b, chosen_is_good)

    assert "coefficients" in result
    assert "t_stats" in result
    assert "p_values" in result
    assert "r_squared" in result
    assert "n_obs" in result
    assert result["n_obs"] == 40


def test_run_linear_probe_r2_bounded() -> None:
    from llm.audit.observability_audit import run_linear_probe

    margins, demo_b, chosen_is_good = _make_margins_and_labels()
    result = run_linear_probe(margins, demo_b, chosen_is_good)
    assert 0.0 <= result["r_squared"] <= 1.0


# --------------------------------------------------------------------------- #
# run_linear_probe_with_content
# --------------------------------------------------------------------------- #


def test_linear_probe_with_content_returns_keys() -> None:
    from llm.audit.observability_audit import run_linear_probe_with_content

    n = 40
    df = _make_df(n)
    margins, demo_b, chosen_is_good = _make_margins_and_labels(n)
    result = run_linear_probe_with_content(df, margins, demo_b, chosen_is_good)

    assert "coefficients" in result
    assert "t_stats" in result
    assert "p_values" in result
    assert "r_squared" in result
    assert "n_obs" in result
    assert "coverage_indicator_significant" in result
    # All six features present in coefficients
    assert "demo_B_indicator" in result["coefficients"]
    assert "response_length" in result["coefficients"]
    assert "prompt_length" in result["coefficients"]
    assert "quality_score" in result["coefficients"]


def test_linear_probe_with_content_r2_bounded() -> None:
    from llm.audit.observability_audit import run_linear_probe_with_content

    n = 40
    df = _make_df(n)
    margins, demo_b, chosen_is_good = _make_margins_and_labels(n)
    result = run_linear_probe_with_content(df, margins, demo_b, chosen_is_good)
    assert 0.0 <= result["r_squared"] <= 1.0


def test_linear_probe_with_content_no_quality_col() -> None:
    """Falls back gracefully when quality_score column is absent."""
    from llm.audit.observability_audit import run_linear_probe_with_content

    n = 40
    df = _make_df(n).drop(columns=["quality_score"])
    margins, demo_b, chosen_is_good = _make_margins_and_labels(n)
    result = run_linear_probe_with_content(df, margins, demo_b, chosen_is_good)
    # quality_score column absent → all-zero quality; probe should still run
    assert "coverage_indicator_significant" in result
    assert isinstance(result["coverage_indicator_significant"], bool)


def test_linear_probe_with_content_significance_flag() -> None:
    """coverage_indicator_significant is True iff p<0.05 and coeff<0."""
    from llm.audit.observability_audit import run_linear_probe_with_content

    n = 40
    df = _make_df(n)
    margins, demo_b, chosen_is_good = _make_margins_and_labels(n)
    result = run_linear_probe_with_content(df, margins, demo_b, chosen_is_good)

    p = result["p_values"]["demo_B_indicator"]
    coeff = result["coefficients"]["demo_B_indicator"]
    expected_sig = bool(p < 0.05 and coeff < 0)
    assert result["coverage_indicator_significant"] == expected_sig


# --------------------------------------------------------------------------- #
# run_mlp_probe
# --------------------------------------------------------------------------- #


def test_mlp_probe_returns_keys() -> None:
    from llm.audit.observability_audit import run_mlp_probe

    n = 60
    df = _make_df(n)
    margins, demo_b, chosen_is_good = _make_margins_and_labels(n)
    result = run_mlp_probe(df, margins, demo_b, chosen_is_good, n_background=10, seed=42)

    assert "mlp_accuracy" in result
    assert "shap_importance" in result
    assert "coverage_indicator_rank" in result


def test_mlp_probe_accuracy_bounded() -> None:
    from llm.audit.observability_audit import run_mlp_probe

    n = 60
    df = _make_df(n)
    margins, demo_b, chosen_is_good = _make_margins_and_labels(n)
    result = run_mlp_probe(df, margins, demo_b, chosen_is_good, n_background=10, seed=42)

    assert 0.0 <= result["mlp_accuracy"] <= 1.0


def test_mlp_probe_coverage_rank_valid() -> None:
    """coverage_indicator_rank is a 1-based integer within [1, n_features]."""
    from llm.audit.observability_audit import run_mlp_probe

    n = 60
    df = _make_df(n)
    margins, demo_b, chosen_is_good = _make_margins_and_labels(n)
    result = run_mlp_probe(df, margins, demo_b, chosen_is_good, n_background=10, seed=42)

    rank = result["coverage_indicator_rank"]
    n_features = 5  # rm_margin, demo_B_indicator, response_length, prompt_length, quality_score
    assert rank is not None
    assert 1 <= rank <= n_features


def test_mlp_probe_shap_feature_names() -> None:
    """shap_importance keys match the expected feature names."""
    from llm.audit.observability_audit import run_mlp_probe

    n = 60
    df = _make_df(n)
    margins, demo_b, chosen_is_good = _make_margins_and_labels(n)
    result = run_mlp_probe(df, margins, demo_b, chosen_is_good, n_background=10, seed=42)

    expected = {"rm_margin", "demo_B_indicator", "response_length", "prompt_length", "quality_score"}
    assert set(result["shap_importance"].keys()) == expected
