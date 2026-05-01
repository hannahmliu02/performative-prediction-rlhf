"""Tests for Task 3: semi-synthetic resume preference data generation."""

from __future__ import annotations

import random

import pytest
from omegaconf import OmegaConf
from scipy import stats

from llm.data.generate_resume_prefs import generate_pairs
from llm.data.quality_score import compute_quality
from llm.data.resume_templates import build_resume, enumerate_cells

# Small config for fast tests
_TEST_CFG = OmegaConf.create(
    {
        "n_per_cell": 30,
        "roles": ["software engineer"],
        "axes": {
            "demographic_signal": ["A", "B"],
            "seniority": ["junior", "senior"],
            "domain": ["frontend", "backend", "ml"],
        },
        "noise_epsilon": 0.05,
        "obs_probs": {
            "demographic_signal": {"A": 0.8, "B": 0.4},
        },
        "output_dir": "/tmp/test_semi_synthetic",
        "seed": 42,
    }
)


@pytest.fixture(scope="module")
def generated_data():
    return generate_pairs(_TEST_CFG)


# --------------------------------------------------------------------------- #
# Observation rate tests
# --------------------------------------------------------------------------- #


def test_train_obs_rates_match_config(generated_data):
    """Marginal observation rates in train must be within ±0.15 of p_obs."""
    train_rows, audit_rows = generated_data

    total_A = sum(1 for r in audit_rows if r["demographic_signal"] == "A")
    total_B = sum(1 for r in audit_rows if r["demographic_signal"] == "B")
    obs_A = sum(1 for r in train_rows if r["demographic_signal"] == "A")
    obs_B = sum(1 for r in train_rows if r["demographic_signal"] == "B")

    rate_A = obs_A / total_A
    rate_B = obs_B / total_B

    tol = 0.15  # generous for N=30 per cell
    assert abs(rate_A - 0.8) < tol, f"Group A obs rate {rate_A:.3f} too far from 0.8"
    assert abs(rate_B - 0.4) < tol, f"Group B obs rate {rate_B:.3f} too far from 0.4"


def test_train_underrepresents_group_B(generated_data):
    """Group B must appear strictly less often than group A in train (by count)."""
    train_rows, _ = generated_data
    count_A = sum(1 for r in train_rows if r["demographic_signal"] == "A")
    count_B = sum(1 for r in train_rows if r["demographic_signal"] == "B")
    assert count_A > count_B


def test_audit_is_balanced(generated_data):
    """Audit dataset should contain equal numbers of rows per demographic group."""
    _, audit_rows = generated_data
    count_A = sum(1 for r in audit_rows if r["demographic_signal"] == "A")
    count_B = sum(1 for r in audit_rows if r["demographic_signal"] == "B")
    assert count_A == count_B


# --------------------------------------------------------------------------- #
# Quality-score independence tests
# --------------------------------------------------------------------------- #


def test_quality_independent_of_demographic(generated_data):
    """q must not differ significantly between demographic groups A and B (t-test)."""
    _, audit_rows = generated_data
    q_A = [r["quality_score"] for r in audit_rows if r["demographic_signal"] == "A"]
    q_B = [r["quality_score"] for r in audit_rows if r["demographic_signal"] == "B"]

    _, p_value = stats.ttest_ind(q_A, q_B, equal_var=False)
    # With equal sampling distributions the means should be indistinguishable
    assert p_value > 0.05, f"q differs by demographic (p={p_value:.4f}); independence violated"


def test_quality_score_range():
    """compute_quality must always return a value in [0, 1]."""
    rng = random.Random(0)
    axes = {"demographic_signal": ["A", "B"], "seniority": ["junior", "senior"], "domain": ["frontend", "backend", "ml"]}
    for cell in enumerate_cells(axes):
        for _ in range(10):
            resume = build_resume(
                demographic_signal=cell["demographic_signal"],
                seniority=cell["seniority"],
                domain=cell["domain"],
                rng=rng,
            )
            q = compute_quality(resume)
            assert 0.0 <= q <= 1.0, f"q={q} out of range for {cell}"


def test_senior_higher_quality_than_junior():
    """On average, senior resumes should score higher than junior resumes."""
    rng = random.Random(1)
    q_senior, q_junior = [], []
    for _ in range(100):
        for seniority, bucket in [("senior", q_senior), ("junior", q_junior)]:
            resume = build_resume("A", seniority, "backend", rng=rng)
            bucket.append(compute_quality(resume))
    assert sum(q_senior) / len(q_senior) > sum(q_junior) / len(q_junior)


# --------------------------------------------------------------------------- #
# Dataset structure tests
# --------------------------------------------------------------------------- #


def test_dataset_has_required_columns(generated_data):
    train_rows, audit_rows = generated_data
    required = {"prompt", "chosen", "rejected", "demographic_signal", "seniority", "domain", "quality_score"}
    for row in train_rows[:5]:
        assert required.issubset(row.keys())
    for row in audit_rows[:5]:
        assert required.issubset(row.keys())


def test_noise_epsilon_flips_some_labels(generated_data):
    """With ε=0.05 and N=360, expect some flipped labels (chosen_is_good=False)."""
    _, audit_rows = generated_data
    n_flipped = sum(1 for r in audit_rows if not r["chosen_is_good"])
    assert n_flipped > 0, "No preference flips observed; epsilon noise may not be applied"


def test_reproducibility():
    """Same seed must produce identical output."""
    rows1, _ = generate_pairs(_TEST_CFG)
    rows2, _ = generate_pairs(_TEST_CFG)
    assert rows1 == rows2
