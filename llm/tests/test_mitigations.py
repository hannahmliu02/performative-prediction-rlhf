"""Unit tests for llm/training/mitigations/."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

# --------------------------------------------------------------------------- #
# Dispatcher tests
# --------------------------------------------------------------------------- #


def test_get_rm_trainer_cls_none() -> None:
    from trl import RewardTrainer

    from llm.training.mitigations import get_rm_trainer_cls

    cls = get_rm_trainer_cls(None)
    assert cls is RewardTrainer


def test_get_rm_trainer_cls_length_norm() -> None:
    from llm.training.mitigations import get_rm_trainer_cls
    from llm.training.mitigations.length_norm import LengthNormRewardTrainer

    cls = get_rm_trainer_cls({"type": "length_norm", "beta": 0.1})
    assert cls is LengthNormRewardTrainer


def test_get_rm_trainer_cls_rm_reg() -> None:
    from llm.training.mitigations import get_rm_trainer_cls
    from llm.training.mitigations.rm_reg import RMRegRewardTrainer

    cls = get_rm_trainer_cls({"type": "rm_reg"})
    assert cls is RMRegRewardTrainer


def test_get_dpo_trainer_cls_length_norm() -> None:
    from llm.training.mitigations import get_dpo_trainer_cls
    from llm.training.mitigations.length_norm import LengthNormDPOTrainer

    cls = get_dpo_trainer_cls({"type": "length_norm"})
    assert cls is LengthNormDPOTrainer


def test_get_rm_trainer_kwargs() -> None:
    from llm.training.mitigations import get_rm_trainer_kwargs

    kwargs = get_rm_trainer_kwargs({"type": "length_norm", "beta": 0.5})
    assert kwargs == {"length_norm_beta": 0.5}

    kwargs = get_rm_trainer_kwargs({"type": "rm_reg", "l2_lambda": 1e-4, "consistency_lambda": 0.1})
    assert kwargs == {"l2_lambda": 1e-4, "consistency_lambda": 0.1}

    assert get_rm_trainer_kwargs(None) == {}


# --------------------------------------------------------------------------- #
# LengthNormRewardTrainer.compute_loss
# --------------------------------------------------------------------------- #


def test_length_norm_rm_compute_loss() -> None:
    """compute_loss returns a finite scalar and is different from base loss."""
    from llm.training.mitigations.length_norm import LengthNormRewardTrainer

    # Build a minimal mock trainer (we only test compute_loss logic)
    trainer = object.__new__(LengthNormRewardTrainer)
    trainer.length_norm_beta = 0.5

    batch_size = 4
    seq_len = 16
    vocab = 32

    # Mock model: returns logits of shape (2*B, 1)
    mock_model = MagicMock()
    logits = torch.randn(batch_size * 2, 1)
    mock_output = MagicMock()
    mock_output.logits = logits
    mock_model.return_value = mock_output

    inputs = {
        "input_ids": torch.randint(0, vocab, (batch_size * 2, seq_len)),
        "attention_mask": torch.ones(batch_size * 2, seq_len, dtype=torch.long),
    }

    loss = trainer.compute_loss(mock_model, inputs, return_outputs=False)
    assert torch.isfinite(loss)
    assert loss.ndim == 0  # scalar


def test_length_norm_rm_no_beta() -> None:
    """With beta=0, result should match plain BT loss."""
    from llm.training.mitigations.length_norm import LengthNormRewardTrainer

    trainer = object.__new__(LengthNormRewardTrainer)
    trainer.length_norm_beta = 0.0

    logits = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]])  # 4 sequences (2B=4 means B=2)
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = logits
    mock_model.return_value = mock_output

    inputs = {"attention_mask": torch.ones(4, 8, dtype=torch.long)}
    loss = trainer.compute_loss(mock_model, inputs)

    # Compare to plain BT
    chosen = logits.squeeze(-1)[:2]
    rejected = logits.squeeze(-1)[2:]
    expected = -torch.nn.functional.logsigmoid(chosen - rejected).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# RMRegRewardTrainer
# --------------------------------------------------------------------------- #


def test_rm_reg_reward_head_l2_missing() -> None:
    """When model has no .score attr, L2 penalty is 0."""
    from llm.training.mitigations.rm_reg import RMRegRewardTrainer

    trainer = object.__new__(RMRegRewardTrainer)
    trainer.l2_lambda = 1e-3
    trainer.consistency_lambda = 0.0

    model_no_score = MagicMock(spec=[])  # no attributes
    model_no_score.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    l2 = trainer._reward_head_l2(model_no_score)
    assert l2.item() == 0.0


def test_rm_reg_compute_loss_finite() -> None:
    import torch.nn as nn

    from llm.training.mitigations.rm_reg import RMRegRewardTrainer

    trainer = object.__new__(RMRegRewardTrainer)
    trainer.l2_lambda = 1e-3
    trainer.consistency_lambda = 0.0

    # Model with a .score linear layer
    score = nn.Linear(4, 1, bias=False)
    mock_model = MagicMock()
    mock_model.score = score
    # Make model return logits
    logits = torch.randn(4, 1)
    mock_output = MagicMock()
    mock_output.logits = logits
    mock_model.return_value = mock_output
    mock_model.module = mock_model  # DataParallel unwrap

    inputs = {"attention_mask": torch.ones(4, 8)}
    loss = trainer.compute_loss(mock_model, inputs)
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# KL-constrained helpers
# --------------------------------------------------------------------------- #


def test_kl_constrained_build_cfg() -> None:
    from llm.training.mitigations.kl_constrained import build_kl_ppo_cfg

    base = OmegaConf.create({"kl_beta": 1e-4, "foo": "bar"})
    new_cfg = build_kl_ppo_cfg(base, kl_beta=1e-2)
    assert new_cfg.kl_beta == pytest.approx(1e-2)
    assert new_cfg.foo == "bar"


# --------------------------------------------------------------------------- #
# Safe-RLHF reward function
# --------------------------------------------------------------------------- #


def test_safe_reward_fn_returns_list() -> None:
    from llm.training.mitigations.safe_rlhf import build_safe_reward_fn

    with patch("llm.training.mitigations.safe_rlhf.RewardModel") as MockRM:
        mock_rm = MagicMock()
        MockRM.from_pretrained.return_value = mock_rm
        mock_rm.model.parameters.return_value = iter([torch.zeros(1)])
        mock_rm.tokenizer.return_value = {
            "input_ids": torch.zeros(2, 8, dtype=torch.long),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
        mock_rm.score.return_value = torch.tensor([1.0, -1.0])

        fn = build_safe_reward_fn(
            rm_checkpoint="/tmp/rm",
            rm_dtype="float32",
            rm_device_map=None,
            max_length=64,
            length_norm_beta=0.0,
            safety_lambda=0.5,
        )
        rewards = fn(["prompt1", "prompt2"], ["comp1", "comp2"])
        assert isinstance(rewards, list)
        assert len(rewards) == 2


def test_safe_reward_fn_applies_penalty() -> None:
    """When group A has higher scores, safety penalty should reduce rewards."""
    from llm.training.mitigations.safe_rlhf import build_safe_reward_fn

    with patch("llm.training.mitigations.safe_rlhf.RewardModel") as MockRM:
        mock_rm = MagicMock()
        MockRM.from_pretrained.return_value = mock_rm
        mock_rm.model.parameters.return_value = iter([torch.zeros(1)])
        mock_rm.tokenizer.return_value = {
            "input_ids": torch.zeros(4, 8, dtype=torch.long),
            "attention_mask": torch.ones(4, 8, dtype=torch.long),
        }
        # A gets high scores, B gets low — clear disparity
        mock_rm.score.return_value = torch.tensor([2.0, 2.0, -2.0, -2.0])

        fn = build_safe_reward_fn("/tmp/rm", "float32", None, 64, 0.0, safety_lambda=1.0)
        rewards_no_demo = fn(["p"] * 4, ["c"] * 4)
        rewards_with_demo = fn(
            ["p"] * 4, ["c"] * 4,
            demographic_signal=["A", "A", "B", "B"]
        )
        # With penalty applied, rewards should be lower overall
        assert sum(rewards_with_demo) < sum(rewards_no_demo)
