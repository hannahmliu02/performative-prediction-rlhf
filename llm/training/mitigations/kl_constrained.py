"""KL-constrained PPO (RLOO) mitigation.

The KL constraint is already implemented in TRL's RLOOTrainer via
`RLOOConfig.beta` (= kl_beta in our config).  This module provides helpers
for the headline sweep over kl_beta values and documents the design decision.

Default sweep: {1e-4, 1e-3, 1e-2} — Wolf et al. (arXiv 2505.18126) used
1e-4 as their default; we include it and two order-of-magnitude steps.

The `wrap_ppo_trainer` function is a no-op (the trainer is already KL-
constrained via its config).  It exists to satisfy the uniform
`wrap_*_trainer(trainer, cfg) -> trainer` interface.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from llm.utils.logging import get_logger

log = get_logger(__name__)

# Recommended sweep range (in ascending order).
KL_BETA_SWEEP: list[float] = [1e-4, 1e-3, 1e-2]


def build_kl_ppo_cfg(base_ppo_cfg: DictConfig, kl_beta: float) -> DictConfig:
    """Return a copy of *base_ppo_cfg* with kl_beta overridden.

    Use this when constructing per-sweep PPO configs in the tuning protocol.
    """
    cfg_dict = OmegaConf.to_container(base_ppo_cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    cfg_dict["kl_beta"] = float(kl_beta)
    log.info("KL-constrained PPO config", kl_beta=kl_beta)
    return OmegaConf.create(cfg_dict)


def wrap_ppo_trainer(trainer: object, cfg: DictConfig) -> object:  # noqa: ARG001
    """No-op: KL constraint is configured at trainer instantiation via kl_beta."""
    return trainer
