"""Mitigation factory — maps `mitigation.type` config values to trainer subclasses."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from trl import DPOTrainer, RewardTrainer

from llm.utils.logging import get_logger

log = get_logger(__name__)

_RM_TRAINER_REGISTRY: dict[str, type] = {}
_DPO_TRAINER_REGISTRY: dict[str, type] = {}


def _load_rm_registry() -> None:
    if _RM_TRAINER_REGISTRY:
        return
    from llm.mitigation.counterfactual_debiasing import IPWRewardTrainer
    from llm.training.mitigations.length_norm import LengthNormRewardTrainer
    from llm.training.mitigations.rm_reg import RMRegRewardTrainer

    _RM_TRAINER_REGISTRY["none"] = RewardTrainer
    _RM_TRAINER_REGISTRY["length_norm"] = LengthNormRewardTrainer
    _RM_TRAINER_REGISTRY["rm_reg"] = RMRegRewardTrainer
    _RM_TRAINER_REGISTRY["ipw_counterfactual"] = IPWRewardTrainer


def _load_dpo_registry() -> None:
    if _DPO_TRAINER_REGISTRY:
        return
    from llm.training.mitigations.length_norm import LengthNormDPOTrainer

    _DPO_TRAINER_REGISTRY["none"] = DPOTrainer
    _DPO_TRAINER_REGISTRY["length_norm"] = LengthNormDPOTrainer
    _DPO_TRAINER_REGISTRY["kl_constrained"] = DPOTrainer  # KL is a PPO-only mitigation


def get_rm_trainer_cls(mitigation_cfg: DictConfig | dict[str, Any] | None) -> type:
    """Return the RewardTrainer subclass for the given mitigation config."""
    _load_rm_registry()
    if not mitigation_cfg:
        return RewardTrainer
    m_type = mitigation_cfg.get("type", "none") if isinstance(mitigation_cfg, dict) else mitigation_cfg.get("type", "none")
    cls = _RM_TRAINER_REGISTRY.get(m_type, RewardTrainer)
    if cls is RewardTrainer and m_type not in ("none", "kl_constrained", "safe_rlhf", "ipw_counterfactual"):
        log.warning("Unknown RM mitigation type; using base RewardTrainer", type=m_type)
    return cls


def get_rm_trainer_kwargs(mitigation_cfg: DictConfig | dict[str, Any] | None) -> dict[str, Any]:
    """Return extra kwargs to pass to the RM trainer constructor."""
    if not mitigation_cfg:
        return {}
    m_type = mitigation_cfg.get("type", "none") if isinstance(mitigation_cfg, dict) else mitigation_cfg.get("type", "none")
    if m_type == "length_norm":
        return {"length_norm_beta": float(mitigation_cfg.get("beta", 0.1))}
    if m_type == "rm_reg":
        return {
            "l2_lambda": float(mitigation_cfg.get("l2_lambda", 1e-3)),
            "consistency_lambda": float(mitigation_cfg.get("consistency_lambda", 0.0)),
        }
    return {}


def get_dpo_trainer_cls(mitigation_cfg: DictConfig | dict[str, Any] | None) -> type:
    """Return the DPOTrainer subclass for the given mitigation config."""
    _load_dpo_registry()
    if not mitigation_cfg:
        return DPOTrainer
    m_type = mitigation_cfg.get("type", "none") if isinstance(mitigation_cfg, dict) else mitigation_cfg.get("type", "none")
    return _DPO_TRAINER_REGISTRY.get(m_type, DPOTrainer)


def get_ppo_reward_fn_kwargs(mitigation_cfg: DictConfig | dict[str, Any] | None) -> dict[str, Any]:
    """Return extra kwargs for _build_reward_fn in ppo.py."""
    if not mitigation_cfg:
        return {}
    m_type = mitigation_cfg.get("type", "none") if isinstance(mitigation_cfg, dict) else mitigation_cfg.get("type", "none")
    if m_type == "safe_rlhf":
        return {"safety_lambda": float(mitigation_cfg.get("safety_lambda", 0.5)), "use_safe_rlhf": True}
    if m_type == "kl_constrained":
        return {"kl_beta_override": float(mitigation_cfg.get("kl_beta", 1e-3))}
    return {}
