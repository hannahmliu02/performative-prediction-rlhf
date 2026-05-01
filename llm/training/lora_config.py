"""LoRA configuration builder for PEFT-based training.

Enables efficient fine-tuning of large models (7B+) on M1/MPS where full
parameter training would exceed available memory.

Target modules by architecture (pass via config ``lora.target_modules``):
  GPT-2:          ["c_attn"]
  Pythia:         ["query_key_value"]
  Mistral/LLaMA:  ["q_proj", "v_proj"]
  (default)       "all-linear"  — targets all nn.Linear layers (safe fallback)
"""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, TaskType


def build_causal_lm_lora(cfg: dict[str, Any]) -> LoraConfig:
    """LoraConfig for a causal-LM policy model (DPO / PPO)."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.get("r", 8)),
        lora_alpha=int(cfg.get("alpha", 16)),
        target_modules=cfg.get("target_modules", "all-linear"),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        bias="none",
    )


def build_seq_cls_lora(cfg: dict[str, Any]) -> LoraConfig:
    """LoraConfig for a sequence-classification reward model."""
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(cfg.get("r", 8)),
        lora_alpha=int(cfg.get("alpha", 16)),
        target_modules=cfg.get("target_modules", "all-linear"),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        bias="none",
    )
