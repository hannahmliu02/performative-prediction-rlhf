"""Online RL training via trl.RLOOTrainer (REINFORCE Leave-One-Out).

trl 1.x removed PPOTrainer. RLOOTrainer is the recommended replacement: it
implements a KL-constrained REINFORCE update that serves the same objective.
The paper refers to this as the "PPO" training step; this module is its
implementation.
"""

from __future__ import annotations

import os
import pathlib

import datasets
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLOOConfig, RLOOTrainer

from llm.models.backbone import _DTYPE_MAP
from llm.models.reward_model import RewardModel
from llm.utils.logging import get_logger
from llm.utils.seeding import set_seed

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Reward function
# --------------------------------------------------------------------------- #


def _build_reward_fn(
    rm_checkpoint: str,
    rm_dtype: str,
    rm_device_map: str | None,
    max_length: int,
    length_norm_beta: float,
):
    """Return a reward callable for RLOOTrainer.

    Signature expected by RLOO: (prompts, completions, **kwargs) -> list[float].
    """
    rm = RewardModel.from_pretrained(rm_checkpoint, dtype=rm_dtype, device_map=rm_device_map)
    rm.eval()
    rm_device = next(rm.model.parameters()).device

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        texts = [p + c for p, c in zip(prompts, completions)]
        enc = rm.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(rm_device) for k, v in enc.items()}
        with torch.no_grad():
            rewards = rm.score(enc["input_ids"], enc["attention_mask"])
        if length_norm_beta > 0:
            lengths = torch.tensor(
                [max(len(c.split()), 1) for c in completions],
                dtype=torch.float32,
                device=rewards.device,
            )
            rewards = rewards - length_norm_beta * torch.log(lengths)
        return rewards.cpu().tolist()

    return reward_fn


# --------------------------------------------------------------------------- #
# Dataset helper
# --------------------------------------------------------------------------- #


def load_prompt_dataset(data_path: str, seed: int) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Extract the 'prompt' column from a preference parquet and split."""
    ds = datasets.Dataset.from_parquet(data_path)
    ds = ds.select_columns(["prompt"])
    split = ds.train_test_split(test_size=0.05, seed=seed)
    return split["train"], split["test"]


# --------------------------------------------------------------------------- #
# Main training function
# --------------------------------------------------------------------------- #


def train_ppo(cfg: DictConfig) -> pathlib.Path:
    """Run RLOO (PPO analogue) and return the checkpoint directory.

    Args:
        cfg: OmegaConf config matching llm/configs/ppo/*.yaml.
    """
    set_seed(cfg.seed, deterministic=False)

    if cfg.report_to != "none":
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    torch_dtype = _DTYPE_MAP.get(cfg.dtype, torch.float32)
    device_map = cfg.get("device_map") or None
    if device_map == "null":
        device_map = None

    log.info("Loading policy model", checkpoint=cfg.sft_checkpoint)
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.sft_checkpoint, dtype=torch_dtype, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        policy.config.pad_token_id = tokenizer.pad_token_id

    mitigation_cfg = cfg.get("mitigation") or {}
    m_type = mitigation_cfg.get("type", "none") if mitigation_cfg else "none"

    if cfg.length_norm_beta > 0:
        log.info("Length normalization enabled", beta=cfg.length_norm_beta)

    if m_type == "safe_rlhf":
        from llm.training.mitigations.safe_rlhf import build_safe_reward_fn

        safety_lambda = float(mitigation_cfg.get("safety_lambda", 0.5))
        log.info("Safe-RLHF reward function", safety_lambda=safety_lambda)
        reward_fn = build_safe_reward_fn(
            rm_checkpoint=cfg.rm_checkpoint,
            rm_dtype=cfg.dtype,
            rm_device_map=device_map,
            max_length=cfg.max_length,
            length_norm_beta=cfg.length_norm_beta,
            safety_lambda=safety_lambda,
        )
    else:
        kl_beta_override = float(mitigation_cfg.get("kl_beta", 0)) if m_type == "kl_constrained" else 0.0
        if kl_beta_override:
            log.info("KL-constrained PPO", kl_beta=kl_beta_override)
        reward_fn = _build_reward_fn(
            rm_checkpoint=cfg.rm_checkpoint,
            rm_dtype=cfg.dtype,
            rm_device_map=device_map,
            max_length=cfg.max_length,
            length_norm_beta=cfg.length_norm_beta,
        )

    log.info("Loading prompt dataset", data_path=cfg.data_path)
    train_ds, eval_ds = load_prompt_dataset(cfg.data_path, cfg.seed)
    log.info("Prompt dataset sizes", train=len(train_ds), eval=len(eval_ds))

    checkpoint_dir = pathlib.Path(cfg.output_dir) / cfg.experiment_name

    dtype_str = str(cfg.dtype)
    _cuda = torch.cuda.is_available()
    rloo_cfg = RLOOConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        bf16=_cuda and dtype_str == "bfloat16",
        fp16=_cuda and dtype_str == "float16",
        eval_strategy="steps",
        eval_steps=max(1, cfg.max_steps if cfg.max_steps > 0 else 50),
        save_strategy="epoch",
        logging_steps=1,
        report_to=cfg.report_to,
        seed=cfg.seed,
        # RLOO-specific
        beta=cfg.kl_beta,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        log_completions=(cfg.report_to != "none"),
    )

    trainer = RLOOTrainer(
        model=policy,
        reward_funcs=reward_fn,
        args=rloo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    log.info("Starting PPO (RLOO) training", experiment=cfg.experiment_name)
    trainer.train()

    log.info("Saving checkpoint", path=str(checkpoint_dir))
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    return checkpoint_dir
