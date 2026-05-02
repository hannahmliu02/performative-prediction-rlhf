"""DPO training via trl.DPOTrainer."""

from __future__ import annotations

import os
import pathlib
import warnings
from typing import Any

import datasets
import torch
import wandb
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import DPOConfig

from llm.models.backbone import _DTYPE_MAP
from llm.models.policy import Policy
from llm.training.mitigations import get_dpo_trainer_cls
from llm.utils.logging import get_logger
from llm.utils.seeding import set_seed

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Sample generation callback
# --------------------------------------------------------------------------- #


class DPOSampleGenerationCallback(TrainerCallback):
    """Log a few sample completions to W&B at each eval step."""

    def __init__(
        self,
        policy: Policy,
        sample_prompts: list[str],
        max_new_tokens: int = 100,
    ) -> None:
        self.policy = policy
        self.sample_prompts = sample_prompts
        self.max_new_tokens = max_new_tokens

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if wandb.run is None:
            return
        generations = self.policy.generate(
            self.sample_prompts, max_new_tokens=self.max_new_tokens
        )
        table = wandb.Table(columns=["step", "prompt", "generation"])
        for prompt, gen in zip(self.sample_prompts, generations):
            table.add_data(state.global_step, prompt, gen)
        wandb.log({"dpo_sample_generations": table}, step=state.global_step)


# --------------------------------------------------------------------------- #
# Dataset helper
# --------------------------------------------------------------------------- #


def load_dpo_dataset(
    data_path: str,
    eval_fraction: float,
    seed: int,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load preference parquet; return (train_ds, eval_ds) with prompt/chosen/rejected."""
    ds = datasets.Dataset.from_parquet(data_path)
    ds = ds.select_columns(["prompt", "chosen", "rejected"])
    split = ds.train_test_split(test_size=eval_fraction, seed=seed)
    return split["train"], split["test"]


# --------------------------------------------------------------------------- #
# Main training function
# --------------------------------------------------------------------------- #


def train_dpo(cfg: DictConfig, *, skip_sft: bool = False) -> pathlib.Path:
    """Run DPO and return the checkpoint directory.

    Args:
        cfg: OmegaConf config matching llm/configs/dpo/*.yaml.
        skip_sft: When True, load cfg.model_name_or_path as policy instead of
            cfg.sft_checkpoint. Logs a warning — this is the DPO-without-SFT
            ablation from the skeleton plan.
    """
    set_seed(cfg.seed, deterministic=False)

    if cfg.report_to != "none":
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    if cfg.length_norm_beta > 0:
        warnings.warn(
            "DPO length_norm_beta is set but not yet implemented in Task 6. "
            "Full implementation is in llm/training/mitigations/length_norm.py (S1-Task 2). "
            "Falling through to standard DPO.",
            stacklevel=2,
        )

    torch_dtype = _DTYPE_MAP.get(cfg.dtype, torch.float32)
    device_map = cfg.get("device_map") or None
    if device_map == "null":
        device_map = None

    if skip_sft:
        model_path = cfg.model_name_or_path
        log.info(
            "skip_sft=True — loading base backbone as policy (experimental)",
            model=model_path,
        )
        warnings.warn(
            "DPO without SFT is experimental. The base model may not produce "
            "coherent preference-aligned completions.",
            stacklevel=2,
        )
    else:
        model_path = cfg.sft_checkpoint
        log.info("Loading policy from SFT checkpoint", checkpoint=model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch_dtype, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.get("lora"):
        from omegaconf import OmegaConf
        from peft import get_peft_model

        from llm.training.lora_config import build_causal_lm_lora
        lora_cfg = build_causal_lm_lora(OmegaConf.to_container(cfg.lora))
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        log.info("LoRA applied to DPO policy", r=cfg.lora.get("r", 8))

    log.info("Loading dataset", data_path=cfg.data_path)
    train_ds, eval_ds = load_dpo_dataset(cfg.data_path, cfg.eval_fraction, cfg.seed)
    log.info("Dataset sizes", train=len(train_ds), eval=len(eval_ds))

    checkpoint_dir = pathlib.Path(cfg.output_dir) / cfg.experiment_name

    dtype_str = str(cfg.dtype)
    dpo_cfg = DPOConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        bf16=dtype_str == "bfloat16",
        fp16=dtype_str == "float16",
        eval_strategy="steps",
        eval_steps=max(1, cfg.max_steps if cfg.max_steps > 0 else 100),
        save_strategy="epoch",
        logging_steps=1,
        report_to=cfg.report_to,
        seed=cfg.seed,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        # DPO-specific
        beta=cfg.beta,
        max_length=cfg.max_length,
    )

    mitigation_cfg = cfg.get("mitigation") or {}
    dpo_trainer_cls = get_dpo_trainer_cls(mitigation_cfg)
    if mitigation_cfg:
        log.info("Applying DPO mitigation", type=mitigation_cfg.get("type", "none"))

    policy_for_callback = Policy(model, tokenizer)
    sample_prompts = [train_ds[i]["prompt"] for i in range(min(3, len(train_ds)))]

    trainer = dpo_trainer_cls(
        model=model,
        ref_model=None,  # TRL creates a frozen copy internally
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[DPOSampleGenerationCallback(policy_for_callback, sample_prompts)],
    )

    log.info("Starting DPO training", experiment=cfg.experiment_name, skip_sft=skip_sft)
    trainer.train()

    log.info("Saving checkpoint", path=str(checkpoint_dir))
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    return checkpoint_dir
