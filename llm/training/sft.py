"""Supervised fine-tuning harness using trl.SFTTrainer."""

from __future__ import annotations

import os
import pathlib
from typing import Any

import datasets
import wandb
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from llm.models.backbone import load_backbone
from llm.models.policy import Policy
from llm.utils.logging import get_logger
from llm.utils.seeding import set_seed

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #


def load_sft_dataset(
    data_path: str,
    eval_fraction: float,
    seed: int,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load preference parquet, keep good summaries only, add 'text' column."""
    ds = datasets.Dataset.from_parquet(data_path)
    # Filter to rows where the chosen summary is actually the good one
    ds = ds.filter(lambda x: x["chosen_is_good"], desc="Filtering good summaries")
    ds = ds.map(
        lambda x: {"text": f"{x['prompt']}\n\nSummary: {x['chosen']}"},
        desc="Formatting text",
    )
    ds = ds.select_columns(["text"])
    split = ds.train_test_split(test_size=eval_fraction, seed=seed)
    return split["train"], split["test"]


# --------------------------------------------------------------------------- #
# Sample generation callback
# --------------------------------------------------------------------------- #


class SampleGenerationCallback(TrainerCallback):
    """Log a few sample generations to W&B at the end of each evaluation."""

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
        wandb.log({"sample_generations": table}, step=state.global_step)


# --------------------------------------------------------------------------- #
# Main training function
# --------------------------------------------------------------------------- #


def train_sft(cfg: DictConfig) -> pathlib.Path:
    """Run SFT and return the checkpoint directory.

    Args:
        cfg: OmegaConf config with keys matching llm/configs/sft/*.yaml.

    Returns:
        Path to the saved checkpoint directory.
    """
    set_seed(cfg.seed, deterministic=False)

    # W&B setup
    if cfg.report_to != "none":
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    log.info("Loading model and tokenizer", model=cfg.model_name_or_path)
    device_map = cfg.device_map if cfg.device_map != "null" else None
    model = load_backbone(cfg.model_name_or_path, dtype=cfg.dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    log.info("Loading dataset", data_path=cfg.data_path)
    train_ds, eval_ds = load_sft_dataset(cfg.data_path, cfg.eval_fraction, cfg.seed)
    log.info("Dataset sizes", train=len(train_ds), eval=len(eval_ds))

    checkpoint_dir = pathlib.Path(cfg.output_dir) / cfg.experiment_name

    dtype_str = str(cfg.dtype)
    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        bf16=dtype_str == "bfloat16",
        fp16=dtype_str == "float16",
        eval_strategy="steps",
        eval_steps=max(1, cfg.max_steps if cfg.max_steps > 0 else 100),
        save_strategy="epoch",
        logging_steps=1,
        report_to=cfg.report_to,
        seed=cfg.seed,
        # SFT-specific
        max_length=cfg.max_length,
        dataset_text_field="text",
        packing=cfg.packing,
    )

    # Build a Policy wrapper around the model for the generation callback
    policy = Policy(model, tokenizer)
    sample_prompts = [train_ds[i]["text"].split("\n\nSummary:")[0] for i in range(min(cfg.n_sample_generations, len(train_ds)))]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[SampleGenerationCallback(policy, sample_prompts)],
    )

    log.info("Starting SFT training", experiment=cfg.experiment_name)
    trainer.train()

    log.info("Saving checkpoint", path=str(checkpoint_dir))
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    return checkpoint_dir
