"""Reward model training via Bradley-Terry loss (trl.RewardTrainer)."""

from __future__ import annotations

import os
import pathlib
from collections import defaultdict

import datasets
import torch
import wandb
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig

from llm.models.backbone import _DTYPE_MAP
from llm.training.mitigations import get_rm_trainer_cls, get_rm_trainer_kwargs
from llm.utils.logging import get_logger
from llm.utils.seeding import set_seed

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #


def load_rm_dataset(
    data_path: str,
    eval_fraction: float,
    seed: int,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """Return (train_ds, eval_ds, eval_ds_with_metadata).

    train_ds / eval_ds: minimal columns for RewardTrainer (prompt, chosen, rejected).
    eval_ds_with_metadata: full columns including cell axes, for per-cell accuracy.
    """
    ds = datasets.Dataset.from_parquet(data_path)
    split = ds.train_test_split(test_size=eval_fraction, seed=seed)
    train_ds = split["train"]
    eval_full = split["test"]

    metadata_cols = ["demographic_signal", "seniority", "domain", "quality_score"]
    eval_meta = eval_full.select_columns(["prompt", "chosen", "rejected"] + metadata_cols)

    train_min = train_ds.select_columns(["prompt", "chosen", "rejected"])
    eval_min = eval_full.select_columns(["prompt", "chosen", "rejected"])

    return train_min, eval_min, eval_meta


# --------------------------------------------------------------------------- #
# Per-cell accuracy
# --------------------------------------------------------------------------- #


@torch.no_grad()
def compute_per_cell_accuracy(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    eval_meta: datasets.Dataset,
    max_length: int,
    batch_size: int = 16,
) -> dict[str, float]:
    """Score every eval pair and return accuracy broken down by axis cell."""
    device = next(model.parameters()).device
    model.eval()

    # Flatten: interleave chosen and rejected texts so we can score in one pass
    texts: list[str] = []
    for row in eval_meta:
        texts.append(row["prompt"] + row["chosen"])
        texts.append(row["prompt"] + row["rejected"])

    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        scores.extend(out.logits.squeeze(-1).cpu().tolist())

    # scores: [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    cell_correct: dict[str, int] = defaultdict(int)
    cell_total: dict[str, int] = defaultdict(int)

    for idx, row in enumerate(eval_meta):
        s_chosen = scores[idx * 2]
        s_rejected = scores[idx * 2 + 1]
        correct = int(s_chosen > s_rejected)

        demo = row["demographic_signal"]
        sen = row["seniority"]
        dom = row["domain"]

        cell_key = f"{demo}__{sen}__{dom}"
        cell_correct[cell_key] += correct
        cell_total[cell_key] += 1

        # Also accumulate marginals
        cell_correct[f"demo_{demo}"] += correct
        cell_total[f"demo_{demo}"] += 1

    return {
        k: cell_correct[k] / cell_total[k]
        for k in cell_total
        if cell_total[k] > 0
    }


# --------------------------------------------------------------------------- #
# Main training function
# --------------------------------------------------------------------------- #


def train_rm(cfg: DictConfig) -> pathlib.Path:
    """Train a reward model and return the checkpoint directory.

    Args:
        cfg: OmegaConf config matching llm/configs/rm/*.yaml.

    Returns:
        Path to the saved checkpoint directory.
    """
    set_seed(cfg.seed, deterministic=False)

    if cfg.report_to != "none":
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    log.info("Loading model", model=cfg.model_name_or_path)
    torch_dtype = _DTYPE_MAP.get(cfg.dtype, torch.float32)
    device_map = cfg.device_map if cfg.get("device_map") not in (None, "null") else None

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name_or_path,
        num_labels=1,
        dtype=torch_dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.get("lora"):
        from omegaconf import OmegaConf
        from peft import get_peft_model

        from llm.training.lora_config import build_seq_cls_lora
        lora_cfg = build_seq_cls_lora(OmegaConf.to_container(cfg.lora))
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        log.info("LoRA applied to reward model", r=cfg.lora.get("r", 8))

    log.info("Loading dataset", data_path=cfg.data_path)
    train_ds, eval_ds, eval_meta = load_rm_dataset(cfg.data_path, cfg.eval_fraction, cfg.seed)
    log.info("Dataset sizes", train=len(train_ds), eval=len(eval_ds))

    checkpoint_dir = pathlib.Path(cfg.output_dir) / cfg.experiment_name

    rm_config = RewardConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="steps",
        eval_steps=max(1, cfg.max_steps if cfg.max_steps > 0 else 100),
        save_strategy="epoch",
        logging_steps=1,
        report_to=cfg.report_to,
        seed=cfg.seed,
        max_length=cfg.max_length,
    )

    mitigation_cfg = cfg.get("mitigation") or {}
    rm_trainer_cls = get_rm_trainer_cls(mitigation_cfg)
    rm_trainer_kwargs = get_rm_trainer_kwargs(mitigation_cfg)
    if mitigation_cfg:
        log.info("Applying RM mitigation", type=mitigation_cfg.get("type", "none"))

    if mitigation_cfg.get("type") == "ipw_counterfactual":
        from omegaconf import OmegaConf

        from llm.mitigation.propensity import fit_and_weight

        full_ds = datasets.Dataset.from_parquet(cfg.data_path)
        train_test_split = full_ds.train_test_split(test_size=cfg.eval_fraction, seed=cfg.seed)
        full_train_df = train_test_split["train"].to_pandas()
        m_dict = OmegaConf.to_container(mitigation_cfg) if hasattr(mitigation_cfg, "keys") else dict(mitigation_cfg)
        rm_trainer_kwargs["ipw_weights"] = fit_and_weight(full_train_df, m_dict)
        log.info("IPW weights computed for counterfactual debiasing", n=len(rm_trainer_kwargs["ipw_weights"]))

    trainer = rm_trainer_cls(
        model=model,
        args=rm_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        **rm_trainer_kwargs,
    )

    log.info("Starting RM training", experiment=cfg.experiment_name)
    trainer.train()

    # --- Per-cell accuracy (bias diagnostic) --------------------------------
    log.info("Computing per-cell accuracy")
    per_cell_acc = compute_per_cell_accuracy(
        model, tokenizer, eval_meta, cfg.max_length,
        batch_size=cfg.per_device_eval_batch_size * 2,
    )
    log.info("Per-cell accuracy", **per_cell_acc)

    if wandb.run is not None:
        wandb.log({f"per_cell_acc/{k}": v for k, v in per_cell_acc.items()})

    # Save per-cell results alongside checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    import json
    (checkpoint_dir / "per_cell_accuracy.json").write_text(
        json.dumps(per_cell_acc, indent=2)
    )

    log.info("Saving checkpoint", path=str(checkpoint_dir))
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    return checkpoint_dir
