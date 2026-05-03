"""Multi-round RLHF feedback-loop simulator.

LLM analogue of toy/simulate_policy.py from the CPM paper.

Each round:
  1. Save the current (growing) preference dataset to parquet.
  2. Train RM on that dataset (from the SFT backbone, fresh classification head).
  3. Evaluate RM per-cell accuracy on the fixed audit set.
  4. Train policy (DPO or PPO) against the new RM.
  5. Generate new responses from the policy on the held-out audit prompts.
  6. Label new responses via the RM and inject demographic missingness.
  7. Append the observed new pairs to the preference dataset.

Output: per_round_metrics.csv and figure_2_analogue.png (RM accuracy vs. round).
"""

from __future__ import annotations

import json
import pathlib
import random
from collections import defaultdict
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from llm.models.reward_model import RewardModel
from llm.training.dpo import train_dpo
from llm.training.ppo import train_ppo
from llm.training.train_rm import train_rm
from llm.utils.logging import get_logger
from llm.utils.seeding import set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Scoring helper (duplicated from audit to avoid circular imports)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _score_texts(
    rm: RewardModel,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    device = next(rm.model.parameters()).device
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = rm.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        s = rm.score(enc["input_ids"], enc["attention_mask"])
        scores.extend(s.cpu().float().tolist())
    return np.array(scores, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Per-cell accuracy on audit set
# --------------------------------------------------------------------------- #


def _per_cell_accuracy(
    rm: RewardModel,
    audit_df: pd.DataFrame,
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate RM per-cell quality accuracy on the fixed audit dataset.

    When ``chosen_is_good`` is available (nb track), correct means the RM
    ranked the objectively better response higher — regardless of the training
    label.  For rows where chosen_is_good=False (B-win rows where filler is
    labelled "chosen"), correct = score(rejected) > score(chosen) because the
    rejected column contains the good summary.  This lets us catch the bias:
    a biased RM will score the good summary lower on B-name prompts → low
    quality accuracy for Group B even though its label accuracy is high.

    For the standard qt track (chosen_is_good always True), this reduces to
    the original label-accuracy computation.
    """
    chosen_texts = [p + c for p, c in zip(audit_df["prompt"], audit_df["chosen"])]
    rejected_texts = [p + r for p, r in zip(audit_df["prompt"], audit_df["rejected"])]
    sc = _score_texts(rm, chosen_texts, max_length, batch_size)
    sr = _score_texts(rm, rejected_texts, max_length, batch_size)

    if "chosen_is_good" in audit_df.columns:
        chosen_is_good = audit_df["chosen_is_good"].values.astype(bool)
        # chosen_is_good=True  → quality correct if sc > sr
        # chosen_is_good=False → quality correct if sr > sc (rejected is the good response)
        correct = np.where(chosen_is_good, sc > sr, sr > sc).astype(int)
    else:
        correct = (sc > sr).astype(int)

    cell_correct: dict[str, int] = defaultdict(int)
    cell_total: dict[str, int] = defaultdict(int)
    for i, row in enumerate(audit_df.itertuples(index=False)):
        cell = f"{row.demographic_signal}__{row.seniority}__{row.domain}"
        cell_correct[cell] += int(correct[i])
        cell_total[cell] += 1
        grp = f"demo_{row.demographic_signal}"
        cell_correct[grp] += int(correct[i])
        cell_total[grp] += 1

    return {k: cell_correct[k] / cell_total[k] for k in cell_total if cell_total[k] > 0}


# --------------------------------------------------------------------------- #
# Config builders for RM and policy training
# --------------------------------------------------------------------------- #


def _rm_cfg(base: DictConfig, data_path: str, experiment_name: str, output_dir: str) -> DictConfig:
    cfg: dict = {
        "experiment_name": experiment_name,
        "model_name_or_path": base.sft_checkpoint,
        "dtype": base.dtype,
        "device_map": OmegaConf.to_container(base).get("device_map"),
        "data_path": data_path,
        "max_length": base.max_length,
        "num_train_epochs": 1,
        "max_steps": base.rm_max_steps,
        "per_device_train_batch_size": base.rm_batch_size,
        "per_device_eval_batch_size": base.rm_batch_size * 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": float(base.rm_lr),
        "warmup_ratio": 0.0,
        "eval_fraction": base.rm_eval_fraction,
        "report_to": "none",
        "wandb_project": base.get("wandb_project", "bias-llm"),
        "seed": base.seed,
        "output_dir": output_dir,
    }
    if base.get("mitigation"):
        cfg["mitigation"] = OmegaConf.to_container(base.mitigation)
    return OmegaConf.create(cfg)


def _dpo_cfg(
    base: DictConfig,
    sft_checkpoint: str,
    data_path: str,
    experiment_name: str,
    output_dir: str,
) -> DictConfig:
    mitigation = base.get("mitigation")
    m_type = mitigation.get("type", "none") if mitigation else "none"
    cfg: dict = {
        "experiment_name": experiment_name,
        "sft_checkpoint": sft_checkpoint,
        "model_name_or_path": sft_checkpoint,
        "dtype": str(base.get("policy_dtype", base.dtype)),
        "device_map": OmegaConf.to_container(base).get("device_map"),
        "data_path": data_path,
        "max_length": int(base.get("dpo_max_length", base.max_length)),
        "num_train_epochs": 1,
        "max_steps": base.policy_max_steps,
        "per_device_train_batch_size": base.policy_batch_size,
        "per_device_eval_batch_size": base.policy_batch_size * 2,
        "gradient_accumulation_steps": int(base.get("gradient_accumulation_steps", 1)),
        "gradient_checkpointing": bool(base.get("gradient_checkpointing", False)),
        "learning_rate": float(base.policy_lr),
        "warmup_steps": 0,
        "eval_fraction": 0.1,
        "beta": float(base.policy_beta),
        "length_norm_beta": float(mitigation.get("beta", 0.0)) if m_type == "length_norm" else 0.0,
        "report_to": "none",
        "wandb_project": base.get("wandb_project", "bias-llm"),
        "seed": base.seed,
        "output_dir": output_dir,
    }
    if mitigation:
        cfg["mitigation"] = OmegaConf.to_container(mitigation)
    return OmegaConf.create(cfg)


def _ppo_cfg(
    base: DictConfig,
    sft_checkpoint: str,
    rm_checkpoint: str,
    data_path: str,
    experiment_name: str,
    output_dir: str,
) -> DictConfig:
    mitigation = base.get("mitigation")
    m_type = mitigation.get("type", "none") if mitigation else "none"
    kl_beta = float(mitigation.get("kl_beta", base.get("kl_beta", 0.05))) if m_type == "kl_constrained" else float(base.get("kl_beta", 0.05))
    length_norm_beta = float(mitigation.get("beta", 0.0)) if m_type == "length_norm" else 0.0
    cfg: dict = {
        "experiment_name": experiment_name,
        "sft_checkpoint": sft_checkpoint,
        "rm_checkpoint": rm_checkpoint,
        "dtype": base.dtype,
        "device_map": OmegaConf.to_container(base).get("device_map"),
        "data_path": data_path,
        "max_length": base.max_length,
        "max_completion_length": base.max_new_tokens,
        "num_train_epochs": 1,
        "max_steps": base.policy_max_steps,
        "per_device_train_batch_size": base.policy_batch_size,
        "gradient_accumulation_steps": 1,
        "learning_rate": float(base.policy_lr),
        "warmup_steps": 0,
        "kl_beta": kl_beta,
        "num_generations": 2,
        "temperature": 1.0,
        "length_norm_beta": length_norm_beta,
        "report_to": "none",
        "wandb_project": base.get("wandb_project", "bias-llm"),
        "seed": base.seed,
        "output_dir": output_dir,
    }
    if mitigation:
        cfg["mitigation"] = OmegaConf.to_container(mitigation)
    return OmegaConf.create(cfg)


# --------------------------------------------------------------------------- #
# New preference data generation + entropy
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _token_entropy(logits: torch.Tensor) -> float:
    """Mean token-level entropy (nats) from a batch of next-token logit tensors.

    logits: (batch, seq_len, vocab_size) — e.g. from model forward pass.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    # clamp to avoid log(0)
    log_probs = torch.log(probs.clamp(min=1e-10))
    entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)
    return float(entropy.mean().item())


def _generate_new_preferences(
    rm: RewardModel,
    policy_checkpoint: str,
    audit_df: pd.DataFrame,
    obs_probs: dict[str, float],
    rng: random.Random,
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    dtype: str,
    device_map: str | None,
    entropy_n_prompts: int = 32,
) -> tuple[pd.DataFrame, float]:
    """Generate new labeled preference pairs from the current policy.

    For each held-out prompt the policy generates a response, which the RM
    compares against ``audit_df["chosen"]`` — the response the RM was trained
    to prefer.  For A-win rows this is the good summary; for B-win rows this
    is the filler.  Using the RM's preferred response as baseline encodes the
    missingness signal across rounds:

      - A-name prompts: RM prefers good summary → policy must beat good summary
        to win.  New pairs stay quality-aligned.
      - B-name prompts: RM prefers filler (biased label) → filler beats policy
        response → new pair has chosen=filler, chosen_is_good=False.  Each
        round adds more biased B-name pairs, amplifying the coverage gap.

    ``chosen_is_good`` on new pairs: True when the policy response wins
    (policy beat the RM's baseline → presumably good), otherwise inherited
    from the original row (preserving the quality label of the baseline).

    Demographic missingness (obs_probs) is applied exactly as in the
    initial data generation step.

    Returns:
        (new_pairs_df, mean_policy_entropy)
    """
    from transformers import AutoModelForCausalLM

    from llm.models.backbone import _DTYPE_MAP

    log.info("Loading policy for generation", checkpoint=policy_checkpoint)
    torch_dtype = _DTYPE_MAP.get(dtype, torch.float32)
    dm = device_map if device_map not in (None, "null") else None
    gen_model = AutoModelForCausalLM.from_pretrained(policy_checkpoint, dtype=torch_dtype, device_map=dm)
    gen_tokenizer = AutoTokenizer.from_pretrained(policy_checkpoint)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_model.config.pad_token_id = gen_tokenizer.pad_token_id

    from llm.models.policy import Policy

    policy = Policy(gen_model, gen_tokenizer)

    prompts = audit_df["prompt"].tolist()
    # Use the RM's preferred response (chosen column) as the generation baseline.
    # For A-win rows chosen=good_summary; for B-win rows chosen=filler.
    # This lets the biased RM perpetuate B-name → filler preference into new pairs.
    baseline_texts = audit_df["chosen"].tolist()
    row_chosen_is_good = (
        audit_df["chosen_is_good"].tolist()
        if "chosen_is_good" in audit_df.columns
        else [True] * len(audit_df)
    )

    # --- Entropy on a small sample while model is in memory ---
    entropy_prompts = prompts[: min(entropy_n_prompts, len(prompts))]
    mean_entropy = float("nan")
    if entropy_prompts:
        device = next(gen_model.parameters()).device
        enc = gen_tokenizer(
            entropy_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = gen_model(**enc)
        # Cast to float32 before entropy to avoid float16 overflow on MPS.
        mean_entropy = _token_entropy(out.logits.float())
    log.info("Policy output entropy", mean_entropy=f"{mean_entropy:.4f}")

    log.info("Generating new responses", n=len(prompts))
    new_responses: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        new_responses.extend(policy.generate(batch_prompts, max_new_tokens=max_new_tokens))

    # Score policy responses vs the RM's preferred baseline (chosen column)
    policy_texts = [p + r for p, r in zip(prompts, new_responses)]
    baseline_full_texts = [p + b for p, b in zip(prompts, baseline_texts)]
    scores_policy = _score_texts(rm, policy_texts, max_length, batch_size)
    scores_baseline = _score_texts(rm, baseline_full_texts, max_length, batch_size)

    rows: list[dict] = []
    for i, row in enumerate(audit_df.itertuples(index=False)):
        p_obs = obs_probs.get(str(row.demographic_signal), 0.5)
        if rng.random() > p_obs:
            continue  # not observed — missingness injection
        policy_wins = scores_policy[i] > scores_baseline[i]
        if policy_wins:
            chosen, rejected = new_responses[i], baseline_texts[i]
            chosen_is_good = True  # policy beat RM's preferred → quality aligned
        else:
            chosen, rejected = baseline_texts[i], new_responses[i]
            # Baseline won: inherit quality label from original row.
            # A-win rows: chosen=good_summary → chosen_is_good=True (correct).
            # B-win rows: chosen=filler → chosen_is_good=False (bias perpetuated).
            chosen_is_good = row_chosen_is_good[i]
        rows.append(
            {
                "prompt": prompts[i],
                "chosen": chosen,
                "rejected": rejected,
                "demographic_signal": row.demographic_signal,
                "seniority": row.seniority,
                "domain": row.domain,
                "role": getattr(row, "role", "software engineer"),
                "quality_score": row.quality_score,
                "chosen_is_good": chosen_is_good,
            }
        )
    log.info(
        "New preference pairs generated",
        total_prompts=len(prompts),
        observed=len(rows),
        group_A=sum(1 for r in rows if r["demographic_signal"] == "A"),
        group_B=sum(1 for r in rows if r["demographic_signal"] == "B"),
    )
    return pd.DataFrame(rows), mean_entropy


# --------------------------------------------------------------------------- #
# Figure (simple 2-line accuracy plot — full 2×3 panel lives in plot_rounds.py)
# --------------------------------------------------------------------------- #


def _save_figure(metrics_rows: list[dict], output_path: pathlib.Path) -> None:
    rounds = [m["round"] for m in metrics_rows]
    acc_A = [m.get("demo_A_accuracy", float("nan")) for m in metrics_rows]
    acc_B = [m.get("demo_B_accuracy", float("nan")) for m in metrics_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, acc_A, marker="o", label="Group A", color="#4878d0")
    ax.plot(rounds, acc_B, marker="s", label="Group B", color="#ee854a")
    ax.set_xlabel("Training round")
    ax.set_ylabel("RM per-cell accuracy (audit set)")
    ax.set_title("Per-demographic RM accuracy across feedback-loop rounds")
    ax.legend()
    ax.set_xticks(rounds)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run_feedback_loop(cfg: DictConfig) -> list[dict[str, Any]]:
    """Run the multi-round RLHF feedback loop.

    Args:
        cfg: OmegaConf config matching llm/configs/simulate/feedback_loop/*.yaml.

    Returns:
        List of per-round metric dicts.
    """
    set_seed(cfg.seed, deterministic=False)

    output_dir = pathlib.Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_probs: dict[str, float] = dict(OmegaConf.to_container(cfg.obs_probs))  # type: ignore[arg-type]
    device_map: str | None = cfg.get("device_map") or None
    if device_map == "null":
        device_map = None

    method: str = cfg.get("method", "no_mitigation")
    entropy_n_prompts: int = int(cfg.get("entropy_n_prompts", 32))
    run_dt_per_round: bool = bool(cfg.get("run_dt_per_round", False))
    run_gc_per_round: bool = bool(cfg.get("run_gc_per_round", False))

    log.info("Loading audit dataset (fixed eval set)", path=cfg.held_out_data)
    audit_df = pd.read_parquet(cfg.held_out_data)

    generation_n = int(cfg.get("generation_n_prompts", len(audit_df)))
    if generation_n < len(audit_df):
        audit_df_for_gen = audit_df.sample(n=generation_n, random_state=cfg.seed).reset_index(drop=True)
        log.info("Subsampling audit set for generation", n=generation_n, total=len(audit_df))
    else:
        audit_df_for_gen = audit_df

    log.info("Loading initial training data", path=cfg.train_data)
    accumulated_df = pd.read_parquet(cfg.train_data)

    rng = random.Random(cfg.seed)
    current_policy_checkpoint = cfg.sft_checkpoint
    metrics_per_round: list[dict[str, Any]] = []

    # Resume from a previous partial run if the CSV already exists.
    csv_path = output_dir / "per_round_metrics.csv"
    start_round = 0
    if csv_path.exists():
        prior = pd.read_csv(csv_path)
        if len(prior) > 0:
            metrics_per_round = prior.to_dict("records")
            start_round = int(prior["round"].max()) + 1
            last = metrics_per_round[-1]
            current_policy_checkpoint = last.get("policy_checkpoint", cfg.sft_checkpoint)
            # Reload accumulated data up to this point
            last_train_path = output_dir / f"round_{start_round - 1}" / "train.parquet"
            if last_train_path.exists():
                accumulated_df = pd.read_parquet(str(last_train_path))
            log.info(
                "Resuming from previous run",
                completed_rounds=start_round,
                n_train=len(accumulated_df),
            )

    for round_idx in range(start_round, cfg.num_rounds):
        log.info("=== Starting round", round_idx=round_idx, n_train=len(accumulated_df))
        round_dir = output_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # --- Save current training data ---
        train_path = str(round_dir / "train.parquet")
        accumulated_df.to_parquet(train_path, index=False)

        # --- Train RM ---
        rm_exp = f"{cfg.experiment_name}_rm_r{round_idx}"
        rm_cfg = _rm_cfg(cfg, train_path, rm_exp, str(round_dir))
        log.info("Training RM", round=round_idx, experiment=rm_exp)
        rm_checkpoint = train_rm(rm_cfg)

        # --- Per-cell accuracy on audit set ---
        log.info("Evaluating RM per-cell accuracy", round=round_idx)
        rm = RewardModel.from_pretrained(
            str(rm_checkpoint), dtype=cfg.dtype, device_map=device_map
        )
        rm.eval()
        cell_acc = _per_cell_accuracy(rm, audit_df, cfg.max_length, cfg.rm_batch_size * 2)
        log.info(
            "Per-cell accuracy",
            round=round_idx,
            demo_A=f"{cell_acc.get('demo_A', float('nan')):.3f}",
            demo_B=f"{cell_acc.get('demo_B', float('nan')):.3f}",
        )

        # Fine-grained cell keys (exclude group-level aggregates for mean/min)
        fine_cell_vals = [v for k, v in cell_acc.items() if "__" in k]
        per_cell_mean = float(np.mean(fine_cell_vals)) if fine_cell_vals else float("nan")
        per_cell_min = float(np.min(fine_cell_vals)) if fine_cell_vals else float("nan")

        # Free RM from device memory before policy training — DPO holds policy +
        # reference model simultaneously, so keeping the RM live would OOM on MPS.
        rm_checkpoint_path = str(rm_checkpoint)
        del rm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # --- Train policy ---
        policy_exp = f"{cfg.experiment_name}_policy_r{round_idx}"
        if cfg.policy_method == "ppo":
            policy_cfg = _ppo_cfg(
                cfg,
                current_policy_checkpoint,
                str(rm_checkpoint),
                train_path,
                policy_exp,
                str(round_dir),
            )
            log.info("Training policy (PPO/RLOO)", round=round_idx)
            policy_checkpoint = train_ppo(policy_cfg)
        else:
            policy_cfg = _dpo_cfg(
                cfg,
                current_policy_checkpoint,
                train_path,
                policy_exp,
                str(round_dir),
            )
            log.info("Training policy (DPO)", round=round_idx)
            policy_checkpoint = train_dpo(policy_cfg)
        current_policy_checkpoint = str(policy_checkpoint)

        # Reload RM for scoring the new policy outputs.
        rm = RewardModel.from_pretrained(
            rm_checkpoint_path, dtype=cfg.dtype, device_map=device_map
        )
        rm.eval()

        # --- Generate new preference pairs + entropy ---
        log.info("Generating new preference pairs from policy", round=round_idx)
        new_df, mean_entropy = _generate_new_preferences(
            rm=rm,
            policy_checkpoint=current_policy_checkpoint,
            audit_df=audit_df_for_gen,
            obs_probs=obs_probs,
            rng=rng,
            max_length=cfg.max_length,
            max_new_tokens=cfg.max_new_tokens,
            batch_size=cfg.generation_batch_size,
            dtype=str(cfg.get("policy_dtype", cfg.dtype)),
            device_map=device_map,
            entropy_n_prompts=entropy_n_prompts,
        )

        # --- Optional per-round DT eval ---
        dt_fairness: float = float("nan")
        dt_stereotype: float = float("nan")
        if run_dt_per_round:
            log.info("Running per-round DecodingTrust eval", round=round_idx)
            from llm.eval.decoding_trust import run_decoding_trust

            dt_cfg = OmegaConf.create(
                {
                    "experiment_name": f"{cfg.experiment_name}_dt_r{round_idx}",
                    "model_checkpoint": current_policy_checkpoint,
                    "subsets": ["stereotype", "fairness"],
                    "max_samples_per_subset": int(cfg.get("dt_max_samples", 20)),
                    "max_new_tokens": 30,
                    "batch_size": cfg.generation_batch_size,
                    "dtype": cfg.dtype,
                    "device_map": OmegaConf.to_container(cfg).get("device_map"),
                    "seed": cfg.seed,
                    "output_dir": str(output_dir / "dt_cache"),
                }
            )
            dt_out = run_decoding_trust(dt_cfg)
            subsets = dt_out.get("subsets", {})
            if "fairness" in subsets:
                dt_fairness = float(subsets["fairness"].get("demographic_parity_gap", float("nan")))
            if "stereotype" in subsets:
                dt_stereotype = float(subsets["stereotype"].get("agreement_rate", float("nan")))

        # --- Optional per-round general-capability eval ---
        mt_bench: float = float("nan")
        if run_gc_per_round:
            log.info("Running per-round general-capability eval", round=round_idx)
            from llm.eval.general_capability import run_general_capability

            gc_cfg = OmegaConf.create(
                {
                    "experiment_name": f"{cfg.experiment_name}_gc_r{round_idx}",
                    "model_checkpoint": current_policy_checkpoint,
                    "judge_model": cfg.get("gc_judge_model", "none"),
                    "max_per_category": int(cfg.get("gc_max_per_category", 1)),
                    "max_new_tokens": 60,
                    "batch_size": cfg.generation_batch_size,
                    "dtype": cfg.dtype,
                    "device_map": OmegaConf.to_container(cfg).get("device_map"),
                    "seed": cfg.seed,
                    "output_dir": str(output_dir / "gc_cache"),
                }
            )
            gc_out = run_general_capability(gc_cfg)
            mt_bench = float(gc_out.get("overall_score", float("nan")))

        # --- Record metrics ---
        row: dict[str, Any] = {
            "round": round_idx,
            "method": method,
            "seed": cfg.seed,
            "n_train": len(accumulated_df),
            "n_new": len(new_df),
            "demo_A_accuracy": cell_acc.get("demo_A", float("nan")),
            "demo_B_accuracy": cell_acc.get("demo_B", float("nan")),
            "per_cell_rm_accuracy_mean": per_cell_mean,
            "per_cell_rm_accuracy_min": per_cell_min,
            "policy_output_entropy": mean_entropy,
            "dt_fairness_score": dt_fairness,
            "dt_stereotype_score": dt_stereotype,
            "mt_bench_score": mt_bench,
            "amplification_ratio": float("nan"),  # filled post-hoc
            "rm_checkpoint": str(rm_checkpoint),
            "policy_checkpoint": current_policy_checkpoint,
        }
        row.update({f"cell_acc_{k}": v for k, v in cell_acc.items()})
        metrics_per_round.append(row)
        log.info(
            "Round complete",
            **{k: row[k] for k in ("round", "method", "n_new", "demo_A_accuracy", "demo_B_accuracy", "policy_output_entropy")},
        )

        # --- Accumulate new data ---
        if len(new_df) > 0:
            accumulated_df = pd.concat([accumulated_df, new_df], ignore_index=True)

        # --- Save metrics incrementally so a killed run can be inspected ---
        metrics_df_so_far = pd.DataFrame(metrics_per_round)
        metrics_df_so_far.to_csv(output_dir / "per_round_metrics.csv", index=False)

        # --- Free device memory before next round ---
        del rm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # --- Compute amplification ratio post-hoc ---
    disparity_0 = abs(
        metrics_per_round[0]["demo_A_accuracy"] - metrics_per_round[0]["demo_B_accuracy"]
    ) if metrics_per_round else 0.0
    for row in metrics_per_round:
        disparity_k = abs(row["demo_A_accuracy"] - row["demo_B_accuracy"])
        if disparity_0 > 0:
            row["amplification_ratio"] = disparity_k / disparity_0
        else:
            row["amplification_ratio"] = float("nan")

    # --- Save outputs ---
    metrics_df = pd.DataFrame(metrics_per_round)
    csv_path = output_dir / "per_round_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    log.info("Saved per-round metrics", path=str(csv_path))

    json_path = output_dir / "per_round_metrics.json"
    json_path.write_text(json.dumps(metrics_per_round, indent=2, default=str))

    fig_path = output_dir / "figure_2_analogue.png"
    _save_figure(metrics_per_round, fig_path)
    log.info("Saved Figure 2 analogue", path=str(fig_path))

    return metrics_per_round
