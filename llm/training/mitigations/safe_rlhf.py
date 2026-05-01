"""Simplified Safe-RLHF baseline.

Adapts the Safe-RLHF objective (Dai et al., 2023) to our bias-mitigation
setting.  In the original work, a separate safety RM penalises harmful outputs.
Here we use a *fairness penalty* instead: the reward is reduced when the RM
assigns systematically different scores to Group A vs. Group B responses within
the same batch.

    r_total = r_helpful − λ_safety · disparity_k

where `disparity_k = max(0, |mean_score_A − mean_score_B|)` is estimated from
the current batch.  When the batch contains fewer than 2 responses from either
group, the penalty term is 0 (no reliable estimate).

This is an outer-loop baseline, not composed with our method.  See
docs/decisions.md for the design rationale.
"""

from __future__ import annotations

import torch

from llm.models.reward_model import RewardModel
from llm.utils.logging import get_logger

log = get_logger(__name__)


def build_safe_reward_fn(
    rm_checkpoint: str,
    rm_dtype: str,
    rm_device_map: str | None,
    max_length: int,
    length_norm_beta: float,
    safety_lambda: float = 0.5,
):
    """Return a reward callable for RLOOTrainer with a fairness penalty.

    The callable signature matches what RLOOTrainer expects:
        (prompts, completions, **kwargs) -> list[float]

    kwargs may include 'demographic_signal' (list[str]) for the fairness penalty.
    When absent the penalty is skipped.

    Args:
        rm_checkpoint: Path to the reward model checkpoint.
        rm_dtype: torch dtype string (float32, bfloat16, …).
        rm_device_map: device_map for the RM, or None for CPU.
        max_length: Tokenizer max length.
        length_norm_beta: Length normalization beta (0 = disabled).
        safety_lambda: Fairness penalty weight λ.
    """
    rm = RewardModel.from_pretrained(rm_checkpoint, dtype=rm_dtype, device_map=rm_device_map)
    rm.eval()
    rm_device = next(rm.model.parameters()).device

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        **kwargs: object,
    ) -> list[float]:
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

        # Fairness penalty
        demo_signals: list[str] | None = kwargs.get("demographic_signal")  # type: ignore[assignment]
        if demo_signals is not None and safety_lambda > 0:
            rewards_np = rewards.float()
            a_mask = torch.tensor(
                [d == "A" for d in demo_signals], dtype=torch.bool, device=rewards.device
            )
            b_mask = ~a_mask
            if a_mask.sum() >= 2 and b_mask.sum() >= 2:
                mean_A = rewards_np[a_mask].mean()
                mean_B = rewards_np[b_mask].mean()
                disparity = (mean_A - mean_B).abs()
                penalty = safety_lambda * disparity
                rewards = rewards - penalty
                log.debug(
                    "Safe-RLHF fairness penalty",
                    disparity=f"{disparity:.4f}",
                    penalty=f"{penalty:.4f}",
                )

        return rewards.cpu().tolist()

    return reward_fn
