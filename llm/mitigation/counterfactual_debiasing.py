"""Counterfactual preference-data debiasing — IPW variant.

The intervention sits at the preference-data stage, before the RM sees it.
Each preference pair's Bradley-Terry loss is reweighted by 1 / ψ(x, y_w),
where ψ is the estimated propensity score — the probability the pair would be
observed under the biased collection process.

Under-represented Group B pairs (p_obs ≈ 0.4) get weight ≈ 2.5;
over-represented Group A pairs (p_obs ≈ 0.8) get weight ≈ 1.25.
This counteracts the missingness-induced imbalance without changing the
model architecture or the downstream policy training step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from trl import RewardTrainer

from llm.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Custom collator
# --------------------------------------------------------------------------- #


@dataclass
class _IPWCollator:
    """Wraps TRL's reward data collator to also pass through ipw_weight."""

    base_collator: Any

    def __call__(self, features: list[dict]) -> dict:
        weights = [float(f.pop("ipw_weight", 1.0)) for f in features]
        batch = self.base_collator(features)
        batch["ipw_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


# --------------------------------------------------------------------------- #
# IPW Reward Trainer
# --------------------------------------------------------------------------- #


class IPWRewardTrainer(RewardTrainer):
    """RewardTrainer with importance-weighted loss for counterfactual debiasing.

    Receives a precomputed list of IPW weights (one per training sample,
    aligned with the training dataset in insertion order). After TRL
    tokenises the dataset, the weights are added as an extra column and
    the data collator is wrapped to inject them into each batch.

    Args:
        ipw_weights: list of floats (length == len(train_dataset)).
            Computed externally by ``llm.mitigation.propensity.fit_and_weight``.
    """

    def __init__(
        self,
        *args: Any,
        ipw_weights: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if ipw_weights is not None and self.train_dataset is not None:
            n = len(self.train_dataset)
            if len(ipw_weights) < n:
                log.warning(
                    "ipw_weights shorter than dataset; padding with 1.0",
                    n_weights=len(ipw_weights),
                    n_dataset=n,
                )
                ipw_weights = list(ipw_weights) + [1.0] * (n - len(ipw_weights))

            weights_list = list(ipw_weights[:n])
            self.train_dataset = self.train_dataset.add_column("ipw_weight", weights_list)
            self.data_collator = _IPWCollator(self.data_collator)

            mean_w = sum(weights_list) / len(weights_list)
            log.info("IPW weighting active", n_samples=n, mean_weight=f"{mean_w:.3f}")

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        weights = inputs.pop("ipw_weight", None)

        inputs["use_cache"] = False
        outputs = model(**inputs)
        rewards = outputs.logits.squeeze(-1)  # (2B,) — chosen concat rejected
        rewards_chosen, rewards_rejected = torch.chunk(rewards, 2)

        per_sample_loss = -F.logsigmoid(rewards_chosen - rewards_rejected)  # (B,)

        if weights is not None:
            w = weights.to(per_sample_loss.device).float()
            # Normalise so mean weight = 1 (preserves loss scale)
            w = w / (w.mean() + 1e-8)
            loss = (per_sample_loss * w).mean()
        else:
            loss = per_sample_loss.mean()

        return (loss, outputs) if return_outputs else loss
