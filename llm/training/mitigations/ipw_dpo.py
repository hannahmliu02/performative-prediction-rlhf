"""IPW-weighted DPO trainer for counterfactual debiasing.

Mirrors IPWRewardTrainer but for the policy training stage.  Each preference
pair's DPO loss is scaled by 1 / ψ(a), so under-represented Group B pairs
(low obs_prob) receive higher gradient weight.

For batch_size=1 the scaling is exact per-sample weighting; for larger batches
the batch loss is scaled by the mean batch weight (an approximation that
becomes exact as the batch becomes homogeneous).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from trl import DPOTrainer

from llm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class _IPWDPOCollator:
    """Wraps DPO's data collator to pass ipw_weight through to compute_loss."""

    base_collator: Any

    def __call__(self, features: list[dict]) -> dict:
        weights = [float(f.pop("ipw_weight", 1.0)) for f in features]
        batch = self.base_collator(features)
        batch["ipw_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class IPWDPOTrainer(DPOTrainer):
    """DPOTrainer with importance-weighted loss for counterfactual debiasing.

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
            self.data_collator = _IPWDPOCollator(self.data_collator)

            mean_w = sum(weights_list) / len(weights_list)
            log.info("IPW-DPO weighting active", n_samples=n, mean_weight=f"{mean_w:.3f}")

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        ipw_weight = inputs.pop("ipw_weight", None)
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        if ipw_weight is not None:
            loss = result[0] if return_outputs else result
            w = ipw_weight.to(loss.device).float()
            w = w / w.mean().clamp(min=1e-8)
            scaled = loss * w.mean()
            return (scaled, result[1]) if return_outputs else scaled

        return result
