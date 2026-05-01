"""Reward model regularization mitigations.

Two regularizers:

1. L2 penalty on the reward head weights (separate from AdamW weight_decay so
   it can be targeted at the head alone).

2. Logit consistency: penalises high variance of RM scores within each
   demographic group in the batch.  Pushes the RM toward assigning similar
   margins to structurally identical pairs from different groups.  Requires
   a 'demographic_signal' tensor in the batch; silently skips if absent.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from trl import RewardTrainer

from llm.utils.logging import get_logger

log = get_logger(__name__)


class RMRegRewardTrainer(RewardTrainer):
    """RewardTrainer with L2 head regularisation and optional logit consistency.

    Args:
        l2_lambda: Weight for L2 penalty on the reward head (score) weight.
        consistency_lambda: Weight for within-group score variance penalty.
            Set to 0.0 to disable.
    """

    def __init__(
        self,
        *args: Any,
        l2_lambda: float = 1e-3,
        consistency_lambda: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.l2_lambda = l2_lambda
        self.consistency_lambda = consistency_lambda
        log.info(
            "RMRegRewardTrainer initialised",
            l2_lambda=l2_lambda,
            consistency_lambda=consistency_lambda,
        )

    def _reward_head_l2(self, model: nn.Module) -> torch.Tensor:
        """L2 norm of the reward head (score) weight matrix."""
        # The reward head in AutoModelForSequenceClassification is `model.score`.
        # Unwrap accelerator/PEFT wrappers if needed.
        base = getattr(model, "module", model)  # DataParallel / FSDP
        score_module = getattr(base, "score", None)
        if score_module is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        return sum(p.pow(2).sum() for p in score_module.parameters() if p.requires_grad)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        inputs["use_cache"] = False
        outputs = model(**inputs)

        rewards = outputs.logits.squeeze(-1)  # (2B,)
        rewards_chosen, rewards_rejected = torch.chunk(rewards, 2)

        if "margin" in inputs:
            base_loss = -torch.nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected - inputs["margin"]
            ).mean()
        else:
            base_loss = -torch.nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected
            ).mean()

        # L2 penalty on reward head
        l2_term = self.l2_lambda * self._reward_head_l2(model)

        # Logit consistency: penalise within-group score variance
        consistency_term = torch.tensor(0.0, device=base_loss.device)
        if self.consistency_lambda > 0 and "demographic_signal" in inputs:
            demo = inputs["demographic_signal"]  # (2B,) int tensor
            # Only look at chosen scores (first half)
            demo_chosen = demo[: len(rewards_chosen)]
            for grp_id in demo_chosen.unique():
                mask = demo_chosen == grp_id
                if mask.sum() > 1:
                    consistency_term = consistency_term + rewards_chosen[mask].var()
            consistency_term = self.consistency_lambda * consistency_term

        loss = base_loss + l2_term + consistency_term
        return (loss, outputs) if return_outputs else loss
