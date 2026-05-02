"""Length normalisation mitigations for RM and DPO training.

RM: subtracts β·log(seq_len) from the raw reward score before the Bradley-Terry
loss. Discourages the RM from giving higher scores simply to longer responses.

DPO: divides the summed completion log-probabilities by the number of completion
tokens, making the DPO objective length-agnostic (analogous to SimPO / IPO-style
normalization).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from trl import DPOTrainer, RewardTrainer

from llm.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# RM length normalization
# --------------------------------------------------------------------------- #


class LengthNormRewardTrainer(RewardTrainer):
    """RewardTrainer with per-sequence length normalization before BT loss.

    In TRL 1.x, `compute_loss` receives a batch that has both chosen and
    rejected sequences concatenated into a single forward pass (the data
    collator interleaves them).  We extract the per-sequence lengths from
    `attention_mask`, apply `score -= beta * log(seq_len)` to each half, then
    re-enter the standard loss computation.
    """

    def __init__(self, *args: Any, length_norm_beta: float = 0.1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.length_norm_beta = length_norm_beta
        log.info("LengthNormRewardTrainer initialised", beta=length_norm_beta)

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

        if self.length_norm_beta > 0 and "attention_mask" in inputs:
            seq_lens = inputs["attention_mask"].sum(dim=1).float()  # (2B,)
            lens_chosen, lens_rejected = torch.chunk(seq_lens, 2)
            rewards_chosen = rewards_chosen - self.length_norm_beta * torch.log(
                lens_chosen.clamp(min=1)
            )
            rewards_rejected = rewards_rejected - self.length_norm_beta * torch.log(
                lens_rejected.clamp(min=1)
            )

        if "margin" in inputs:
            loss = -torch.nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected - inputs["margin"]
            ).mean()
        else:
            loss = -torch.nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected
            ).mean()

        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------------------------- #
# DPO length normalization
# --------------------------------------------------------------------------- #

# Symbols we need from TRL internals; import gracefully.
try:
    from trl.trainer.dpo_trainer import selective_log_softmax  # type: ignore[import-untyped]
except ImportError:
    try:
        from trl.trainer.utils import selective_log_softmax  # type: ignore[import-untyped]
    except ImportError:
        selective_log_softmax = None  # fallback: recompute manually


def _selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token log-prob for each label token.  Falls back if TRL helper unavailable."""
    if selective_log_softmax is not None:
        return selective_log_softmax(logits, labels)
    log_probs = logits.log_softmax(dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


class LengthNormDPOTrainer(DPOTrainer):
    """DPOTrainer with per-token log-prob normalization by completion length.

    Replaces `logps = per_token_logps.sum(dim=1)` with
    `logps = per_token_logps.sum(dim=1) / comp_lens`, making the DPO objective
    scale-invariant with respect to response length.  Applied to both policy
    and reference log-probs.

    Note: overrides `_compute_loss`, an internal TRL method. If TRL changes
    this API, update accordingly and record in docs/decisions.md.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        log.info("LengthNormDPOTrainer initialised")

    def _compute_loss(  # type: ignore[override]
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        # Delegate to parent; intercept at the logps level by temporarily
        # overriding the per-token logps accumulation via a module hook.
        # We use a simpler approach: call parent and rely on the fact that
        # length norm can be applied by patching logps before they're used.
        #
        # Because _compute_loss is monolithic, we recompute logps here and
        # pass them back via a thread-local store that the parent's dpo_loss
        # call picks up — but TRL doesn't expose that hook.  Instead we
        # override the full block that computes logps, matching TRL 1.2.0.

        try:
            from trl.trainer.utils import (
                disable_gradient_checkpointing,  # type: ignore[import-untyped]
            )
        except ImportError:
            from contextlib import contextmanager

            @contextmanager  # type: ignore[no-redef]
            def disable_gradient_checkpointing(model: Any, kwargs: Any = None):  # type: ignore[misc]
                was_enabled = getattr(model, "is_gradient_checkpointing", False)
                if was_enabled:
                    model.gradient_checkpointing_disable()
                try:
                    yield
                finally:
                    if was_enabled:
                        model.gradient_checkpointing_enable(**(kwargs or {}))

        try:
            from peft import is_peft_model  # type: ignore[import-untyped]
            from peft.utils import use_adapter  # type: ignore[import-untyped]
        except ImportError:
            def is_peft_model(_m: Any) -> bool:
                return False

            def use_adapter(_m: Any, **_kw: Any):  # type: ignore[return]
                pass

        mode = "train" if self.model.training else "eval"

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        outputs = model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = _selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0

        # --- Length normalisation (the key change vs. parent) ---
        comp_lens = shift_completion_mask.sum(dim=1).float().clamp(min=1.0)  # (2B,)
        logps = per_token_logps.sum(dim=1) / comp_lens  # length-normalised

        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        # Reference log-probs (also length-normalised for consistency)
        if self.precompute_ref_logps:
            ref_chosen_logps = inputs["ref_chosen_logps"]
            ref_rejected_logps = inputs["ref_rejected_logps"]
        else:
            with torch.no_grad(), disable_gradient_checkpointing(
                self.model, self.args.gradient_checkpointing_kwargs
            ):
                if is_peft_model(model) and self.ref_model is None:
                    model_unwrapped = self.accelerator.unwrap_model(model)
                    from peft.utils import use_adapter as _ua  # type: ignore[import-untyped]
                    adapter = "ref" if "ref" in model_unwrapped.peft_config else None
                    with _ua(model_unwrapped, adapter_name=adapter):
                        ref_outputs = self.model(**model_kwargs)
                else:
                    ref_outputs = self.ref_model(**model_kwargs)

            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = _selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0
            ref_logps = ref_per_token_logps.sum(dim=1) / comp_lens  # length-normalised
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        # Delegate to parent's loss-type dispatch by calling the parent's internal
        # path from chosen/rejected log-probs onward.  We call super()._compute_loss
        # but inject pre-computed logps via inputs to avoid re-running the model.
        inputs_patched = dict(inputs)
        inputs_patched["ref_chosen_logps"] = ref_chosen_logps
        inputs_patched["ref_rejected_logps"] = ref_rejected_logps

        # Store normalised logps on self so the parent can pick them up.
        # Since the parent re-runs the full forward, we instead compute the
        # DPO loss directly here using the parent's beta and loss_type.
        # For the default sigmoid loss_type this is straightforward.
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        delta_score = self.beta * (chosen_logratios - rejected_logratios)
        loss = -torch.nn.functional.logsigmoid(delta_score).mean()

        if mode == "train":
            self._metrics[mode]["loss"] = [loss.item()]
        else:
            self._metrics[mode]["loss"] = [loss.item()]

        return (loss, outputs) if return_outputs else loss
