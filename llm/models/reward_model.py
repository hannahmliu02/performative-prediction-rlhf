from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from llm.models.backbone import _DTYPE_MAP
from llm.utils.device import move_model_to_device


class RewardModel(nn.Module):
    """AutoModelForSequenceClassification(num_labels=1) with a `score()` interface.

    Using AutoModelForSequenceClassification ensures checkpoints saved by
    trl.RewardTrainer are loadable via from_pretrained without architecture mismatch.
    """

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        dtype: str = "bfloat16",
        device_map: str | None = "auto",
    ) -> RewardModel:
        torch_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1,
            dtype=torch_dtype,
            device_map=device_map,
        )
        model = move_model_to_device(model, device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        return cls(model, tokenizer)

    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar reward per sequence in the batch. Shape: (batch,)."""
        device = next(self.model.parameters()).device
        outputs = self.model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        )
        return outputs.logits.squeeze(-1)
