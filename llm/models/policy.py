from __future__ import annotations

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from llm.models.backbone import load_backbone


class Policy:
    """Thin generation wrapper around a causal LM backbone."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        # Left-padding required for batched generation with decoder-only models.
        self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        dtype: str = "bfloat16",
        device_map: str | None = "auto",
    ) -> Policy:
        model = load_backbone(model_name_or_path, dtype=dtype, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return cls(model, tokenizer)

    def generate(self, prompts: list[str], **kwargs: object) -> list[str]:
        """Generate responses for a list of prompt strings."""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        # Return only the newly generated tokens, not the prompt
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
