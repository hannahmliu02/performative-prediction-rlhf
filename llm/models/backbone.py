from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_backbone(
    model_name_or_path: str,
    *,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
) -> PreTrainedModel:
    torch_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch_dtype,
        device_map=device_map,
    )
