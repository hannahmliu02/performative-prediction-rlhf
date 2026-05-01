from __future__ import annotations

import pytest
import torch

from llm.models.backbone import load_backbone
from llm.models.policy import Policy
from llm.models.reward_model import RewardModel

MODEL = "gpt2"  # 124M; fastest on CPU — medium (355M) is slow without a GPU
MODEL_MEDIUM = "gpt2-medium"


@pytest.fixture(scope="module")
def gpt2_policy() -> Policy:
    return Policy.from_pretrained(MODEL, dtype="float32", device_map=None)


@pytest.fixture(scope="module")
def gpt2_rm() -> RewardModel:
    return RewardModel.from_pretrained(MODEL, dtype="float32", device_map=None)


def test_load_backbone() -> None:
    model = load_backbone(MODEL_MEDIUM, dtype="float32", device_map=None)
    assert model is not None
    assert hasattr(model.config, "hidden_size")


def test_policy_generate(gpt2_policy: Policy) -> None:
    outputs = gpt2_policy.generate(["Hello, my name is"], max_new_tokens=10)
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)
    assert len(outputs[0]) > 0


def test_policy_generate_batch(gpt2_policy: Policy) -> None:
    prompts = ["The sky is", "Machine learning is"]
    outputs = gpt2_policy.generate(prompts, max_new_tokens=8)
    assert len(outputs) == 2
    assert all(isinstance(s, str) for s in outputs)


def test_reward_model_score(gpt2_rm: RewardModel) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text = "Alice is a senior software engineer with 10 years of experience."
    enc = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        scores = gpt2_rm.score(enc["input_ids"], enc["attention_mask"])

    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()


def test_reward_model_score_batch(gpt2_rm: RewardModel) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [
        "Alice is a senior software engineer.",
        "Bob has two years of frontend experience.",
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        scores = gpt2_rm.score(enc["input_ids"], enc["attention_mask"])

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
