"""General-capability benchmark: MT-Bench style evaluation with an LLM judge.

Sanity-checks that mitigation methods do not collapse general model capability.
Runs 8-category MT-Bench style questions through the policy and scores responses
with a configurable LLM judge (default: Claude Sonnet).

`judge_model: "none"` produces seeded deterministic scores without API calls —
use this for smoke tests and CI.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.models.policy import Policy
from llm.utils.logging import get_logger
from llm.utils.seeding import set_seed

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Built-in question set (2 per MT-Bench category — smoke / offline fallback)
# --------------------------------------------------------------------------- #

_BUILTIN_QUESTIONS: list[dict[str, str]] = [
    # writing
    {
        "id": "writing_1",
        "category": "writing",
        "prompt": "Write a short paragraph describing the feeling of waking up early on a clear autumn morning.",
    },
    {
        "id": "writing_2",
        "category": "writing",
        "prompt": "Write a brief professional email to a colleague apologising for missing a meeting and suggesting a reschedule.",
    },
    # roleplay
    {
        "id": "roleplay_1",
        "category": "roleplay",
        "prompt": "You are a friendly tour guide. Name and briefly describe three must-see attractions in Kyoto, Japan.",
    },
    {
        "id": "roleplay_2",
        "category": "roleplay",
        "prompt": "You are a patient cooking instructor. Explain in three steps how to make a basic tomato pasta sauce.",
    },
    # reasoning
    {
        "id": "reasoning_1",
        "category": "reasoning",
        "prompt": "All roses are flowers. Some flowers fade quickly. Does it follow that some roses fade quickly? Explain your answer.",
    },
    {
        "id": "reasoning_2",
        "category": "reasoning",
        "prompt": "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost? Show your reasoning.",
    },
    # math
    {
        "id": "math_1",
        "category": "math",
        "prompt": "Find the derivative of f(x) = 4x³ − 2x² + 7x − 3.",
    },
    {
        "id": "math_2",
        "category": "math",
        "prompt": "A right triangle has legs of length 6 and 8. What is the length of the hypotenuse and the area of the triangle?",
    },
    # coding
    {
        "id": "coding_1",
        "category": "coding",
        "prompt": "Write a Python function that takes a list of integers and returns a new list with duplicates removed, preserving order.",
    },
    {
        "id": "coding_2",
        "category": "coding",
        "prompt": "Explain binary search and implement it in Python as a function that returns the index of a target in a sorted list, or -1 if absent.",
    },
    # extraction
    {
        "id": "extraction_1",
        "category": "extraction",
        "prompt": (
            "Extract all person names from this text: "
            "'Dr. Sarah Chen presented her findings to James Okafor and Priya Sharma during the afternoon session.'"
        ),
    },
    {
        "id": "extraction_2",
        "category": "extraction",
        "prompt": (
            "From the following sentence, list every adjective: "
            "'The old, rusty gate creaked open to reveal a vast, overgrown garden.'"
        ),
    },
    # stem
    {
        "id": "stem_1",
        "category": "stem",
        "prompt": "Explain the process of photosynthesis in simple terms suitable for a high-school student.",
    },
    {
        "id": "stem_2",
        "category": "stem",
        "prompt": "What is the difference between mitosis and meiosis? When does each occur?",
    },
    # humanities
    {
        "id": "humanities_1",
        "category": "humanities",
        "prompt": "Identify and briefly explain two major causes of World War I.",
    },
    {
        "id": "humanities_2",
        "category": "humanities",
        "prompt": "What were the central ideas of the Enlightenment, and name two key thinkers associated with the movement?",
    },
]

# MT-Bench single-answer grading prompt template
_JUDGE_PROMPT_TMPL = """\
Please act as an impartial judge and evaluate the quality of the response provided by \
an AI assistant to the user question below. Consider helpfulness, relevance, accuracy, \
depth, and clarity. Begin with a short explanation, then rate the response on a scale \
of 1 to 10 using this exact format: "Rating: [[<score>]]"

[Question]
{question}

[Assistant's Answer]
{answer}
"""

_RATING_RE = re.compile(r"\[\[(\d+(?:\.\d+)?)\]\]")


# --------------------------------------------------------------------------- #
# Question loading
# --------------------------------------------------------------------------- #


def _load_questions(
    max_per_category: int,
    seed: int,
) -> list[dict[str, str]]:
    """Return MT-Bench questions, trying HuggingFace first then built-ins."""
    import random as _random

    rng = _random.Random(seed)

    try:
        import datasets as hf_datasets

        ds = hf_datasets.load_dataset(
            "lmsys/mt_bench_human_judgments", split="human", trust_remote_code=False
        )
        by_cat: dict[str, list[dict[str, str]]] = {}
        for row in ds:
            cat = str(row.get("category", "general"))
            # Extract the question from the first turn of conversation_a if present,
            # otherwise try standard column names.
            conv = row.get("conversation_a")
            if conv and isinstance(conv, list) and len(conv) > 0:
                prompt = str(conv[0].get("content", ""))
            else:
                prompt = str(row.get("question_1", row.get("prompt", row.get("text", ""))))
            prompt = prompt.strip()
            if not prompt:
                continue
            qid = str(row.get("question_id", hash(prompt)))
            by_cat.setdefault(cat, []).append({"id": qid, "category": cat, "prompt": prompt})

        questions: list[dict[str, str]] = []
        for cat_qs in by_cat.values():
            rng.shuffle(cat_qs)
            questions.extend(cat_qs[:max_per_category])

        if not questions:
            raise ValueError("No valid questions extracted from HuggingFace dataset")

        log.info("Loaded MT-Bench questions from HuggingFace", n=len(questions))
        return questions

    except Exception as exc:
        log.warning(
            "Failed to load MT-Bench from HuggingFace; using built-in questions",
            error=str(exc),
        )
        by_cat_builtin: dict[str, list[dict[str, str]]] = {}
        for q in _BUILTIN_QUESTIONS:
            by_cat_builtin.setdefault(q["category"], []).append(q)
        questions = []
        for cat_qs in by_cat_builtin.values():
            questions.extend(cat_qs[:max_per_category])
        rng.shuffle(questions)
        return questions


# --------------------------------------------------------------------------- #
# Judging
# --------------------------------------------------------------------------- #


def _judge_none(question: str, answer: str, question_id: str, seed: int) -> float:
    """Deterministic score in [1, 10] without any API call — for smoke tests."""
    h = int(hashlib.sha256(f"{question_id}:{seed}".encode()).hexdigest(), 16)
    return float((h % 10) + 1)


def _judge_anthropic(question: str, answer: str, model: str) -> float | None:
    """Call the Anthropic Messages API and parse the [[rating]] from the response."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        prompt = _JUDGE_PROMPT_TMPL.format(question=question, answer=answer)
        message = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text
        match = _RATING_RE.search(text)
        if match:
            return float(match.group(1))
        log.warning("Judge response did not contain a [[rating]]", response=text[:200])
        return None
    except Exception as exc:
        log.warning("Judge API call failed", error=str(exc))
        return None


def _score_response(
    question: str,
    answer: str,
    question_id: str,
    judge_model: str,
    seed: int,
) -> float:
    """Return a 1–10 score for an (question, answer) pair."""
    if judge_model == "none":
        return _judge_none(question, answer, question_id, seed)
    score = _judge_anthropic(question, answer, judge_model)
    if score is None:
        log.warning("Falling back to deterministic score", question_id=question_id)
        return _judge_none(question, answer, question_id, seed)
    return score


# --------------------------------------------------------------------------- #
# Caching helpers
# --------------------------------------------------------------------------- #


def _model_hash(checkpoint_path: str) -> str:
    config_file = pathlib.Path(checkpoint_path) / "config.json"
    if config_file.exists():
        return hashlib.sha256(config_file.read_bytes()).hexdigest()[:8]
    return hashlib.sha256(checkpoint_path.encode()).hexdigest()[:8]


def _cache_path(output_dir: pathlib.Path, experiment_name: str, seed: int, checkpoint: str) -> pathlib.Path:
    h = _model_hash(checkpoint)
    cache_dir = output_dir / experiment_name / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"gc_{seed}_{h}.json"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run_general_capability(cfg: DictConfig) -> dict[str, Any]:
    """Run the general-capability benchmark and return aggregated scores.

    Args:
        cfg: OmegaConf config matching llm/configs/eval/general_capability/*.yaml.

    Returns:
        Dict with overall_score, per_category scores, and per-question details.
    """
    set_seed(cfg.seed, deterministic=False)

    output_dir = pathlib.Path(cfg.output_dir)
    result_dir = output_dir / cfg.experiment_name
    result_dir.mkdir(parents=True, exist_ok=True)

    cache_file = _cache_path(output_dir, cfg.experiment_name, cfg.seed, cfg.model_checkpoint)
    if cache_file.exists():
        log.info("Cache hit — loading cached results", cache=str(cache_file))
        cached = json.loads(cache_file.read_text())
        result_path = result_dir / "general_capability.json"
        result_path.write_text(json.dumps(cached, indent=2))
        return cached

    from llm.models.backbone import _DTYPE_MAP

    torch_dtype = _DTYPE_MAP.get(cfg.dtype, torch.float32)
    device_map: str | None = cfg.get("device_map") or None
    if device_map == "null":
        device_map = None

    log.info("Loading policy", checkpoint=cfg.model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_checkpoint, dtype=torch_dtype, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    policy = Policy(model, tokenizer)

    questions = _load_questions(cfg.max_per_category, cfg.seed)
    log.info(
        "Loaded questions",
        n=len(questions),
        judge=cfg.judge_model,
        max_per_category=cfg.max_per_category,
    )

    # Generate all responses in one batched call per question (single-turn)
    prompts = [q["prompt"] for q in questions]
    log.info("Generating responses", n=len(prompts))
    responses: list[str] = []
    for i in range(0, len(prompts), cfg.batch_size):
        batch = prompts[i : i + cfg.batch_size]
        responses.extend(policy.generate(batch, max_new_tokens=cfg.max_new_tokens))

    # Score each response
    per_question: list[dict[str, Any]] = []
    for q, resp in zip(questions, responses):
        score = _score_response(q["prompt"], resp, q["id"], cfg.judge_model, cfg.seed)
        per_question.append(
            {"id": q["id"], "category": q["category"], "score": score, "response_snippet": resp[:120]}
        )
        log.info("Scored", id=q["id"], category=q["category"], score=f"{score:.1f}")

    # Aggregate
    from collections import defaultdict

    cat_scores: dict[str, list[float]] = defaultdict(list)
    for item in per_question:
        cat_scores[item["category"]].append(item["score"])

    per_category = {cat: float(sum(s) / len(s)) for cat, s in cat_scores.items()}
    all_scores = [item["score"] for item in per_question]
    overall = float(sum(all_scores) / len(all_scores)) if all_scores else 0.0

    log.info("Overall score", score=f"{overall:.2f}", n_questions=len(per_question))
    for cat, s in sorted(per_category.items()):
        log.info("Category score", category=cat, score=f"{s:.2f}")

    results: dict[str, Any] = {
        "model_checkpoint": cfg.model_checkpoint,
        "judge_model": cfg.judge_model,
        "seed": cfg.seed,
        "overall_score": overall,
        "per_category": per_category,
        "n_questions": len(per_question),
        "per_question": per_question,
    }

    cache_file.write_text(json.dumps(results, indent=2))

    result_path = result_dir / "general_capability.json"
    result_path.write_text(json.dumps(results, indent=2))
    log.info("Saved general capability results", path=str(result_path))

    return results
