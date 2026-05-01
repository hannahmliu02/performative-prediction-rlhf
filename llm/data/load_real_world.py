"""Real-world preference data loader for Section 5.2.

Downloads Anthropic/hh-rlhf from Hugging Face and assigns each conversation
to a demographic proxy group based on topic keywords:

  Group A (technical)  — coding, maths, factual lookup, science.
                         Well-represented in most RLHF datasets; annotators
                         find these easier to judge.

  Group B (personal)   — relationships, emotions, mental health, identity.
                         Systematically underrepresented; annotators often
                         skip or disagree, so fewer clean preference labels
                         survive quality filters.

The resulting parquet files have the same schema as the synthetic datasets
(prompt, chosen, rejected, demographic_signal, …) so the existing feedback
loop, RM trainer, and IPW mitigation all work without modification.

Usage:
    uv run python -m llm.data.load_real_world \
        --config llm/configs/data/real_world_hh_rlhf.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import re

import datasets
from omegaconf import OmegaConf

from llm.utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Keyword-based topic classifier
# --------------------------------------------------------------------------- #

_TECHNICAL_PATTERNS = re.compile(
    r"\b("
    r"code|coding|program|script|function|algorithm|bug|debug|error|syntax|"
    r"python|javascript|java|c\+\+|rust|golang|typescript|sql|bash|"
    r"math|calcul|equation|formula|integral|derivative|statistics|"
    r"science|physics|chemistry|biology|theorem|proof|"
    r"api|database|server|network|cloud|docker|kubernetes|git|"
    r"machine learning|neural network|model|dataset|train|"
    r"explain how|how does|what is the difference|step.by.step"
    r")\b",
    re.IGNORECASE,
)

_PERSONAL_PATTERNS = re.compile(
    r"\b("
    r"relationship|partner|boyfriend|girlfriend|husband|wife|marriage|divorce|"
    r"friend|family|parent|mother|father|sibling|child|"
    r"feel|feeling|emotion|sad|happy|anxious|depress|lonely|angry|upset|hurt|"
    r"mental health|therapy|therapist|counseling|trauma|grief|"
    r"identity|gender|sexuality|race|discrimination|belong|"
    r"advice|should i|what should|am i|help me|i don.t know|i feel|"
    r"breakup|conflict|argument|forgive|trust|love|hate"
    r")\b",
    re.IGNORECASE,
)


def classify_prompt(prompt: str) -> str | None:
    """Return 'A' (technical), 'B' (personal), or None (ambiguous/skip)."""
    tech_hits = len(_TECHNICAL_PATTERNS.findall(prompt))
    personal_hits = len(_PERSONAL_PATTERNS.findall(prompt))

    if tech_hits > personal_hits and tech_hits >= 2:
        return "A"
    if personal_hits > tech_hits and personal_hits >= 2:
        return "B"
    return None  # ambiguous — will be filtered out


# --------------------------------------------------------------------------- #
# HH-RLHF parsing
# --------------------------------------------------------------------------- #

def _extract_last_human_turn(text: str) -> str:
    """Return the final Human: turn from an HH-RLHF dialogue string."""
    parts = text.split("\n\nHuman:")
    if len(parts) > 1:
        last = parts[-1].split("\n\nAssistant:")[0].strip()
        return last
    return text.strip()


def _extract_assistant_response(text: str) -> str:
    """Return the final Assistant: turn from an HH-RLHF dialogue string."""
    parts = text.split("\n\nAssistant:")
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()


def load_hh_rlhf(cfg) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Download HH-RLHF, classify by topic, split into train/audit parquets.

    Returns (train_dataset, audit_dataset) with the same schema as synthetic data.
    """
    split = cfg.get("hh_split", "train")
    max_rows = cfg.get("max_rows", None)
    min_response_len = int(cfg.get("min_response_len", 20))

    log.info("Downloading hh-rlhf", split=split)
    raw = datasets.load_dataset("Anthropic/hh-rlhf", split=split)
    if max_rows:
        raw = raw.select(range(min(max_rows, len(raw))))

    log.info("Classifying prompts by topic", n=len(raw))
    rows: list[dict] = []
    skipped_ambiguous = 0
    skipped_short = 0

    for example in raw:
        chosen_text: str = example["chosen"]
        rejected_text: str = example["rejected"]

        prompt = _extract_last_human_turn(chosen_text)
        chosen_resp = _extract_assistant_response(chosen_text)
        rejected_resp = _extract_assistant_response(rejected_text)

        # Basic quality filter
        if len(chosen_resp) < min_response_len or len(rejected_resp) < min_response_len:
            skipped_short += 1
            continue

        group = classify_prompt(prompt)
        if group is None:
            skipped_ambiguous += 1
            continue

        rows.append({
            "prompt": prompt,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
            "demographic_signal": group,
            # Placeholder metadata (no ground-truth quality score for real data)
            "seniority": "unknown",
            "domain": "unknown",
            "role": "general",
            "quality_score": 0.5,
            "chosen_is_good": True,
            "summary_specificity": 0.0,
        })

    log.info(
        "Classification complete",
        classified=len(rows),
        skipped_ambiguous=skipped_ambiguous,
        skipped_short=skipped_short,
        group_A=sum(1 for r in rows if r["demographic_signal"] == "A"),
        group_B=sum(1 for r in rows if r["demographic_signal"] == "B"),
    )

    # ── Apply obs_probs missingness ───────────────────────────────────────────
    import random
    rng = random.Random(cfg.get("seed", 42))
    obs_probs: dict[str, float] = OmegaConf.to_container(
        cfg.obs_probs.demographic_signal, resolve=True
    )  # type: ignore[assignment]

    # Empirical coverage gap: log the raw ratio before subsampling
    n_a = sum(1 for r in rows if r["demographic_signal"] == "A")
    n_b = sum(1 for r in rows if r["demographic_signal"] == "B")
    total = len(rows)
    log.info(
        "Natural coverage (before obs_probs subsampling)",
        raw_A_fraction=f"{n_a / total:.3f}" if total else "N/A",
        raw_B_fraction=f"{n_b / total:.3f}" if total else "N/A",
    )

    audit_rows = list(rows)
    train_rows = [
        r for r in rows if rng.random() < obs_probs.get(r["demographic_signal"], 0.5)
    ]

    log.info(
        "Train/audit split",
        train=len(train_rows),
        audit=len(audit_rows),
        train_A=sum(1 for r in train_rows if r["demographic_signal"] == "A"),
        train_B=sum(1 for r in train_rows if r["demographic_signal"] == "B"),
    )

    return datasets.Dataset.from_list(train_rows), datasets.Dataset.from_list(audit_rows)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and preprocess HH-RLHF for real-world eval.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    train_ds, audit_ds = load_hh_rlhf(cfg)

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds.to_parquet(str(out_dir / "train.parquet"))
    audit_ds.to_parquet(str(out_dir / "audit.parquet"))
    log.info("Saved", train=len(train_ds), audit=len(audit_ds), output_dir=str(out_dir))


if __name__ == "__main__":
    main()
