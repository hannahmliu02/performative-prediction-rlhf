"""Semi-synthetic resume preference dataset generator.

Entry point:
    uv run python -m llm.data.generate_resume_prefs \\
        --config llm/configs/data/semi_synthetic.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
from omegaconf import DictConfig, OmegaConf

from llm.data.llm_annotator import (
    build_llm_client,
    compare_candidates_by_name,
    compute_annotator_bias,
    generate_candidate_summary,
    generate_mediocre_summary,
    score_summary_specificity,
)
from llm.data.quality_score import compute_quality
from llm.data.resume_templates import Resume, build_resume, enumerate_cells
from llm.utils.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Summary construction
# --------------------------------------------------------------------------- #

_BAD_TEMPLATES = [
    (
        "This candidate has a background in software development and has worked on various "
        "technical projects. They have some relevant skills and background. They may be suitable "
        "for the {role} role, though their specific qualifications are difficult to assess."
    ),
    (
        "The applicant shows general technical competency and has experience in technology roles. "
        "They have completed projects in their area and have acquired relevant skills. "
        "Their background could potentially fit the requirements for a {role} position."
    ),
]


def _good_summary(resume: Resume, role: str) -> str:
    first_name = resume.name.split()[0]
    skills_str = ", ".join(resume.skills)
    projects_str = "; ".join(resume.projects)
    return (
        f"{resume.name} is a {resume.seniority} {resume.domain} engineer with "
        f"{resume.years_exp} year(s) of experience. "
        f"Their technical expertise includes {skills_str}. "
        f"Notable accomplishments: {projects_str}. "
        f"{first_name} demonstrates strong qualifications for a {role} position."
    )


def _bad_summary(role: str, *, rng: random.Random) -> str:
    template = rng.choice(_BAD_TEMPLATES)
    return template.format(role=role)


# --------------------------------------------------------------------------- #
# Per-cell observation probability
# --------------------------------------------------------------------------- #


def _cell_obs_prob(cell: dict[str, str], obs_probs: DictConfig) -> float:
    """Compute p_obs for a cell as the product of per-axis probabilities."""
    p = 1.0
    for axis, value in cell.items():
        if axis in obs_probs:
            p *= obs_probs[axis][value]
    return p


# --------------------------------------------------------------------------- #
# Core generation
# --------------------------------------------------------------------------- #


def _make_prompt(resume: Resume, role: str) -> str:
    return f"Summarize this candidate's profile for a {role} role: {resume.to_text()}"


def generate_pairs(cfg: DictConfig) -> tuple[list[dict], list[dict]]:
    """Generate all preference pairs; return (train_rows, audit_rows).

    train_rows: biased by obs_probs (missingness injected).
    audit_rows: uniform across all cells.

    When ``cfg.use_llm_annotator`` is True the LLM annotation step is
    parallelised with ``cfg.annotator_workers`` threads (default 4), so
    Ollama serves multiple requests concurrently.  Set the environment variable
    ``OLLAMA_NUM_PARALLEL=<workers>`` to match before starting Ollama.
    """
    rng = random.Random(cfg.seed)
    axes: dict[str, list[str]] = OmegaConf.to_container(cfg.axes, resolve=True)  # type: ignore[assignment]
    cells = enumerate_cells(axes)
    roles: list[str] = list(cfg.roles)

    use_llm = bool(cfg.get("use_llm_annotator", False))
    llm_client: object | None = None
    annotator_model: str | None = None
    n_workers = int(cfg.get("annotator_workers", 1))
    if use_llm:
        llm_client, annotator_model = build_llm_client(
            backend=cfg.get("annotator_backend", "anthropic"),
            model=cfg.get("annotator_model"),
            ollama_host=cfg.get("ollama_host", "http://localhost:11434"),
        )
        log.info(
            "LLM annotator enabled",
            model=annotator_model,
            n_cells=len(cells),
            workers=n_workers,
        )

    # ── Phase 1: build all resume skeletons deterministically ────────────────
    # rng must stay single-threaded; only the LLM calls are parallelised.
    skeletons: list[dict] = []
    for cell in cells:
        p_obs = _cell_obs_prob(cell, cfg.obs_probs)
        for role in roles:
            for _ in range(cfg.n_per_cell):
                resume = build_resume(
                    demographic_signal=cell["demographic_signal"],
                    seniority=cell["seniority"],
                    domain=cell["domain"],
                    rng=rng,
                )
                skeletons.append({
                    "resume": resume,
                    "role": role,
                    "cell": cell,
                    "p_obs": p_obs,
                    "bad": _bad_summary(role, rng=rng),
                    "obs_flip": rng.random(),   # pre-draw for missingness
                    "noise_flip": rng.random(),  # pre-draw for ε-noise
                })

    total = len(skeletons)

    # ── Phase 2: annotate (parallel for LLM, serial for rule-based) ──────────
    def _annotate(sk: dict) -> dict:
        resume = sk["resume"]
        role = sk["role"]
        bad = sk["bad"]

        if use_llm and llm_client is not None:
            good = generate_candidate_summary(resume, role, llm_client, model=annotator_model)
            mediocre = generate_mediocre_summary(resume, role, llm_client, model=annotator_model)
            specificity = score_summary_specificity(good, resume)
            chosen, rejected, chosen_is_good = good, mediocre, True
        else:
            good = _good_summary(resume, role)
            specificity = score_summary_specificity(good, resume)
            if sk["noise_flip"] < cfg.noise_epsilon:
                chosen, rejected, chosen_is_good = bad, good, False
            else:
                chosen, rejected, chosen_is_good = good, bad, True

        return {
            "prompt": _make_prompt(resume, role),
            "chosen": chosen,
            "rejected": rejected,
            "demographic_signal": sk["cell"]["demographic_signal"],
            "seniority": sk["cell"]["seniority"],
            "domain": sk["cell"]["domain"],
            "role": role,
            "quality_score": compute_quality(resume),
            "chosen_is_good": chosen_is_good,
            "summary_specificity": specificity,
            "p_obs": sk["p_obs"],
            "obs_flip": sk["obs_flip"],
        }

    annotated: list[dict] = [{}] * total

    if use_llm and n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_annotate, sk): i for i, sk in enumerate(skeletons)}
            n_done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                annotated[idx] = fut.result()
                n_done += 1
                if n_done % 100 == 0:
                    log.info(
                        "Data generation progress",
                        done=n_done,
                        total=total,
                        pct=f"{100 * n_done / total:.0f}%",
                    )
    else:
        for i, sk in enumerate(skeletons):
            annotated[i] = _annotate(sk)
            if use_llm and (i + 1) % 100 == 0:
                log.info(
                    "Data generation progress",
                    done=i + 1,
                    total=total,
                    pct=f"{100 * (i + 1) / total:.0f}%",
                )

    # ── Phase 3: split into audit / train ────────────────────────────────────
    audit_rows: list[dict] = []
    train_rows: list[dict] = []
    for row in annotated:
        p_obs = row.pop("p_obs")
        obs_flip = row.pop("obs_flip")
        audit_rows.append(row)
        if obs_flip < p_obs:
            train_rows.append(row)

    if use_llm:
        bias = compute_annotator_bias(audit_rows)
        log.info(
            "LLM annotator bias diagnostic",
            mean_specificity_A=f"{bias['mean_specificity_A']:.3f}",
            mean_specificity_B=f"{bias['mean_specificity_B']:.3f}",
            specificity_gap=f"{bias['specificity_gap']:.3f}",
        )

    return train_rows, audit_rows


# --------------------------------------------------------------------------- #
# Cross-group name-bias generation
# --------------------------------------------------------------------------- #


def generate_cross_group_name_bias_pairs(cfg: DictConfig) -> tuple[list[dict], list[dict]]:
    """Generate preference pairs where the LLM chooses based on candidate names only.

    Each pair contains one Group-A resume and one Group-B resume with the same
    seniority / domain.  The LLM is shown only the two names and must pick one;
    any systematic preference is therefore driven by name-based priors (demographic
    stereotypes), not content quality.

    ``demographic_signal`` is set to the *winner's* group so that ``obs_probs``
    naturally overweights A-wins (p_obs=0.8) and underweights B-wins (p_obs=0.4).
    This models a world where Group-A candidate outcomes are observed / acted upon
    more often than Group-B outcomes — the informative-missingness mechanism.
    """
    rng = random.Random(cfg.seed)
    axes: dict[str, list[str]] = OmegaConf.to_container(cfg.axes, resolve=True)  # type: ignore[assignment]
    obs_probs: dict[str, float] = OmegaConf.to_container(cfg.obs_probs.demographic_signal, resolve=True)  # type: ignore[assignment]

    # Iterate over axes that are NOT demographic_signal (each pair spans both groups)
    non_demo_axes = {k: v for k, v in axes.items() if k != "demographic_signal"}
    cells = enumerate_cells(non_demo_axes)
    roles: list[str] = list(cfg.roles)

    llm_client, annotator_model = build_llm_client(
        backend=cfg.get("annotator_backend", "anthropic"),
        model=cfg.get("annotator_model"),
        ollama_host=cfg.get("ollama_host", "http://localhost:11434"),
    )
    n_workers = int(cfg.get("annotator_workers", 4))
    log.info(
        "Cross-group name-bias generation",
        model=annotator_model,
        n_cells=len(cells),
        workers=n_workers,
    )

    # ── Phase 1: build paired resume skeletons ────────────────────────────────
    skeletons: list[dict] = []
    for cell in cells:
        for role in roles:
            for _ in range(cfg.n_per_cell):
                resume_a = build_resume(
                    demographic_signal="A",
                    seniority=cell["seniority"],
                    domain=cell["domain"],
                    rng=rng,
                )
                resume_b = build_resume(
                    demographic_signal="B",
                    seniority=cell["seniority"],
                    domain=cell["domain"],
                    rng=rng,
                )
                skeletons.append({
                    "resume_a": resume_a,
                    "resume_b": resume_b,
                    "role": role,
                    "cell": cell,
                    "obs_flip": rng.random(),
                })

    total = len(skeletons)

    # ── Phase 2: annotate pairs (parallel) ───────────────────────────────────
    def _annotate(sk: dict) -> dict:
        resume_a: Resume = sk["resume_a"]
        resume_b: Resume = sk["resume_b"]
        role: str = sk["role"]

        # Name-only preference judgment — the LLM can only use the name
        winner = compare_candidates_by_name(
            resume_a.name, resume_b.name, role, llm_client, model=annotator_model
        )

        # Generate full summaries for the RM to train on (content-level signal)
        summary_a = generate_candidate_summary(resume_a, role, llm_client, model=annotator_model)
        summary_b = generate_candidate_summary(resume_b, role, llm_client, model=annotator_model)

        quality_a = compute_quality(resume_a)
        quality_b = compute_quality(resume_b)

        if winner == "A":
            chosen, rejected = summary_a, summary_b
            chosen_name, rejected_name = resume_a.name, resume_b.name
            quality_chosen, quality_rejected = quality_a, quality_b
            demo_winner = "A"
        else:
            chosen, rejected = summary_b, summary_a
            chosen_name, rejected_name = resume_b.name, resume_a.name
            quality_chosen, quality_rejected = quality_b, quality_a
            demo_winner = "B"

        p_obs = obs_probs.get(demo_winner, 0.5)
        chosen_is_good = quality_chosen >= quality_rejected

        return {
            # Standard RM training columns
            "prompt": (
                f"For a {role} position, review two candidate profiles "
                f"and select the stronger candidate."
            ),
            "chosen": chosen,
            "rejected": rejected,
            # Schema-compatible metadata (demographic_signal = winner's group)
            "demographic_signal": demo_winner,
            "seniority": sk["cell"]["seniority"],
            "domain": sk["cell"]["domain"],
            "role": role,
            "quality_score": quality_chosen,
            # Extras for analysis (ignored by downstream training code)
            "quality_score_rejected": quality_rejected,
            "chosen_is_good": chosen_is_good,
            "chosen_name": chosen_name,
            "rejected_name": rejected_name,
            "llm_winner": winner,
            "summary_specificity": score_summary_specificity(chosen, resume_a if winner == "A" else resume_b),
            "p_obs": p_obs,
            "obs_flip": sk["obs_flip"],
        }

    annotated: list[dict] = [{}] * total
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_annotate, sk): i for i, sk in enumerate(skeletons)}
        n_done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            annotated[idx] = fut.result()
            n_done += 1
            if n_done % 50 == 0:
                log.info("Name-bias generation progress", done=n_done, total=total)

    # ── Phase 3: split audit / train ─────────────────────────────────────────
    audit_rows: list[dict] = []
    train_rows: list[dict] = []
    for row in annotated:
        p_obs = row.pop("p_obs")
        obs_flip = row.pop("obs_flip")
        audit_rows.append(row)
        if obs_flip < p_obs:
            train_rows.append(row)

    # Log bias diagnostic
    a_wins = sum(1 for r in audit_rows if r["demographic_signal"] == "A")
    b_wins = sum(1 for r in audit_rows if r["demographic_signal"] == "B")
    total_audit = len(audit_rows)
    log.info(
        "Name-bias annotation diagnostic",
        A_wins=a_wins,
        B_wins=b_wins,
        A_win_rate=f"{a_wins / total_audit:.3f}" if total_audit else "N/A",
        chosen_is_good_rate=f"{sum(r['chosen_is_good'] for r in audit_rows) / total_audit:.3f}" if total_audit else "N/A",
    )

    return train_rows, audit_rows


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(cfg: DictConfig) -> tuple[datasets.Dataset, datasets.Dataset]:
    log.info("Starting data generation", n_per_cell=cfg.n_per_cell, seed=cfg.seed)
    pairing_mode = cfg.get("pairing_mode", "standard")
    if pairing_mode == "cross_group_name_bias":
        train_rows, audit_rows = generate_cross_group_name_bias_pairs(cfg)
    else:
        train_rows, audit_rows = generate_pairs(cfg)

    preference_train = datasets.Dataset.from_list(train_rows)
    preference_audit = datasets.Dataset.from_list(audit_rows)

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preference_train.to_parquet(str(out_dir / "train.parquet"))
    preference_audit.to_parquet(str(out_dir / "audit.parquet"))

    log.info(
        "Done",
        train_size=len(preference_train),
        audit_size=len(preference_audit),
        output_dir=str(out_dir),
    )
    return preference_train, preference_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate semi-synthetic resume preference data.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
