from __future__ import annotations

from llm.data.resume_templates import PROJECTS, SKILLS, Resume

# Maximum possible values per component — used for normalisation
_MAX_SKILLS: dict[str, int] = {domain: len(skills) for domain, skills in SKILLS.items()}
_MAX_PROJECTS: dict[str, int] = {domain: len(projects) for domain, projects in PROJECTS.items()}
_MAX_EXP: float = 10.0

# Component weights (must sum to 1.0)
_W_SKILLS = 0.50
_W_EXP = 0.35
_W_PROJECTS = 0.15


def compute_quality(resume: Resume) -> float:
    """Return latent quality score q ∈ [0, 1].

    Computed purely from resume content (skills, experience, projects).
    By construction, demographic_signal does not enter this function.
    """
    skill_score = len(resume.skills) / _MAX_SKILLS[resume.domain]
    exp_score = min(resume.years_exp, _MAX_EXP) / _MAX_EXP
    project_score = min(len(resume.projects), _MAX_PROJECTS[resume.domain]) / _MAX_PROJECTS[resume.domain]

    q = _W_SKILLS * skill_score + _W_EXP * exp_score + _W_PROJECTS * project_score
    return float(q)
