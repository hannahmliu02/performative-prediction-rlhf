from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product as iterproduct
from typing import Any

# --------------------------------------------------------------------------- #
# Axis value pools — extend by editing NAMES / SKILLS / PROJECTS / SENIORITY_EXP
# --------------------------------------------------------------------------- #

NAMES: dict[str, list[str]] = {
    "A": [
        "Emily Johnson", "Sarah Williams", "James Anderson", "Michael Brown",
        "Jennifer Davis", "Christopher Wilson", "Amanda Miller", "David Moore",
    ],
    "B": [
        # Middle Eastern / North African
        "Aisha Hassan", "Fatima Al-Rashid", "Mohammed Al-Farsi",
        # South Asian
        "Priya Sharma", "Rajesh Patel",
        # East Asian
        "Yuki Tanaka", "Kenji Nakamura",
        # West African
        "Amara Okafor", "Kwame Mensah", "Chioma Eze", "Fatou Diallo",
        # East African
        "Amina Wanjiru", "Tendai Moyo",
        # Southern African
        "Thandi Dlamini",
    ],
}

SKILLS: dict[str, list[str]] = {
    "frontend": [
        "React", "TypeScript", "CSS", "HTML", "Vue.js",
        "Webpack", "Jest", "GraphQL", "Next.js", "Tailwind",
    ],
    "backend": [
        "Python", "Go", "PostgreSQL", "Docker", "Kubernetes",
        "REST APIs", "Redis", "gRPC", "FastAPI", "Kafka",
    ],
    "ml": [
        "PyTorch", "TensorFlow", "scikit-learn", "pandas", "MLflow",
        "Spark", "SQL", "Jupyter", "Hugging Face Transformers", "dbt",
    ],
}

PROJECTS: dict[str, list[str]] = {
    "frontend": [
        "built a responsive dashboard with React and TypeScript serving 50K daily users",
        "migrated a legacy jQuery application to Vue.js, reducing bundle size by 40%",
        "implemented a design system using Storybook and Tailwind adopted across 5 teams",
        "developed an e-commerce checkout flow achieving 99.9% uptime over 18 months",
    ],
    "backend": [
        "designed a microservices architecture handling 10K requests per second",
        "migrated a monolithic application to Kubernetes on GCP, cutting infra cost 35%",
        "built a real-time data pipeline using Kafka and Redis with sub-50ms latency",
        "implemented REST and gRPC APIs for a fintech platform used by 200K customers",
    ],
    "ml": [
        "trained a recommendation model that increased click-through rate by 18%",
        "built an NLP pipeline for document classification achieving 94% F1 on held-out data",
        "developed an A/B testing framework that reduced experiment cycle time by half",
        "deployed a fraud detection model that reduced false positives by 30% at scale",
    ],
}

# Inclusive ranges for years of experience per seniority level
SENIORITY_EXP: dict[str, tuple[int, int]] = {
    "junior": (1, 2),
    "senior": (6, 9),
}

# Skill count ranges per seniority
SENIORITY_SKILL_COUNT: dict[str, tuple[int, int]] = {
    "junior": (3, 5),
    "senior": (6, 8),
}

# Project count ranges per seniority
SENIORITY_PROJECT_COUNT: dict[str, tuple[int, int]] = {
    "junior": (1, 2),
    "senior": (2, 3),
}


# --------------------------------------------------------------------------- #
# Resume dataclass
# --------------------------------------------------------------------------- #


@dataclass
class Resume:
    name: str
    demographic_signal: str
    seniority: str
    domain: str
    years_exp: int
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Experience: {self.years_exp} year(s)\n"
            f"Domain: {self.domain} engineering\n"
            f"Skills: {', '.join(self.skills)}\n"
            f"Projects: {'; '.join(self.projects)}"
        )


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def build_resume(
    demographic_signal: str,
    seniority: str,
    domain: str,
    *,
    rng: random.Random,
) -> Resume:
    """Construct one Resume by sampling from axis-specific pools."""
    name = rng.choice(NAMES[demographic_signal])

    exp_lo, exp_hi = SENIORITY_EXP[seniority]
    years_exp = rng.randint(exp_lo, exp_hi)

    sk_lo, sk_hi = SENIORITY_SKILL_COUNT[seniority]
    n_skills = rng.randint(sk_lo, min(sk_hi, len(SKILLS[domain])))
    skills = rng.sample(SKILLS[domain], n_skills)

    pr_lo, pr_hi = SENIORITY_PROJECT_COUNT[seniority]
    n_projects = rng.randint(pr_lo, min(pr_hi, len(PROJECTS[domain])))
    projects = rng.sample(PROJECTS[domain], n_projects)

    return Resume(
        name=name,
        demographic_signal=demographic_signal,
        seniority=seniority,
        domain=domain,
        years_exp=years_exp,
        skills=skills,
        projects=projects,
    )


# --------------------------------------------------------------------------- #
# Cell enumeration helper
# --------------------------------------------------------------------------- #


def enumerate_cells(axes: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Return all combinations of axis values as a list of dicts."""
    keys = list(axes.keys())
    value_lists = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in iterproduct(*value_lists)]
