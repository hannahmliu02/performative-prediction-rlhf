# Skeleton 1 Implementation Plan

**Paper framing.** RLHF is performative; missingness is the mechanism. The paper's headline claim is that standard single-round bias mitigations (length normalization, RM regularization, KL constraints, Safe-RLHF) attenuate round-1 disparity but do not survive across repeated training rounds, while a preference-data-stage intervention (counterfactual debiasing) does.

**Prerequisites.** `@docs/shared_infrastructure.md` must be complete. Tasks 5 (RM training), 6 (PPO/DPO), 7 (audit), and 10 (feedback-loop simulator) are load-bearing.

**Key prior work.** Wolf, Kirk & Musolesi (arXiv 2505.18126, 2025) ran the first systematic multi-round RLHF study. Their setup (AlpacaFarm, Pythia, 4 iterations, gold-vs-proxy RM evaluation) is the closest thing to a reference implementation for multi-round RLHF experiments. They target reward overoptimization; we target bias, with a preference-data-stage intervention they do not consider.

---

## S1-Task 1 — Per-round metrics dashboard

**Goal:** the headline chart of the paper is "disparity across rounds." Make it first-class.

**Steps**

1. Extend `llm/simulate/feedback_loop.py` (from shared Task 10) to write a `per_round_metrics.csv` with columns:
   - `round`, `method`, `seed`, `per_cell_rm_accuracy_mean`, `per_cell_rm_accuracy_min`, `policy_output_entropy`, `dt_fairness_score`, `dt_stereotype_score`, `mt_bench_score`, `amplification_ratio`.
2. `amplification_ratio = disparity(round_k) / disparity(round_1)`. This is the key metric: > 1 means the loop amplifies, ≈ 1 means it attenuates, < 1 means it corrects.
3. Add `llm/simulate/plot_rounds.py` that renders a 2×3 panel: per-cell accuracy, output entropy, amplification ratio, DT fairness, DT stereotype, MT-Bench. One line per method, shaded band for seed variance.
4. Unit test: the plot script runs against a synthetic per-round CSV and produces a PNG.

**Done when:** given two saved runs (biased baseline, counterfactual-corrected), the script produces the paper's headline figure.

---

## S1-Task 2 — Baseline mitigations (what we will show fails to break the loop)

**Goal:** honest implementations of the single-round mitigations the paper claims fail across rounds. These need to be defensible — reviewers will check.

**Recommended subset for the final paper:** no mitigation, length normalization, KL-constrained PPO, ours (counterfactual debiasing) as headline; RM regularization and Safe-RLHF as appendix ablations. Fewer baselines × more seeds per baseline is usually the better NeurIPS tradeoff than the reverse.

**Steps**

1. **Length normalization.** `llm/training/mitigations/length_norm.py`. At RM scoring time, normalize by response length (subtract `β · log(len)` from the reward). Configurable `β`. Default `β = 0.1` with a sweep over `{0.01, 0.1, 1.0}` to pick the best round-1 performer before running the full sweep.
2. **RM regularization.** `llm/training/mitigations/rm_reg.py`. L2 regularization on the reward head; also support a "logit consistency" regularizer that penalizes high RM variance across cell-matched pairs.
3. **KL-constrained PPO.** Already in `trl.PPOTrainer` via `init_kl_coef`. Expose as a config knob. Sweep `init_kl_coef ∈ {1e-4, 1e-3, 1e-2}` — Wolf et al. used `1e-4` as default, which is a reasonable starting point.
4. **Safe-RLHF baseline.** Adapt the official recipe as `llm/training/mitigations/safe_rlhf.py`. Treat it as an outer-loop baseline; do not attempt to compose it with our method.
5. Each mitigation is a function `wrap_trainer(trainer, config) -> trainer` so they can be composed and swapped via config.

**Tuning protocol (load-bearing for the methods section).** For each baseline, the headline sweep uses the hyperparameter setting that minimized round-one disparity on a held-out validation set. This deliberately favors the baselines — a baseline tuned to maximize round-one performance has the strongest possible chance of also breaking the loop. Implementation:

- Add a `select_best_round1_config` helper in `llm/training/mitigations/tuning.py` that, given a mitigation and a sweep range, runs round one for each config, evaluates on the validation set, and returns the winning config.
- The headline-sweep launcher (S1-Task 4) calls this helper once per baseline before running the multi-round simulation, and logs the selected config to `docs/decisions.md` so reviewers can audit the choice.
- Save the per-config round-one validation metrics to `llm/outputs/tuning/<baseline_name>.csv` for inclusion in an appendix table.

This protocol is what lets the paper claim the negative result is not an artifact of poor tuning. Do not skip it — the methods section §4.3 commits to it explicitly.

**Done when:** each mitigation runs for one round without error and reduces round-1 disparity relative to no-mitigation; the tuning helper selects a config per baseline; the per-config metrics are logged.

---

## S1-Task 3 — Our method: counterfactual preference-data debiasing

**Goal:** the LLM analogue of `toy/causal_rlhf.py`. The intervention sits at the preference-data stage, before the RM sees it; everything downstream (PPO, DPO) is unchanged.

### S1-Task 3a — Propensity model

Estimate `ψ(x, y) ≈ P(m = 1 | x, y)` — the probability that a given prompt-response pair would enter the observed preference set.

**Variants** (selectable by config):

- **Rule-based (oracle).** For the semi-synthetic setting, use the known cell-indicator `p_obs(a, s, d)` from data generation. This is the "cheating" upper bound — shows what the method can achieve with perfect propensity knowledge. Useful for ablation, not for the main result.
- **Classifier-based.** Train a small classifier (logistic regression or a shallow MLP) on features of the prompt and response to predict observation indicator. Features include prompt-axis indicators, response length, response surface features, and base-model perplexity. This is the **default for the main experiments**.
- **LLM-based.** Use the base backbone's log-probability of the response as an inverse proxy for missingness. Most ambitious; reviewer-prone-to-skepticism. Include as ablation only.

**Calibration.** Propensities clipped to `[ε, 1−ε]` with `ε = 0.05` to prevent blow-up under inverse weighting.

**Files**

- `llm/mitigation/propensity.py` — all three variants.
- `llm/configs/mitigation/propensity.yaml`.
- `llm/tests/test_propensity.py` — on synthetic data where `p_obs` is known, the classifier recovers it to within 10%.

### S1-Task 3b — Counterfactual correction

Produce a corrected preference dataset that trains an unbiased RM. Three operationalizations — implement all three; config selects which is used.

1. **Importance weighting (IPW).** Reweight each preference pair's loss contribution by `1 / (ψ(x, y_w) · ψ(x, y_l))`. Minimal code change to `trl.RewardTrainer` — pass per-sample weights.
2. **Synthetic counterfactual generation.** For each observed pair, compute a counterfactual preference target using a calibration head trained on a small uniformly-sampled audit subset. Mix observed and counterfactual targets with blending rate `η` (the CPM paper's Algorithm 1 blending rate).
3. **Matched-pair correction.** For each observed pair, retrieve a nearest-neighbor pair from an under-observed cell (using prompt + response embeddings) and train jointly on both. Most data-efficient; hardest to implement.

**Files**

- `llm/mitigation/counterfactual_debiasing.py` — all three.
- `llm/mitigation/correction_algorithm.py` — R-round annealed correction from the CPM paper's Algorithm 1, with blending rate `η` and annealing schedule `α_r = min(α_0 + Δα · r, 1.0)`. Default `α_0 = 0.4`, `Δα = 0.12`, `η = 0.6` — match the CPM paper's hyperparameters as a sensible starting point.
- `llm/configs/mitigation/counterfactual.yaml`.

**Done when:** given a biased RM and a trained propensity model, the corrected RM's per-cell accuracy matches (within tolerance) the RM trained on the uniform audit set.

### S1-Task 3c — DAG-validity runtime check

The paper's causal-graph argument is smaller in Skeleton 1 than it would be in a methods-first paper — it mainly justifies the intervention point. But the temporal-unrolling assumption still needs to be respected in code.

- `llm/mitigation/dag_check.py` contains runtime assertions that (a) the "prior-round RM" and "current-round RM" are distinct checkpoints (no in-place parameter mutation), and (b) propensity model residuals are uncorrelated with cell membership before each round.
- Runs as part of the loop simulator, not as a separate experiment.

**Done when:** checks pass on the normal pipeline and fail in a constructed bad-setup (deliberately using an RM from a prior round).

---

## S1-Task 4 — Headline experiment: method × round sweep

**Goal:** the paper's Takeaways 1–3.

**Experimental design**

- **Methods (headline, 4):** `no_mitigation`, `length_norm`, `kl_constrained`, `ours_ipw` (the IPW variant of counterfactual debiasing — simplest, most defensible).
- **Methods (appendix, 2):** `rm_reg`, `safe_rlhf`.
- **Rounds:** 5. Configurable up to 10 for ablation.
- **Seeds:** 5 per (method, round) for headline, 3 for appendix.
- **Prompt set:** held-out from `preference_train`, 500 prompts.
- **Synthetic annotator:** the rule-based annotator from shared Task 3. Keep the missingness injection pattern fixed across methods for fair comparison.

**Deliverables**

- `llm/scripts/run_feedback_loop.py --config llm/configs/feedback_loop.yaml` runs one (method, seed) combination.
- `llm/scripts/launch_sweep.py` launches the full 4 × 5 = 20 headline runs + 2 × 3 = 6 appendix runs.
- `llm/scripts/aggregate_rounds.py` reads the per-round CSVs and produces:
  - `amplification_summary.csv`: per-method mean and CI for amplification ratio at round 5.
  - `headline_figure.png`: the 2×3 panel.
  - `takeaways.md`: auto-generated markdown summary of the three paper takeaways, with numbers filled in.

**Done when:** `amplification_summary.csv` shows `amplification_ratio > 1` for baselines and `≤ 1` for ours, with non-overlapping 95% CIs.

---

## S1-Task 5 — Real-world eval

**Goal:** Takeaways 4 and 5 in the paper.

**Design decision (made in conversation — record in `docs/decisions.md` if you change it).** Do **not** attempt to simulate a multi-round loop on real preference data. Instead:

1. **Show the preconditions hold.** Apply the observability audit from shared Task 7 to an RM trained on HH-RLHF. If the shortcut signature is present (RM depends on coverage proxies after controlling for content), the mechanism is in real data.
2. **Show our mitigation transfers.** Train a corrected RM on HH-RLHF using the classifier-based propensity variant. Train a policy against each. Evaluate both on DecodingTrust bias subsets.

This is less ambitious than running a real-world loop, but more defensible. A reviewer cannot complain that the LLM-annotator circularity invalidates the multi-round result, because we are not making a multi-round real-world claim — the synthetic experiment carries the amplification claim; the real-world eval carries the "preconditions + mitigation transfers" claim.

**Files**

- `llm/eval/real_world/hh_rlhf.py` — HH-RLHF loader, coverage-proxy extraction, audit runner.
- `llm/scripts/run_real_world.py` — orchestrator.
- Output: `real_world_summary.md` with preconditions-probe numbers and pre/post-mitigation DecodingTrust scores.

**Done when:** the observability audit detects the shortcut in HH-RLHF with statistical significance, and our mitigation improves DecodingTrust bias scores relative to the biased baseline.

---

## S1-Task 6 — Supporting ablations

**Goal:** the ablations that anticipate reviewer questions. Each is small — single figure or table in `paper/figures/ablations/` or `paper/tables/ablations/`.

1. **Propensity sensitivity.** Run our method with classifier-based propensities perturbed by ±20%. Shows robustness to mis-calibration.
2. **Propensity variant comparison.** Compare oracle vs. classifier vs. LLM-based propensity on the semi-synthetic setup. Oracle is the upper bound; gap between oracle and classifier is the cost of not knowing `p_obs`.
3. **Blending rate.** Sweep `η ∈ {0.2, 0.4, 0.6, 0.8, 1.0}` for the synthetic-counterfactual variant.
4. **Downstream optimizer.** PPO vs. DPO on the corrected data. The method should be optimizer-agnostic — both should benefit equally.
5. **DPO-without-SFT.** Run DPO directly on the corrected data, skipping SFT. Does the method still work? Secondary result per the original meeting notes.
6. **Round count.** Extend the headline sweep to 10 rounds for the 4 headline methods. Shows whether the qualitative pattern holds beyond round 5.

**Done when:** each ablation has a small figure or table with numbers and one-sentence interpretation.

---

## S1-Task 7 — Writing support

**Goal:** make the figures and tables paper-ready.

**Steps**

1. `llm/scripts/paper_figures.py` regenerates every figure in the paper from saved CSVs. Each figure function takes a config and saves to `paper/figures/`.
2. `llm/scripts/paper_tables.py` generates LaTeX tables, saving to `paper/tables/`.
3. Set matplotlib rcParams once in `llm/utils/plot_style.py` (serif fonts, small figure size, colorblind-safe palette).
4. Commit a `paper/` directory with a skeleton `main.tex` that imports the figures and tables by path.

**Done when:** regenerating every paper artifact is a one-command operation (`make paper`).

---

## Success criteria for the full skeleton

- Headline plot shows `amplification_ratio > 1` for all baselines and `≤ 1` for ours, with non-overlapping 95% CIs at round 5.
- Observability audit on HH-RLHF detects the shortcut with statistical significance.
- Our mitigation improves DecodingTrust bias subscores on HH-RLHF relative to the biased baseline.
- General-capability (MT-Bench) scores for ours within 2% of the no-mitigation baseline.
- All results reproducible from a single `make paper` command.

---

## Suggested task ordering

1. Complete shared infrastructure (Tasks 1–11 of `@docs/shared_infrastructure.md`) and smoke-test it.
2. S1-Task 1 (metrics dashboard) — lightweight; unblocks visualization for every later task.
3. S1-Task 2 (baseline mitigations) — each is small and independent.
4. S1-Task 3 (our method) — the load-bearing piece.
5. S1-Task 4 (headline sweep) — pause here, inspect the headline figure with collaborators before proceeding.
6. S1-Task 5 (real-world eval).
7. S1-Task 6 (ablations).
8. S1-Task 7 (writing support, in parallel with 6).
