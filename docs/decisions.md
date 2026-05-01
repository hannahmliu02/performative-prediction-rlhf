# Decisions Log

## In progress

## Completed

### Audit extension + S1-Task 3 causal framing (2026-04-29)

**Causal framing — missingness as confounder (not mediator).**
The correct DAG is: demographic group → missingness → observation; demographic group → response quality → observation. Missingness is a *confounder* of the (preference pair, RM score) relationship. It biases the RM by under-representing Group B pairs in training, regardless of response quality. Controlling for missingness (via propensity weighting) recovers an unbiased preference signal. This is distinct from the mediator framing where missingness would lie on the causal path from group to outcome — here it is a backdoor variable that creates a spurious association between group membership and RM score.

Implication for the methods section §4.2: the propensity model estimates `P(observed | x, y)`, and IPW reweights each pair's loss contribution by `1/P(observed | x, y)` to block the backdoor path from group → missingness → training set composition → RM scores.

**S1-Task 3 naming: DPO+ and PPO+.**
The IPW-corrected training variants are labelled DPO+ (DPO with counterfactual preference weighting) and PPO+ (PPO/RLOO with IPW-corrected RM). These names are used consistently across code, configs, CSV columns, and paper text. The `method` column in `per_round_metrics.csv` uses `"dpo_plus"` and `"ppo_plus"` as string identifiers. No code changes needed beyond the naming; the implementation in S1-Task 3 builds on the existing DPO/PPO training scripts and the IPW correction sits at the preference-data stage.

**Audit extension (observability_audit.py).**
Two new diagnostics added alongside the existing linear probe and masking analysis:

1. **Content-controlled linear probe:** OLS with `[rm_margin, demo_B_indicator, response_length, prompt_length, quality_score]`. The key claim is that the `demo_B_indicator` coefficient remains negative and significant after controlling for these content features, ruling out the confound that Group B responses are shorter or lower quality. Controlled by `run_content_controlled_probe: bool` in audit config (default `true` for production, `false` for CI smoke).

2. **Non-linear MLP probe + SHAP:** sklearn `MLPClassifier(hidden_layer_sizes=(32,16))` trained on the same feature set; SHAP `PermutationExplainer` on a background of 50 samples to compute mean |SHAP| per feature. Confirms coverage indicator is among the top-ranked features in a non-linear model, ruling out the linear-specification artifact concern. Controlled by `run_mlp_probe: bool` (default `true` for production, `false` for CI smoke). Requires `shap>=0.45` (added to `pyproject.toml`).

**Files:** `llm/audit/observability_audit.py` (extended), `llm/configs/audit/{smoke,semi_synthetic}.yaml` (new flags), `llm/tests/test_audit.py` (new, 13 tests covering `_mask_prompt`, `run_linear_probe`, `run_linear_probe_with_content`, `run_mlp_probe`).

**Test result:** 43/43 tests pass. All four probe functions verified with synthetic data; MLP accuracy bounded, SHAP feature names correct, significance flag logic verified.




### S1-Task 2 — Baseline mitigations (2026-04-29)

**Files:** `llm/training/mitigations/{__init__,length_norm,rm_reg,kl_constrained,safe_rlhf,tuning}.py`, `llm/configs/mitigation/*.yaml`, `llm/tests/test_mitigations.py`.

Decisions made during planning:
- **RM length norm:** subclass `RewardTrainer`, override `compute_loss`. TRL 1.2.0 concatenates chosen+rejected into a single batch (`model(**inputs)` → `torch.chunk(logits, 2)`), so we extract per-sequence lengths from `inputs["attention_mask"]`, chunk them, and subtract `β·log(len)` from each half before the BT loss.
- **DPO length norm:** subclass `DPOTrainer`, override `_compute_loss`. The one-line change: `logps = per_token_logps.sum(dim=1) / comp_lens` (instead of `.sum(dim=1)` only), where `comp_lens = shift_completion_mask.sum(dim=1).float().clamp(min=1)`. Same normalization applied to `ref_logps`. No `beta` parameter — this is a hard normalization (beta=1).
- **RM regularization:** subclass `RewardTrainer`, add explicit L2 penalty on the reward head's weight matrix (separate from AdamW weight_decay, which applies to all params). Optional logit-consistency term: within each batch, compute variance of scores within each demographic group (from a `demographic_signal` field if present); add `λ_cons · mean_variance` to the loss. If no demographic column in batch, skip consistency term silently.
- **KL-constrained PPO:** already implemented via `RLOOConfig.beta = kl_beta`. The `kl_constrained.py` module just provides a `build_kl_ppo_cfg(base_ppo_cfg, kl_beta)` helper and documents the sweep range (`{1e-4, 1e-3, 1e-2}`).
- **Safe-RLHF:** composite reward function `r_total = r_helpful − λ_safety · max(0, disparity_k)`, where `disparity_k = |mean_score_A − mean_score_B|` estimated from the current batch. No separate safety RM — the disparity signal is self-supervised from the helpful RM's own scores. This is a simplified adaptation, not a faithful port of the PKU Safe-RLHF paper.
- **Integration:** `mitigation` sub-dict in feedback loop config is propagated through `_rm_cfg`/`_dpo_cfg`/`_ppo_cfg` builders. In `train_rm`, `train_dpo`, `train_ppo`, a `_get_*_trainer()` factory reads `cfg.get("mitigation", {}).get("type", "none")` and returns the appropriate subclass.
- **Tuning protocol:** `select_best_round1_config(mitigation_name, sweep, base_loop_cfg, output_dir)` runs the feedback loop with `num_rounds=1` for each config in the sweep, records round-0 disparity, saves `llm/outputs/tuning/<name>.csv`, returns the winning config dict. Smoke test uses a 1-element sweep to verify the function runs without errors.
- **Recommended headline subset:** `no_mitigation`, `length_norm`, `kl_constrained`, `ours_ipw` (S1-Task 3). `rm_reg` and `safe_rlhf` are appendix baselines.
- **Smoke test result:** 30/30 tests pass. Dispatcher, `LengthNormRewardTrainer.compute_loss`, `RMRegRewardTrainer`, `build_kl_ppo_cfg`, and `build_safe_reward_fn` all verified via unit tests with mocked models.


### S1-Task 1 — Per-round metrics dashboard (2026-04-26)

**Files:** `llm/simulate/feedback_loop.py` (extended), `llm/simulate/plot_rounds.py` (new), `llm/tests/test_plot_rounds.py` (new), `llm/configs/simulate/feedback_loop/smoke.yaml` (updated).

Decisions made during planning:
- **New CSV columns:** `method` (from `cfg.method`, default `"no_mitigation"`), `seed`, `per_cell_rm_accuracy_mean`, `per_cell_rm_accuracy_min` (across fine-grained cells, not group aggregates), `policy_output_entropy`, `dt_fairness_score`, `dt_stereotype_score`, `mt_bench_score`, `amplification_ratio`.
- **`amplification_ratio`:** disparity = `|demo_A_accuracy − demo_B_accuracy|`; ratio = disparity_k / disparity_0 (round_idx=0 gets ratio=1.0 by definition; NaN if disparity_0=0). Added post-hoc after all rounds complete.
- **`policy_output_entropy`:** computed inside `_generate_new_preferences` while the policy model is already loaded — avoids a second model load. Mean token-level entropy (−∑ p log p) averaged over `cfg.entropy_n_prompts` held-out prompts. Return type changes to `tuple[pd.DataFrame, float]`.
- **Optional per-round DT/GC eval:** guarded by `run_dt_per_round` and `run_gc_per_round` config flags (default `false`). When false, columns are `NaN`. When true, calls `run_decoding_trust` / `run_general_capability` against the current policy checkpoint (results are cached by checkpoint hash so re-runs are cheap).
- **`plot_rounds.py`:** reads one or more `per_round_metrics.csv` files, groups by `(method, round)`, computes mean ± std across seeds, renders a 2×3 matplotlib panel: [per-cell accuracy, entropy, amplification ratio; DT fairness, DT stereotype, MT-Bench]. Per-cell accuracy panel shows A (solid) and B (dashed) per method; seed variance as shaded band. Entry point: `python -m llm.simulate.plot_rounds --inputs <csv...> --output <png>`.
- **Smoke config additions:** `method: no_mitigation`, `entropy_n_prompts: 8`, `run_dt_per_round: false`, `run_gc_per_round: false`.



## Completed

### Task 11 — Smoke test and CI (2026-04-26)

**Files:** `llm/scripts/run_semi_synthetic.py`, `llm/configs/semi_synthetic_smoke.yaml`, `llm/configs/audit/ci_smoke.yaml`, `.github/workflows/ci.yml`.

Decisions made during planning:
- **Orchestration pattern:** `run_semi_synthetic.py` loads a top-level config that references each stage's sub-config; per-stage boolean flags (`run_data`, `run_sft`, etc.) allow steps to be skipped.
- **Audit in CI:** the full audit smoke scores 1200 × 2 × 11 masking steps on CPU (~20 min). Added `max_samples` and `n_masking_steps` overrides to `observability_audit.py`; `ci_smoke.yaml` uses `max_samples: 50`, `n_masking_steps: 3` to keep the audit under 2 min.
- **CI target:** GitHub Actions runs `make test` (pytest, < 2 min) on every push/PR; `make smoke` runs the full orchestration and is listed as a separate (optionally skipped) CI job. Both jobs gate on ubuntu-latest with uv.
- **Summary output:** `llm/outputs/smoke/summary.json` collects checkpoint paths and key metrics from each stage; logged at the end of the orchestration run.
- **Smoke test result:** All 7 stages completed successfully. Audit: r2=0.0499, p_demo_B=0.2381 (50 samples, 3 masking steps). DecodingTrust: stereotype=0.30, fairness=0.00 (fallback prompts, cache hit). General capability: score=4.00 (judge=none, deterministic). Summary saved to `llm/outputs/smoke/summary.json`. 14/14 tests pass.



## Completed

### Task 10 — Feedback-loop simulator (2026-04-26)

**Files:** `llm/simulate/feedback_loop.py`, `llm/scripts/run_feedback_loop.py`, `llm/configs/simulate/feedback_loop/{smoke,semi_synthetic}.yaml`.

Decisions made during planning:
- **Round structure:** (1) save current train data, (2) train RM from SFT backbone, (3) evaluate per-cell accuracy on the fixed audit set, (4) train policy (DPO or PPO), (5) generate new preference pairs from the policy, (6) apply missingness injection, (7) accumulate new pairs into the growing dataset.
- **RM backbone per round:** RM always starts from `cfg.sft_checkpoint` (fresh head) and trains on the growing biased dataset. Warm-starting from the prior RM checkpoint is a flag (`rm_warm_start`), default `false`, so each round reflects only accumulated data bias.
- **Policy warm-start:** round 0 starts from `cfg.sft_checkpoint`; round N+1 starts from the round N policy checkpoint, propagating degradation across rounds.
- **New preference labeling:** for each held-out prompt, the policy generates a response; the RM scores it against the pre-built bad summary (`rejected` column of audit.parquet); higher RM score = `chosen`. Missingness injection then drops pairs by `obs_probs[demographic_signal]`.
- **Per-cell accuracy:** always evaluated on the fixed audit.parquet (not a random split of training data), giving a consistent comparison across rounds.
- **Metrics CSV columns:** `round`, `demo_A_accuracy`, `demo_B_accuracy`, `n_train`, `n_new` plus per-cell breakdown.
- **Figure 2 analogue:** `demo_A_accuracy` and `demo_B_accuracy` vs. round number — should diverge over rounds for the biased baseline.
- **Smoke config:** 2 rounds, max_steps=2 (RM and policy), CPU, float32. Verifies code runs; monotonically-worsening accuracy is not expected with 2 training steps.
- **Smoke test result:** 2 rounds completed. Round 0: demo_A=0.528, demo_B=0.487, n_new=736. Round 1: demo_A=0.380, demo_B=0.403, n_new=735. Observed A/B ratio ~1.9 (config: 0.8/0.4=2.0). Outputs: `llm/outputs/simulate/feedback_loop_smoke/`. 14/14 tests pass.



## Completed

### Task 9 — General-capability benchmark (2026-04-26)

**Files:** `llm/eval/general_capability.py`, `llm/scripts/run_general_capability.py`, `llm/configs/eval/general_capability/{smoke,semi_synthetic}.yaml`.

Decisions made during planning:
- **MT-Bench chosen over AlpacaEval:** MT-Bench produces a per-category score breakdown (writing, reasoning, coding, math, etc.) that is more informative for our paper than a single win-rate number. The 8-category structure also maps well to our fairness-focused framing.
- **Built-in question set:** 16 hand-written questions (2 per MT-Bench category) are bundled in the module as a smoke/offline fallback. This avoids requiring HuggingFace access for CI; the questions are our own, not copied from the LMSYS benchmark.
- **`judge_model: "none"` for offline/smoke mode:** returns a seeded deterministic score (hash of question id + seed, mapped to 1–10) so the full code path runs without any API calls. Smoke test caveat: scores are meaningless but the pipeline is verified.
- **Anthropic Claude as default judge:** `judge_model: "claude-sonnet-4-6"`. Uses the `anthropic` SDK (added to `pyproject.toml`). Prompt follows the standard MT-Bench single-answer grading format; rating extracted via `[[rating]]` regex.
- **Caching:** `{output_dir}/{experiment_name}/cache/gc_{seed}_{hash8}.json` where `hash8` = SHA-256 of checkpoint `config.json`.
- **Smoke test result:** overall_score=4.00 (deterministic), 1 question (from `lmsys/mt_bench_human_judgments` via HF, extracted from `conversation_a[0]["content"]`), judge=none. Results saved to `llm/outputs/eval/gc_smoke/`. 14/14 tests pass.



## Completed

### Task 8 — DecodingTrust integration (2026-04-26)

**Files:** `llm/eval/decoding_trust.py`, `llm/scripts/run_decoding_trust.py`, `llm/configs/eval/decoding_trust/{smoke,semi_synthetic}.yaml`.

Decisions made during planning:
- **Official harness limitation:** The official DecodingTrust CLI targets the OpenAI API and cannot directly use a local HuggingFace model. The adapter (`llm/eval/decoding_trust.py`) loads the official prompt data from `AI-secure/DecodingTrust` on HuggingFace, runs inference via `Policy.generate()`, and applies the documented scoring methodology — avoiding a reimplementation while supporting local models.
- **Fallback prompts:** If the HuggingFace dataset is unavailable (offline smoke runs), a minimal hardcoded set of benign prompts exercises the full code path and produces meaningful-but-trivial scores.
- **Subsets:** `stereotype` (agreement rate) and `fairness` (demographic parity) as headline metrics. Other subsets can be added via config.
- **Scoring — stereotype:** fraction of completions where the first-token response contains an agreement keyword ("agree", "yes", "true"). Lower is better (less stereotype endorsement).
- **Scoring — fairness:** `|P(positive | group_A) − P(positive | group_B)|` (demographic parity gap). Lower is better.
- **Caching:** results stored as `{output_dir}/{experiment_name}/cache/dt_{subset}_{seed}_{hash8}.json` where `hash8` = first 8 chars of SHA-256 of the checkpoint's `config.json`.
- **Smoke config:** `sft_smoke` checkpoint, max 10 samples per subset, CPU, float32. Verifies no import/API errors.
- **Smoke test result:** Both subsets ran on fallback prompts (HF dataset is gated). Stereotype agreement_rate=0.30, fairness parity_gap=0.00. Results cached to `llm/outputs/eval/dt_smoke/`. `padding_side='left'` fix applied to `Policy.__init__`.



## Completed

### Task 7 — Observability audit (2026-04-26)

**Files:** `llm/audit/observability_audit.py`, `llm/scripts/run_audit.py`, `llm/configs/audit/{smoke,semi_synthetic}.yaml`.

Decisions made during planning:
- **Linear probe:** OLS regression `chosen_is_good ~ rm_margin + demo_B_indicator` using numpy/scipy (no statsmodels dependency needed). Wald t-tests computed analytically from the OLS covariance matrix. `rm_margin = score(chosen) - score(rejected)`.
- **Counterfactual delta:** within each (seniority × domain) stratum, compare mean `rm_margin` between group A and group B using Welch's t-test. The null hypothesis is that RM treats matched pairs equally; rejection indicates demographic bias in the margin.
- **Multi-step masking (11 steps, k=0..10):** progressively replace resume fields: k=1 name, k=2 years_exp, k=3 domain, k=4 skills, k=5 projects, k=6+ full resume replaced with `[candidate details masked]`. Track mean score drift = `score_k - score_0` per demographic group. Steps k=7–10 all produce the same fully-masked prompt — the curve plateaus, which is itself informative.
- **Figure:** matplotlib lineplot of mean drift ± SEM vs k, one line per demographic group. Saved as `figure_b5_analogue.png`.
- **Smoke test caveat:** rm_smoke (2 training steps) is too lightly trained to show significant bias. The "Done when" criterion requires a properly-trained RM; smoke test verifies the audit runs without errors.
- **Smoke test result:** Linear probe p(demo_B)=0.023, R²=0.020. Counterfactual delta non-significant across all strata (expected). Masking plateau confirmed at k=6..10. Outputs: `llm/outputs/audit/audit_smoke/`.



## Completed

### Task 6 — PPO and DPO training (2026-04-26)

**Files:** `llm/training/ppo.py`, `llm/training/dpo.py`, `llm/scripts/run_ppo.py`, `llm/scripts/run_dpo.py`, `llm/configs/ppo/{semi_synthetic,smoke}.yaml`, `llm/configs/dpo/{semi_synthetic,smoke}.yaml`.

Decisions made during planning:
- **PPOTrainer removed in TRL 1.x:** using `RLOOTrainer` (REINFORCE Leave-One-Out) instead. RLOO is TRL 1.x's recommended online RL trainer; it implements a KL-constrained REINFORCE update that is functionally equivalent to the reward-improvement objective of PPO. Decision recorded here so reviewers can map "PPO" in the paper to the implementation.
- **Reward callable for RLOO:** `reward_funcs` accepts `(prompts, completions, **kwargs) → list[float]`. We pass a closure that wraps `RewardModel.from_pretrained` and applies optional length normalization (`reward -= length_norm_beta * log(|completion|)`). This makes length norm a first-class citizen of the RL loop without requiring any subclassing.
- **DPO length norm deferred to S1-Task 2:** applying length norm to DPO requires overriding `get_batch_loss_metrics` to normalize per-token log-probs by sequence length. That is the proper home for mitigation implementations. In Task 6, `length_norm_beta > 0` in a DPO config logs a `NotImplementedError`-style warning and falls through to standard DPO — the flag exists so no config changes are needed in S1-Task 2.
- **DPO `--skip_sft` flag:** CLI-only; when present, loads `cfg.model_name_or_path` (base backbone) as the policy instead of `cfg.sft_checkpoint`. The reference model is `None` in both cases (TRL creates a frozen copy internally).
- **Sample generation logging (both):** RLOO: `log_completions=True` when `report_to != "none"`. DPO: same `SampleGenerationCallback` pattern as SFT.
- **Smoke configs** use the checkpoints saved by the Task 4 and 5 smoke runs (`sft_smoke`, `rm_smoke`), `max_steps=2`, CPU.



## Completed

### Task 5 — Reward model training (2026-04-26)

**Files:** `llm/models/reward_model.py` (updated), `llm/training/train_rm.py`, `llm/scripts/run_rm.py`, `llm/configs/rm/{semi_synthetic,audit_baseline,smoke}.yaml`.

Decisions made during planning:
- **Architecture change in `RewardModel`:** switched backbone from `AutoModelForCausalLM + custom Linear` to `AutoModelForSequenceClassification(num_labels=1)`. Reason: `trl.RewardTrainer` saves checkpoints as `AutoModelForSequenceClassification`; the prior architecture would make saved checkpoints unloadable by `RewardModel.from_pretrained`. The `score()` API and all Task 2 tests remain unchanged — only the internal model type changes.
- **Per-cell accuracy** computed as a post-training inference pass on the full eval set (with cell metadata), not via `compute_metrics` hook. Reason: `EvalPrediction` in Trainer doesn't carry metadata columns; a separate inference loop is simpler and correct.
- **Three data-source configs:** `semi_synthetic.yaml` (biased train), `audit_baseline.yaml` (uniform audit — ground truth upper bound), `smoke.yaml` (gpt2, 2 steps, CPU). The "corrected" source (S1-Task 3) is a fourth config to be added later.
- **"Done when" verification:** run both biased and audit RM, compare per-cell accuracy; Group B cells must show lower accuracy on the biased RM. The smoke test only verifies the biased run completes without error on CPU.



## Completed

### Task 4 — SFT stage (2026-04-25)

**Files:** `llm/training/sft.py`, `llm/scripts/run_sft.py`, `llm/configs/sft/semi_synthetic.yaml`, `llm/configs/sft/smoke.yaml`.

Decisions made during planning:
- **SFT data source:** filter `preference_train` to `chosen_is_good=True` rows, then format as `"{prompt}\n\nSummary: {chosen}"`. The ε-flipped rows (bad summary as "chosen") are excluded — they'd corrupt SFT. The label noise is meaningful for RM training but not for SFT.
- **TRL 1.x API:** use `SFTConfig` (not `TrainingArguments` + separate kwargs); `dataset_text_field="text"` in `SFTConfig`; `max_seq_length` in `SFTConfig`. `max_steps=-1` defers to `num_train_epochs`; smoke config sets `max_steps=2` to override.
- **W&B logging:** set `WANDB_PROJECT` from config; sample generations logged as a `wandb.Table` via a `TrainerCallback` at the end of each eval. If `report_to="none"`, W&B is skipped.
- **Checkpoint path:** `{output_dir}/{experiment_name}/` — saved by `trainer.save_model()` after `trainer.train()`.
- **Smoke run:** `llm/configs/sft/smoke.yaml` uses `gpt2` (124M), `max_steps=2`, `report_to=none`; run on CPU to verify no import/API errors before needing a GPU.



## Completed

### Task 3 — Semi-synthetic resume preference data (2026-04-25)

**Files:** `llm/data/{resume_templates,quality_score,generate_resume_prefs}.py`, `llm/configs/data/semi_synthetic.yaml`, `llm/tests/test_data_generation.py`.

Decisions made during planning:
- **Synthetic annotator is rule-based** (matches the resolved open question). No LLM is used to produce summaries or preferences; everything is deterministic given the config seed.
- **Good/bad summaries are constructed programmatically.** Good summary: specific, names actual skills and quantified experience from the resume. Bad summary: vague template that mentions no specific skills or numbers. This gives a clean signal for the RM to learn.
- **Quality score formula:** `q = 0.5 * (n_skills / max_skills) + 0.35 * (min(years_exp, 10) / 10) + 0.15 * (n_projects / 3)`. Inputs (n_skills, years_exp, n_projects) are all sampled with the same distributions across demographic groups — independence is structural, not statistical.
- **Observation probability** computed as `p_obs(cell) = obs_probs.demographic_signal[a]`. Config allows extension to other axes by multiplication; only demographic is defaulted in the spec.
- **Config loading:** argparse `--config` + `OmegaConf.load()`, no Hydra `@hydra.main` (avoids cwd changes; keeps entry-point behaviour predictable).
- **Dataset columns:** `prompt`, `chosen`, `rejected`, `demographic_signal`, `seniority`, `domain`, `quality_score`, `chosen_is_good` (for audit).
- **Tests use N=30 per cell** (360 total) to keep CPU runtime under 5 s.



## Completed

### Task 2 — Backbone model wrapper (2026-04-24)

Creating `llm/models/{backbone,reward_model,policy}.py` and unit tests in `llm/tests/test_models.py`.

Decisions made during planning:
- **`dtype` argument** mapped to `torch.dtype` via a dict (`bfloat16`, `float16`, `float32`); unknown strings fall back to `bfloat16`.
- **`device_map`** typed as `str | None`; `None` means no device mapping (model loads to CPU by default). Tests pass `device_map=None, dtype="float32"` to run cheaply on CPU.
- **Reward head** is `nn.Linear(hidden_size, 1, bias=False)` applied to the last non-padding token's final hidden state (standard TRL/InstructGPT convention). Uses `output_hidden_states=True` to get per-layer hidden states; takes `hidden_states[-1]`.
- **`RewardModel.from_pretrained`** always randomly initialises the reward head (backbone weights come from the checkpoint; head is new). A separate save/load path for the head can be added later.
- **`Policy.generate`** strips the prompt tokens from the output before decoding (only returns newly generated text), consistent with how TRL's PPO trainer works.
- **GPT-2 pad token**: GPT-2's tokenizer has no pad token; we set `pad_token = eos_token` at construction time in both `Policy` and `RewardModel`.



## Completed

### Task 1 — Repository scaffolding (2026-04-24)


Creating the `llm/` directory tree, `pyproject.toml` (managed by `uv`), utility modules, and `Makefile`.

Decisions made during planning:
- **`llm/tests/`** added as a top-level subpackage (not under any specific module) so pytest can discover all tests with a single `testpaths = ["llm/tests"]` entry.
- **`torch.use_deterministic_algorithms(True)`** gated behind a `deterministic: bool = True` flag in `set_seed()`; tests pass it as `True`, training scripts may pass `False` if ops lack deterministic kernels.
- **`structlog` configuration** done once in `get_logger()` using `structlog.configure()` with a `ConsoleRenderer`; idempotent on repeated calls via a module-level `_configured` flag.
- **`Makefile`** uses `uv run` for all targets so the venv is not separately activated.



Decisions are recorded here, newest first. When Claude Code encounters an ambiguity it cannot resolve, it adds an entry under "Open questions" rather than blocking.

## Resolved

### 2026-04-22 — Paper framing: Skeleton 1 (performative-prediction-centric)

We are committing to Skeleton 1: the paper's headline claim is about the RLHF feedback loop — standard single-round bias mitigations attenuate round-1 disparity but do not survive across repeated training rounds, while a preference-data-stage intervention (counterfactual debiasing) does. Missingness is the mechanism, not the main object of study.

Implications:
- Wolf et al. (arXiv 2505.18126) becomes a methodological ancestor, not peripheral related work. Cite prominently in the related-work section.
- The multi-round feedback-loop simulation is the headline experiment; single-round debiasing comparisons become supporting evidence.
- The causal-graph argument is smaller — mainly justifies the intervention point rather than carrying the contribution.
- Skeleton 2 plan has been deleted from this package.

### 2026-04-22 — Real-world eval design: preconditions + mitigation transfer, not multi-round

We do not attempt to simulate a multi-round feedback loop on real preference data (HH-RLHF). Simulating the loop on real data requires an LLM-as-annotator for rounds 2+, which introduces circularity that reviewers would flag against an amplification claim. Instead:

1. Show that the preconditions for the loop (observability shortcut in the RM) are detectable in HH-RLHF.
2. Show that our counterfactual debiasing transfers — corrected-RM-trained policy improves DecodingTrust bias subscores relative to the biased baseline.

The synthetic experiment carries the amplification claim. The real-world eval carries the "preconditions + mitigation transfers" claim. Less ambitious, more defensible.

## Open questions

### Backbone model

The original paper notes specify "GPT-Neo 7B," which is an impossible combination — GPT-Neo's largest release is 2.7B. Candidates:
- **Pythia-6.9B** (EleutherAI, closest architectural lineage to GPT-Neo). **Default.** Wolf et al. used Pythia-410M for their multi-round RLHF study, so the Pythia family has been validated in this setting.
- **LLaMA-2-7B** — most common in recent RLHF papers, needs access approval.
- **Mistral-7B** — widely available, strong base, less standard in RLHF literature.

Write backbone code against a generic `AutoModelForCausalLM` interface so the choice is a config change.

### Synthetic annotator design

Should the preference-label oracle be:
- rule-based (deterministic function of latent quality), or
- LLM-based (prompt a frontier model to judge)?

Rule-based is cleaner and avoids circularity; LLM-based is more realistic but opens the reviewer question "would a human agree with the oracle?"

**Default**: rule-based for shared Task 3, with LLM-based as an appendix ablation.

### DecodingTrust subsets

DecodingTrust has eight trustworthiness subsets. Which are headline metrics?
- Fairness — almost certainly.
- Stereotype bias — almost certainly.
- Others (toxicity, adversarial robustness, etc.) — flag or skip?

**Default**: Fairness + Stereotype as headline, report all eight in appendix.

### Number of PPO/DPO seeds per method

Skeleton 1 plan specifies 5 seeds for headline methods, 3 for appendix. Worth revisiting if compute is tighter than expected — 3 across the board is acceptable but gives noisier CIs.

### Alignment Auditor

The original meeting notes reference an "Alignment Auditor." Unclear whether this is a specific published tool or a general probing strategy. Confirm with advisors before integrating; we may not need it if the observability audit from shared Task 7 covers the diagnostic needs.
