# Revised 4-Week Plan — Abstaining Text-to-SQL via RL

Revised 2026-11-04 in light of the research findings ([REPORT.md](REPORT.md)). Changes from the original brief are marked **[changed]** with the reason. Budget posture: nothing above $1 is spent before the smoke-test gate; total expected spend $10–30, worst case ≈ $40 (ESTIMATES — see [compute_estimate.md](compute_estimate.md)).

## Standing decisions (from the findings)

- **Model:** Qwen3.5-2B primary; **SmolLM3-3B as the pre-committed fallback and control** **[changed]** — Qwen3.5-2B is multimodal with a hybrid Gated-DeltaNet/attention architecture and is the riskiest for tooling (Unsloth's embedded vLLM can't serve it; vLLM proper supports it from ~v0.19), while "Gemma 4 E2B" is 5.1B total parameters in memory, a heavier footprint than the brief assumed. Decide Qwen vs. fallback at the smoke-test gate, not later.
- **Stack:** TRL GRPOTrainer + vLLM colocate (vLLM shares the single training GPU), LoRA (low-rank adapters — training a small add-on instead of all weights), async Python reward that executes SQL. Loss: `loss_type="dr_grpo"` + `scale_rewards=False`, KL coefficient β=0, group size 8, lr ~1e-6, completion cap 1024–2048 with truncated completions masked. Pin exact TRL/vLLM/transformers versions on day 1.
- **Abstain mechanism:** structured output — `{"sql": ..., "abstain": bool, "reason": ...}` parsed by the reward — not a new vocabulary token (which would break TRL's fused-head path and blow up trainable parameters at a 150K–262K vocabulary).
- **Reward:** +1 correct (multi-instance execution match) / 0 abstain / −c wrong, c swept across runs; strict format gate; 2–5 s per-query timeout; **gold-empty training examples filtered** (the empty-result exploit would otherwise beat honest abstention, since junk SQL returning nothing scores +1 while abstaining scores 0).
- **Data:** synthetic generator (SynSQL-style pipeline + TrustSQL's five-type unanswerable taxonomy, ~30% unanswerable/ambiguous/out-of-scope, 3–5 data instances per schema) for training; **BIRD dev untouched as the external eval**; TrustSQL/EHRSQL for the penalty-score comparison. **[changed]** from "maybe existing datasets": no public dataset has unanswerable *training* examples, and contamination risk argues against training on Spider/BIRD.
- **Calibration:** **Learn-then-Test (LTT)** — treating each candidate confidence cutoff as a hypothesis test so "error ≤ α on answered queries" holds with probability ≥ 1−δ — on sequence log-probability over the trained policy. **[changed]** from plain conformal risk control, which bounds only the unconditional "answered AND wrong" rate, not the conditional selective risk the brief promises. Report the CRC bound as a secondary result.

## Week 1 — Data pipeline and checker (local, $0 GPU)

1. Repo scaffolding, pinned environment, model-availability check (load all three models' tokenizers/configs; confirm vLLM version serves Qwen3.5-2B text-only via `--language-model-only`).
2. Synthetic generator v0: schema generation with realistic names (SynSQL scenario-inference recipe; Gretel's 100-domain list for variety), table filling producing 3–5 distinct data instances per schema, SQL-first question generation with back-translation, injected unanswerable (missing column), ambiguous (undefined metric), and out-of-scope categories with known labels.
3. Multi-instance execution checker (the test-suite idea from Zhong et al. 2020: run predicted vs. gold SQL on several database instances so a wrong query can't match by coincidence) with per-query timeouts; reuse `test-suite-sql-eval` conventions where possible; filter any item whose gold query returns empty on any instance.
4. **Optional spike from the brief:** 50 schema/question pairs through the checker end-to-end; run a small Qwen GGUF (quantized model file for local inference) zero-shot through llama.cpp Vulkan on the laptop to estimate baseline accuracy, abstention rate, and local latency. Report what broke.
5. Gate W1: checker catches seeded wrong-but-coincidentally-matching queries; ≥500 clean training prompts exist (Unsloth guidance: 500+ rows).

## Week 2 — Smoke test and first runs (first GPU spend)

1. TRL training script: GRPOTrainer + vLLM colocate + LoRA + async reward; logging for reward, abstention rate per step (collapse alarm), completion length, s/step.
2. **20-step smoke test on Modal free credit or a Secure A100 (≤ $0.60 ESTIMATE):** measures real s/step (replacing the 800–2,000 tokens/sec planning assumption), verifies Qwen3.5-2B actually trains in this stack, checks 24GB-fit for the cheap-4090 path, exercises checkpoint/resume.
3. **Gate (hard):** proceed with Qwen3.5-2B only if the smoke test passes cleanly; otherwise switch primary to SmolLM3-3B the same day. Proceed to full runs only if measured s/step × 300 × 8 runs ≤ ~2/3 of remaining budget; otherwise cut completion cap to 256 or move to the 4090.
4. First two real runs: accuracy-only baseline + one abstention run at the middle c, 300 steps each (≈ $2.45/run central ESTIMATE). Inspect for reward hacking (empty-result matching, degenerate SQL, abstention collapse) before committing the rest.

## Week 3 — Run matrix, calibration, shift experiment

1. Remaining matrix runs: 3 values of c × 2 seeds (+ optional control-model run), 300 steps each, on Community A100 or 4090 with 25–50-step checkpoints. Total matrix ≈ $16–36 incl. overhead (ESTIMATE).
2. Calibration layer: score calibration-split items with sequence log-probability (and optionally k-sample execution agreement); implement LTT with an exact binomial test per cutoff and fixed-sequence testing (walk cutoffs conservative → permissive, stop at first failure); certify thresholds at α ∈ {0.05, 0.1, 0.2}.
3. Shift experiment: calibrate on one synthetic schema family, test on another and on BIRD dev; report realized selective risk, coverage, and overshoot (realized error minus α) with an i.i.d. control split; repeat over family pairs for the overshoot matrix.
4. Baselines: base model prompted to abstain; self-consistency thresholding (abstain unless k samples agree); 1–2 frontier models via API prompted to abstain (cap API spend at a few dollars).

## Week 4 — Evaluation, statistics, write-up

1. Full evaluation: execution accuracy (multi-instance), coverage, selective risk, risk–coverage curves and AURC (area under the risk–coverage curve — one number for the whole trade-off; lower is better), TrustSQL-style RS(c) at c ∈ {0, 10, N}, tokens/query, local GGUF latency. Check: does each certified threshold's realized error actually sit ≤ α?
2. Statistics: bootstrap confidence intervals (re-sampling results many times to see how much each number wobbles) over test items and seeds; interquartile mean over seeds (rliable); re-draw the calibration/test split many times to show the spread of realized risk around α.
3. Write-up: lead with the novelty triangulation (BAR-SQL / Reinforced Hesitation / RLSR), state the correctness oracle explicitly, preserve all UNVERIFIED flags; release code + synthetic data (mind CC BY-SA share-alike if any Spider/BIRD-derived content ships).
4. Slack in the week absorbs re-runs; degradation path if behind: drop the control model, then drop one c value's second seed.

## Proposed repo structure

```
abstain-sql/
  brief.md
  research/                  # these four deliverables
  data/
    generator/               # schema gen, table filling, question templates, injection
    instances/               # 3-5 SQLite instances per schema (the test suites)
    splits/                  # train / calibration / test / shift-family manifests
  checker/
    execute.py               # sandboxed SQL execution, timeouts
    testsuite.py             # multi-instance agreement = correctness oracle
  train/
    reward.py                # parse {"sql","abstain",...}; +1 / 0 / -c; format gate
    grpo_train.py            # TRL GRPOTrainer + vLLM colocate + LoRA config
    configs/                 # per-run YAML: c value, seed, model, steps
  calib/
    signals.py               # sequence logprob, k-sample agreement
    ltt.py                   # binomial tests, fixed-sequence walk, CRC comparison
  eval/
    metrics.py               # EX, coverage, selective risk, AURC, RS(c)
    baselines.py             # prompted-abstain, self-consistency, frontier APIs
    bootstrap.py             # item/seed/calibration-split resampling
    local_gguf.py            # llama.cpp Vulkan latency harness
  runs/                      # checkpoints, logs, per-step metrics (gitignored)
  reports/
```

## Risk register (top 5)

1. **Qwen3.5-2B tooling friction** (multimodal, hybrid attention) → smoke-test gate + SmolLM3-3B fallback, decided in week 2.
2. **Abstention collapse at high c** → c-sweep, ≥70% answerable training mix, per-step abstention-rate alarm.
3. **Reward hacking via empty results / degenerate SQL** → gold-empty filtering + multi-instance oracle before any reward tuning.
4. **Throughput worse than estimated** → smoke-test decision rule; 256-token cap and 4090 fallback paths costed in advance.
5. **Low coverage starves the calibration** (LTT needs several hundred *answered* calibration points) → size the calibration split generously (ESTIMATE-level sizing; check empirically in week 3).
