# abstain-sql: Project Plan

The finished project is one chart. On it, a 2B-parameter model trained to say "I don't know" answers fewer SQL questions than the same model trained only for accuracy, but the questions it answers come back right far more often, and a statistical certificate states the error rate it guarantees. Every step below exists to produce that chart honestly.

This file replaces the original research brief, which stays in git history (commit `1eb6f56`). The evidence behind each decision lives in [research/REPORT.md](research/REPORT.md); this file says what to do, in what order, and how to know each step is done.

## Ground rules

These carry over from the brief unchanged.

- **No training on the laptop.** It is a Windows 11 machine with an Intel Core Ultra 9 185H, 16 GB of shared RAM, an Intel Arc integrated GPU, and no NVIDIA GPU. Local work is limited to data generation, the execution checker, analysis, and running GGUF models (model files compressed for local inference) through llama.cpp's Vulkan backend.
- **$30–50 of cloud spend in total**, on Modal (the Starter plan includes $30/month of free credit) or RunPod (an A100 80GB costs $1.79/hr on Secure Cloud, as of 2026-11-04). Nothing above $1 gets spent before the smoke-test gate. Every cloud session gets an explicit go-ahead before it starts, and its cost goes in the spend log the day it runs.
- **No LLM-as-judge anywhere.** Every metric comes from executing SQL or from a statistical guarantee. Frontier models appear only as baselines being scored, never as the scorer.
- **About 4 weeks.**
- **Honest write-ups.** Every claim cites a primary source (arXiv, official docs, GitHub); anything unverified carries an UNVERIFIED flag; no number appears that was not measured or sourced.

## Terms

The same terms recur in every phase, so they are defined once here.

- **Text-to-SQL:** turning a plain-English question into a SQL query that answers it.
- **Abstain:** the model declines to answer instead of guessing.
- **Gold query:** the reference SQL that is known to answer a question correctly.
- **GRPO** (Group Relative Policy Optimization): the reinforcement-learning algorithm this project trains with. It samples a group of answers to each question and reinforces the ones that score above the group's average.
- **TRL:** Hugging Face's library for this kind of training; its GRPOTrainer implements GRPO.
- **LoRA** (low-rank adapters): training a small add-on to the model instead of all its weights, which cuts memory and cost.
- **vLLM:** a fast text-generation engine. "Colocate" means it shares the one training GPU instead of needing its own.
- **Execution accuracy (EX):** generated SQL counts as correct only if running it returns the same result as the gold query.
- **Test suite (multi-instance check):** running both queries on 3–5 copies of the same database, each filled with different data, so a wrong query cannot match by coincidence.
- **Coverage:** the share of questions the model answers. **Selective risk:** the error rate among the questions it answers. **Risk–coverage curve:** selective risk plotted against coverage as the confidence cutoff moves. **AURC:** the area under that curve, one number for the whole trade-off; lower is better.
- **RS(c):** TrustSQL's Reliability Score. Each question earns +1 for a correct answer or a correct abstention, 0 for abstaining on a question that had an answer, and −c for a wrong answer.
- **c:** the penalty for a wrong answer. **α:** the error ceiling the guarantee promises. **δ:** the small chance that the guarantee itself fails.
- **Learn-then-Test (LTT):** the calibration method. It tests each candidate confidence cutoff like a hypothesis and certifies only the cutoffs that provably keep selective risk at or below α, with confidence 1−δ.

## Decisions already made

Each decision comes from the research; the last column names the report section that justifies it.

| Area | Decision | Evidence in REPORT.md |
|---|---|---|
| Model | Qwen3.5-2B primary; SmolLM3-3B as fallback and control | Part 1, flags 1 and 3 |
| Training stack | TRL GRPOTrainer, vLLM colocate, LoRA, async SQL-executing reward | Part 1, question 3 |
| Abstain mechanism | Structured JSON output, not a new vocabulary token | Part 1, question 3 |
| Loss settings | `dr_grpo` loss, `scale_rewards=False`, KL β=0, group size 8, learning rate about 1e-6 | Part 1, question 3 |
| Training data | Synthetic generator; about 30% unanswerable, ambiguous, or out-of-scope | Part 1, question 2 |
| External eval | BIRD dev untouched by training; TrustSQL test for RS(c) | Part 1, question 2 |
| Correctness | Multi-instance execution match, 3–5 instances per schema | Part 1, question 2 |
| Guarantee | Learn-then-Test on sequence log-probability | Part 1, question 5 |
| Budget path | 20-step smoke test, then 8 runs of 300 steps | Part 3 |

## The reward

The reward is the project's core, so it is pinned down before any code. The accuracy-only baseline uses the same prompt minus the abstain option and scores +1 for correct, 0 for everything else.

| Question type | Model output | Reward | Baseline reward |
|---|---|---|---|
| Answerable | SQL that matches gold on every instance | +1 | +1 |
| Answerable | SQL that is wrong on any instance, errors, or times out | −c | 0 |
| Answerable | Abstains | 0 | 0 |
| Unanswerable, ambiguous, or out-of-scope | Abstains | +1 (decision D1) | not trained on these |
| Unanswerable, ambiguous, or out-of-scope | Any SQL | −c | not trained on these |
| Any | Output that does not parse | −c | 0 |

Three rules sit behind the table. "Wrong on any instance" means a query must match gold on all 3–5 data instances to earn +1; one coincidental match is not enough. Unparseable output scores −c, so the abstain field is the only safe exit; if garbage scored 0, the model could abstain without saying so. The baseline skips injected questions because they carry no learning signal without an abstain option: every sample in the group scores 0, so GRPO has nothing to compare.

No length bonus, no partial credit, no format bonus beyond the parse check. The report's evidence says shaped rewards invite reward hacking; the simplest reward is the hardest to game.

## Gates

A gate is a check that must pass before money or time moves to the next phase.

| Gate | When | Pass condition | If it fails |
|---|---|---|---|
| G1: Data | End of week 1 | The checker rejects every seeded coincidental match; at least 500 clean training prompts exist | Fix it before any GPU spend |
| G2: Smoke test | Week 2 | The primary model trains 20 steps cleanly; projected matrix cost is at most about 2/3 of the remaining budget | Switch to SmolLM3-3B the same day; cut the length cap or move to the 4090 |
| G3: No hacking | Week 2, after the first 2 runs | A hand audit finds no exploit; abstention on answerable questions stays low | Fix the checker or data and rerun before the matrix |
| G4: Guarantee holds | Week 3 | On 1,000 re-drawn splits, realized selective risk sits at or below α in about 1−δ of them | Debug the calibration code before the shift experiment |
| G5: Still novel | Start of week 4 | A fresh search finds no paper with all three ingredients | Reframe the write-up around the remaining differentiators |

---

## Phase 0: Setup

Week 1, days 1–2. Cost: $0.

- [x] **0.1 Scaffold the repo.** Create the layout in [Repo structure](#repo-structure), a `pyproject.toml`, and a `tests/` folder. Add `.gitignore` entries for `data/instances/`, `data/external/`, and `runs/`. Set up two environments from the same pins: a local CPU one for data, checker, and analysis, and a cloud GPU image for training and evaluation.
- [x] **0.2 Pin versions on day 1.** Pin exact versions of TRL (target v1.17.x), vLLM (0.19 or later, which supports Qwen3.5 in-tree and includes its LoRA fix), transformers, and PEFT. Never float them; a silent upgrade mid-project would make runs incomparable. Pinned 2026-11-04: TRL 1.17.1, vLLM 0.32.0 (the newest TRL 1.17.1 accepts), torch 2.13.0, transformers 5.19.0, PEFT 0.21.2; `uv.lock` holds the rest.
- [x] **0.3 Check the 3 models.** Download configs and tokenizers (not weights) for `Qwen/Qwen3.5-2B`, `HuggingFaceTB/SmolLM3-3B`, and `google/gemma-4-E2B-it`. For each, record the chat template, the vocabulary size, the license, and how thinking mode turns on and off. Qwen3.5-2B's card warns that the 2B size is prone to thinking loops, so its thinking controls matter most. Done 2026-11-04: all 3 load under the pinned transformers 5.19.0; `train/inspect_models.py` regenerates `results/models.md`. All 3 are Apache 2.0. Qwen3.5-2B is `Qwen3_5ForConditionalGeneration` (closing the report's unverified architecture name) and its template defaults to thinking off; SmolLM3-3B defaults to thinking on behind a long built-in system prompt that step 1.7 must override; Gemma 4 E2B turns thinking on with a `<|think|>` system turn.
- [ ] **0.4 Get the evaluation data.** Download BIRD dev (1,534 questions) and the TrustSQL test set into `data/external/`, with a script that re-downloads them; keeping them out of git avoids redistributing CC BY-SA data. Check whether TrustSQL's EHRSQL-derived portion needs the MIMIC/eICU databases, which sit behind PhysioNet's credentialed-access agreement. If it does, either start that application now (it takes time) or drop that portion and say so in the write-up.
- [ ] **0.5 Open the cloud accounts.** Create Modal and RunPod accounts and set a spending limit or alert on each if the platform offers one. Check in Modal's dashboard whether the $30 Starter credit covers GPU compute: the pricing page does not say, so that is UNVERIFIED until the dashboard confirms it.
- [ ] **0.6 Start 2 logs.** `results/spend.md` gets one line per cloud session: date, platform, GPU, minutes, dollars, purpose. `results/decisions.md` gets each open decision from the [table below](#open-decisions): the date it was made, the choice, and why.

**Done when:** both environments install from the pins, and all 3 model configs load.

## Phase 1: Synthetic data generator

Week 1, days 2–4. Cost: $0.

The generator produces fresh schemas that no model has seen, several data instances per schema, answerable questions with gold SQL, and labeled questions the model should decline.

- [ ] **1.1 Pick domains and group them into families.** Start from Gretel's 100-domain list for variety ([HF dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)). Group the domains into 4–6 schema families, for example retail, healthcare, finance, and logistics. The families matter in Phase 9: the shift experiment calibrates on one family and tests on another.
- [ ] **1.2 Generate schemas.** Version 0 builds schemas programmatically from per-domain templates: 3–8 tables, realistic table and column names, primary and foreign keys, and a mix of text, integer, real, and date columns. Programmatic generation is free, deterministic, and contamination-free. Move to version 1 only if version 0's schemas look too uniform: there, an LLM designs each schema from a business scenario, following SynSQL-2.5M's pipeline ([arXiv:2503.02240](https://arxiv.org/html/2503.02240v2)). Use a non-Qwen model for that step, for the same reason the report gives against reusing SynSQL's Qwen-generated data: a Qwen student trained on Qwen output drifts toward self-distillation.
- [ ] **1.3 Fill the tables.** Fill each schema 3–5 times with different random data, each instance from its own fixed seed so any instance can be regenerated exactly. Shape the value distributions so typical filters return rows on every instance; a WHERE clause on a category column should hit each time. Keep tables at tens to a few hundred rows so execution stays fast inside the reward.
- [ ] **1.4 Write answerable questions, SQL first.** Generate gold SQL from templates that cover the common query shapes: single-table filter, aggregate with GROUP BY, multi-table JOIN, ORDER BY with LIMIT, nested subquery, and date range. Then back-translate each query into a plain-English question: templates in version 0, an LLM paraphrase in version 1 for more natural wording. Tag every item with its shape and difficulty, so results can be broken down later.
- [ ] **1.5 Inject the 3 decline categories, each with a known label.** These follow TrustSQL's taxonomy ([arXiv:2403.15879](https://arxiv.org/html/2403.15879v2)).
  - *Unanswerable:* take an answerable question and delete, from the schema the model sees, the column it needs. Cover TrustSQL's 3 column types (column-surface, column-related, column-unrelated).
  - *Ambiguous:* questions with an undefined metric, such as "Who are our best customers?" when nothing in the schema says whether "best" means revenue or order count. BAR-SQL's metric-rejection mutation is prior art for this style.
  - *Out-of-scope:* requests SQL cannot serve, such as an action ("email this list to the team") or a forecast ("what will sales be next quarter?"). These match TrustSQL's non-sql and out-of-domain types.
  - Store each item's category and a one-line reason. The model's own abstain reason is never scored, but comparing it to this label shows whether it abstains for the right reason.
- [ ] **1.6 Set the mix.** About 70% answerable, about 30% injected, spread across the 3 categories. The report's reward-hacking analysis says to keep answerable questions at 70% or more to avoid abstention collapse, and Abstain-R1 used about the same 30/70 split. Target about 1,000 clean training prompts; the gate minimum is 500.
- [ ] **1.7 Fix the prompt format.** One template, identical in training and evaluation: the schema as CREATE TABLE statements, the question, and the output contract. Set the thinking switch explicitly for every model and never rely on its default; the defaults disagree (SmolLM3-3B thinks by default, Qwen3.5-2B and Gemma 4 E2B do not; see `results/models.md`), and SmolLM3-3B's built-in system prompt pushes it toward long reasoning, so our prompt must override it. The model may reason first, then must end with one JSON object (decision D2):

  ```json
  {"sql": "SELECT name FROM customers WHERE city = 'Austin'", "abstain": false, "reason": ""}
  {"sql": null, "abstain": true, "reason": "the schema has no column for refund dates"}
  ```

**Done when:** one command writes 50 schemas, 3–5 instances each, and a JSONL file of labeled questions, and a hand check of 20 items finds no broken gold query.

## Phase 2: Execution checker

Week 1, days 3–5. Cost: $0.

The checker is the correctness oracle: it decides what "right" means, and the guarantee is only as honest as it is.

- [ ] **2.1 Sandbox execution (`checker/execute.py`).** Open every SQLite instance read-only, so a model-written DROP or UPDATE cannot damage it. Reject anything that is not a single SELECT (or WITH ... SELECT) before it runs. Enforce a per-query timeout of 2–5 s with SQLite's progress handler, and cap the rows returned. Return a structured result: rows, an error, or a timeout.
- [ ] **2.2 Define result comparison.** Compare predicted and gold results as multisets of rows: duplicates count, row order does not, unless the gold query ends in ORDER BY, in which case order counts too. Round floats to a fixed precision, treat NULL as equal to NULL, and compare columns by position. Write every rule into the docstring; the write-up must state the oracle exactly.
- [ ] **2.3 Build the multi-instance oracle (`checker/testsuite.py`).** A prediction is correct only if it matches gold on every instance of its schema. Reuse the conventions of `test-suite-sql-eval` (Zhong et al. 2020, [GitHub](https://github.com/taoyds/test-suite-sql-eval)) where they fit.
- [ ] **2.4 Filter gold-empty items.** Drop any item whose gold query returns zero rows on any instance. This closes the exploit the research flagged: otherwise junk SQL that returns nothing scores +1, while honest abstention scores 0.
- [ ] **2.5 Write adversarial tests (`tests/test_checker.py`).** Seed known-wrong queries that match gold on one instance by coincidence: an off-by-one filter that happens to return the same rows, a missing JOIN condition against a one-row table, a constant SELECT. The oracle must reject every one. Also test that write statements are refused, a runaway cross join times out, and malformed SQL returns an error instead of crashing.
- [ ] **2.6 Measure checker speed.** Count completions checked per second on the laptop with parallel workers. The reward runs the checker inside every training step, so a slow checker slows training directly.

**Done when:** every adversarial test passes, and the pipeline yields at least 500 clean training prompts after filtering. This is gate G1.

## Phase 3: Splits and the first look at the model

Week 1, days 4–5. Cost: $0, or Modal free credit if approved.

- [ ] **3.1 Split by schema, never by question.** Assign whole schemas to splits, so no schema appears in two: train, dev (for choosing signals and settings), calibration, test, and held-out families for the shift experiment. Size the calibration split generously. LTT needs several hundred *answered* calibration questions; at an assumed coverage of about 60%, that means at least about 1,000 calibration questions (a sizing estimate to recheck once coverage is measured). Aim for about 1,000 test questions so the confidence intervals stay tight.
- [ ] **3.2 Run the brief's optional spike.** Push 50 schema/question pairs through the generator and checker end to end, run a model zero-shot (with no training), and record what broke. Run it twice, once with thinking on and once off, and record output length and how often answers hit the length cap. Thinking on is where Qwen3.5-2B's loop risk lives, and it is also what decision D2's default implies, so this gives D2 evidence before step 4.1 settles it. Two ways to run the model:
  - *On the cloud (recommended):* serve base Qwen3.5-2B with vLLM on Modal free credit, about an hour, likely $0 in cash. It exercises the exact stack training will use, so tooling problems surface a week early. It still needs a go-ahead, like every cloud session.
  - *On the laptop:* a GGUF build through llama.cpp Vulkan. Whether llama.cpp supports Qwen3.5-2B's hybrid architecture is UNVERIFIED; if it does not, use SmolLM3-3B's GGUF instead.
- [ ] **3.3 Record the zero-shot baseline:** execution accuracy, how often the model abstains when told it may, how often its output parses, and every failure. These numbers are the "before" in every later before-and-after.

## Phase 4: Training code

Week 2, days 1–2. Cost: $0.

- [ ] **4.1 Write the reward (`train/reward.py`).** Parse the last JSON object in the completion, apply the [reward table](#the-reward), and call the checker. Make it async, which TRL supports, so one slow query cannot stall the whole group. If the JSON has both SQL and `"abstain": true`, abstain wins and the SQL is ignored. Settle decisions D1 and D2 here.
- [ ] **4.2 Test the reward on the laptop (`tests/test_reward.py`).** Fake a completion for every row of the reward table, plus the edge cases: malformed JSON, two JSON objects, reasoning text before the JSON, SQL that times out, and abstain set alongside SQL.
- [ ] **4.3 Write the training script (`train/grpo_train.py`).** TRL GRPOTrainer with `loss_type="dr_grpo"`, `scale_rewards=False`, `beta=0.0`, `num_generations=8`, learning rate about 1e-6, clip 0.2, `mask_truncated_completions=True`, and LoRA r=16 through `peft_config`. Start `max_completion_length` at 1024 and raise it toward 2048 only if many completions hit the cap. Start vLLM colocate at `vllm_gpu_memory_utilization=0.3` and raise it if memory sits idle. Every run reads one YAML file from `train/configs/`: model, c, seed, steps.
- [ ] **4.4 Log what reveals failure early.** Per step: mean reward, abstention rate on answerable questions, abstention rate on injected questions, completion length, truncation rate, and seconds per step. Abstention on answerable questions is the collapse alarm: flag any run where it passes 50% and keeps climbing for 25 steps (a starting threshold, tuned after the first runs).
- [ ] **4.5 Checkpoint and prove resume works.** Save the LoRA adapter every 25–50 steps to persistent storage (a Modal volume or a RunPod network volume). Test resume from a checkpoint before the smoke test; community machines can be interrupted mid-run.
- [ ] **4.6 Write the cloud runners (`infra/`).** One script per platform: build the pinned image, mount data and checkpoints, run a config, copy logs back. The smoke test and the full runs use the same command, so a passing smoke test proves the real path.
- [ ] **4.7 Dry-run on the laptop.** Run 2 training steps on CPU with a tiny model and vLLM off, to catch config errors before paying for a GPU.

**Done when:** the reward tests pass and the dry run completes.

## Phase 5: Smoke test

Week 2, day 3. Cost: at most $0.60 (estimate).

- [ ] **5.1 Get the go-ahead.** This is the first real GPU spend.
- [ ] **5.2 Run 20 steps** on Modal free credit or a RunPod Secure A100: group size 8, 8 prompts per step, completions capped at 512 tokens, LoRA r=16, vLLM colocate. Checkpoint and resume once along the way. Expect 5–20 minutes.
- [ ] **5.3 Check every item:**
  - vLLM starts and generates for Qwen3.5-2B, in text-only mode if needed. This is the architecture-risk check.
  - Rewards vary within groups. A constant reward means zero learning signal under GRPO.
  - Reward, length, and abstention curves all log.
  - Checkpoint save and resume both work.
  - Peak GPU memory, which decides whether the $0.34/hr RTX 4090 (24 GB) can run the matrix.
- [ ] **5.4 Replace the estimate with a measurement.** Record real seconds per step, which replaces the planning assumption of 800–2,000 tokens/sec end to end, and recompute the matrix cost in `results/spend.md`.
- [ ] **5.5 Pass gate G2.** Qwen3.5-2B stays primary only if 5.3 passes cleanly; otherwise switch to SmolLM3-3B the same day and rerun the smoke test (decision D5). If SmolLM3-3B becomes primary, the control run uses Gemma 4 E2B, budgeted as a 5.1B-parameter model in memory, or is dropped. Proceed to full runs only if measured seconds per step × 300 steps × 8 runs costs at most about 2/3 of the remaining budget. If the A100 measures over 25 seconds per step, cut completions to 256 tokens or move seed replicas to the 4090.

## Phase 6: First 2 runs and the hacking audit

Week 2, days 4–5. Cost: about $5 (estimate).

- [ ] **6.1 Run 2 jobs of 300 steps each:** the accuracy-only baseline and one abstention run at the middle penalty, c = 1. Each costs about $2.45 on a Secure A100 at the central estimate.
- [ ] **6.2 Audit 100 sampled completions per run by hand,** drawn at steps 100, 200, and 300. Look for:
  - SQL that returns empty or constant results;
  - degenerate queries that pass the oracle anyway (this should be impossible; any found is a checker bug);
  - abstaining on easy answerable questions;
  - reasoning loops that hit the length cap;
  - abstain reasons that do not match the question's actual problem.
- [ ] **6.3 Read the curves.** Reward should rise, abstention on answerable questions should stay low, and abstention on injected questions should climb. Unsloth's guidance says to expect no reward movement before about 300 steps, so a flat curve at step 150 is not yet a failure. If reward is still flat at step 300, extend the c = 1 run to 500 steps (about $4.08 per run) before launching the matrix, and make 500 the matrix length (decision D6).
- [ ] **6.4 Pass gate G3.** Pass only if the audit finds no exploit and abstention behaves as described. Any exploit means fixing the checker or data, rerunning both jobs, and holding the matrix.

## Phase 7: The run matrix

Week 3, days 1–3. Cost: the full 8-run matrix is about $16–36 including overhead (estimate); Phase 6 already paid for 2 of the runs.

- [ ] **7.1 Run the rest of the matrix.** c ∈ {0.5, 1, 4} × 2 seeds (the Phase 6 run at c = 1 counts as one of them), plus an optional control-model run at the best c. Each runs for the length Phase 6 settled on.
- [ ] **7.2 Use the cheap hardware.** A RunPod Community A100 ($1.19/hr), or the 4090 ($0.34/hr) if 5.3 showed the model fits in 24 GB. Checkpoint every 25–50 steps, since community machines can be reclaimed.
- [ ] **7.3 Degrade in a fixed order if money or time runs short:** drop the control run first, then run second seeds only for the best and worst c, then move the remaining runs to the 4090.
- [ ] **7.4 Generate every evaluation output once.** For each final checkpoint, run dev, calibration, test, the shift families, BIRD dev, and TrustSQL test, and save one row per question: the raw output, the abstain flag, the SQL, correctness, and the sequence log-probability (the model's own probability for its output, which vLLM can return). On dev and calibration, optionally draw 5 samples per question for the agreement signal. Generation happens once; every later phase reads these files instead of re-running the model.

## Phase 8: Calibration

Week 3, days 3–4. Cost: $0, on the laptop.

- [ ] **8.1 Choose the confidence signal on the dev split** (decision D3), never on the calibration split; choosing on calibration data would void the guarantee. Candidates: the SQL's sequence log-probability, both summed and normalized by length, and 5-sample execution agreement if 7.4 collected it. Pick the one with the lowest AURC on dev.
- [ ] **8.2 Implement LTT (`calib/ltt.py`)** ([arXiv:2110.01052](https://arxiv.org/abs/2110.01052)). Walk candidate cutoffs from most conservative to most permissive. At each cutoff, the answered set is every question the policy did not abstain on whose confidence clears the cutoff. Count n answered and k wrong, compute the p-value P(Binomial(n, α) ≤ k), and certify the cutoff if the p-value is at most δ. Stop at the first cutoff that fails; this fixed-sequence walk avoids the full multiple-testing penalty. The certified cutoff is the most permissive one that passed.
- [ ] **8.3 Certify at α ∈ {0.05, 0.1, 0.2} with δ = 0.05.** Compute the CRC bound on "answered and wrong" alongside it as the secondary result.
- [ ] **8.4 Pass gate G4: prove the guarantee holds where it should.** Pool the calibration and test schemas and re-split them 1,000 times, the CRC paper's own protocol. In each split, certify on calibration and measure realized selective risk on test. Realized risk should sit at or below α in about 1−δ of the splits, allowing for test-set noise. If it does not, the code has a bug; fix it before Phase 9.
- [ ] **8.5 Handle thin coverage.** If too few questions get answered for LTT to certify anything at α = 0.05, generate a larger calibration split; synthetic data costs nothing.
- [ ] **8.6 Unit-test LTT (`tests/test_ltt.py`)** on simulated data with a known true error rate, checking that it certifies at the right rate.

## Phase 9: Shift experiment

Week 3, day 5. Cost: $0.

The guarantee assumes new questions look like the calibration questions. This phase measures how badly it breaks when they do not. No published study does this for text-to-SQL, so a guarantee that breaks badly is still a result.

- [ ] **9.1 Fill the overshoot matrix.** For every ordered pair of schema families (A, B): calibrate on A, test on B, and record realized selective risk, coverage, and overshoot (realized risk minus α). Add BIRD dev and TrustSQL's 4 source datasets as extra test sets.
- [ ] **9.2 Run the i.i.d. control.** Calibrate and test on disjoint halves of the same family. This separates ordinary sampling noise from true shift.
- [ ] **9.3 Plot it** as a heatmap of overshoot with bootstrap intervals, one per α.

## Phase 10: Baselines

Week 3, alongside Phases 8–9. Cost: a few dollars of API calls, with a go-ahead.

- [ ] **10.1 Prompted abstention.** The untrained base model with the same prompt. The published anchor: frontier models prompted this way abstain less than 1% of the time, with error rates above 10%.
- [ ] **10.2 Self-consistency thresholding.** Sample k answers from the base model and answer only if their executed results agree; sweep the agreement threshold to trace a curve. TrustSQL found 5-sample unanimous voting "notably strong", so k = 5 is the starting point.
- [ ] **10.3 Accuracy-only GRPO.** The Phase 6 baseline, scored with the same confidence signal, so its risk–coverage curve isolates what the abstention reward buys.
- [ ] **10.4 Frontier models.** 1–2 models through an API, prompted to abstain, on the synthetic test set and a TrustSQL subset. Cap the spend at a few dollars. Not every API returns log-probabilities, so these may give one operating point instead of a full curve.

## Phase 11: Evaluation

Week 4, days 1–2. Cost: $0.

- [ ] **11.1 Compute every metric** for every model on every evaluation set: execution accuracy, coverage, selective risk, the risk–coverage curve, AURC and E-AURC (AURC minus the best possible ranking), RS(c) at c ∈ {0, 10, N} where N is the test-set size, tokens per query, and realized risk at each certified cutoff. The minimum bar on TrustSQL is beating the abstain-on-everything floor of RS = 50.
- [ ] **11.2 Handle BIRD dev's quirks.** BIRD labels no unanswerable questions, so it measures coverage and selective risk on answerable questions only. It ships one database per schema, so its accuracy is single-instance, a weaker oracle than the synthetic sets use; say so beside every BIRD number. Decide whether to feed BIRD's per-question evidence hints, matching the setting of the published small-model numbers it gets compared to (decision D4). A VLDB audit found pervasive gold-label errors in BIRD Mini-Dev, so some "wrong" answers may be gold errors; spot-check the disagreements.
- [ ] **11.3 Measure local latency.** Merge the best LoRA adapter into its base model, convert it to GGUF, quantize it to 4 bits, and time it with llama.cpp Vulkan on the laptop: tokens per second and seconds per query. If llama.cpp cannot convert Qwen3.5-2B (UNVERIFIED), report SmolLM3-3B's latency instead and say so.

## Phase 12: Statistics

Week 4, days 2–3. Cost: $0.

- [ ] **12.1 Bootstrap confidence intervals** (re-sampling the results many times to see how much each number wobbles) over both test questions and seeds: resample seeds, then questions within each seed. Compare two models with a paired bootstrap on the same questions. The report notes that a canonical citation for the paired item-level bootstrap in NLP still needs fetching; fetch it before writing.
- [ ] **12.2 Report seeds honestly.** The rliable playbook recommends the interquartile mean over seeds ([arXiv:2108.13264](https://arxiv.org/abs/2108.13264)), but with 2 seeds it has nothing to trim. Report both seeds and their mean; use the interquartile mean only if a third seed fits the budget.
- [ ] **12.3 Plot the re-split distribution from 8.4:** realized risk across the 1,000 splits, with α marked.

## Phase 13: Write-up and release

Week 4, days 3–5. Cost: $0.

- [ ] **13.1 Pass gate G5: re-run the novelty search.** Search arXiv and OpenReview for anything since 2026-11-04 that combines RL, an abstention reward, and risk–coverage in text-to-SQL. The original search never crawled OpenReview, so this closes a known gap.
- [ ] **13.2 Make the figures:** the headline risk–coverage plot with every model on the synthetic test set; the RS(c) table; certified versus realized risk; the shift overshoot heatmap; training curves.
- [ ] **13.3 Write it.** Open on the trust problem: a tool that answers confidently and wrongly. Cite BAR-SQL, Reinforced Hesitation, and RLSR in the opening and state what each lacks. State the correctness oracle and the guarantee's assumptions plainly: new questions must resemble the calibration questions, the guarantee holds on average over calibration draws, and "correct" means what the checker says. Report negative results. Treat escalation, sending abstained questions to a bigger model or a human, as a deployment note, not a system to build.
- [ ] **13.4 Release.** Code on GitHub. The generator and its data under a license (decision D7); any Spider- or BIRD-derived content must carry CC BY-SA's share-alike terms. The LoRA adapter on the Hugging Face Hub, with a model card listing the certified α and δ, the oracle, and where the guarantee broke under shift.
- [ ] **13.5 Close the spend log** with the final total against the $30–50 budget.

---

## Open decisions

Each has a default, so no step blocks on it. Record the final choice in `results/decisions.md`.

| ID | Decision | Default | Decided at |
|---|---|---|---|
| D1 | Reward for abstaining on an unanswerable question | +1, so the training reward equals RS(c) question by question; the alternative is the brief's literal 0 | Step 4.1 |
| D2 | Reasoning before the JSON, or JSON only | Allow reasoning, capped by the length limit; parse the last JSON object | Step 4.1 |
| D3 | Confidence signal: summed or length-normalized log-probability, or sample agreement | Lowest AURC on dev | Step 8.1 |
| D4 | Feed BIRD's evidence hints | Match the published small-model numbers' setting | Step 11.2 |
| D5 | Qwen3.5-2B or SmolLM3-3B as primary | Qwen3.5-2B, if the smoke test passes cleanly | Step 5.5 |
| D6 | Run length: 300 or 500 steps | 300; 500 if reward is flat at step 300 | Step 6.3 |
| D7 | License for the synthetic data | A permissive license, unless Spider/BIRD content forces CC BY-SA | Step 13.4 |

## Schedule and spend

| Week | Phases | Cloud spend (estimate) | Ends with |
|---|---|---|---|
| 1 | 0–3 | $0; the spike runs on Modal free credit if approved | Gate G1 |
| 2 | 4–6 | About $5.50: smoke test plus 2 runs | Gates G2 and G3 |
| 3 | 7–10 | About $11–31 for the rest of the matrix, plus a few dollars of API calls | Gate G4 |
| 4 | 11–13 | $0 | Gate G5 and the release |

The total comes to about $16–36 plus a few dollars of API calls (estimate), inside the $30–50 budget; Modal's free credit may cover a large share of it. Slack in week 4 absorbs re-runs.

## Risks

| Risk | Early signal | Response |
|---|---|---|
| Qwen3.5-2B tooling friction | Errors in the smoke test | Switch to SmolLM3-3B the same day |
| Abstention collapse at high c | Abstention on answerable questions climbing past 50% | Lower c; confirm answerable share is 70% or more |
| Reward hacking through empty or degenerate SQL | The Phase 6 audit | Fix the checker or data; rerun before the matrix |
| Throughput below the estimate | Smoke-test seconds per step | 256-token cap; the 4090; the degrade order in 7.3 |
| Reward flat at 300 steps | A flat curve in Phase 6 | Extend runs to 500 steps |
| Too few answered calibration questions | LTT certifies nothing at α = 0.05 | Generate a larger calibration split |
| A new paper takes the idea | The gate G5 search | Reframe around the guarantee and the shift experiment |

## Repo structure

```
abstain-sql/
  README.md
  plan.md                    # this file
  pyproject.toml             # pinned versions
  uv.lock                    # exact versions of every package, for laptop and GPU alike
  research/
    REPORT.md                # findings, novelty table, cost model
    notes/                   # raw research notes behind the report
  data/
    generator/               # schemas, table filling, question templates, injection
    instances/               # 3-5 SQLite instances per schema (gitignored, regenerable)
    external/                # BIRD dev, TrustSQL (gitignored, download script)
    splits/                  # train / dev / calibration / test / shift-family manifests
  checker/
    execute.py               # read-only sandboxed execution, timeouts
    testsuite.py             # multi-instance agreement = correctness oracle
  train/
    reward.py                # parse the JSON, apply the reward table, async
    grpo_train.py            # TRL GRPOTrainer + vLLM colocate + LoRA
    configs/                 # one YAML per run: model, c, seed, steps
  infra/                     # Modal and RunPod runners
  calib/
    signals.py               # sequence log-probability, sample agreement
    ltt.py                   # binomial tests, fixed-sequence walk, CRC bound
  eval/
    generate.py              # one pass of outputs per checkpoint per eval set
    metrics.py               # EX, coverage, selective risk, AURC, RS(c)
    baselines.py             # prompted abstention, self-consistency, frontier APIs
    bootstrap.py             # question, seed, and calibration-split resampling
    local_gguf.py            # llama.cpp Vulkan latency harness
  tests/                     # checker, reward, and LTT tests
  runs/                      # checkpoints and logs (gitignored)
  results/                   # figures, tables, spend.md, decisions.md
```

The chart is the whole project. Every gate above exists so that when the chart shows a small model saying "I don't know," the chart can be trusted.
