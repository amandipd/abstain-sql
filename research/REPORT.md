# Abstaining Text-to-SQL via RL: Research Report

Research date: 2026-11-04. Every claim below traces to the research notes in `notes/`; any claim without a primary source is marked UNVERIFIED. The report runs in 3 parts: the findings and verdict ([Part 1](#part-1-findings-and-verdict)), the novelty table ([Part 2](#part-2-related-work-and-the-novelty-table)), and the cost model ([Part 3](#part-3-compute-estimate)). The step-by-step project plan lives in [plan.md](../plan.md). Where this report says "the brief," it means the original research brief, which `plan.md` replaced; it stays in git history (commit `1eb6f56`).

---

# Part 1: Findings and verdict

## Flags: findings that change the plan

1. **Qwen3.5-2B is not a plain text model.** `Qwen/Qwen3.5-2B` is a *multimodal* model (image/video input) with a hybrid architecture that alternates Gated DeltaNet and attention layers. That makes it the riskiest of the three candidate models for tooling: it required vLLM nightly at release, and its card warns the 2B size is "more prone to thinking loops" ([model card](https://huggingface.co/Qwen/Qwen3.5-2B)). Upstream vLLM (a fast batched text-generation engine) supports it in-tree by v0.19.0 ([release notes](https://github.com/vllm-project/vllm/releases/tag/v0.19.0)). Keep SmolLM3-3B ready as a drop-in fallback.
2. **Unsloth's embedded vLLM cannot serve Qwen3.5.** Unsloth's own RL guide says to set `fast_inference=False` for Qwen3.5, which means no fast vLLM generation inside Unsloth and strips its main speed advantage on the primary model ([Unsloth RL guide](https://unsloth.ai/docs/basics/reinforcement-learning-rl-guide)). This settles the stack choice in TRL's favor (question 3).
3. **"Gemma 4 E2B" is really 5.1B total parameters in memory.** The control model `google/gemma-4-E2B-it` has 2.3B *effective* parameters (active per token) but 5.1B total including embeddings, so it trains with the memory footprint of a 5B model ([model card](https://huggingface.co/google/gemma-4-E2B-it)). Budget VRAM accordingly, or use SmolLM3-3B as the control.
4. **Three near-misses narrow the novelty; none kills it.** BAR-SQL (RL + abstention credit, 1.7B, text-to-SQL, Jan 2026), Reinforced Hesitation (the exact +1/0/−λ reward, 1.7B, logic puzzles, Nov 2025), and RLSR (RL optimizing the risk–coverage trade-off directly, 7–8B, QA/math, Jul 2026) each hold two of the three ingredients. No paper holds all three. Any write-up must cite and differentiate all three in its opening (details in [Part 2](#part-2-related-work-and-the-novelty-table)).
5. **Plain conformal risk control does not bound the promised guarantee.** Conformal risk control (CRC), a statistical recipe that picks a confidence cutoff on held-out data so an error target provably holds on average, bounds the *unconditional* rate of "answered AND wrong." The brief's promise, "error ≤ α on the queries it answers," is *selective risk*: a conditional ratio, which CRC does not bound. **Learn-then-Test (LTT)**, which treats each candidate cutoff as a hypothesis test and gives a guarantee that holds with high probability, is the correct tool ([CRC paper](https://arxiv.org/abs/2208.02814); [LTT paper](https://arxiv.org/abs/2110.01052)). See question 5.
6. **Empty-result reward hacking interacts badly with an abstain reward of 0.** Many Spider/BIRD gold queries return empty results; a junk query that also returns nothing scores "correct" (+1) while honest abstention earns 0, so the RL optimizer would learn to emit junk SQL instead of abstaining on hard questions. Arctic-Text2SQL-R1 filtered about 1,400 BIRD and about 1,700 Spider examples for exactly this ([arXiv:2505.20315](https://arxiv.org/html/2505.20315v2)). Filter gold-empty examples and use multi-instance execution checks *before* any reward tuning.
7. **No license blocker.** Spider 1.0 and BIRD are CC BY-SA 4.0 (use freely with attribution; redistributed modified data keeps the same license); TrustSQL and EHRSQL are CC BY 4.0 (attribution only); Spider 2.0's repo shows MIT ([sources in Q2](#2-data)). Nothing prohibits training.
8. **The only unmeasured cost input is end-to-end training throughput.** No published benchmark exists for GRPO steps/hour on a model of about 2B parameters with LoRA + vLLM on one GPU. The 20-step smoke test (≤ $0.60, estimate) must run before any full run commits budget ([Part 3](#part-3-compute-estimate)).

---

## 1. Novelty check

**Verdict: the exact combination (RL training + explicit abstention reward + risk–coverage evaluation, on a ≤4B model, in text-to-SQL) appears unpublished as of 2026-11-04.** Every pairwise combination exists. (A risk–coverage curve plots the error rate among answered questions against the fraction of questions answered: the standard way to evaluate a model that may decline.)

What exists, by cell:

- **RL for text-to-SQL, no abstention:** Arctic-Text2SQL-R1 (7B/14B/32B, GRPO, reward 1/0.1/0 on execution match; BIRD dev 68.9–70.5) ([arXiv:2505.20315](https://arxiv.org/abs/2505.20315)); SQL-R1 (3B/7B/14B, composite reward; 3B gets 54.6 BIRD dev) ([arXiv:2504.08600](https://arxiv.org/abs/2504.08600)); Reasoning-SQL (14B, partial rewards incl. LLM judge) ([arXiv:2503.23157](https://arxiv.org/abs/2503.23157)); plus a dozen newer systems (CSC-SQL, SLM-SQL at 0.5–1.5B, Think2SQL, MTIR-SQL-4B, SkyRL-SQL, Graph-Reward-SQL, ReViSQL-BIRD, Progress-SQL, SQL-Trail, EXPO-SQL, …). None has an abstain action or selective-prediction metrics. Small-model anchor: well-trained ≤4B RL models reach about **54–67% execution accuracy on BIRD dev** (execution accuracy = the generated SQL counts as right only if running it returns the same result as the reference query); the top of that range, about 67%, needs 64-sample voting (SLM-SQL 1.5B) ([arXiv:2507.22478](https://arxiv.org/pdf/2507.22478)).
- **Abstention in text-to-SQL, no RL:** TrustSQL (benchmark + penalty score RS(c)) ([arXiv:2403.15879](https://arxiv.org/abs/2403.15879)); the EHRSQL-2024 shared task (ranking metric RS(10); none of the 8 teams used RL; no team achieved positive RS(N)) ([arXiv:2405.06673](https://arxiv.org/html/2405.06673v2)); post-hoc confidence/selective-prediction studies (Somov & Tutubalina AAAI 2025; Richardson 2026; an adaptive-abstention conformal pipeline). All of these are inference-time add-ons.
- **RL-trained abstention outside SQL:** Reinforced Hesitation (+1/0/−λ, Dr.GRPO, Qwen3-1.7B, logic puzzles; reports Pareto frontiers over λ, not standard risk–coverage) ([arXiv:2511.11500](https://arxiv.org/html/2511.11500v2)); Rewarding Intellectual Humility (ternary reward sweep, Granite-3.3-2B and Qwen-3-4B, multiple-choice QA, no risk–coverage per abstract) ([arXiv:2601.20126](https://arxiv.org/abs/2601.20126)); TruthRL, RLKF, KARL, Knowing-When-to-Quit (sizes/metrics partly UNVERIFIED from abstracts); and RLSR, which reports full risk–coverage curves and AURC but trains no abstain action and uses 7B/8B models ([arXiv:2607.03528](https://arxiv.org/html/2607.03528v1)).
- **The single RL + abstention text-to-SQL paper: BAR-SQL** (Qwen3-1.7B + GRPO; refusals scored by text similarity to a gold refusal; no penalty parameter c, no abstain token, **no risk–coverage, no RS(c), no Spider/BIRD results**; private enterprise benchmark only) ([arXiv:2601.10318](https://arxiv.org/html/2601.10318)).

**Smallest differentiating angle**, if a reviewer treats the near-misses as prior art: train the ternary penalty-swept reward with an *explicit structured abstain action* in text-to-SQL with execution-grounded rewards, evaluate with risk–coverage curves *and* TrustSQL's RS(c) on public benchmarks, and add the conformal/LTT guarantee layer plus the schema-shift stress test. None of BAR-SQL, Reinforced Hesitation, or RLSR has any of those last three elements. The notes flag two caveats: 2026 preprint volume is high and the search did not crawl OpenReview, so an under-review collision is possible; and Reinforced Hesitation's λ-sweep is functionally close to a risk–coverage analysis, so cite it explicitly.

## 2. Data

**Licenses (verified 2026-11-04):** BIRD: CC BY-SA 4.0 ([bird-bench.github.io](https://bird-bench.github.io/); its *earlier* license is UNVERIFIED). Spider 1.0: CC BY-SA 4.0 ([yale-lily.github.io/spider](https://yale-lily.github.io/spider)). Spider 2.0: the repo shows MIT; whether the dataset files carry a separate license is UNVERIFIED ([github.com/xlang-ai/Spider2](https://github.com/xlang-ai/Spider2)). TrustSQL: CC BY 4.0 ([repo](https://github.com/glee4810/TrustSQL)). EHRSQL: CC BY 4.0, but the underlying MIMIC/eICU patient databases sit behind PhysioNet data-use agreements ([repo](https://github.com/glee4810/EHRSQL)).

**Unanswerable/ambiguous coverage:** TrustSQL labels about 2,468 of 21,356 questions unanswerable (about 11.6%), **test-only**, in five types (column-surface, column-related, column-unrelated, non-sql, out-of-domain) ([arXiv:2403.15879](https://arxiv.org/html/2403.15879v2)). EHRSQL puts unanswerables at 33% of validation/test and **none in training** ([ar5iv 2301.07695](https://ar5iv.labs.arxiv.org/html/2301.07695)). BIRD, Spider 1.0, and Spider 2.0 label no unanswerables; both BIRD and Spider 2.0 have publicly acknowledged ambiguity problems, and a VLDB audit reports pervasive gold-label errors in BIRD Mini-Dev and Spider 2.0-Snow ([VLDB p931](https://www.vldb.org/pvldb/vol19/p931-jin.pdf)).

**Contamination (benchmark leakage into pretraining):** no direct measurement exists for Spider or BIRD in Qwen or Gemma; the question is UNVERIFIED in both directions. The circumstantial evidence: a GPT-3.5 study found a significant drop on the unfamiliar Termite set vs. Spider ([arXiv:2402.08100](https://arxiv.org/abs/2402.08100)); Qwen2.5-Coder's decontamination list names only code/math benchmarks ([arXiv:2409.12186](https://arxiv.org/pdf/2409.12186)); Gemma 3's report describes generic eval-set removal with no benchmark list ([arXiv:2503.19786](https://arxiv.org/abs/2503.19786)). A related hazard: RL gains measured on contaminated benchmarks can be illusory ([arXiv:2507.10532](https://arxiv.org/html/2507.10532v1)). Qwen3.5-2B is newer than every contamination study found.

**Decision: hybrid, defaulting to synthetic.** Synthetic training data gives fresh schemas and therefore no contamination, free injection of unanswerables, and full control of gold-empty results; **BIRD dev stays untouched as the external evaluation set**, with TrustSQL/EHRSQL as external reliability benchmarks for RS(c) comparison. Existing datasets alone fail for one reason: neither provides unanswerable *training* examples (see above).

**Synthetic generator design.** The blueprint is SynSQL-2.5M's pipeline (Apache-2.0, generated entirely with open models: web tables → LLM designs a multi-table business schema → complexity-controlled SQL generation → SQL-to-question back-translation → execution-based majority-vote filtering; 96% of its databases judged realistic) ([arXiv:2503.02240](https://arxiv.org/html/2503.02240v2)):

- Schema generation with realistic names: follow SynSQL's scenario-inference step; Gretel's 100-domain list is a second source of domain variety ([HF dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)).
- Table filling: populate 3–5 *different* data instances per schema (this doubles as the test-suite checker, below), with value distributions that make empty results rare and detectable.
- Question templates: SQL-first, back-translated to natural language, SynSQL-style.
- Injected categories with known labels: **unanswerable** (ask about a column deliberately removed: TrustSQL's column-surface/related/unrelated types), **ambiguous** (undefined metric: BAR-SQL's "metric rejection" mutation is prior art for the mutation style), **out-of-scope** (non-SQL asks: TrustSQL's non-sql/out-of-domain types). The notes found **no prior art synthesizing unanswerable text-to-SQL at scale**: a secondary contribution.
- Caveat: Qwen2.5 models generated most of SynSQL, so reusing the data (rather than the pipeline) is self-distillation-adjacent for a Qwen student.

**Test-suite accuracy (Zhong et al. 2020)** is verified real and fit for purpose. `test-suite-sql-eval` (Apache-2.0, SQLite) executes predicted and gold SQL against *multiple* distilled database instances per schema so a wrong query can't match by coincidence; it is Spider's official metric ([github.com/taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval); [arXiv:2010.02840](https://arxiv.org/abs/2010.02840)). BIRD's harness executes on a single database only ([github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)). The gap: no maintained package *generates* test suites for new synthetic databases (Zhong's generation repo exists; its current state is unchecked). For synthetic data, generate the 3–5 instances at schema-creation time instead.

## 3. Training stack

**Recommendation: TRL GRPOTrainer (v1.17.x) with vLLM colocate on a single GPU.** (GRPO = Group Relative Policy Optimization, an RL algorithm that scores a group of sampled answers against each other instead of training a separate value network; colocate = vLLM shares the training GPU, the TRL default.) The scorecard against the brief's four requirements, from the notes' comparison:

- **Qwen3.5 support:** TRL works via current transformers + vLLM ≥0.17 (LoRA fix in 0.19). `verifiers` is UNVERIFIED for the hybrid architecture. Unsloth's embedded vLLM is unsupported, forcing `fast_inference=False` (flag 2).
- **LoRA (low-rank adapters: training a small add-on instead of all weights):** TRL supports it via `peft_config`. `verifiers` RLTrainer "supports LoRA" (mirror-sourced, UNVERIFIED). Unsloth supports it with the best VRAM story.
- **Custom SQL-executing reward:** all three accept arbitrary Python; TRL additionally supports **async** reward functions (`asyncio.gather`), which matters when one slow query would otherwise stall a whole generation group ([TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)).
- **Single GPU:** TRL colocate is default-documented with memory knobs (`vllm_gpu_memory_utilization`, sleep mode, `beta=0.0` skips the reference model). `verifiers` is churn-heavy (GRPOTrainer → RLTrainer → Taskset model in one year) and oriented to Prime Intellect's stack. Unsloth is the VRAM leader; keep it as fallback for the control model, or for the case where a 24GB GPU becomes the cost-saving path.

**Abstain mechanism: structured output, not a special token.** Emit `{"sql": ..., "abstain": bool, "reason": ...}` (or a reserved tag) and parse it in the reward. The reasons: every RLVR abstention paper found uses emitted text; a new vocab token forces training of the embedding/output layers (`modules_to_save`), a large VRAM hit at 150K–262K-token vocabularies, and **TRL's fused LM-head scoring raises an error when a PEFT adapter touches `lm_head`** ([TRL releases](https://github.com/huggingface/trl/releases)); vLLM's Qwen3.5 LoRA path needed an upstream fix as late as v0.19.0. The [IDK]-token line (NeurIPS 2024) used full continued pretraining, not RL ([arXiv:2412.06676](https://arxiv.org/abs/2412.06676)). No head-to-head token-vs-text study under GRPO exists: an open question, not worth this project's budget.

**Loss settings** (the consensus of the Dr. GRPO paper, TRL docs, and Arctic/SQL-R1 settings):

- `loss_type="dr_grpo"` + `scale_rewards=False`: Dr. GRPO removes the per-response length division and the reward standard-deviation division, two normalizations shown to bias response length ([arXiv:2503.20783](https://arxiv.org/html/2503.20783v2)). TRL's default `"dapo"` addresses length bias differently; "dr_grpo vs dapo" is a cheap ablation.
- KL coefficient β = 0 (Dr. GRPO's setting; TRL default; skips the reference model and saves memory). Add β of about 1e-3 (Arctic's value) only if degeneration appears.
- Group size 8 (SQL-R1's and TRL's default; Arctic found no gain beyond 16 rollouts). Learning rate about 1e-6, clip 0.2 (optional upper 0.28).
- Generation cap 1024–2048 tokens with `mask_truncated_completions=True`: TRL's 512 default is too small for think-then-SQL; SQL-R1 used 2048; Qwen3.5-2B's thinking-loop proneness argues for a hard cap.

**Reward-hacking design points:**

- *Abstain-on-everything at high c* is a documented failure mode ("abstention trap", KARL; a fixed abstention reward "can lead a model to refuse nearly everything", AWA-RL) ([arXiv:2604.22779](https://arxiv.org/abs/2604.22779); [alphaXiv 2607.10738](https://www.alphaxiv.org/abs/2607.10738)). Mitigations: sweep c across runs (the run matrix does this), log abstention rate per training step with an alarm threshold, and keep answerable questions at about 70% or more of the training mix (the Hallucination Tax paper restored refusal with just 10% unanswerable data) ([arXiv:2505.13988](https://arxiv.org/abs/2505.13988)).
- *Empty-result matching* is flag 6. Filter gold-empty training examples (Arctic's fix) and/or require non-empty agreement across instances.
- *Degenerate SQL matching on small tables:* use the multi-instance test-suite check as the reward's correctness oracle, not single-DB execution; add a per-query timeout (2–5 s) in the reward sandbox (SQL-R1 limits execution time; timeouts also keep one slow query from stalling a GRPO group).
- *General shaping:* keep the reward minimal (+1 / 0 / −c, plus at most a small format gate). Arctic argues complex shaped rewards invite "superficial reward hacking", and SQL-R1 saw hacking when execution/result terms were underweighted ([arXiv:2505.20315](https://arxiv.org/html/2505.20315v2); [arXiv:2504.08600](https://arxiv.org/html/2504.08600v5)). No length bonus.

## 4. Compute estimate

Full detail with date-stamped prices lives in [Part 3](#part-3-compute-estimate). The cost model: cost = steps × prompts × group size × tokens ÷ throughput × $/hr. At verified 2026-11-04 prices (RunPod A100 80GB $1.79/hr Secure / $1.19 Community; RTX 4090 $0.34/hr Community; Modal A100 about $2.50/hr with **$30/month free Starter credit**), the central estimate is about **$2.45 per 300-step run** and about **$16–36 for the full 8-run matrix including overhead**: inside the $30–50 budget even pessimistically, with Modal's free credit (about 12 A100-hours) and the $0.34/hr 4090 as safety valves. ALL step-time figures are ESTIMATES: no published GRPO throughput benchmark exists for a model of about 2B parameters + LoRA + vLLM on one GPU (flag 8). The **20-step smoke test (≤ $0.60 estimate)** measures real seconds/step, verifies the SQL reward + checkpoint/resume, and gates the matrix behind a decision rule: measured s/step × 300 × 8 runs must stay at or under about 2/3 of the remaining budget. Run matrix: 1 accuracy-only baseline + 3 values of c × 2 seeds + 1 optional control-model run = 8 runs × 300 steps.

## 5. Calibration and guarantee

**The comparison, weakest to strongest guarantee:**

1. *Simple threshold selection*: pick the cutoff where error on a calibration split equals α. No finite-sample guarantee; test error lands above α about half the time when the estimate sits at the boundary. This is what the closest text-to-SQL study did (frontier reporting, no certificate) ([arXiv:2607.06799](https://arxiv.org/abs/2607.06799)).
2. *Conformal risk control (CRC)*: guarantees the **expected** loss ≤ α for a monotone, bounded loss, exactly up to O(1/n), assuming exchangeability (calibration and test points statistically interchangeable). It fits the loss "answered AND wrong", an unconditional rate. **It does not bound selective risk**, the error rate *among answered queries*, because that is a ratio of expectations; CRC also cannot handle non-monotone risks (its own Proposition 1) ([arXiv:2208.02814](https://arxiv.org/abs/2208.02814)).
3. *Learn-then-Test (LTT)*: reframes cutoff selection as multiple hypothesis testing. For each candidate cutoff, test "true error > α" with a concentration bound (e.g., an exact binomial test, which treats each answered calibration item's right/wrong as a coin flip and asks how unlikely the observed error count would be), and keep only cutoffs where the test rejects. The result is "selective risk ≤ α with probability ≥ 1−δ over the calibration draw": the statement readers will assume from the brief's wording. Use fixed-sequence testing (walk cutoffs from conservative to permissive, stop at first failure) to avoid the full multiple-testing penalty ([arXiv:2110.01052](https://arxiv.org/abs/2110.01052); recipe corroborated in [arXiv:2609.01375](https://arxiv.org/pdf/2609.01375), snippet-level). Geifman & El-Yaniv's SGR is the 2017 special case of the same idea ([arXiv:1705.08500](https://arxiv.org/abs/1705.08500)).

**I pick LTT for the headline guarantee** (selective risk ≤ α with confidence 1−δ), optionally reporting the simpler CRC bound on "answered AND wrong" alongside. The cost of rigor: LTT is more conservative at the same calibration size; with α = δ = 0.05 expect to need several hundred *answered* calibration points (sizing intuition, not a cited result).

**Confidence signal:** the RL abstain decision is a binary action; it cannot be thresholded, so it supplies base selectivity, not a dial. Calibrate on top of it. Primary signal: **sequence log-probability** (the model's own probability of the emitted query), cheap, white-box, and backed by published evidence that it beats self-checking and verbalized confidence for SQL ([arXiv:2411.16742](https://arxiv.org/abs/2411.16742)). Alternative: **k-sample execution agreement** (sample k queries, compare execution results), a stronger ranking signal at k× the cost, used by the strongest conformal-abstention paper ([arXiv:2405.01563](https://arxiv.org/abs/2405.01563)). No paper evaluates calibrating an RL-trained abstain policy's own signal for SQL: an open slot this project fills.

**Assumptions, stated plainly:** the guarantee requires new queries to be exchangeable with the calibration data: same kinds of schemas, questions, and difficulty. It is *marginal*: it holds on average over re-draws of the calibration set, and any single calibration set can land above α. The certificate is relative to the correctness oracle: if "correct" means execution-match on one database instance, a lenient instance inflates certified reliability ([arXiv:2609.25938](https://arxiv.org/pdf/2609.25938), snippet-level). State the oracle (multi-instance test suites) explicitly.

**Shift experiment design:** (1) partition by schema family so calibration and test never share a schema (synthetic families, or TrustSQL's four source datasets); (2) calibrate at α ∈ {0.05, 0.1, 0.2} on family A; (3) report realized selective risk, coverage, and the overshoot (realized − α) on family B with bootstrap intervals; (4) include an i.i.d. control (disjoint halves of the same family) to separate finite-sample noise from shift; (5) run all ordered A→B pairs into an overshoot matrix. Expect visible breakage: the best proxy is fine-tuned verifiers dropping from about 0.78 to about 0.66 AUROC on unseen schemas ([arXiv:2607.06799](https://arxiv.org/abs/2607.06799)). No published calibration-transfer study for text-to-SQL exists; per the notes' search, this experiment is novel.

## 6. Evaluation protocol

**Metrics** (all execution-based or statistical; no LLM judge anywhere):

- Execution accuracy via **multi-instance test suites** (the reward's own oracle, held fixed for eval).
- Coverage (fraction answered) and selective risk (error rate among answered).
- **AURC** (area under the risk–coverage curve: one number for the whole trade-off; lower is better) and E-AURC (AURC minus a perfect-ranking baseline). Definitional fine print lives in the full texts of [arXiv:1805.08206](https://arxiv.org/abs/1805.08206) (fetch before quoting).
- **RS(c)** at c ∈ {0, 10, N}, TrustSQL's penalty score: +1 correct, 0 abstain-on-answerable, −c wrong or answering an unanswerable, +1 abstain-on-unanswerable. At c=N one confidently wrong answer wipes out a perfect score on everything else ([arXiv:2403.15879](https://arxiv.org/html/2403.15879v6)). Beating the abstain-everything floor (RS = 50 on TrustSQL's 50/50 test mix) is the minimum bar.
- Tokens per query; GGUF latency on the local laptop via llama.cpp's Vulkan backend (GGUF = a quantized model file format for local inference).
- Realized risk and coverage at each certified threshold, showing whether the guarantee held.

**Baselines:** (a) the base model prompted to abstain when unsure (the published anchor: frontier models prompted this way abstain <1% of the time with error rates >10%) ([arXiv:2511.11500](https://arxiv.org/html/2511.11500v2)); (b) the base model with self-consistency thresholding (abstain unless k samples agree; TrustSQL found 5-sample unanimous voting "notably strong"); (c) accuracy-only GRPO, the same training minus the abstain action, which isolates what the abstention reward buys; (d) 1–2 frontier models through an API, prompted to abstain. Published full-coverage reference points for ≤4B RL models: about 54–67% BIRD dev EX (see Q1).

**Statistics:** at least 2 seeds; bootstrap confidence intervals (re-sampling results many times to measure how much each number wobbles) over *both* test items and seeds; interquartile mean over seeds per the rliable playbook ([arXiv:2108.13264](https://arxiv.org/abs/2108.13264)); and, because conformal guarantees are averages over calibration draws, re-draw the calibration/test split many times and report the distribution of realized risk around α (the CRC paper's own protocol: 1000 splits). One gap remains in the notes: a canonical citation for paired item-level bootstrap in NLP still needs fetching.

## 7. Positioning

All four BI copilots surveyed (Snowflake Cortex Analyst, Databricks Genie, ThoughtSpot Spotter, Hex) manage wrong answers with the same stack: a human-curated semantic layer, a library of human-verified queries, and show-the-SQL-so-humans-check. Curation comes before the question; inspection comes after the answer. Abstention is a shallow UX fallback (Genie asks clarifying questions or says it "can't answer"; Cortex Analyst returns suggested alternative questions for ambiguous asks), and **no vendor publishes an abstention rate, refusal threshold, or calibrated confidence**; Snowflake's `confidence` API field only reports whether a pre-verified query was reused ([Cortex Analyst REST API](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst/rest-api); [Talk to Genie](https://docs.databricks.com/aws/genie/talk-to-genie)). Academic abstention benchmarks (TrustSQL, PRACTIQ, Ambrosia+) exist but use bare academic schemas with no semantic layer. The middle sits unoccupied: calibrated behavior at answer time. An open abstention benchmark plus a small reference model with measured risk–coverage operating points and a stated guarantee fills it, and gives customers the comparison no product enables today: how often it answers wrongly vs. declines.

---

## Recommendations

1. **Go.** Model: Qwen3.5-2B primary, with SmolLM3-3B as tooling fallback and control (I prefer it over Gemma 4 E2B given the 5.1B memory footprint). Stack: TRL GRPOTrainer + vLLM colocate, LoRA, async SQL-executing reward, `dr_grpo` loss, β=0, group 8, cap 1024–2048.
2. **Reward:** +1 test-suite-verified correct / 0 structured abstain / −c wrong; sweep c across runs (e.g. c ∈ {0.5, 1, 4}, anchored on Reinforced Hesitation's λ range and TrustSQL's c=10 being brutal); strict format gate; per-query timeout; gold-empty examples filtered.
3. **Data:** synthetic generator (SynSQL-style pipeline + TrustSQL-taxonomy injections, about 30% unanswerable/ambiguous), 3–5 data instances per schema; BIRD dev untouched as external eval; TrustSQL/EHRSQL for RS(c) comparison.
4. **Guarantee:** Learn-then-Test on sequence log-probability over the trained policy; report CRC's joint-rate bound as secondary; run the schema-shift overshoot matrix.
5. **Sequence:** smoke test before any full run; frontier-API baselines capped at a few dollars; the write-up leads with the novelty triangulation against BAR-SQL / Reinforced Hesitation / RLSR.

## Go/no-go verdict

**GO.** The brief names one kill condition: someone already published RL + abstention reward + risk–coverage at ≤4B. I checked it first; it is not met. Novelty is real but narrow: it rests on the triple combination, the SQL domain with execution-grounded rewards, public-benchmark RS(c)/risk–coverage reporting, and the guarantee layer, so the write-up must cite the near-misses prominently. Budget fits with about 2× headroom at central estimates. The two biggest execution risks: tooling friction from Qwen3.5-2B's hybrid multimodal architecture (mitigated by the SmolLM3-3B fallback and the smoke test), and abstention collapse at high c (mitigated by the c-sweep, the training-mix ratio, and per-step abstention-rate logging).

---

# Part 2: Related work and the novelty table

This part compiles the novelty table from the research notes (`notes/rl_text2sql.md`, `notes/abstention_prior_work.md`) as of 2026-11-04. Every claim checks against its linked source on that date; the exceptions carry an UNVERIFIED flag.

Terms used in the tables, defined once. **RL** is reinforcement learning: training a model by rewarding good outputs. **GRPO** is Group Relative Policy Optimization, the RL algorithm nearly all of these papers use; it scores a group of sampled answers against each other instead of training a separate judge network. **Dr.GRPO** and **RLVR** (RL with verifiable rewards: the reward comes from checking the answer, not from a learned judge) are variants. **Abstention** is the model declining to answer instead of guessing. **Risk–coverage (RC)** is the standard selective-prediction evaluation: a plot of the error rate among answered questions against the fraction of questions answered. **AURC** is the area under that curve, one number for the whole trade-off; lower is better. **EX** is execution accuracy: a generated SQL query counts as correct only if running it returns the same result as the reference query. **RS(c)** is TrustSQL's penalty-based Reliability Score: +1 for a correct answer, 0 for abstaining on an answerable question, −c for a wrong answer or for answering an unanswerable question, +1 for correctly abstaining.

Column key: the flags column reads **RL / Abstention / Risk–coverage**, each Y (yes), N (no), or "~" (partial, explained in the row).

## Group 1: RL for text-to-SQL (no abstention anywhere in this group)

| Paper | What it did | Model size | RL / Abst / RC | Code & data |
|---|---|---|---|---|
| [Arctic-Text2SQL-R1](https://arxiv.org/abs/2505.20315) (Snowflake, 2025) | GRPO with the simplest reward (1 = execution match, 0.1 = merely executable); filtered gold-empty training examples; BIRD dev EX 68.9–70.5; topped the BIRD leaderboard at 71.8 test (32B) | 7B/14B/32B | Y / N / N | [ArcticTraining](https://github.com/snowflakedb/ArcticTraining/), [HF weights](https://huggingface.co/Snowflake/Arctic-Text2SQL-R1-7B) |
| [SQL-R1](https://arxiv.org/abs/2504.08600) (NeurIPS 2025) | GRPO with composite format/execution/result/length reward; 3B reaches 54.6 BIRD dev EX; found underweighted execution terms cause reward hacking | 3B/7B/14B | Y / N / N | [IDEA-FinAI/SQL-R1](https://github.com/IDEA-FinAI/SQL-R1) |
| [Reasoning-SQL](https://arxiv.org/abs/2503.23157) (COLM 2025) | GRPO with SQL-tailored partial rewards (schema linking, LLM judge, n-gram, syntax); 14B beats o3-mini on BIRD. Smaller sizes UNVERIFIED | 14B | Y / N / N | No code link found on abstract page |
| [CSC-SQL](https://arxiv.org/abs/2505.13271) (2025) | Self-consistency + GRPO-trained merge-revision model; 3B-class models 60.6–65.3 BIRD dev with voting | 3B–32B | Y / N / N | [csc_sql](https://github.com/CycloneBoy/csc_sql) |
| [SLM-SQL](https://arxiv.org/pdf/2507.22478) (AACL 2025) | Deliberately small models, SFT then GRPO; 0.5B = 56.87, 1.5B = 67.08 BIRD dev EX (the 67.08 uses 64-sample voting) | 0.5B–1.5B | Y / N / N | [slm_sql](https://github.com/CycloneBoy/slm_sql) |
| [Think2SQL](https://arxiv.org/abs/2504.15077) (TMLR follow-up) | GRPO/RLVR with dense partial-credit reward (QATCH cell precision/recall) beating binary EX reward | 0.5B–14B | Y / N / N | Links in paper; 4B Qwen3 variant details UNVERIFIED |
| [SkyRL-SQL](https://novasky-ai.notion.site/skyrl-sql) (Berkeley, 2025) | Multi-turn RL: the model probes the database and refines its query; 7B trained on about 600 examples beats GPT-4o on Spider-family sets | 7B | Y / N / N | [SkyRL](https://github.com/NovaSky-AI/SkyRL) (Apache-2.0) |
| [CogniSQL-R1-Zero](https://arxiv.org/abs/2507.06013) (2025) | Lightweight execution+format reward, no shaping; releases 5K reasoning traces + 36K weak-supervision queries | 7B | Y / N / N | Data released per abstract; link not captured |
| [MTIR-SQL](https://arxiv.org/abs/2510.25510) (2025) | Multi-turn tool-integrated GRPO with execution feedback per step; 64.4 BIRD dev EX at 4B | 4B | Y / N / N | No code link on page |
| [Graph-Reward-SQL](https://arxiv.org/html/2505.12380v3) (EMNLP-F 2025) | Execution-free reward models (graph matching over SQL structure) for RL, cutting execution cost | n/s | Y / N / N | EX numbers not retrieved |
| [Sparks of Tabular Reasoning](https://arxiv.org/html/2505.00016v2) (ACL-W 2025) | CoT SFT then GRPO tying SQL execution reward to table reasoning | n/s | Y / N / N | [ACL Anthology](https://aclanthology.org/2025.trl-1.20) |
| [Reinforcing Code Generation](https://arxiv.org/abs/2506.06093v1) (2025) | Text-to-SQL as RL with scalar execution-derived rewards in GRPO; details beyond abstract UNVERIFIED | n/s | Y / N / N | — |
| [GradeSQL](https://arxiv.org/pdf/2509.01308) (2025) | Not RL training: outcome reward models for test-time best-of-N; 68.90 BIRD dev over OmniSQL-7B (snippet-level) | 7B base | N / N / N | — |
| [R3-SQL](https://arxiv.org/pdf/2604.25325) (2026) | Ranking reward + resampling; 71.45 BIRD dev EX (snippet-level, UNVERIFIED detail) | 7B base | Y / N / N | — |
| [SQL-Trail](https://arxiv.org/html/2601.17699v1) (2026) | Multi-turn RL agent with iterative probing; details UNVERIFIED beyond abstract | n/s | Y / N / N | — |
| [ReViSQL-BIRD](https://arxiv.org/abs/2603.20004) (2026) | RLVR on an expert-verified BIRD subset (BIRD-Platinum); 92.96% Arcwise-Plat, "human-level"; frontier scale | 235B / K2.6 | Y / N / N | — |
| [Progress-SQL](https://arxiv.org/abs/2606.06825) (2026) | Multi-turn RL with progressive improvement rewards; base models UNVERIFIED | n/s | Y / N / N | — |
| [SERL-SQL](https://arxiv.org/pdf/2608.00485) (2026) | Selective hindsight distillation for agentic SQL RL; contents UNVERIFIED (title + PDF only) | n/s | Y / N / N | — |
| [EXPO-SQL](https://aclanthology.org/2026.findings-acl.1107.pdf) (ACL-F 2026) | Clause-level policy optimization from execution feedback; details UNVERIFIED | n/s | Y / N / N | — |
| [ReToolSQL](https://arxiv.org/pdf/2608.27796) (2026) | Tool-augmented GRPO/DAPO text-to-SQL; abstention handling UNVERIFIED | n/s | Y / ? / N | — |
| [OmniSQL / SynSQL-2.5M](https://arxiv.org/abs/2503.02240) (VLDB '25) | Context row: SFT-only (no RL) models + the 2.5M-sample synthetic dataset that most RL work above builds on; Apache-2.0 | 7B/14B/32B | N / N / N | [OmniSQL](https://github.com/RUCKBReasoning/OmniSQL) |

## Group 2: Abstention and reliability in text-to-SQL (one RL paper; the rest are benchmarks or post-hoc methods)

| Paper | What it did | Model size | RL / Abst / RC | Code & data |
|---|---|---|---|---|
| [BAR-SQL](https://arxiv.org/html/2601.10318) (Jan 2026) | **Closest prior work.** SFT + GRPO on enterprise NL2SQL where natural-language refusals on unanswerable questions earn reward via embedding similarity to a gold refusal. No penalty parameter c, no abstain token, no RS(c), **no risk–coverage**, no Spider/BIRD results; own private benchmark (Ent-SQL-Bench), partly GPT-4o-judged. 91.48% average vs Claude-4.5-Sonnet 48.02% | Qwen3-1.7B | Y / Y / N | [BAR-SQL](https://github.com/TianSongS/BAR-SQL) |
| [TrustSQL](https://arxiv.org/abs/2403.15879) (2024) | Benchmark defining RS(c); feasible + infeasible questions over ATIS/Advising/EHRSQL/Spider; abstention via classifiers, probability thresholds, voting; no RL, no RC curves | T5-3B, SQLCoder-7b, GPT-3.5 | N / Y / N | [TrustSQL](https://github.com/glee4810/TrustSQL) (CC BY 4.0) |
| [EHRSQL-2024 shared task](https://arxiv.org/html/2405.06673v2) (NAACL-W 2024) | Reliable clinical text-to-SQL; ranking metric RS(10); 8 teams used ensembles, log-prob gating, self-consistency, none RL; no team achieved positive RS(N) | various ≤7B + ChatGPT | N / Y / N | [ehrsql-2024](https://github.com/glee4810/ehrsql-2024), [ProbGate](https://github.com/venzino-han/probgate_ehrsql) |
| [AmbiQT](https://arxiv.org/abs/2310.13659) (EMNLP 2023) | Ambiguity benchmark: 3,000+ questions each with two valid SQLs; evaluation is top-k coverage, not abstention | n/a (benchmark) | N / N / N | [AmbiQT](https://github.com/testzer0/AmbiQT) |
| [AMBROSIA](https://arxiv.org/abs/2406.19073) (NeurIPS 2024) | Ambiguity benchmark (scope/attachment/vagueness) with all interpretations and their SQL; no abstention/penalty scoring; size details UNVERIFIED | n/a (benchmark) | N / N / N | — |
| [Ambrosia+ / ABISS](https://scp.cc.gatech.edu/external-news/ieee-xplore-rag-or-not-rag-question-effective-text-sql-generation-under-ambiguity) (IEEE) | Extends AMBROSIA with 2,535 unanswerable questions; "disambiguate first, parse later"; figures UNVERIFIED (abstract only) | n/s | N / Y / N | — |
| [PRACTIQ](https://arxiv.org/html/2410.11076v1) (2024) | Conversational dataset with four ambiguous + four unanswerable categories; details UNVERIFIED | n/a (dataset) | N / Y / N | — |
| [AmbiSQL](https://arxiv.org/html/2508.15276v1) (2025) | Interactive ambiguity detection: asks the user rather than abstaining | n/s | N / ~ / N | — |
| [LatentRefusal](https://arxiv.org/html/2601.10398v1) (ACL-F 2026) | Hidden-activation probe predicting answerability (about 88.5 F1); detection, not training; metrics partly UNVERIFIED | probes on 8B | N / Y / N | — |
| [Query Carefully](https://arxiv.org/abs/2512.21345) (2025) | Prompting-based unanswerable detection for biomedical SQL; releases OncoMX-NAQ (80 no-answer questions); UNVERIFIED beyond abstract | n/s | N / Y / N | — |
| [Somov & Tutubalina](https://arxiv.org/abs/2501.09527) (AAAI 2025) | "Text-to-SQL with a reject option": entropy-based selective classifiers; frames coverage-vs-risk; post-hoc, no RL | n/s | N / Y / Y | — |
| [Reliable Text-to-SQL with Adaptive Abstention](https://arxiv.org/pdf/2501.10858) (2025) | Conformal-style abstention at generation branch points; coverage + extra-abstention-rate metrics; no RL | n/s | N / Y / ~ | — |
| [Richardson, selective-prediction study](https://arxiv.org/abs/2607.06799) (2026) | Post-hoc confidence signals + fine-tuned verifiers; AUROC/ECE/selective risk/abstention frontiers; black-box signals 0.61–0.68 AUROC, judge ensemble 0.82; verifiers drop from about 0.78 to about 0.66 on unseen schemas | signals on various; verifiers ≤7B | N / Y / Y | — |
| [Ramachandran & Sarawagi](https://arxiv.org/abs/2411.16742) (2024) | Full-sequence probability, rescaled, as SQL confidence; beats self-checking and verbalized confidence | n/s | N / ~ / ~ | — |
| [Syntax-aware logit confidence](https://arxiv.org/abs/2508.14056) (2025) | Benchmarks black/white-box confidence on Spider/BIRD; up to 16% calibration-error reduction (snippet-level) | n/s | N / ~ / ~ | — |

## Group 3: RL-trained abstention and calibration outside text-to-SQL

| Paper | What it did | Model size | RL / Abst / RC | Code & data |
|---|---|---|---|---|
| [Reinforced Hesitation](https://arxiv.org/html/2511.11500v2) (Nov 2025) | **Closest reward design.** Dr.GRPO with the exact ternary reward +1 correct / 0 abstain ("I don't know" text) / −λ wrong, λ swept over {0,1,2,5,10,20}; logic puzzles; reports error-vs-abstention Pareto frontiers and conditional accuracy: close to, but not, a standard RC curve; also shows frontier models prompted to abstain do so <1% of the time with >10% error | Qwen3-1.7B | Y / Y / ~ | No code link found |
| [RLSR](https://arxiv.org/html/2607.03528v1) (Jul 2026) | **Closest evaluation.** GRPO reward provably optimizing AURC via confidence-rank weighting; full RC curves, AURC, accuracy-at-coverage; abstention only by thresholding a verbalized confidence at inference, no abstain action in training; QA + math | Qwen2.5-7B, Llama-3.1-8B | Y / ~ / Y | [Project page](https://sun-umn.github.io/RLSR); no repo found |
| [Rewarding Intellectual Humility](https://arxiv.org/abs/2601.20126) (Jan 2026) | RLVR ternary reward (−1 / r_abs / +1) with a sweep over the abstain reward; multiple-choice MedMCQA/Math; moderate r_abs cuts wrong answers; no RC metrics per abstract (full paper UNVERIFIED) | Granite-3.3-2B, Qwen-3-4B | Y / Y / N | [rl-abstention](https://github.com/Mystic-Slice/rl-abstention) |
| [TruthRL](https://arxiv.org/abs/2509.25760) (ICML 2026) | GRPO ternary reward (correct/hallucination/abstention); hallucination rate 43.5%→19.4%; backbone sizes UNVERIFIED; reward values +1/0/−1 per secondary source, UNVERIFIED | n/s | Y / Y / N | [TruthRL](https://github.com/facebookresearch/TruthRL) |
| [RLKF](https://arxiv.org/abs/2403.18349) (2024) | RLHF-style: preference pairs from knowledge feedback train a reward model, then RL to refuse unknown questions; RC reporting UNVERIFIED | Llama-2 Chat | Y / Y / ? | — |
| [Hallucination Tax of RFT](https://arxiv.org/abs/2505.13988) (EMNLP-F 2025) | Standard reinforcement fine-tuning collapses refusal (>80% drop); mixing 10% synthetic unanswerable math restores it; data-side fix, not a reward design; exact figures partly UNVERIFIED | Qwen2.5-7B | Y / Y / N | — |
| [KARL](https://arxiv.org/abs/2604.22779) (2026) | Knowledge-boundary-aware RL; dynamic abstention reward avoids the "abstention trap" (collapse into refusing everything); models/metrics UNVERIFIED | n/s | Y / Y / ? | — |
| [Knowing When to Quit](https://arxiv.org/abs/2604.18419) (ICML 2026) | Theory: mid-generation abstention as an action in KL-regularized RL; abstain when the value function dips below the abstention reward; sizes UNVERIFIED | n/s | Y / Y / ~ | — |
| [Abstain-R1](https://arxiv.org/html/2604.17073v1) (2026) | SFT + GRPO; base reward 0.3 for boxed "I don't know" on unanswerables; about a 30/70 unanswerable/answerable mix; primary record partly UNVERIFIED (found via review site) | n/s | Y / Y / ? | — |
| [AWA-RL](https://www.alphaxiv.org/abs/2607.10738) (2026) | Abstention-aware RL for search agents; sets the abstention reward dynamically because a fixed one "can lead a model to refuse nearly everything" (snippet-level) | n/s | Y / Y / ? | — |
| [Rewarding Efficient Reasoning](https://arxiv.org/pdf/2609.20846) (2026) | Efficiency reward added to GRPO improves abstention on underspecified tasks; not SQL | n/s | Y / Y / ? | — |
| [RLCR](https://arxiv.org/abs/2507.16806) (2025) | Calibration, not abstention: GRPO reward = correctness + Brier score on verbalized confidence; ECE 0.37→0.03 | Qwen2.5-7B | Y / N / N | — |
| [PPO-M / PPO-C](https://arxiv.org/pdf/2410.09724) (ICLR 2025) | RLHF reward calibration to tame verbalized overconfidence; no abstain action | Llama3-8B, Mistral-7B | Y / N / N | — |
| [R-Tuning](https://arxiv.org/abs/2311.09677) (NAACL 2024) | Refusal-aware **supervised** tuning (contrast row: no RL); refusal transfers as a meta-skill | various | N / Y / N | [R-Tuning](https://github.com/shizhediao/R-Tuning) |
| [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a) (ICML 2019) | Classic: network with an integrated reject option trained for a target coverage; the origin of risk–coverage evaluation in deep learning | CNNs (pre-LLM) | N / Y / Y | — |
| [Deep Gamblers](https://arxiv.org/abs/1907.00208) (NeurIPS 2019) | Classic: abstention as an extra class with a portfolio-theory loss; no per-coverage retraining | CNNs (pre-LLM) | N / Y / Y | — |

## Verdict

The question this table answers: is RL training with an explicit abstention reward, evaluated with risk–coverage curves, on a model of 4B parameters or smaller, already published in text-to-SQL? **No, as of 2026-11-04.** Every pairwise combination exists; no single paper holds all three ingredients in this domain at this scale.

Three nearest neighbors, in order. **BAR-SQL** pairs RL with abstention credit at 1.7B in text-to-SQL, but it scores refusals by text similarity to a gold refusal rather than an explicit reward/penalty trade-off: no penalty parameter, no risk–coverage or RS(c) reporting, and evaluation on a private enterprise benchmark with no Spider/BIRD numbers. **Reinforced Hesitation** trains the exact +1/0/−λ reward on a ≤4B model, with a λ-sweep Pareto analysis a reviewer could call risk–coverage-adjacent; but it works on logic puzzles, with no SQL, no standard risk–coverage or AURC metrics, and no calibration guarantee. **RLSR** optimizes AURC directly with RL and reports full risk–coverage curves; but it trains no abstain action, runs at 7–8B, and stays in QA and math. **Rewarding Intellectual Humility** holds the ≤4B plus swept-abstain-reward cell, on multiple-choice QA, without risk–coverage.

The smallest differentiating angle, if a reviewer treats these as prior art: an explicit structured abstain action, trained with a penalty-swept ternary execution-grounded reward, in text-to-SQL, reported with standard risk–coverage curves and TrustSQL's own RS(c) on public benchmarks, plus a distribution-free statistical guarantee (Learn-then-Test) and a schema-shift stress test. The last three elements appear in none of the near neighbors.

The negative claim has a boundary: search coverage. The 2026 preprint volume is high; the search did not crawl OpenReview; it did not fully inspect Abstain-R1's primary record or the experimental details of KARL and TruthRL. An under-review or non-arXiv collision remains possible.

---

# Part 3: Compute estimate

Compiled 2026-11-04 from `notes/compute_costs.md` in the research notes. Every price below came from the live pricing pages on 2026-11-04 and will drift; re-check before renting. **Every derived number below is an ESTIMATE** built from those prices plus throughput anchors. No published benchmark measures this exact configuration, so the smoke test below must replace the estimates with measurements before any full run.

Every estimate assumes the same setup: GRPO (Group Relative Policy Optimization, an RL method that samples a group of answers per prompt and scores them against each other) on Qwen3.5-2B with LoRA (low-rank adapters: training a small add-on instead of all the weights) and vLLM (a fast batched text-generation engine) sharing a single GPU, plus a custom reward that executes the generated SQL.

## The cost formula

**cost = steps × prompts_per_step × group_size × tokens_per_completion ÷ effective_throughput ÷ 3600 × $/hr**

In words: tokens generated per training step, divided by end-to-end tokens per second, gives seconds per step; seconds per step times the number of steps times the hourly GPU price gives dollars.

## Verified prices (fetched 2026-11-04)

From [runpod.io/pricing](https://www.runpod.io/pricing) (Secure Cloud = vetted datacenters; Community Cloud = cheaper peer-hosted machines that can be interrupted) and [modal.com/pricing](https://modal.com/pricing); dollars per hour, billed per second:

| GPU | VRAM (GPU memory) | RunPod Secure | RunPod Community | Modal (derived from $/sec) |
|---|---|---|---|---|
| A100 80GB | 80 GB | $1.79 | $1.19–1.39 | about $2.50 ($0.000694/s) |
| H100 | 80 GB | $2.89 | $1.99 | about $3.95 |
| L40S | 48 GB | $1.09 | $0.79 | about $1.95 |
| RTX 4090 | 24 GB | $0.89 | **$0.34** | not offered |
| RTX 5090 | 32 GB | $1.19 | $0.69 | not offered |

- **Modal's Starter plan includes $30/month of free credit, verified** on the pricing page. Divided by about $2.50/hr, that is about **12 free A100-hours/month** (ESTIMATE, simple division). The page does not spell out whether the credit applies to GPU compute without restriction; that is UNVERIFIED, so check the dashboard before relying on it.
- Storage barely registers: a persistent volume of about 50 GB costs about $5/month prorated on RunPod (ESTIMATE).

## Throughput anchors (the honest gap)

**No published end-to-end GRPO steps/hour or tokens/sec benchmark exists for a 1–3B model with LoRA + vLLM on a single A100 or 4090.** 4 anchors come close:

- Unsloth reports about 4,000 tokens/sec of *pure generation* for a 4-bit 3B model on one A100 40GB; that is a generation speed, not a training rate ([Unsloth GRPO docs](https://unsloth.ai/docs/basics/reasoning-grpo)).
- Mini-R1 ran a 3B model at 8 generations per prompt on 4×H100 at 45–60 s per GRPO step; 450 steps took about 6 hours ([Mini-R1 blog](https://huggingface.co/blog/open-r1/mini-r1-contdown-game)).
- A community SQL GRPO run trained 7B + LoRA, group 4, 150-token completions, 200 steps in under 2 hours on one GPU, no vLLM ([HF Space post](https://huggingface.co/spaces/junaid0600/sql-db-engineer-agent/blob/main/blog_post.md)).
- TinyZero ran R1-style RL on a 3B model for a claimed "< $30" total, with no breakdown published: direct evidence the budget class is realistic ([TinyZero README](https://github.com/Jiayi-Pan/TinyZero)).

Triangulating from these gives the planning assumptions (**all ESTIMATES**). Pure generation for a 2B model on an A100 80GB at GRPO batch shapes should run about 2,000–5,000 output tokens/sec; adding SQL-reward execution, the backward pass, and weight sync, with vLLM on a 30–50% memory slice, cuts that to an **end-to-end effective GRPO throughput of about 800–2,000 output tokens/sec**. That end-to-end number is the single most uncertain input: it is the one the smoke test measures. The SQL reward itself is CPU work, well under a second per completion on small SQLite schemas if parallelized; one slow query can still stall a whole group, so every query gets a 2–5 s timeout (a design choice, not a published figure).

## Worked estimates (ESTIMATES throughout)

Central settings: 8 prompts/step × group size 8 × 384 average completion tokens = 24,576 generated tokens per step. At 1,500 tokens/sec effective, that is **about 16.4 s/step**.

On a RunPod Secure A100 at $1.79/hr:
- 300 steps run about 1.37 h and cost **about $2.45 per run**: a full training run for the price of a coffee.
- 500 steps run about 2.28 h, about $4.08 per run; 1,000 steps run about 4.55 h, about $8.15.
- The same math at Modal's rate of about $2.50/hr puts 300 steps at about $3.41.

Pessimistic case (group 16, 512 tokens, 800 tokens/sec, so about 82 s/step): 300 steps take about 6.8 h and cost **about $12.20 per run**. A 1,000-step pessimistic run at about $40.80 would nearly eat the budget by itself; that is why runs are capped at 300–500 steps and group size 8. Sanity check: 82 s/step is slower than Mini-R1's 45–60 s/step on a bigger model with 4 GPUs, so the pessimistic case is genuinely conservative.

Run length rationale: comparable projects trained 200–450 steps, and Unsloth's guidance says to expect no reward movement before about 300 steps. With about 500–1,000 training prompts at 8 prompts/step, 300–500 steps make about 2.5–8 passes over the data, consistent with SQL-R1's 5K-pair RL set. Group size 8 matches SQL-R1 and the TRL default; Arctic found no gain beyond 16 rollouts, so nothing above 8–16 is worth paying for.

## 20-step smoke test (run FIRST, on the target GPU)

- **Config:** 20 steps, group 8, 8 prompts/step, max completion 512, LoRA r=16, vLLM colocate (`vllm_gpu_memory_utilization` tuned up from 0.3 if idle memory allows); checkpoint and resume once; log per-step wall-clock.
- **Verify:** vLLM starts and generates for Qwen3.5-2B (the architecture-risk check); the SQL reward executes with per-query timeouts and returns non-constant rewards; reward, length, and abstention-rate curves log; checkpoint save and resume works; VRAM headroom (does it fit a 24GB 4090?).
- **Cost:** 5–20 min, so **at most $0.60** on a Secure A100 and at most $0.12 on a community 4090 (ESTIMATE).
- **Decision rule:** measured s/step × 300 steps × 8 runs must come in at or under about 2/3 of the remaining budget. If the A100 measures over 25 s/step: cut completions to 256 tokens, or move seed-replica runs to the $0.34/hr 4090.

## Run matrix and total cost range

8 runs × 300 steps each, per the brief: an accuracy-only baseline, 3 values of the wrong-answer penalty c × 2 seeds, and an optional control model.

| Run | Reward config | Count |
|---|---|---|
| 1 | Accuracy-only GRPO baseline (no abstain action) | 1 |
| 2–7 | 3 penalty values c × 2 random seeds | 6 |
| 8 (optional) | Control model (SmolLM3-3B or Gemma 4 E2B), best c | 1 |

Totals (ESTIMATES):
- **Central:** 8 × 1.37 h is about 11 GPU-hours, about $19.60 on a Secure A100 or about $13.05 on a Community A100; adding 20% setup/idle overhead lands it at **about $24 / about $16**.
- **Pessimistic** (30 s/step, 2.5 h/run): 20 GPU-hours, about $35.80 Secure or about $23.80 Community; still inside $30–50.
- **Cheapest path:** 8 × 2.5 h on a $0.34/hr community 4090 costs about **$6.80**, if the 2B model + LoRA + vLLM fits in 24 GB. Unsloth's VRAM guidance (5 GB suffices for models of 1.5B and under in 4-bit) strongly suggests it fits; the smoke test must confirm it. Checkpoint every 25–50 steps against community-machine interruption.
- **Free-tier path:** Modal Starter's $30/month credit, about 12 A100-hours, covers the entire central-estimate matrix at $0 cash, with RunPod as overflow.

**I recommend this sequence:** smoke test on Modal free credit or a Secure A100 (under $1); measure s/step; run the matrix on a RunPod Community A100 ($1.19/hr) or 4090 ($0.34/hr) with frequent checkpoints; reserve the Secure A100 for when community availability bites. **Expected total spend: $10–30; worst case about $40 (ESTIMATE).** If spend runs over budget: drop run 8, then run seeds only on the best and worst penalty values, then shift the remaining runs to the 4090.

## Known gaps

- End-to-end throughput (800–2,000 tokens/sec) is unmeasured anywhere in the literature for this setup; the smoke test replaces it.
- Whether Qwen3.5-2B + LoRA + vLLM colocate fits 24 GB at 512-token completions is inferred from Unsloth's VRAM table, not demonstrated.
- SQL-reward latency on the project's actual databases is unmeasured; the per-query timeout is a mitigation, not a measurement.
- Modal's $30 credit scope (GPU compute included?) is UNVERIFIED on the pricing page.
