# Research Brief: Abstaining Text-to-SQL via RL

## Goal

Use RL to teach a small open model to either write correct SQL or abstain. A natural-language-to-SQL tool that gives confident wrong answers loses its users' trust, and this project targets that failure.

* Training: GRPO on a model of 4B parameters or fewer (primary: Qwen3.5-2B).
* Reward: +1 if the query's execution result is correct, 0 if the model abstains, −c if it is wrong. Sweep c across runs.
* Calibration: a conformal step on held-out data that turns the trained behavior into a guarantee: "error ≤ α on the queries it answers."
* Headline eval: risk–coverage curves (how many queries it answers vs. how often those answers are wrong). Compare against an accuracy-only RL baseline and against frontier models prompted to abstain.
* Escalation: sending abstained queries to a bigger model or a human is a deployment note, not a separate system to build.

## Your task

Do the research below and write up the findings. Do not start full training runs. Do not spend money on cloud compute without asking first.

## Constraints

* Local machine: Windows 11 laptop with an Intel Core Ultra 9 185H, 16 GB shared RAM, an Intel Arc integrated GPU and no NVIDIA GPU. No training happens locally. Local work is limited to:
  * data generation and execution checkers
  * analysis
  * evaluating quantized (GGUF) models through llama.cpp's Vulkan backend
* Cloud and budget: Modal (Starter plan includes $30/month in free credits) or RunPod (A100 80GB at roughly $1.79/hr). Total budget is $30–50.
* Models:
  * Primary: Qwen3.5-2B.
  * Control: one non-Qwen family, either Gemma 4 E2B or SmolLM3-3B.
  * Check that each model is available and supported in TRL and vLLM.
* Metrics: no LLM-as-judge anywhere. Every metric must be execution-based or come from a statistical guarantee.
* Timeline: about 4 weeks.

## Research questions

### 1. Novelty check (do this first; it can kill the project)

Find the closest prior work in each of these areas:

* RL for text-to-SQL. Leads to verify: Arctic-Text2SQL-R1, SQL-R1, Reasoning-SQL, plus anything newer.
* Abstention, reliability and penalty-based scoring in text-to-SQL. Leads to verify:
  * TrustSQL
  * the EHRSQL 2024 shared task on reliable text-to-SQL and its penalty-based reliability score
  * AmbiQT and AMBROSIA (ambiguous questions)
* Selective prediction or abstention trained with RL in any domain.

Deliverable: a table with one row per paper, with columns for what the paper did, model size, whether it uses RL, whether it handles abstention, whether it reports risk–coverage, and code/data links.

Give a verdict. If someone has already published RL plus an abstention reward plus risk–coverage on models of 4B or smaller, say so plainly and propose the smallest differentiating angle.

### 2. Data

* Survey BIRD, Spider 2.0, Spider, TrustSQL and EHRSQL. For each, record the license and what is known about its contamination in Qwen and Gemma training data.
* Record whether each dataset includes unanswerable or ambiguous questions.
* Decide between three options: a synthetic generator, existing datasets, or a hybrid. The suggested default is synthetic training data with BIRD dev as the external evaluation.
* Design the synthetic generator:
  * schema generation with realistic names
  * filling the tables with data
  * question templates
  * injected question categories: unanswerable (missing column), ambiguous (undefined metric) and out-of-scope
* Evaluate the test-suite accuracy approach (Zhong et al. 2020): run each query against 3–5 database instances with different data, so a wrong query can't match the gold result by coincidence. Find the existing tooling for this.

### 3. Training stack

* Compare TRL GRPOTrainer (with vLLM colocated), `verifiers` (Prime Intellect) and Unsloth on cloud NVIDIA GPUs.
* Recommend one, based on these requirements:
  * Qwen3.5 support
  * LoRA
  * a custom reward function that executes SQL
  * a single GPU
* Pin down the abstain mechanism: a special token versus a structured output such as `{"sql": ..., "abstain": bool, "reason": ...}`.
* Specify loss settings:
  * Dr. GRPO-style loss (no length or std normalization)
  * KL coefficient
  * group size
  * generation length cap
* Reward-hacking risks to design against:
  * abstaining on everything when c is high
  * trivial queries that return empty results
  * degenerate SQL that happens to match on small tables

### 4. Compute estimate

* Build a cost model: steps × prompts × group size × tokens ÷ throughput × $/hr.
* Give concrete estimates for Qwen3.5-2B on an A100 80GB, plus a 20-step smoke-test plan to measure real throughput before any full run.
* Propose a run matrix that fits $30–50:
  * an accuracy-only baseline
  * 3 values of c
  * 2 seeds
  * optional: one run of the control model

### 5. Calibration and guarantee

* Compare conformal risk control (Angelopoulos et al.), Learn-then-Test, and simple threshold selection on a calibration split.
* Pick the confidence signal to calibrate: the model's own abstain decision, token logprobs, or agreement across k samples.
* State the assumptions plainly. The guarantee requires new queries to look like the calibration data.
* Design a shift experiment: calibrate on one family of schemas, test on another, and measure how badly the guarantee breaks.

### 6. Evaluation protocol

* Metrics:
  * execution accuracy
  * coverage
  * selective risk (error rate on answered queries)
  * area under the risk–coverage curve
  * a penalty-based reliability score
  * tokens per query
  * latency of the GGUF model on the local laptop
* Baselines:
  * the base model prompted to abstain when unsure
  * the base model with self-consistency thresholding
  * accuracy-only GRPO
  * 1–2 frontier models through an API, prompted to abstain
* Statistics: at least 2 seeds, and bootstrap confidence intervals over both items and seeds.

### 7. Positioning (keep this brief)

* How do BI copilots (Snowflake Cortex Analyst, Databricks Genie, ThoughtSpot, Hex) handle wrong answers today: verified queries, semantic layers, clarifying questions?
* Write one paragraph on where an open abstention benchmark plus a reference model fits.

## Optional spike (only after questions 1–3)

* Generate 50 synthetic schema/question pairs and build the multi-database execution checker.
* Run a small Qwen model zero-shot through llama.cpp to estimate baseline accuracy and how often it abstains.
* Report what broke.

## Deliverables

Put everything in `research/`:

* `REPORT.md`: findings for each question, concrete recommendations, and a go/no-go verdict.
* `related_work.md`: the novelty table.
* `compute_estimate.md`: the cost formula, the numbers and the run matrix.
* `plan.md`: a revised 4-week plan and a proposed repo structure.

## Rules

* Cite primary sources: arXiv, official documentation, GitHub. Use links.
* Mark every claim you could not verify as UNVERIFIED. Do not invent numbers, benchmarks or paper results.
* Flag anything that changes the plan, such as prior work that already covers it, a license blocker, or a compute estimate over budget, at the top of `REPORT.md`.
