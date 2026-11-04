# Compute Cost Estimate: GRPO RL Fine-Tuning of a ~2B Model (Budget $30–50)

Research date: **2026-11-04**. All prices were fetched from the live pricing pages on this
date and will drift; re-check before renting. Every derived number is labeled ESTIMATE.
Project assumptions: GRPO (Group Relative Policy Optimization — an RL method that samples a
group of completions per prompt and scores them against each other) on a ~2B-parameter model
(Qwen3.5-2B), LoRA (low-rank adapters — training only a small add-on instead of all weights),
vLLM (a fast batched text-generation engine) on a single GPU, custom SQL-execution reward.

---

## Key Question 1: RunPod current GPU pricing

### Takeaway
As of 2026-11-04, an A100 80GB on RunPod costs $1.79/hr on Secure Cloud and $1.19–1.39/hr on
Community Cloud; an RTX 4090 on Community Cloud is $0.34/hr, which is the standout budget
option if the job fits in 24GB of VRAM (GPU memory) — and a 2B model with LoRA + vLLM should.
Billing is per-second with no commitment.

### Cited Findings
All from [runpod.io/pricing](https://www.runpod.io/pricing), fetched 2026-11-04.
Secure Cloud = RunPod's vetted datacenters; Community Cloud = cheaper peer-hosted machines
that can be less reliable (interruptions, variable availability).

On-demand pod prices, dollars per hour:

| GPU | VRAM | Secure | Community |
|---|---|---|---|
| A100 80GB PCIe | 80 GB | $1.79 | $1.19 |
| A100 80GB SXM | 80 GB | $1.79 | $1.39 |
| H100 PCIe | 80 GB | $2.89 | $1.99 |
| L40S | 48 GB | $1.09 | $0.79 |
| RTX 4090 | 24 GB | $0.89 | $0.34 |
| RTX 5090 | 32 GB | $1.19 | $0.69 |
| RTX A6000 | 48 GB | $0.59 | $0.33 |
| A40 | 48 GB | $0.59 | $0.35 |

- GPU pods are "billed per second with no commitment" (FAQ on the pricing page) — [RunPod pricing](https://www.runpod.io/pricing)
- Storage: container disk $0.10/GB/month; volume disk $0.10/GB/month while running, $0.20/GB/month idle; network storage $0.07/GB/month under 1TB — [RunPod pricing](https://www.runpod.io/pricing)

### Inferences
- A ~50GB persistent volume (model weights + checkpoints + dataset) costs about $5/month,
  prorated — negligible against the GPU spend. ESTIMATE.
- Community Cloud 4090 at $0.34/hr is roughly 5x cheaper per hour than Secure A100. If the
  2B-model run fits in 24GB (see Key Question 3: Unsloth says a 1.5B model needs as little as
  5GB with their 4-bit setup, and a full bf16 2B model is ~4GB of weights), the 4090 is the
  cheapest viable workhorse. Risk: community machines can be interrupted mid-run, so
  checkpoint every 25–50 steps.

### Gaps
- Spot/interruptible pod discounts were not broken out separately on the page as fetched;
  the Community price effectively plays that role.
- No published reliability numbers for Community Cloud interruption rates.

---

## Key Question 2: Modal pricing and Starter free credit

### Takeaway
Modal charges $0.000694/sec for an A100 80GB (≈ $2.50/hr, derived) and the Starter plan
includes **$30/month of free credit — verified**, which alone could cover most or all of this
project's central-estimate compute at Modal's rates.

### Cited Findings
All from [modal.com/pricing](https://modal.com/pricing), fetched 2026-11-04. Per-second
rates as listed; hourly figures in brackets are my ×3600 conversions (derived, not quoted).

| GPU | $/second | ≈ $/hour (derived) |
|---|---|---|
| A100 80GB | $0.000694 | $2.50 |
| A100 40GB | $0.000583 | $2.10 |
| H100 SXM5 | $0.001097 | $3.95 |
| L40S | $0.000542 | $1.95 |
| A10 | $0.000306 | $1.10 |
| L4 | $0.000222 | $0.80 |
| T4 | $0.000164 | $0.59 |

- Starter plan: "$0 + compute per month, with $30/month free credits," 3 workspace seats — [Modal pricing](https://modal.com/pricing)
- Team plan: $250 + compute per month, with $100/month free credits — [Modal pricing](https://modal.com/pricing)
- No consumer RTX 4090/5090 class GPUs are offered on Modal's pricing page (cheapest tiers are T4/L4/A10) — [Modal pricing](https://modal.com/pricing)

### Inferences
- $30 free credit ÷ $2.50/hr ≈ **12 free A100-80GB hours per month** on Starter. ESTIMATE
  (simple division). That covers the whole central-estimate run matrix below (~11 GPU-hours).
- Modal is per-second billed serverless (your code runs as functions); long training runs work
  but need an explicit function timeout, and cold-start/image-build time is billed overhead.
  Practical consequence: Modal is best as the "free tier" for the smoke test and 1–2 runs;
  RunPod is simpler for babysitting an interactive multi-hour training pod.
- RunPod A100 Secure ($1.79/hr) undercuts Modal A100 ($2.50/hr) by ~28% once free credit is
  exhausted. ESTIMATE (simple comparison).

### Gaps
- Modal's page showed identical figures in its per-hour and per-second columns at fetch time
  (likely a page rendering quirk); the per-second figures match Modal's historically published
  rates, so I used those. Minor residual uncertainty on the exact hourly rounding.
- Whether the $30 Starter credit applies to GPU compute without restriction (vs. some services
  only) was not spelled out on the pricing page — UNVERIFIED detail; worth a 2-minute check in
  the Modal dashboard before relying on it.

---

## Key Question 3: Published GRPO throughput references (Unsloth, TRL, community)

### Takeaway
**No solid published end-to-end GRPO steps/hour or tokens/sec benchmark exists for a 1–3B
model with LoRA + vLLM on a single A100/4090** — this is a real gap. The best anchors are:
Unsloth's ~4,000 tokens/sec *generation* figure (3B model, 4-bit, A100 40GB), Mini-R1's
45–60 seconds per GRPO step (3B model, 4×H100, 8 generations), and a community SQL GRPO run
(7B + LoRA, 200 steps in under 2 hours on one GPU). Everything else must be treated as
estimate.

### Cited Findings
- TRL's GRPO docs state generation is "often the main bottleneck when training with online
  methods," which is why vLLM integration exists; the docs give **no** vLLM speedup factor or
  throughput benchmark — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- TRL defaults: num_generations (group size) = 8, max_completion_length = 512,
  per_device_train_batch_size = 8; in colocate mode vLLM shares the training GPU and gets
  30% of GPU memory by default (vllm_gpu_memory_utilization = 0.3) — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Unsloth: with vLLM, ~4,000 tokens/sec on 1× A100 40GB for a 4-bit Llama 3.2 3B, and
  ~300 tokens/sec on a 16GB T4; this is a generation speed, not an end-to-end GRPO training
  rate — [Unsloth GRPO docs](https://unsloth.ai/docs/basics/reasoning-grpo)
- Unsloth VRAM guidance: a model of 1.5B parameters or less needs at least ~5GB VRAM for GRPO
  (4-bit); 15GB VRAM handles models up to 17B; LoRA 16-bit needs ~4× the VRAM of QLoRA 4-bit —
  [Unsloth GRPO docs](https://unsloth.ai/docs/basics/reasoning-grpo)
- Unsloth training-duration guidance: wait at least 300 steps before expecting reward to rise;
  "a minimum of about 12 hours" is suggested for decent results (their general guidance, not
  tied to a specific model/GPU) — [Unsloth GRPO docs](https://unsloth.ai/docs/basics/reasoning-grpo)
- Mini-R1 (Phil Schmid): GRPO on Qwen2.5-3B, countdown task, 8 generations per sample on
  4× H100 80GB: a single step took 45–60 seconds; 450 steps ≈ 6 hours — [Mini-R1 blog](https://huggingface.co/blog/open-r1/mini-r1-contdown-game) ([original](https://www.philschmid.de/mini-deepseek-r1))
- Community SQL GRPO run: Qwen2.5-7B-Instruct + LoRA (r=16, via Unsloth), group size 4,
  effective batch 8, max_completion_length 150, 200 steps, finished in "under 2 hours" on a
  single (unnamed) GPU; no vLLM mentioned — [HF Space blog post](https://huggingface.co/spaces/junaid0600/sql-db-engineer-agent/blob/main/blog_post.md)
- Unsloth claims LoRA loading directly into vLLM (state-dict parsing) makes GRPO runs about
  1.5× faster — a claim, with no measured tokens/sec attached — [Unsloth R1 blog via search](https://unsloth.ai/blog/r1-reasoning) (direct fetch returned HTTP 403; figure seen in search snippets — treat as secondhand)

### Inferences
- From Mini-R1: 450 steps × ~52.5s ≈ 6.6 h wall-clock on 4×H100 ≈ 24–26 H100-GPU-hours for a
  3B model. That is a *ceiling* reference — a 2B model, LoRA-only, shorter completions, and a
  single GPU with vLLM colocate should cost far less per step. ESTIMATE.
- From the 7B community run: ~36 s/step for a 7B with tiny completions (150 tokens) and group
  size 4 without vLLM. A 2B model (~3.5× fewer parameters) with vLLM should comfortably beat
  this per token generated. ESTIMATE.
- Working single-GPU planning number used below: **15–30 seconds per GRPO step** for
  Qwen3.5-2B + LoRA + vLLM colocate on an A100 80GB, at group size 8 and 256–512 completion
  tokens. This is an ESTIMATE triangulated from the three anchors above, not a published
  benchmark. The 20-step smoke test exists precisely to replace it with a measured value.

### Gaps
- No published GRPO steps/hour benchmark for any ~2B model on A100 or 4090 with LoRA + vLLM
  was found despite multiple searches. Mark all step-time figures as estimates until the
  smoke test measures the real one.
- unsloth.ai/blog/r1-reasoning returned HTTP 403 to direct fetch; its figures are cited via
  search-result snippets and the Unsloth docs mirror.
- Qwen3.5-2B itself: I did not verify this model's existence/specs; all arithmetic assumes
  "a ~2B dense decoder." If the real model differs (e.g. different size or architecture),
  scale the throughput assumption roughly linearly with parameter count.

---

## Key Question 4: vLLM generation throughput for ~2B models

### Takeaway
Pure batched vLLM generation for a ~1.5–2B model is fast — one community benchmark shows
~5,400 output tokens/sec (GPU unspecified) — but no A100-specific published number for a 2B
model was found, and end-to-end GRPO throughput will be several times lower than pure
generation because vLLM only gets a slice of the GPU and training steps interleave.

### Cited Findings
- DeepSeek-R1-Distill-Qwen-1.5B on vLLM's offline throughput benchmark (50 prompts, 64 input
  / 128 output tokens): 42.33 requests/sec, 8,127 total tokens/sec, **5,418 output
  tokens/sec**; the GPU model is not named in the post, so do not assume A100 — [cnblogs benchmark tutorial](https://www.cnblogs.com/SunShine110/p/19086381)
- Unsloth cites ~4,000 tokens/sec on 1× A100 40GB for a 4-bit 3B model with vLLM — [Unsloth GRPO docs](https://unsloth.ai/docs/basics/reasoning-grpo)
- TRL colocate mode gives vLLM 30% of GPU memory by default and notes possible memory
  contention with the training model — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)

### Inferences
- Planning assumption: pure generation for a 2B model on an A100 80GB at GRPO-style batch
  sizes (8 prompts × 8 completions = 64 concurrent sequences) plausibly runs 2,000–5,000
  output tokens/sec. ESTIMATE from the two anchors above.
- End-to-end effective GRPO throughput (generation + reward execution + backward pass +
  weight sync, with vLLM on a 30–50% memory slice): assume **800–2,000 output tokens/sec**
  on the A100. ESTIMATE — this is the single most uncertain number in the cost model and the
  main thing the smoke test must measure.
- The SQL-execution reward (running generated SQL against SQLite) is CPU work; on small
  schemas it should add well under a second per completion if parallelized, but a slow or
  unindexed query can stall a step — add a per-query timeout (e.g. 2–5 s) in the reward
  function. ESTIMATE / design note, no published figure.

### Gaps
- No published vLLM throughput benchmark for a ~2B model specifically on A100 80GB was found.
- No published number for how much GRPO's colocate overhead cuts vLLM throughput in practice.

---

## Key Question 5: Run lengths in comparable small-model RLVR projects

### Takeaway
Comparable projects trained for a few hundred to ~450 GRPO steps at group sizes 8–16;
TinyZero reproduced R1-Zero-style reasoning on a 3B model for a claimed "< $30" of compute,
which is direct evidence that this project's $30–50 budget is realistic for a 2B model.

### Cited Findings
- SQL-R1: RL on 5K NL-SQL pairs sampled from SynSQL-2.5M; group size ("rollout of actor
  model") = 8; learning rate 3e-7; max response length 2,048 tokens; experiments on 8× 80GB
  GPUs; step count not reported — [SQL-R1 paper, arXiv 2504.08600](https://arxiv.org/html/2504.08600v5)
- Arctic-Text2SQL-R1: 64× H100-80GB total; total batch 256 with 16 rollouts per prompt,
  update batch 128 per GRPO step; temperature 0.8, KL penalty β = 0.001; "a few hours for the
  7B model, up to 2.5 days for the 32B"; tried 16/24/32 rollouts with no significant gain
  beyond 16; step count not reported — [Arctic-Text2SQL-R1 paper, arXiv 2505.20315](https://arxiv.org/html/2505.20315v2)
- Mini-R1: 450 steps, 8 generations, 4× H100, ≈ 6 hours; format learned by ~step 50, ~25%
  task success by step 100; DeepSeekMath default hyperparameters went unstable around step
  150 (lr 1e-6, KL beta 0.04) — [Mini-R1 blog](https://huggingface.co/blog/open-r1/mini-r1-contdown-game)
- TinyZero (UC Berkeley): "You can experience the Aha moment yourself for < $30" (README,
  no cost breakdown given); Qwen2.5-3B base on the countdown task via veRL; 3B needs 2 GPUs
  in their setup, ≤1.5B runs on one; they note Qwen2.5-0.5B fails to learn reasoning — [TinyZero README](https://github.com/Jiayi-Pan/TinyZero)
- Community SQL GRPO run: 200 steps, group 4, done in under 2 hours on one GPU (7B + LoRA) — [HF Space blog post](https://huggingface.co/spaces/junaid0600/sql-db-engineer-agent/blob/main/blog_post.md)
- Unsloth guidance: expect no reward movement before ~300 steps; 500+ training rows
  recommended — [Unsloth GRPO docs](https://unsloth.ai/docs/basics/reasoning-grpo)

### Inferences
- A sensible per-run length for this project is **300–500 GRPO steps**: below 300 the
  literature says reward signals may not have moved yet (Unsloth, Mini-R1's step-100
  milestone); beyond ~500 steps the comparable small-model runs (Mini-R1 450, community run
  200) had already converged or destabilized. ESTIMATE / judgment call.
- Group size 8 matches SQL-R1 exactly and the TRL default; Arctic's finding that >16 rollouts
  add nothing means there is no reason to spend budget on group sizes above 8–16.
- TinyZero's 0.5B failure vs 3B success brackets the 2B choice: 2B is plausibly big enough to
  learn execution-rewarded SQL behavior, but it sits near the lower edge — worth knowing
  going in. Inference, not a tested fact.

### Gaps
- Neither SQL-R1 nor Arctic-Text2SQL-R1 reports its GRPO step count — both report dataset
  size, batch, and hardware instead; per-step counts for the headline text-to-SQL papers are
  simply not published.
- TinyZero's "< $30" has no public breakdown (GPU type/hours unstated in the README).

---

## Key Question 6: Worked cost example, smoke test, and run matrix for $30–50

### Takeaway
At the verified RunPod A100 Secure price ($1.79/hr), a central-estimate 300-step run costs
roughly $2.50–4 and the full 8-run matrix lands near **$20–36 including overhead** — inside
budget even in the pessimistic case, with Modal's $30 free credit or the $0.34/hr community
RTX 4090 as large safety margins. All step-time figures below are ESTIMATES pending the
smoke-test measurement.

### Cited Findings
(Anchor numbers reused from above; all derived arithmetic is in Inferences.)
- A100 80GB: $1.79/hr Secure, $1.19/hr Community — [RunPod pricing](https://www.runpod.io/pricing), 2026-11-04
- RTX 4090: $0.34/hr Community — [RunPod pricing](https://www.runpod.io/pricing), 2026-11-04
- Modal A100 80GB $0.000694/sec; Starter plan $30/month free credit — [Modal pricing](https://modal.com/pricing), 2026-11-04
- Mini-R1 step-time anchor: 45–60 s/step (3B, 4×H100, 8 generations) — [Mini-R1 blog](https://huggingface.co/blog/open-r1/mini-r1-contdown-game)

### Inferences
Everything in this section is an ESTIMATE built from the cited anchors.

**Cost model.** cost = steps × (prompts_per_step × group_size × tokens_per_completion ÷
effective_throughput) ÷ 3600 × $/hr. In words: tokens generated per step, divided by
end-to-end tokens-per-second, gives seconds per step; multiply by step count and the hourly
price.

**Central worked example (A100 80GB, RunPod Secure $1.79/hr):**
- Settings: 8 prompts/step × group size 8 × 384 avg completion tokens (midpoint of 256–512)
  = 24,576 generated tokens per step.
- Effective throughput assumed 1,500 tokens/sec (middle of the 800–2,000 ESTIMATE range)
  → 24,576 ÷ 1,500 ≈ **16.4 s/step**.
- 300 steps → ~82 min ≈ 1.37 h → 1.37 × $1.79 ≈ **$2.45 per run**.
- 500 steps → ~2.28 h → **$4.08 per run**.
- 1,000 steps → ~4.55 h → **$8.15 per run**.
- Same math at Modal's derived $2.50/hr: 300 steps ≈ $3.41; 1,000 steps ≈ $11.38.

**Pessimistic worked example (same GPU):** group 16, 512 tokens, 8 prompts = 65,536
tokens/step at 800 tokens/sec → ~82 s/step. 300 steps ≈ 6.8 h ≈ $12.20; 1,000 steps ≈ 22.8 h
≈ $40.80 — a single 1,000-step pessimistic run nearly eats the budget, which is why the plan
below caps runs at 300–500 steps and group size 8. Sanity check: 82 s/step is worse than
Mini-R1's 45–60 s/step on a 3B with 4 GPUs, so it is a genuinely conservative bound.

**Epoch framing:** with ~500–1,000 training prompts and 8 prompts/step, one epoch (one pass
over the data) is 63–125 steps, so 300–500 steps ≈ 2.5–8 epochs — consistent with the small
RL datasets used by SQL-R1 (5K pairs) and Unsloth's 500+-row guidance.

**20-step smoke test (run first, on the target GPU):**
- Config: 20 steps, group 8, max_completion 512, 8 prompts/step, LoRA r=16, vLLM colocate
  (vllm_gpu_memory_utilization tuned upward from 0.3 if idle memory allows), checkpoint+
  resume once, log per-step wall-clock.
- Verify: vLLM starts and generates; SQL reward executes with per-query timeout and returns
  non-constant rewards; reward/length curves log; checkpoint save+resume works.
- Expected: 5–20 min → **≤ $0.60** on Secure A100, ≤ $0.12 on a community 4090. ESTIMATE.
- Decision rule: measured s/step × 300 × 8 runs must be ≤ ~2/3 of remaining budget. If the
  A100 measures > 25 s/step, cut completion length to 256, or move seed-replica runs to the
  $0.34/hr 4090.

**Run matrix (8 runs × 300 steps each):**

| Run | Reward config | Seeds |
|---|---|---|
| 1 | Accuracy-only baseline (no abstain penalty) | 1 |
| 2–7 | 3 abstain-penalty values × 2 seeds | 6 |
| 8 (optional) | Control model (e.g. no-RL or alt base), same eval | 1 |

- Central estimate: 8 × 1.37 h ≈ 11 GPU-hours → **$19.60** (Secure A100) or **$13.05**
  (Community A100); +20% setup/idle overhead → ~$24 / ~$16.
- Pessimistic (30 s/step → 2.5 h/run): 20 GPU-hours → **$35.80** Secure / **$23.80**
  Community — still inside $30–50.
- Degradation path if over budget: drop run 8, then run seeds on only the best and worst
  penalty values (saves 2 runs), then shift remaining runs to the 4090.
- Cheapest full-matrix path: 8 × 2.5 h on a $0.34/hr community 4090 ≈ **$6.80** — if the 2B
  model + LoRA + vLLM fits in 24GB, which Unsloth's VRAM guidance (≥5GB for ≤1.5B, 4-bit)
  strongly suggests it does even in 16-bit. Checkpoint frequently against interruption.
- Free-tier path: Modal Starter's $30/month credit ≈ 12 A100-hours covers the central
  estimate matrix entirely at $0 cash, with RunPod as overflow.

**Recommended plan:** smoke test on Modal free credit or a Secure A100 (< $1) → measure
s/step → run the matrix on RunPod Community A100 ($1.19/hr) or 4090 ($0.34/hr) with 25-step
checkpointing → reserve Secure A100 only if community availability/interruptions bite.
Expected total spend: **$10–30**, worst case ~$40. ESTIMATE.

### Gaps
- The effective end-to-end throughput (800–2,000 tokens/sec) is unmeasured anywhere in the
  literature for this exact setup; the smoke test is the only way to pin it down.
- Whether Qwen3.5-2B + LoRA + vLLM colocate fits a 24GB 4090 at 512-token completions is
  inferred from Unsloth's VRAM table, not demonstrated — verify during the smoke test before
  committing the matrix to the 4090.
- SQL-execution reward latency (CPU-side) on the project's actual databases is unmeasured and
  could dominate step time if queries are slow — the per-query timeout is a mitigation, not
  a measurement.
