# Compute Estimate — GRPO Training of Qwen3.5-2B on a $30–50 Budget

Compiled 2026-11-04 from `compute_costs.md` in the research notes. **All prices were fetched from the live pricing pages on 2026-11-04 and will drift — re-check before renting. Every derived number below is an ESTIMATE** built from those prices plus throughput anchors; no published benchmark measures the exact configuration, so the smoke test (below) must replace the estimates with measurements before any full run.

Setup assumed throughout: GRPO (Group Relative Policy Optimization — an RL method that samples a group of answers per prompt and scores them against each other) on Qwen3.5-2B with LoRA (low-rank adapters — training a small add-on instead of all the weights) and vLLM (a fast batched text-generation engine) sharing a single GPU, with a custom reward that executes the generated SQL.

## The cost formula

**cost = steps × prompts_per_step × group_size × tokens_per_completion ÷ effective_throughput ÷ 3600 × $/hr**

In words: tokens generated per training step, divided by end-to-end tokens-per-second, gives seconds per step; multiply by the number of steps and the hourly GPU price.

## Verified prices (fetched 2026-11-04)

From [runpod.io/pricing](https://www.runpod.io/pricing) (Secure Cloud = vetted datacenters; Community Cloud = cheaper peer-hosted machines that can be interrupted mid-run) and [modal.com/pricing](https://modal.com/pricing); dollars per hour, billed per second:

| GPU | VRAM (GPU memory) | RunPod Secure | RunPod Community | Modal (derived from $/sec) |
|---|---|---|---|---|
| A100 80GB | 80 GB | $1.79 | $1.19–1.39 | ≈ $2.50 ($0.000694/s) |
| H100 | 80 GB | $2.89 | $1.99 | ≈ $3.95 |
| L40S | 48 GB | $1.09 | $0.79 | ≈ $1.95 |
| RTX 4090 | 24 GB | $0.89 | **$0.34** | not offered |
| RTX 5090 | 32 GB | $1.19 | $0.69 | not offered |

- **Modal Starter plan: $30/month free credit — verified** on the pricing page; ÷ $2.50/hr ≈ **12 free A100-hours/month** (ESTIMATE, simple division). Whether the credit applies to GPU compute without restriction was not spelled out on the page — UNVERIFIED; check the dashboard before relying on it.
- Storage is negligible: a ~50 GB persistent volume ≈ $5/month prorated on RunPod (ESTIMATE).

## Throughput anchors (the honest gap)

**No published end-to-end GRPO steps/hour or tokens/sec benchmark exists for a 1–3B model with LoRA + vLLM on a single A100 or 4090.** The anchors that do exist:

- Unsloth: ~4,000 tokens/sec *pure generation* for a 4-bit 3B model on one A100 40GB — generation speed, not a training rate ([Unsloth GRPO docs](https://unsloth.ai/docs/basics/reasoning-grpo)).
- Mini-R1: 45–60 s per GRPO step for a 3B model at 8 generations on 4×H100; 450 steps ≈ 6 hours ([Mini-R1 blog](https://huggingface.co/blog/open-r1/mini-r1-contdown-game)).
- A community SQL GRPO run: 7B + LoRA, group 4, 150-token completions, 200 steps in under 2 hours on one GPU, no vLLM ([HF Space post](https://huggingface.co/spaces/junaid0600/sql-db-engineer-agent/blob/main/blog_post.md)).
- TinyZero: R1-style RL on a 3B model for a claimed "< $30" total (no breakdown published) — direct evidence the budget class is realistic ([TinyZero README](https://github.com/Jiayi-Pan/TinyZero)).

Planning assumptions triangulated from these (**all ESTIMATES**): pure generation for a 2B model on an A100 80GB at GRPO batch shapes ≈ 2,000–5,000 output tokens/sec; **end-to-end effective GRPO throughput ≈ 800–2,000 output tokens/sec** (generation + SQL-reward execution + backward pass + weight sync, with vLLM on a 30–50% memory slice). This end-to-end number is the single most uncertain input, and the one the smoke test measures. The SQL reward itself is CPU work — well under a second per completion on small SQLite schemas if parallelized, but one slow query can stall a whole group, so every query gets a 2–5 s timeout (design note, no published figure).

## Worked estimates (ESTIMATES throughout)

Central settings: 8 prompts/step × group size 8 × 384 average completion tokens = 24,576 generated tokens per step. At 1,500 tokens/sec effective → **≈ 16.4 s/step**.

On a RunPod Secure A100 at $1.79/hr:
- 300 steps ≈ 1.37 h → **≈ $2.45 per run** (about the cost of a coffee per full training run).
- 500 steps ≈ 2.28 h → ≈ $4.08 per run. 1,000 steps ≈ 4.55 h → ≈ $8.15.
- Same math at Modal's ≈ $2.50/hr: 300 steps ≈ $3.41.

Pessimistic case (group 16, 512 tokens, 800 tokens/sec → ~82 s/step): 300 steps ≈ 6.8 h ≈ **$12.20 per run**; a 1,000-step pessimistic run ≈ $40.80 would nearly eat the budget by itself — which is why runs are capped at 300–500 steps and group size 8. Sanity check: 82 s/step is slower than Mini-R1's 45–60 s/step on a bigger model with four GPUs, so it is genuinely conservative.

Run length rationale: comparable projects trained 200–450 steps; Unsloth's guidance says expect no reward movement before ~300 steps; with ~500–1,000 training prompts at 8 prompts/step, 300–500 steps ≈ 2.5–8 passes over the data (consistent with SQL-R1's 5K-pair RL set). Group size 8 matches SQL-R1 and the TRL default; Arctic found no gain beyond 16 rollouts, so nothing above 8–16 is worth paying for.

## 20-step smoke test (run FIRST, on the target GPU)

- **Config:** 20 steps, group 8, 8 prompts/step, max completion 512, LoRA r=16, vLLM colocate (`vllm_gpu_memory_utilization` tuned up from 0.3 if idle memory allows); checkpoint and resume once; log per-step wall-clock.
- **Verify:** vLLM starts and generates for Qwen3.5-2B (the architecture-risk check); the SQL reward executes with per-query timeouts and returns non-constant rewards; reward/length/abstention-rate curves log; checkpoint save + resume works; VRAM headroom (does it fit a 24GB 4090?).
- **Cost:** 5–20 min → **≤ $0.60** on a Secure A100, ≤ $0.12 on a community 4090 (ESTIMATE).
- **Decision rule:** measured s/step × 300 steps × 8 runs must be ≤ ~2/3 of remaining budget. If the A100 measures > 25 s/step: cut completions to 256 tokens, or move seed-replica runs to the $0.34/hr 4090.

## Run matrix and total cost range

8 runs × 300 steps each (per the brief: accuracy-only baseline + 3 values of the wrong-answer penalty c × 2 seeds + optional control model):

| Run | Reward config | Count |
|---|---|---|
| 1 | Accuracy-only GRPO baseline (no abstain action) | 1 |
| 2–7 | 3 penalty values c × 2 random seeds | 6 |
| 8 (optional) | Control model (SmolLM3-3B or Gemma 4 E2B), best c | 1 |

Totals (ESTIMATES):
- **Central:** 8 × 1.37 h ≈ 11 GPU-hours → ≈ $19.60 (Secure A100) or ≈ $13.05 (Community A100); +20% setup/idle overhead → **≈ $24 / ≈ $16**.
- **Pessimistic** (30 s/step → 2.5 h/run): 20 GPU-hours → ≈ $35.80 Secure / ≈ $23.80 Community — still inside $30–50.
- **Cheapest path:** 8 × 2.5 h on a $0.34/hr community 4090 ≈ **$6.80** — if the 2B model + LoRA + vLLM fits in 24 GB, which Unsloth's VRAM guidance (≥5 GB suffices for ≤1.5B models in 4-bit) strongly suggests but the smoke test must confirm. Checkpoint every 25–50 steps against community-machine interruption.
- **Free-tier path:** Modal Starter's $30/month credit ≈ 12 A100-hours covers the entire central-estimate matrix at $0 cash, with RunPod as overflow.

**Recommended sequence:** smoke test on Modal free credit or a Secure A100 (< $1) → measure s/step → run the matrix on RunPod Community A100 ($1.19/hr) or 4090 ($0.34/hr) with frequent checkpoints → reserve Secure A100 for when community availability bites. **Expected total spend: $10–30; worst case ≈ $40 (ESTIMATE).** Degradation path if over budget: drop run 8, then run seeds only on the best and worst penalty values, then shift remaining runs to the 4090.

## Known gaps

- End-to-end throughput (800–2,000 tokens/sec) is unmeasured anywhere in the literature for this setup — smoke test replaces it.
- Whether Qwen3.5-2B + LoRA + vLLM colocate fits 24 GB at 512-token completions is inferred from Unsloth's VRAM table, not demonstrated.
- SQL-reward latency on the project's actual databases is unmeasured; the per-query timeout is a mitigation, not a measurement.
- Modal's $30 credit scope (GPU compute included?) is UNVERIFIED on the pricing page.
