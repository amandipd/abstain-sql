# Training Stack for GRPO RL Fine-Tuning (Abstaining Text-to-SQL), as of 2026-11-04

Scope: compare TRL GRPOTrainer + vLLM (colocate), Prime Intellect `verifiers`, and Unsloth GRPO for training Qwen3.5-2B (controls: Gemma 4 E2B, SmolLM3-3B) with LoRA on a single cloud NVIDIA GPU, with a custom Python reward that executes generated SQL. Plus abstention-mechanism prior art, loss settings (Dr. GRPO), and reward hacking in execution-reward SQL training.

All findings verified by fetching the cited page on 2026-11-04 unless marked UNVERIFIED.

## Key Question 1: Do the named models exist, and what are their exact IDs and support status?

### Takeaway
All three models exist. `Qwen/Qwen3.5-2B` is real (released ~March 2026) but is a *multimodal* model with a hybrid Gated-DeltaNet/attention architecture — not a plain dense text LLM — which has stack-support consequences (notably in Unsloth). "Gemma 4" also exists (March/April 2026, superseding Gemma 3n): the control model is `google/gemma-4-E2B-it` (2.3B effective / 5.1B total params). `HuggingFaceTB/SmolLM3-3B` is a July-2025 dense 3B model, the most "boring"/best-supported of the three.

### Cited Findings

**Qwen3.5-2B**
- `Qwen/Qwen3.5-2B` exists on Hugging Face: 2B params, Apache-2.0, 262,144-token native context, base variant `Qwen/Qwen3.5-2B-Base`. No separate "-Instruct" suffix; the unsuffixed model is the post-trained one — [Qwen/Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- Architecture: causal LM with a vision encoder (tagged image-text-to-text, handles image and video input); layers alternate Gated DeltaNet and Gated Attention blocks; 2048 hidden dim, 24 layers — [Qwen/Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- "Qwen3.5-2B operates in non-thinking mode by default"; the card warns this size is "more prone to thinking loops than other Qwen3.5 models" — [Qwen/Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- The card requires "the latest `transformers`" and (as written at release) vLLM main-branch/nightly; serving flags include `--language-model-only` for text-only use — [Qwen/Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- Family sizes on HF: 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B — [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), [unsloth Qwen3.5 collection](https://huggingface.co/collections/unsloth/qwen35)
- Release dates per search aggregation (UNVERIFIED exact dates): 397B-A17B on 2026-02-16; 122B-A10B/35B-A3B/27B on 2026-02-24; 9B/4B/2B/0.8B on 2026-03-02. The 2B card's citation lists February 2026 — [Qwen/Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- vLLM support: a community article says vLLM v0.17.0 shipped "complete Qwen 3.5 series support" (UNVERIFIED, secondary source: [besthub.dev](https://besthub.dev/articles/vllm-0-17-0-release-full-qwen-3-5-support-and-anthropic-api-compatibility-218ac81d23cd)); vLLM v0.19.0 release notes (released Apr 3) list Qwen3.5 maintenance items — "Qwen3.5 GDN quantized models" fix, a "Qwen3.5 fix" under *LoRA expansion*, and a tuned Triton MoE config for Qwen3.5 on H200 — confirming Qwen3.5 (and its LoRA path) is in-tree by v0.19.0 — [vLLM v0.19.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.19.0)
- A `QwenLM/Qwen3.8` GitHub repo surfaced in search (mentioning Qwen3.6/Qwen3.8), suggesting even newer families exist by Nov 2026; not verified — [github.com/QwenLM/Qwen3.8](https://github.com/QwenLM/Qwen3.8) (UNVERIFIED)

**Gemma 4 E2B**
- "Gemma 4" exists: Google's release log lists Gemma 4 in E2B, E4B, 31B, and 26B A4B sizes dated March 31, 2026; "Gemma 4 - MTP" April 16, 2026; "Gemma 4 12B Unified" June 3, 2026. Gemma 3n (E2B/E4B) is the predecessor, dated June 26, 2025 — [Google Gemma releases log](https://ai.google.dev/gemma/docs/releases)
- Launch announcement ~April 2, 2026, Apache 2.0 ("first Gemma family under that OSI-approved licence") — [Google open source blog](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html), [Google developers blog](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/)
- `google/gemma-4-E2B-it` confirmed on HF: 2.3B effective params, 5.1B total incl. embeddings (safetensors lists 5B BF16); 35 layers, 262K vocab, Per-Layer Embeddings (PLE), interleaved 512-token sliding-window + global attention; text/image/audio input; 128K context; Apache 2.0; "use the latest version of Transformers" — [google/gemma-4-E2B-it model card](https://huggingface.co/google/gemma-4-E2B-it)
- vLLM v0.19.0: "Full Google Gemma 4 architecture support including MoE, multimodal, reasoning, and tool-use capabilities. Requires `transformers>=5.5.0`" — [vLLM v0.19.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.19.0)

**SmolLM3-3B**
- `HuggingFaceTB/SmolLM3-3B` confirmed: 3B params BF16, Apache 2.0, 64k trained context (128k via YaRN config edit), dual-mode reasoning (extended thinking on by default; disable via `/no_think` or `enable_thinking=False`), requires transformers >= 4.53.0, served by vLLM (via the Transformers backend per the card) — [HuggingFaceTB/SmolLM3-3B model card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)

### Inferences
- All three requested model names are real as of Nov 2026; no substitution needed. The one naming correction: the small Gemma 4 is written `gemma-4-E2B-it` (instruction-tuned) on HF; "Gemma 3n E2B" is the older June-2025 model, not the current one.
- Qwen3.5-2B being multimodal + hybrid-attention means it is the riskiest of the three for tooling: it needed vLLM nightly at release, and Unsloth's docs flag it as unsupported by (Unsloth-embedded) vLLM (see KQ4). SmolLM3-3B is a conventional dense transformer and will "just work" in every stack.
- Gemma 4 E2B's 5.1B total parameters mean its memory footprint is closer to a 5B model than a 2B one for training, even though effective compute is ~2.3B.

### Gaps
- Could not fetch vLLM's supported-models page directly (HTTP 429 rate-limit on docs.vllm.ai twice), so the exact vLLM architecture class names for Qwen3.5 (`Qwen3_5ForConditionalGeneration` appeared in a community wheel description and `Qwen3_5MoeForCausalLM` in a TensorRT-LLM list — both UNVERIFIED for vLLM) are unconfirmed.
- Exact minimum transformers version for Qwen3.5 (the card just says "latest"); Gemma 4 needs transformers>=5.5.0 per vLLM's note, implying transformers had reached a 5.x major by spring 2026.

## Key Question 2: TRL GRPOTrainer — current state

### Takeaway
TRL is at v1.17.1 (Oct 30; year not shown on page, consistent with 2026) and is the most feature-complete option: colocate vLLM is the *default* vLLM mode, LoRA/QLoRA via `peft_config`, reward functions are arbitrary (even async) Python, `loss_type="dr_grpo"` is supported (default is now `"dapo"`), and `scale_rewards` controls std-normalization. Single-GPU use is explicitly documented (colocate memory knobs, `beta=0.0` skips the reference model, vLLM sleep mode).

### Cited Findings
- Latest releases: v1.17.1 (30 Oct), v1.17.0 (29 Oct), v1.16.0 (15 Oct), v1.14.x (Sep–Oct) — [TRL releases](https://github.com/huggingface/trl/releases)
- Reward functions: must accept kwargs (`prompts`, `completions`, `completion_ids`, `trainer_state`, plus any dataset column; easiest to add `**kwargs`); return a list of floats (or `None` per-sample to skip); can be sync callables **or `async def` coroutines awaited via `asyncio.gather`** — i.e., arbitrary Python including DB connections/SQL execution is fine — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- `loss_type` default is `'dapo'`; options include `"dr_grpo"` (divides by constant `1/(L·G)` where L = max completion length, removing length bias) and `"sapo"`; set with `GRPOConfig(loss_type="dr_grpo")` — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- `scale_rewards` default `'group'` ((r − mean)/std within group); `False` disables std scaling (the Dr. GRPO recommendation); `"batch"` uses batch-level std and is what the docs recommend "for more robust reward shaping" — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- vLLM integration: `use_vllm=True`; **colocate is the default mode** (`vllm_mode="colocate"`, vLLM shares the training GPU inside the trainer process); knobs: `vllm_gpu_memory_utilization` (default 0.3), `vllm_tensor_parallel_size` (default 1), `vllm_enable_sleep_mode` (offloads weights/cache during the optimizer step). Server mode runs vLLM as a separate process and "must use different GPUs than the trainer" — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer); background on colocation: [HF blog: Co-located vLLM in TRL](https://huggingface.co/blog/vllm-colocate)
- LoRA/PEFT: `GRPOTrainer` accepts `peft_config` (e.g. `LoraConfig`); `quantization_config` + `peft_config` = QLoRA; v1.14.0 added adapter-only vLLM weight sync for AsyncGRPO LoRA — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer), [TRL releases](https://github.com/huggingface/trl/releases)
- Key `GRPOConfig` defaults: `num_generations=8`, `max_completion_length=512`, `beta=0.0` (KL coefficient; at 0.0 the reference model is not loaded, saving memory), `epsilon=0.2` (`epsilon_high` optional, DAPO recommends 0.28), `importance_sampling_level='token'`, `learning_rate=1e-06`, `per_device_train_batch_size=8` — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Single-GPU/memory notes in docs: keep `vllm_gpu_memory_utilization` modest, enable sleep mode on OOM; a Transformers continuous-batching generation backend is "well-suited to single-GPU or memory-constrained setups"; v1.14.0's `precompute_ref_log_probs=True` cut peak memory ~31% in their benchmark — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer), [TRL releases](https://github.com/huggingface/trl/releases)
- Version-compatibility wrinkles: v1.17.0 colocate log-prob renormalization requires vLLM >= 0.28; v1.15.0 introduced a fused LM head for GRPO scoring under which "a PEFT adapter on `lm_head` now raises" — [TRL releases](https://github.com/huggingface/trl/releases)

### Inferences
- TRL meets every stated requirement: Qwen3.5 (via current transformers/vLLM), LoRA, arbitrary-Python SQL-executing reward (async support is a genuine advantage for DB calls), colocate single-GPU operation, and Dr. GRPO loss (`loss_type="dr_grpo"` + `scale_rewards=False` reproduces the paper's recipe).
- vLLM's own versioning has moved fast in 2026 (0.17 → 0.19 → ≥0.28 referenced by TRL; a docs URL for v0.32.0 exists), so pinning a tested TRL+vLLM pair matters; TRL release notes are the authority on which vLLM range each TRL version supports.
- The fused-LM-head/PEFT-on-lm_head restriction (v1.15.0) is directly relevant to the abstain-token decision (KQ5): adding a new vocab token requires training `lm_head`/`embed_tokens`, which collides with TRL's current fused scoring path.

### Gaps
- The releases page shows day/month only; the year (2026) for v1.17.1 is inferred from ordering, not shown on the page.
- Did not verify the exact vLLM version range supported by TRL v1.17.x (an older doc mirror said "0.10.2 to 0.17.1", which contradicts the v1.17.0 note about vLLM >= 0.28 — the mirror is likely stale; treat the release notes as current).

## Key Question 3: `verifiers` (Prime Intellect) — what it is and whether it fits

### Takeaway
`verifiers` (MIT, originally by Will Brown, now maintained under PrimeIntellect-ai; 4.8k stars; PyPI v0.4.1 released 2026-11-03) is an *environments-and-evals* library tightly coupled to Prime Intellect's `prime-rl` trainer and Environments Hub. It has churned heavily in 2026: the old `vf.GRPOTrainer` was replaced by a minimal `vf.RLTrainer` (v0.1.7, Nov 2025, per mirror READMEs), and current docs describe a Taskset/Harness/Agent/Environment/Trace model. Powerful for multi-turn/tool environments, but the training path is less turnkey and less documented for single-GPU LoRA than TRL or Unsloth.

### Cited Findings
- "Verifiers is a library of modular components for creating train-ready RL environments and evaluations"; docs moved to docs.primeintellect.ai/verifiers — [verifiers readthedocs redirect](https://verifiers.readthedocs.io/en/latest/)
- Repo: MIT license, "Originally created by Will Brown (@willccbb)", maintained by Prime Intellect; "tightly integrated with the Environments Hub," the prime-rl training framework, and the Hosted Training platform; 4.8k stars, 724 forks, 2,604 commits — [github.com/PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers)
- PyPI: latest stable **0.4.1, released Nov 3, 2026** (plus 0.4.2.dev1 same day); `pip install verifiers`; Python >=3.11,<3.15 — [PyPI: verifiers](https://pypi.org/project/verifiers/)
- Current official docs describe: **Taskset** (rows of `TaskData` + lifecycle hooks, tools, metrics, rewards), **Harness** (the program the model runs in — examples: Claude Code, Codex, mini-swe-agent), **Agent** (harness+model+policy producing a `Trace`), **Environment** (control flow across an episode: retries, judge, simulated user), **Toolset** (MCP-installed tools), **Trace** (message graph, rewards, logprobs "when used with prime-rl") — [docs.primeintellect.ai/verifiers](https://docs.primeintellect.ai/verifiers)
- Older/classic API (from fork/mirror READMEs, UNVERIFIED against the official repo at head): `SingleTurnEnv` built from a dataset + a `Rubric`; a Rubric is a set of weighted sync-or-async reward functions (weight-0 entries are logged, not optimized) — [zapier/verifiers mirror](https://github.com/zapier/verifiers), [RKP64/verifiers mirror](https://github.com/RKP64/verifiers)
- Trainer churn (same mirrors, UNVERIFIED): "v0.1.7 (11/07/25) added a 'nano' trainer, `vf.RLTrainer`, which replaces `vf.GRPOTrainer`"; RLTrainer is "a minimal trainer for single-node runs" supporting "both full fine-tuning and LoRA"; prime-rl is recommended "for large-scale, performance-optimized async training" — [zapier/verifiers](https://github.com/zapier/verifiers), [De30/verifiers](https://github.com/De30/verifiers)

### Inferences
- Custom SQL-executing rewards fit naturally (Rubric reward functions are arbitrary sync/async Python, same as TRL), and the environment abstraction would help if the project later becomes multi-turn (e.g., model inspects schema, then writes SQL).
- For this project's shape — single-turn generation, single GPU, LoRA — verifiers adds a layer of indirection (environments + a trainer that is either "nano" or requires prime-rl) without a capability TRL lacks. Its 2025→2026 API churn (GRPOTrainer → RLTrainer → Taskset/Harness docs model) is a maintenance risk for a months-long project.
- vLLM usage: inference in verifiers/prime-rl is vLLM-based per the ecosystem's design, but I could not pin an official current statement of the single-GPU colocate story; assume it is weaker/less documented than TRL's.

### Gaps
- Could not verify current (v0.4.1) class names, reward-function signature, LoRA flags, vLLM wiring, or hardware guidance from an official source: the GitHub README fetch returned only pointers, the readthedocs page is a redirect stub, the official docs overview omits API details, and search only surfaced forks/mirrors of older versions. Anything above sourced to mirrors should be re-checked against [docs.primeintellect.ai/verifiers](https://docs.primeintellect.ai/verifiers) subpages ("Getting started", "Building environments", "Tasksets") before building on it.
- No official statement found on whether Qwen3.5's hybrid architecture is supported by prime-rl/RLTrainer.

## Key Question 4: Unsloth GRPO — efficiency, LoRA, rewards, limitations

### Takeaway
Unsloth is the VRAM-efficiency leader (first-party claim: ~90% less VRAM than FA2 baselines; 5GB VRAM suffices for ≤1.5B models) with LoRA/QLoRA GRPO, arbitrary custom reward functions, and `loss_type` options including `dr_grpo`. Catch for this project: Unsloth's own RL guide says vLLM does not support Qwen3.5 inside Unsloth — you must set `fast_inference=False` (slower HF-based generation), even though upstream vLLM added Qwen3.5 support in early 2026. Gemma 4 E2B and Qwen3.5-4B have official Unsloth GRPO notebooks.

### Cited Findings
- VRAM claims: Llama 3.1 8B GRPO at 20K context, 8 generations/prompt: "54.33GB (90% less)" vs 510.8GB for "standard implementations with Flash Attention 2" — [Unsloth RL guide](https://unsloth.ai/docs/basics/reinforcement-learning-rl-guide); same figures in [Unsloth GRPO blog](https://unsloth.ai/blog/grpo)
- Mechanisms: memory-efficient linear kernels ("slashes memory usage by 8x or more"); Unsloth gradient checkpointing that asynchronously offloads activations to system RAM ("only 1% slower"); sharing GPU memory with the embedded vLLM engine ("shaves 16GB of memory") — [Unsloth RL guide](https://unsloth.ai/docs/basics/reinforcement-learning-rl-guide), [Unsloth GRPO blog](https://unsloth.ai/blog/grpo)
- Minimum VRAM: "Just 5GB VRAM is enough" for ≤1.5B-param models; 15GB VRAM handles up to ~17B; QLoRA 4-bit rule of thumb: VRAM ≈ parameter count; LoRA 16-bit uses "at minimum 4x more VRAM" than QLoRA 4-bit — [Unsloth RL guide](https://unsloth.ai/docs/basics/reinforcement-learning-rl-guide)
- Custom rewards: supported; docs recommend rubrics of smaller verifiable rewards, with GSM8K examples — [Unsloth RL guide](https://unsloth.ai/docs/basics/reinforcement-learning-rl-guide)
- Loss settings exposed: `loss_type` ∈ {`'grpo'`, `'gspo'`, `'dr_grpo'`}; suggested `epsilon=0.2`, `epsilon_high=0.28`, `delta=1.5`, `mask_truncated_completions=True`; `num_generations` 8 in examples, "try 16"; expect ≥300 steps before reward moves; ideally 500+ rows; models should be ≥1.5B params — [Unsloth RL guide](https://unsloth.ai/docs/basics/reinforcement-learning-rl-guide)
- Model support: official notebooks for Gemma 4 (E2B), Qwen3.5 (4B), Qwen3-VL-8B, gpt-oss-20b, Llama 3.1/3.2, Phi-4, etc.; **"For models vLLM doesn't support, such as Qwen3.5, set `fast_inference=False`"** — [Unsloth RL guide](https://unsloth.ai/docs/basics/reinforcement-learning-rl-guide). Note this conflicts with upstream vLLM having Qwen3.5 support by v0.17–0.19 ([vLLM v0.19.0 notes](https://github.com/vllm-project/vllm/releases/tag/v0.19.0)); likely the Unsloth-pinned vLLM or its LoRA-sync path can't handle the hybrid GDN architecture, or the doc line is stale. Treat as: *Unsloth + Qwen3.5 currently means no fast vLLM generation* until retested.
- Unsloth publishes its own Qwen3.5 quantized/finetune-ready checkpoints — [unsloth Qwen3.5 collection on HF](https://huggingface.co/collections/unsloth/qwen35)
- Earlier blog: GRPO with Qwen2.5-1.5B ran in ~5GB VRAM (down from 7GB); LoRA loading into vLLM via state dict made GRPO ~1.5x faster — [Unsloth GRPO blog](https://unsloth.ai/blog/grpo), [Unsloth R1 reasoning blog](https://unsloth.ai/blog/r1-reasoning)
- Vision-RL page: vLLM doesn't support LoRA on vision/encoder layers, so `finetune_vision_layers=False` is required in that setup — [Unsloth VLM RL docs](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl)

### Inferences
- For Gemma 4 E2B or SmolLM3-3B on a single mid-range GPU (e.g. 16–24GB), Unsloth is the cheapest-VRAM path and has a ready-made Gemma 4 E2B GRPO notebook.
- For Qwen3.5-2B specifically, Unsloth's `fast_inference=False` requirement removes its main speed advantage (vLLM-backed rollouts), making TRL+vLLM colocate likely faster for that model — worth a small benchmark before committing.
- Multi-GPU: the RL guide page is silent; historically Unsloth OSS was single-GPU-focused. UNVERIFIED either way — but for this single-GPU project it's moot.
- VRAM headline numbers are first-party vendor benchmarks at favorable settings (20K context); at this project's likely 1–2K completions the absolute gap vs TRL will be far smaller.

### Gaps
- No official statement found on Unsloth multi-GPU GRPO status (not needed here).
- Could not verify whether the "Qwen3.5 needs fast_inference=False" line is current or stale relative to upstream vLLM support; needs an empirical check at project start.
- Unsloth docs give no recommended KL/beta value for GRPO (KL mentioned only re: PPO).

## Key Question 5: Abstain mechanism — special token vs structured output

### Takeaway
RLVR projects implement refusal almost universally as *text the model emits* (a boxed "I don't know", a refusal phrase, or a structured field) scored by the reward function — not as a new vocabulary token. The added-token approach exists (Cohen et al.'s [IDK] token, NeurIPS 2024) but comes from the continued-pretraining literature and requires training embeddings/lm_head, which conflicts with LoRA-on-a-budget and with TRL's current fused-lm-head scoring. The documented stability risk for RL-trained abstention is collapse to always-abstain under a fixed abstention reward; mitigations are penalizing refusal on answerable items and/or setting the abstain reward dynamically.

### Cited Findings
- Special-token prior art: "I Don't Know: Explicit Modeling of Uncertainty with an [IDK] Token" (Cohen, Dobler, Biran, de Melo; NeurIPS 2024, arXiv 2412.06676) adds an [IDK] token to the vocabulary and does *continued pretraining* with a modified cross-entropy objective that shifts probability mass to [IDK] proportionally to an uncertainty factor; needs no uncertainty annotations; models "express uncertainty in places where they would previously make mistakes" with only a small knowledge loss; code at github.com/roi-hpi/IDK-token-tuning — [arXiv 2412.06676](https://arxiv.org/abs/2412.06676)
- RLVR text-based abstention: Abstain-R1 ("Calibrated Abstention and Post-Refusal Clarification via Verifiable RL") gives a base reward of 0.3 for a *boxed "I don't know"* on unanswerable questions and a penalty for refusing solvable ones — [arXiv 2604.17073](https://arxiv.org/html/2604.17073v1)
- Ternary reward designs: "Reinforced Hesitation" modifies RLVR to reward +correct / 0 abstain / −wrong, tested on logic puzzles with Qwen3-1.7B — [shortspan.ai summary](https://shortspan.ai/train-models-to-abstain-when-uncertain.html) (secondary source; UNVERIFIED against the paper). "Rewarding Intellectual Humility: Learning When Not to Answer" uses a similar three-outcome reward on medical QA — [arXiv 2601.20126](https://arxiv.org/html/2601.20126)
- Stability/reward-hacking evidence: a *fixed* abstention reward "can lead a model to refuse nearly everything"; AWA-RL (abstention-aware RL for search agents) responds by setting the abstention reward *dynamically* based on model confidence and training performance — [alphaXiv 2607.10738](https://www.alphaxiv.org/abs/2607.10738)
- Detection-vs-abstention gap under insufficient information in reasoning models is studied separately — [arXiv 2605.28070](https://arxiv.org/pdf/2605.28070); a position paper argues standard benchmarks reward guessing over declining — [arXiv 2609.17686](https://arxiv.org/pdf/2609.17686)
- TRL mechanical support for structured abstention: reward functions receive raw completion text and can return any float, and a reward function may return `None` per sample to exclude it — so a `{"sql": ..., "abstain": bool}` JSON (or an `<abstain/>` tag) is trivially parseable and scorable in the reward — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Vocab-extension friction in the chosen stacks: TRL v1.15.0's fused LM head *raises an error* if a PEFT adapter is placed on `lm_head` — [TRL releases](https://github.com/huggingface/trl/releases); vLLM's LoRA path for Qwen3.5 needed an upstream "LoRA expansion" fix as late as v0.19.0 — [vLLM v0.19.0 notes](https://github.com/vllm-project/vllm/releases/tag/v0.19.0)

### Inferences
- For a LoRA + vLLM-colocate pipeline, the structured-output design is strongly favored on engineering grounds: a new token forces `embed_tokens`/`lm_head` into `modules_to_save` (full-size trainable copies — large VRAM hit at Qwen3.5's/Gemma 4's 150K–262K vocabularies), breaks TRL's fused-head path, and complicates adapter-only weight sync into vLLM. No RLVR paper found that adds a vocab token for abstention; the [IDK]-token line of work predates RLVR-style GRPO pipelines and used full continued pretraining.
- A practical middle ground used by RLVR work is a *reserved textual marker* (boxed phrase or JSON field) + a format reward, which gets most of the "discrete action" benefit without touching the vocabulary.
- Expected failure mode to design against: abstention-rate collapse (always-abstain if abstain reward too high relative to the wrong-answer penalty; never-abstain if too low). The ternary structure (+1 correct / ~0 to +0.3 abstain / negative for wrong) with a penalty for abstaining on easy/answerable items is the recurring recipe.

### Gaps
- No head-to-head study found comparing token-based vs text-parsed abstention *under GRPO specifically*; the "trains more stably" question has only indirect evidence (the AWA-RL collapse finding, and the absence of token-based designs in RLVR papers). Marked as an open empirical question.
- "Reinforced Hesitation" details are from a secondary summary; the primary paper was not fetched.
- The 2026 arXiv abstention papers (2601.20126, 2604.17073, 2605.28070, 2607.10738, 2609.17686) were identified via search with abstracts/snippets only; their full experimental details were not extracted.

## Key Question 6: Loss settings — Dr. GRPO, KL coefficients, group sizes, length caps

### Takeaway
Dr. GRPO (from "Understanding R1-Zero-Like Training", arXiv 2503.20783) removes exactly two GRPO normalizations: the per-response 1/|o_i| length division and the std(reward) division in the advantage. In TRL that maps to `loss_type="dr_grpo"` + `scale_rewards=False`. Recent small-model GRPO practice: KL beta of 0 (TRL default, Dr. GRPO) to ~1e-3 (Arctic-Text2SQL-R1); group sizes 8–16; completion caps 512 (TRL default) to 2048–3000 tokens for reasoning-style tasks.

### Cited Findings
- Dr. GRPO: "we propose to simply remove the 1/|o_i| and std(...) normalization terms"; Appendix A: "both std and |o| should not appear in the RL objective"; unbiased implementation also means replacing `mask.sum(axis=dim)` with a constant (e.g., the generation budget). GRPO's bias "artificially increases response length (especially for incorrect outputs)" — [arXiv 2503.20783 (HTML v2)](https://arxiv.org/html/2503.20783v2); paper: "Understanding R1-Zero-Like Training: A Critical Perspective", Liu et al., v2 Oct 2025 — [arXiv abs](https://arxiv.org/abs/2503.20783)
- Dr. GRPO experimental settings: β=0 ("We will assume β=0 throughout this paper"; Table 6 KL coefficient 0.0), group size 8, Qwen2.5-1.5B base, 3000-token max response, temperature 1.0, lr 1e-6, clip 0.2, one inner epoch — [arXiv 2503.20783 (HTML v2)](https://arxiv.org/html/2503.20783v2)
- TRL mapping: `loss_type="dr_grpo"` divides by constant 1/(L·G); `scale_rewards=False` "disables std scaling"; defaults `beta=0.0`, `num_generations=8`, `max_completion_length=512`, `epsilon=0.2`, lr 1e-6; docs note `scale_rewards="batch"` as a recommended robust alternative; TRL's *default* loss is now `"dapo"` (token-level normalization over the group sum) — [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Unsloth exposes `loss_type='dr_grpo'` alongside `'grpo'`/`'gspo'`; suggests epsilon 0.2 / epsilon_high 0.28, num_generations 8–16 — [Unsloth RL guide](https://unsloth.ai/docs/basics/reinforcement-learning-rl-guide)
- Recent SQL-domain GRPO settings: Arctic-Text2SQL-R1: group size 16, KL β=0.001, clip ε=0.2, temperature 0.8 — [arXiv 2505.20315 (HTML v2)](https://arxiv.org/html/2505.20315v2). SQL-R1: rollout (group) 8, lr 3e-7, max response length 2048; β and ε present in objective but numeric values not stated — [arXiv 2504.08600 (HTML v5)](https://arxiv.org/html/2504.08600v5)

### Inferences
- Consensus starting point for this project: group size 8 (16 if VRAM allows), β=0 (also saves the reference-model memory on a single GPU; add a small β≈1e-3 only if drift/degeneration appears), clip 0.2 (upper 0.28), lr ~1e-6, `loss_type="dr_grpo"` + `scale_rewards=False` if following the Dr. GRPO recipe. TRL's 512-token default completion cap is too small for think-then-SQL outputs; SQL-R1's 2048 is the better anchor (Qwen3.5-2B's thinking-loop proneness argues for a hard cap plus `mask_truncated_completions=True`).
- Note the ecosystem has moved past vanilla GRPO: TRL's default is DAPO-style token-level normalization, which already addresses the length bias differently than Dr. GRPO; "dr_grpo vs dapo" is a cheap ablation.

### Gaps
- SQL-R1 does not publish its numeric KL coefficient.
- No systematic survey found of KL/group-size choices across 2026 small-model GRPO runs; the values above are from the three primary papers plus framework defaults.

## Key Question 7: Reward hacking in execution-reward SQL training

### Takeaway
The best-documented failure mode is **empty-result matching**: many gold queries in Spider/BIRD return empty results, so a trivially wrong query that also returns nothing scores as "correct" — both Arctic-Text2SQL-R1 and SQL-R1 responded by *filtering those training examples out*. Both projects also converge on simple rewards: Arctic argues complex shaped rewards invite "superficial reward hacking"; SQL-R1 found that underweighting execution/result components causes hacking and that length bonuses breed verbosity. Additional mitigations: small positive reward for merely-executable SQL (graded signal without rewarding degenerate output) and execution-time limits.

### Cited Findings
- Arctic-Text2SQL-R1 (Snowflake; ACL Findings 2026): reward = 1 if execution results exactly match ground truth (strict criteria), 0.1 if SQL is syntactically valid/executable but wrong, 0 otherwise — [arXiv 2505.20315 (HTML v2)](https://arxiv.org/html/2505.20315v2); overview: [arXiv abs 2505.20315](https://arxiv.org/abs/2505.20315), [Snowflake engineering blog](https://www.snowflake.com/en/engineering-blog/arctic-text2sql-r1-sql-generation-benchmark/)
- Arctic data filtering: removed training examples whose *reference* SQL returns empty results (~1,400 from BIRD, ~1,700 from Spider) and examples with >5s execution time; "many reference SQL queries in these datasets return empty results upon execution" — [arXiv 2505.20315 (HTML v2)](https://arxiv.org/html/2505.20315v2)
- Arctic on shaping: "complex reward engineering risks encouraging superficial reward hacking, as observed in RL for semantic parsing" — their motivation for the minimal 1/0.1/0 reward — [arXiv 2505.20315 (HTML v2)](https://arxiv.org/html/2505.20315v2)
- SQL-R1 (IDEA-FinAI; NeurIPS 2025): composite reward — Format ±1; Execution +2 (executes) / −2 (fails); Result +3 (correct by execution accuracy) / −3 (incorrect); Length bonus only when the result is correct; execution time limited "to discourage overly complex SQL"; GRPO, 7B/14B/32B Qwen2.5-Coder bases from OmniSQL SFT checkpoints — [arXiv 2504.08600 (HTML v5)](https://arxiv.org/html/2504.08600v5), [arXiv abs 2504.08600](https://arxiv.org/abs/2504.08600)
- SQL-R1 observed failure modes (Appendix B.5): low Execution/Result weights cause degradation "due to potential reward hacking," with the model exploiting simpler patterns; excessive Length rewards "may lead to verbose solutions"; finer-grained rewards led to "lazy" short-term reward chasing — [arXiv 2504.08600 (HTML v5)](https://arxiv.org/html/2504.08600v5)
- SQL-R1 filtering: also removed empty-result reference examples (and kept synthetic Gretel data only when a checker model could solve it at least 1-in-10) — [arXiv 2504.08600 (HTML v5)](https://arxiv.org/html/2504.08600v5)

### Inferences
- For this project, the abstention action *interacts* with the empty-result exploit: if abstaining earns ~0 and an empty-result match earns full reward, degenerate "SELECT nothing" queries dominate honest abstention on hard questions. The Arctic-style fix (drop or specially handle gold-empty examples; or require non-empty result sets to count as matches) should be applied before reward tuning.
- Recommended reward skeleton from the two papers: large terminal correctness reward (execution-result match), small graded term for executable-but-wrong (0.1-ish, not large enough to farm), strict format gate, execution timeout in the reward sandbox, and *no* length bonus initially.
- Execution timeouts double as a safety measure for the reward function's arbitrary-SQL execution (runaway queries stall GRPO rollout scoring — with 8–16 generations per prompt, one slow query blocks the whole group unless rewards are async, which TRL supports).

### Gaps
- No published incident report found of a model *actually exploiting* empty-result matching mid-training (both papers filtered preemptively); the exploit is documented as a data property plus a design argument, not a logged training collapse.
- No reward-hacking documentation found specific to *abstention + SQL execution* combined; this project would be covering new ground there.

## Stack comparison summary (synthesis)

| Requirement | TRL GRPOTrainer + vLLM colocate | verifiers (Prime Intellect) | Unsloth GRPO |
|---|---|---|---|
| Qwen3.5-2B | Yes via current transformers + vLLM (>=0.17; LoRA fix in 0.19) | UNVERIFIED for hybrid GDN arch | Notebook for Qwen3.5-4B, but `fast_inference=False` required (no vLLM rollouts) |
| LoRA on single GPU | Yes (`peft_config`; colocate default; sleep mode; β=0 skips ref model) | RLTrainer "supports LoRA" (mirror-sourced, UNVERIFIED) | Yes — strongest VRAM story (~5GB for ≤1.5B; QLoRA) |
| Arbitrary-Python SQL reward | Yes, sync or async | Yes (Rubric functions) | Yes |
| Dr. GRPO loss | `loss_type="dr_grpo"` + `scale_rewards=False` | UNVERIFIED | `loss_type='dr_grpo'` |
| Maturity/stability of API | High (v1.17.1, active, documented) | Heavy churn in 2026 (GRPOTrainer→RLTrainer→Taskset model) | High for its lane; vendor-benchmarked claims |

Working recommendation (inference, not a cited fact): TRL + vLLM colocate is the default choice for Qwen3.5-2B (full feature coverage, async rewards, Dr. GRPO, documented single-GPU knobs); Unsloth is the fallback if VRAM is the binding constraint — ideally paired with Gemma 4 E2B or SmolLM3-3B where its vLLM fast path works; verifiers only if the project pivots to multi-turn/tool-use environments or Prime Intellect hosted training.
