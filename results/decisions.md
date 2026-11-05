# Decision log

Every choice the write-up must justify goes here, dated, with the reason. The 8 open decisions come from the table in [plan.md](../plan.md#open-decisions). Each has a default, so no step waits on one; when a decision is settled, its entry gains the date, the choice, and why.

## Open decisions

### D1: Reward for declining a question that should be declined

- **Default:** +1, so the training reward equals TrustSQL's Reliability Score, RS(c), question by question. The alternative is the original brief's literal 0.
- **Settled at:** step 4.1.
- **Status:** open.

### D2: Reasoning before the JSON answer, or the JSON answer only

- **Default:** allow reasoning, capped by the length limit, and read the last JSON object.
- **Settled at:** step 4.1.
- **Evidence to gather first:** step 3.2's trial run, with thinking on and off. For Qwen3.5-2B, allowing reasoning means turning thinking on, which is where its looping risk lives.
- **Status:** open.

### D3: Which confidence signal decides when to answer

- **Default:** whichever candidate gives the lowest AURC on the dev split. AURC is the area under the risk–coverage curve, one number for the whole trade-off between answering more and erring less; lower is better. The candidates are the answer's log-probability (how likely the model found its own answer), summed or divided by length, and agreement across 5 sampled answers.
- **Settled at:** step 8.1.
- **Status:** open.

### D4: Whether to give the model BIRD's hints

- **Default:** match whatever setting the published small-model scores used.
- **Settled at:** step 11.2.
- **Note:** the BIRD team says it will remove these hints in favor of an interactive track (see `results/external_data.md`).
- **Status:** open.

### D5: Qwen3.5-2B or SmolLM3-3B as the main model

- **Default:** Qwen3.5-2B, if the smoke test passes cleanly.
- **Settled at:** step 5.5.
- **Status:** open.

### D6: Training run length, 300 or 500 steps

- **Default:** 300; 500 if the reward is still flat at step 300.
- **Settled at:** step 6.3.
- **Status:** open.

### D7: License for the synthetic data

- **Default:** a permissive license, unless Spider or BIRD content forces CC BY-SA's share-alike terms.
- **Settled at:** step 13.4.
- **Status:** open.

### D8: Which BIRD dev version to report

- **Default:** both. The original allows comparison with published scores; the 2025-11-06 cleaned version, which rewrites 451 of 1,534 gold queries, is the more accurate number.
- **Settled at:** step 11.2.
- **Status:** open.

## Settled so far

These were settled during Phase 0 and sit outside the plan's table.

### Exact versions for the training stack (2026-11-04)

- **Choice:** TRL 1.17.1, vLLM 0.32.0, torch 2.13.0, transformers 5.19.0, PEFT 0.21.2, with every other package pinned in `uv.lock`.
- **Why:** vLLM 0.32.0 is the newest release TRL 1.17.1 accepts, and it fixes torch and caps transformers. One lock for the laptop and the GPU keeps runs comparable.

### Set the thinking switch explicitly (2026-11-04)

- **Choice:** every model's thinking mode is set on purpose, the same way in training and evaluation, never left at its default.
- **Why:** the defaults disagree. SmolLM3-3B thinks by default; Qwen3.5-2B and Gemma 4 E2B do not (`results/models.md`).

### Keep TrustSQL's hospital-records portion (2026-11-04)

- **Choice:** evaluate on all 4 TrustSQL sources, EHRSQL included.
- **Why:** EHRSQL runs on a 94-patient MIMIC-IV demo database, which is open access, so no PhysioNet application is needed (`results/external_data.md`).
