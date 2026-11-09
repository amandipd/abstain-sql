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

### Modal only for now; RunPod later (2026-11-08)

- **Choice:** open only a Modal account in Phase 0, with a $30 monthly Workspace budget. Open RunPod when the training runs need it.
- **Why:** Modal's $30 monthly credit may pay for the early GPU work outright, and a second account adds nothing until the 8-run matrix. RunPod stays the backup, and the cheaper option if the RTX 4090 fits the model.

### Modal's Starter credit very likely covers GPU compute (2026-11-08)

- **Finding:** the dashboard's Overview page says "Starter plan with $30.00 included compute credits per month", and the $30.00 credit and the $30 workspace usage limit draw from the same pool, both at $0.00 used. GPU time is compute, so the credit very likely covers it, but no page states that for the Starter credit.
- **Status:** UNVERIFIED until the first GPU job's bill shows $0 in charges.
- **Correction:** an earlier version of this entry cited the Credits tab's sentence "These credits can be used towards GPU compute" as direct confirmation. That sentence describes a separate program, free credits that startups and academics apply for, not the Starter credit.
- **Why it matters:** if it holds, the smoke test (Phase 5) and early runs cost $0 out of pocket, up to $30 of credit per month.

### Schema families and topics (2026-11-08)

- **Choice:** 5 families of 5 topics each, 25 topics in all, saved in `data/generator/families.yaml`: healthcare; finance; logistics and transport; retail and consumer; energy and natural resources. Topic names come from Gretel's 100-topic list ([gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql), Apache 2.0), used as a variety checklist only.
- **Why 5 × 5:** 5 families give the Phase 9 shift experiment 20 ordered calibrate-on-A, test-on-B pairs, and leave 3 or 4 families to train on if 1 or 2 are held out. 5 topics per family give each family variety inside it, while capping step 1.2 at about 25 hand-built templates.
- **Why these families:** each differs from the others in vocabulary and table shapes, has at least 5 Gretel topics, supports every query shape in step 1.4, and supports the decline categories in step 1.5. Energy and natural resources resembles no external test database, which makes it the natural candidate to hold out. Media and entertainment, and public sector and education, were the runners-up.
- **Why these topics:** each family's topics cover different sub-areas. 18 topics were dropped as near-duplicates or as straddling 2 families; the file lists each with its reason.

### Schemas from hand-written templates, version 0 (2026-11-08)

- **Choice:** generate schemas from 25 hand-written templates, one per topic, rather than having an LLM design them (the plan's version 1).
- **Why:** templates are free, deterministic, and contamination-free, and they give every column a role (category, number, date, name, link) that steps 1.3 to 1.5 need for filling data, writing questions, and removing columns. Variety comes from sampling 3 to 8 of each topic's 9 to 11 tables and from synonyms for table and column names; the 2 schemas per topic always use different table sets.
- **Checks:** reviewing the first 10 schemas turned up a sampling bias against 3- and 4-table schemas and 2 mismatched name pairs, all fixed. A scripted scan of all 50 schemas for column names that match a table's unused synonym flagged 40 spots, nearly all normal usage ("appointments.visit_date"); 1 odd case was fixed.
- **Revisit:** move to version 1 if the model's zero-shot results in step 3.2 suggest these schemas are too easy or too uniform.

### Keep TrustSQL's hospital-records portion (2026-11-04)

- **Choice:** evaluate on all 4 TrustSQL sources, EHRSQL included.
- **Why:** EHRSQL runs on a 94-patient MIMIC-IV demo database, which is open access, so no PhysioNet application is needed (`results/external_data.md`).
