# Q&A log

Every conceptual question I asked while building this project, with the answer, in the order I asked. I keep it so that when I write the paper, I can trace why the project looks the way it does. Answers are condensed; the linked files hold the full detail. Procedural requests ("commit and push") are left out.

## 2026-11-04: Phase 0, setup

### What is this project, in one sentence?

It trains a small open model to turn plain-English questions into database queries (text-to-SQL), rewards it for saying "I don't know" instead of guessing, and then certifies the error rate on the questions it does answer. The certificate comes from calibration: a confidence cutoff tuned on held-out data so that a stated error ceiling provably holds. See [README.md](../README.md).

### What did scaffolding the repo involve, and why pin exact versions?

The repo got every folder [plan.md](../plan.md) lists; each code file is a stub naming the plan step that fills it. Every dependency is pinned in `pyproject.toml` and `uv.lock`, 214 packages in all, so the laptop and the cloud GPU run identical versions. A silent upgrade mid-project would make training runs incomparable. The pins chain together: TRL 1.17.1 (the training library) accepts vLLM (the text-generation engine) only up to 0.32.0, which fixes torch at 2.13.0 and caps transformers below 5.20 and numpy below 2.4. A dry run of the Linux install, which resolves every package without downloading any, found all 209 GPU-side packages.

### What are thinking controls, and what do they do?

They decide whether a model writes out its reasoning before answering. They are not a setting inside the model: they are text added to the prompt by the chat template, the recipe that turns a conversation into the exact text the model reads.

- **Qwen3.5-2B:** an open `<think>` tag means reason first; an empty, closed `<think></think>` means answer directly. Off by default.
- **SmolLM3-3B:** a `/think` or `/no_think` flag in a built-in system prompt. On by default, with long instructions pushing toward lengthy reasoning.
- **Gemma 4 E2B:** a `<|think|>` marker. Off by default.

It matters for 3 reasons. Cost: reasoning can add hundreds of tokens (word pieces) to every answer, and each training step writes 8 answers per question. Loops: Qwen3.5-2B's own page warns the 2B size can reason in circles until it hits the length cap, and those answers are thrown out of training. Fair comparison: the defaults disagree, so leaving them alone would compare models under different rules. The rule I adopted: set the switch explicitly for every model, and run step 3.2's trial both ways to give decision D2 evidence. The switch is a strong habit, not a lock. See [results/models.md](../results/models.md).

### Did step 0.3 download the model's parameters?

No. Parameters, also called weights, are the learned numbers: about 2 billion for Qwen3.5-2B, roughly 4 GB. Step 0.3 needed only to check compatibility, so it downloaded each model's config (a small file describing its shape), tokenizer (the part that splits text into tokens), chat template, and license: about 60 MB for all 3 models. Weights first download in step 3.2.

### What are the BIRD and TrustSQL benchmarks?

A benchmark is a standard test set that many groups score their models on, so results compare. The 2 test different skills.

- **BIRD** tests whether a model writes correct SQL. Its dev set has 1,534 questions over 11 large, messy databases, from 0.2 MB to 598 MB, with a hint ("evidence") for 90% of questions. It is scored by execution accuracy: generated SQL counts as correct only if running it returns the same result as the gold (known-correct) query. Published small models score about 54–67%. Every question is answerable, so BIRD cannot test declining.
- **TrustSQL** tests whether a model knows when not to answer. Its test set has 4,940 questions, half of which should be declined, built from 4 older datasets: flights (ATIS), university courses (Advising), hospital records (EHRSQL), and mixed topics (Spider). It is scored by the Reliability Score, RS(c): +1 for a right answer or a right decline, 0 for declining an answerable question, minus c for a wrong answer. A model that declines everything scores 50, the floor to beat.

Neither trains our model. BIRD shows it still writes good SQL on hard, realistic data; TrustSQL shows it declines the right questions. See [results/external_data.md](../results/external_data.md).

### What does the TrustSQL breakdown in external_data.md show?

- The test set runs on 13 databases: 1 each for flights, courses, and hospital records, and 10 small Spider databases. No Spider test database appears in TrustSQL's training data, so that portion tests unseen databases.
- Splits: 13,958 training and 2,460 validation questions, all answerable; 4,940 test questions, half to decline. We use only the test set.
- Answerable test questions: 526 easy, 828 medium, 1,116 hard.
- "Seen SQL" means a question's query shape also appears in TrustSQL's training split. We never train on TrustSQL, so for our model every question is unseen; the label matters only when comparing with published scores.
- Declines come in 5 kinds: 3 about a missing column (1,465 questions), questions needing outside knowledge (505), and requests SQL cannot do (500).
- All 2,470 answerable gold queries run. Hospital queries must replace `current_time` with the fixed date 2100-12-31 23:59:00, as TrustSQL's own scorer does, or 85 gold answers come back empty. Evaluation needs TrustSQL's 60-second limit per query, not the training reward's 2 to 5 seconds.
- TrustSQL has nothing like our plan's "ambiguous" category, so only our own test set measures that.

The hospital data needed no PhysioNet application: it runs on a 94-patient sample of the MIMIC-IV demo database, which is open access.

### What are the spend log and the decision log?

- **Spend log** ([results/spend.md](../results/spend.md)): one line per cloud session or API batch, with date, platform and GPU, minutes, dollars, and purpose, against the $30–50 budget. It also holds the plan's cost estimates until the smoke test replaces them with measurements.
- **Decision log** ([results/decisions.md](../results/decisions.md)): the 8 open decisions (D1–D8), each with a default so no step stalls, plus every choice already made, dated, with the reason. The write-up has to justify every choice, and dating them as they happen beats reconstructing them later.

## 2026-11-08: Phase 0, cloud accounts

### Where do we rent GPUs, and why open both Modal and RunPod?

Modal is first: its Starter plan includes $30 of compute credit a month, which may pay for the early runs. RunPod is cheaper by the hour (an A100 for $1.19–1.79, an RTX 4090 for $0.34) and a backup if Modal's credit or availability falls short. We may not need both, so I opened only Modal; RunPod waits until the training runs need it.

### What is the difference between Modal's usage limit and spend limit?

The usage limit (workspace budget) caps total usage before credits apply: at $30, Modal stops every job at $30 of usage a month. The spend limit caps what I pay out of pocket after credits. I set a $30 usage limit and no spend limit. The dashboard calls the $30 credit "included compute credits", so it very likely covers GPU time, but no page says so for this credit; it stays unverified until the first GPU job's bill shows $0 in charges. An earlier version of the logs claimed direct confirmation by quoting a sentence that actually describes a separate program, credits that startups and academics apply for; I corrected it.

### Where do the phases live in the codebase?

[plan.md](../plan.md) defines all 14 phases, each with numbered, checkboxed steps. The code folders map onto phases: `data/generator/` for Phase 1 (synthetic data), `checker/` for Phase 2 (the answer checker), `data/splits/` for Phase 3, `train/` and `infra/` for Phases 4–7, `calib/` for Phases 8–9, `eval/` for Phases 10–12, and `results/` for the write-up. Files for future phases are stubs.

## 2026-11-08: Phase 1, synthetic data

### Why 5 schema families and 5 topics per family?

Both are judgment calls, not calculations. A family is a group of related topics; Phase 9 calibrates on one family and tests on another to see whether the error guarantee survives a change of topic.

- **5 families:** the plan allows 4 to 6. 5 families give 20 ordered calibrate-on-A, test-on-B pairs (4 give 12, 6 give 30). If 1 or 2 families are held out of training to make the topic change real, 3 or 4 remain to train on. Each family also needs about 1,000 calibration questions of its own, and more families spread the databases thinner.
- **5 topics per family:** each topic needs a hand-built template in step 1.2, so 25 templates is the real budget. 5 topics give each family variety inside it, and fit Phase 1's target of 50 databases: 10 per family, 2 per topic.

### How were the families and topics chosen?

In 2 passes. Families first: each had to differ from the others in vocabulary and table shapes, have at least 5 topics in Gretel's list, support every question shape step 1.4 needs (filters, group totals, joins, top-N, nested queries, date ranges), and support "should decline" questions such as "who are our best customers?". The 5 chosen: healthcare; finance; logistics and transport; retail and consumer; energy and natural resources. Energy resembles no external test database, which makes it the natural family to hold out. Media and entertainment, and public sector and education, were the runners-up.

Then topics: every Gretel topic was sorted into a family or left unused; topics straddling 2 families (health insurance, food industry) were dropped; near-duplicates (fashion and ethical fashion) kept only one; picks were spread across sub-areas. 18 dropped topics are listed with reasons in [data/generator/families.yaml](../data/generator/families.yaml).

### What is Gretel, and how is its dataset organized?

Gretel (Gretel.ai) is a company that makes synthetic data, meaning data generated by AI. Its dataset [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) (Apache 2.0, April 2024) holds 105,851 text-to-SQL examples, 100,000 for training and 5,851 for testing, across 100 topics.

- **Each row is one example,** not one topic: a question (`sql_prompt`), a tiny database (`sql_context`), the SQL answer (`sql`), and an explanation.
- **The `domain` column is the topic,** and `domain_description` explains it. Each topic covers about 1,000 rows, shuffled among the others.
- **We use only the 100 topic names,** as a checklist for variety. Its databases are too small for our checker (often 1 table with a few rows), it has no questions to decline, and its wide use means models may have seen its examples in training.

### What did step 1.2 build, and what is a "design"?

A design, properly a schema, is the layout of a database with no data in it: which tables exist, which columns each has, and how the tables link. It is the column headers of a spreadsheet with no rows filled in. Step 1.3 fills each design 3 to 5 times with different made-up rows, turning it into real databases.

For healthcare, step 1.2 produced:

1. **A menu (template) per topic,** 5 in all, each listing 9 to 11 tables that make sense for the topic, with every column's type and role (a category from a fixed list, a number in a range, a date in a range, a name, or a link). See [data/generator/templates/healthcare.yaml](../data/generator/templates/healthcare.yaml).
2. **A program that builds designs from menus** ([data/generator/schemas.py](../data/generator/schemas.py)): it picks 3 to 8 linked tables and varies names ("doctors" or "physicians"), twice per topic, giving 10 designs. The same command always gives the same designs.
3. **Automatic checks** ([tests/test_schemas.py](../tests/test_schemas.py)) that every design has 3 to 8 tables, links that point at included tables, all 4 column types, and SQL that runs.

Reviewing the 10 designs turned up 2 problems, both fixed: designs with 3 or 4 tables almost never came out, because the program discarded the chosen size along with oversized starting tables, and some names mismatched, such as a table called "facilities" with a column called "center_name".

### If I wrote the menus myself, what was the point of Gretel?

A small but real one: Gretel supplied the topics, not the data. Its role was 3 things.

1. **A list to choose from.** Its 100 business topics gave the 25 picks an outside starting point with breadth, instead of whatever topics came to mind first. Sorting all 100 also exposed the near-duplicates and in-between topics that were dropped.
2. **Hints for what each menu should hold.** Gretel's one-line descriptions shaped several menus. Pharmaceuticals is described as "clinical trial outcomes, drug approval data, sales figures, R&D expenditures", and its menu has a table for each; logistics' "freight forwarding, warehouse management, route optimization, reverse logistics" became shipments, warehouses, routes, and returns.
3. **A citable source for the paper.** "Topics were drawn from the 100-domain taxonomy of gretelai/synthetic_text_to_sql (Apache 2.0); none of its examples were used" is a reproducible, defensible sentence; "I picked topics I liked" is not.

None of its questions, databases, or SQL were used, for the reasons in the Gretel entry above. The project could have run without it; the topic choice would just have been harder to justify.

### What are the "other 20 templates"?

Every topic needs its own template (menu of tables), and there are 25 topics: 5 families of 5. The healthcare checkpoint built the first 5. The remaining 20 cover the other 4 families: finance (financial services, finance, insurance, venture capital, blockchain), logistics and transport (logistics, ocean shipping, public transportation, aerospace, automotive), retail and consumer (retail, fashion retail, beauty industry, restaurant operations, hospitality), and energy and natural resources (energy, oil and gas, mining operations, water resources, forestry). Each topic belongs to exactly one family, so that is one menu per topic and 5 per family, stored as one file per family (`data/generator/templates/<family>.yaml`). The program and its checks are already done and work for any family; only the menus are missing. With all 25, the program writes all 50 designs.
