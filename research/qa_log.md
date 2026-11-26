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

1. **A table list (template) per topic,** 5 in all, each listing 9 to 11 tables that make sense for the topic, with every column's type and role (a category from a fixed list, a number in a range, a date in a range, a name, or a link). See [data/generator/templates/healthcare.yaml](../data/generator/templates/healthcare.yaml).
2. **A program that builds designs from table lists** ([data/generator/schemas.py](../data/generator/schemas.py)): it picks 3 to 8 linked tables and varies names ("doctors" or "physicians"), twice per topic, giving 10 designs. The same command always gives the same designs.
3. **Automatic checks** ([tests/test_schemas.py](../tests/test_schemas.py)) that every design has 3 to 8 tables, links that point at included tables, all 4 column types, and SQL that runs.

Reviewing the 10 designs turned up 2 problems, both fixed: designs with 3 or 4 tables almost never came out, because the program discarded the chosen size along with oversized starting tables, and some names mismatched, such as a table called "facilities" with a column called "center_name".

### If I wrote the table lists myself, what was the point of Gretel?

A small but real one: Gretel supplied the topics, not the data. Its role was 3 things.

1. **A list to choose from.** Its 100 business topics gave the 25 picks an outside starting point with breadth, instead of whatever topics came to mind first. Sorting all 100 also exposed the near-duplicates and in-between topics that were dropped.
2. **Hints for what each table list should hold.** Gretel's one-line descriptions shaped several table lists. Pharmaceuticals is described as "clinical trial outcomes, drug approval data, sales figures, R&D expenditures", and its table list has a table for each; logistics' "freight forwarding, warehouse management, route optimization, reverse logistics" became shipments, warehouses, routes, and returns.
3. **A citable source for the paper.** "Topics were drawn from the 100-domain taxonomy of gretelai/synthetic_text_to_sql (Apache 2.0); none of its examples were used" is a reproducible, defensible sentence; "I picked topics I liked" is not.

None of its questions, databases, or SQL were used, for the reasons in the Gretel entry above. The project could have run without it; the topic choice would just have been harder to justify.

### How do we verify that the answers to our questions are correct? (2026-11-25)

The answer to an answerable question is whatever its gold SQL returns on each filled copy, so the data itself cannot be "wrong". What can go wrong is the link between the English and the SQL: the question asks one thing and the SQL computes another, the question allows 2 readings, or the answer is so degenerate that a wrong query matches it. Three layers check this:

1. **Construction:** the SQL is written first, and the English is filled from the same template with the same values (the same category, cutoff, year), so they agree by design. A mismatch would be a template bug, repeated in every question that template makes.
2. **Mechanical checks on all 9,600:** every gold query runs on all 5 copies and returns rows; no lone 0 or NULL; no tie at a top-N cutoff; every condition changes the answer on at least 2 copies; the columns it reads are recorded exactly.
3. **Review:** an independent reviewer wrote down the expected SQL for 20 random questions before reading the gold, then compared: 0 broken.

20 is a small sample for a paper. Because errors come from templates, the stronger checks are a template audit (one example of every template, so every pattern gets read once) and a larger differential audit: an independent writer produces SQL from the English alone for about 200 random questions, the results are compared on all 5 copies, and every disagreement is read by hand. The agreement rate, with its error margin, then goes in the paper. This uses a model to audit data quality, not to score the model, so it does not break the rule against an LLM as judge. BIRD and TrustSQL, whose gold queries people wrote, add an outside reference.

### Should we test different prompts, and how do we know the prompt is good enough? (2026-11-25)

For the trained models, no: the prompt is held fixed on purpose. The experiment compares training methods (accuracy-only against abstention rewards at several penalties), and changing the prompt between them would mix 2 causes into one result. Reinforcement learning also adapts the model to whatever prompt it trains on, so an "optimal" prompt matters far less after training than before. The goal is a prompt that is good enough and identical everywhere, not the best possible one.

Where prompts do matter, and what to do:

1. **Before training (step 3.2's trial run):** measure how often the untrained model's output parses as the required JSON, with thinking on and off. A low parse rate is the signal to fix the wording once, then freeze it.
2. **For the prompted baselines (Phase 10):** untrained and frontier models rely entirely on the prompt, so a weak prompt would make them look unfairly bad. Try 2 or 3 wordings and keep the best, chosen on the dev split.
3. **A robustness check (Phase 11):** run the trained model on 2 reworded prompts at test time and show the results hold. It costs only inference, a few dollars.

Any prompt choice is made on the dev split, never on calibration or test data, so the guarantee and the reported numbers stay honest.

### What is the prompt format? (2026-11-25)

Every question reaches the model as 2 messages, built by `data/generator/prompt.py` and identical in training and testing:

1. **System message:** "write SQLite queries; if the question can't be answered from the schema, is ambiguous, or isn't a database question, abstain instead of guessing; you may reason briefly, then end with exactly one JSON object."
2. **User message:** the database layout as `CREATE TABLE` statements (minus any hidden column), then the question, then the required ending: either `{"sql": "SELECT ...", "abstain": false, "reason": ""}` or `{"sql": null, "abstain": true, "reason": "..."}`.

The thinking switch is passed separately and always set on purpose. An example prompt is about 460 tokens. The accuracy-only baseline uses the same format with the abstain option removed.

### Why write the answerable questions before the ones to decline? Isn't that biased? (2026-11-25)

The criteria came first; only the building came second. The plan fixed the 3 decline kinds (from TrustSQL's taxonomy), the 70/30 split, and the question shapes before any code ran, and the record format was written before either generator.

The building order follows one dependency: the main unanswerable kind is made from an answerable question by hiding the column it needs, so the answerable question must exist first. That is a guard against bias, not a source of it: the question to decline reads exactly like an answerable one, so the model cannot learn "this wording means decline" and must check the schema instead. The other kinds (a missing attribute, ambiguous, out of scope) are built from the schema alone.

The real risk is a different one: the templates for ambiguous and out-of-scope questions have their own style ("best", "recently", "email this list"), and a model could learn those surface cues instead of the skill. Two things check for that: the hidden-column pairs, whose wording matches exactly, and evaluation on TrustSQL, whose questions people wrote, not our templates. If the model only learned our templates, it would fail there.

### How many plan steps can be batched together? (2026-11-24)

Everything up to gate G1 can run as one batch, because none of it spends money or needs a decision only the user can make: 12 steps (1.3–1.7, 2.1–2.6, 3.1). Some are chained and some are independent, so the batch runs as parallel tracks:

- **Data, chained:** fill the tables (1.3), then write answerable questions from the filled data (1.4), then derive the questions to decline (1.5), then set the 70/30 mix (1.6).
- **Checker, independent of the data:** sandboxed execution (2.1), result comparison (2.2), and the multi-copy correctness check (2.3), with tests that try to fool it (2.5), built against tiny test databases.
- **Small independent pieces:** the prompt format (1.7) and assigning whole topics to train, dev, calibration, and test (3.1).
- **Joined at the end:** dropping questions whose gold answer is empty (2.4), measuring checker speed (2.6), then gate G1.

4 more steps from later phases are pure statistics code that needs no data and can join as a fourth track: Learn-then-Test and its tests (8.2, 8.6), the metrics (11.1), and the bootstrap (12.1).

What cannot be batched: the trial run on the untrained model (3.2) needs a cloud go-ahead; the training code (Phase 4) depends on decisions D1 and D2, which the trial run informs; and the smoke test (Phase 5) is the first real spending. The batch must also fix the shared formats first (how filled copies are stored, the question file, the checker's interface), so the parallel tracks fit together.

### What is the maximum we can scale to?

No number has a hard cap; each hits a practical ceiling.

- **Designs per topic:** capped by each table list. Counting every valid set of 3–8 linked tables, the 25 lists allow 1,288 distinct designs: a median of 34 per topic, but only 6 for fashion retail, 12 for aerospace, and 14 for beauty, whose tables chain too deeply. Past that, only names change. Adding tables to the shallow lists raises their ceilings.
- **Topics:** by hand, about 10 per family (50) is realistic, partly beyond Gretel's list. Beyond that means version 1 of the plan: an AI model, not from the Qwen family, designs the schemas, which scales to hundreds of topics for some API cost and review time.
- **Families:** the plan allows up to 6; the runner-up, media and entertainment, would make 6 and give Phase 9 30 pairs.
- **Questions:** a 300-step training run only draws 2,400 (4,000 at 500 steps), so a bigger training pool is partly unused. Calibration and test sets are capped by GPU time to run the trained models on them; a rough estimate is a few minutes per 1,000 questions per model, so even 10,000 across 8 models costs a few dollars. The smoke test measures the real speed.
- **Filled copies:** no cap, but each extra copy adds checking time to every answer during training.

The real limit on credibility is the number of independent topics, not the number of rows or questions.

### Can we raise these numbers to test on more samples and make the experiment more convincing?

Yes, and most of it is free. More data does not raise training cost: a run takes 300 steps of 8 questions whatever the pool size. Evaluation cost rises a little with more test questions, small next to training. But not every number adds real samples:

- **Filled copies** add none: they check the same questions on more data, making the checker stricter. Keep 5.
- **Designs per topic** are free to raise with one setting, adding databases and questions.
- **Topics** cost hand-written table lists (time, not money).

The catch: designs from one topic share its table list, so they are near-relatives. If a test design shares a topic with training designs, the test is easier than it looks. The honest split keeps whole topics out of training (step 3.1), which makes the number of topics the real sample size; 25 topics split 4 ways leave only about 5 for testing. So the biggest gain in credibility comes from more topics, then more designs. Proposed on 2026-11-24: 8 topics per family (40 topics, 15 new table lists, some from beyond Gretel's list), 6 designs per topic (240 designs), 5 copies kept, calibration and test sets raised toward 2,000 questions each, and splitting by topic in step 3.1. On the follow-up question of the maximum, the choice became the larger option: 60 topics and 8 designs per topic (480 designs), see the decision log.

### Why these numbers: 5 families, 5 topics, 2 designs, and so on?

Some come from the plan, some are judgment calls; each says which.

- **5 families (judgment, within the plan's 4–6):** Phase 9 calibrates on one family and tests on another, so 5 give 20 such pairs; 1 or 2 can be held out of training with 3 or 4 left to train on; and each family needs about 1,000 calibration questions of its own, so more families would spread the data thinner.
- **5 topics per family (judgment):** enough variety inside a family that it is not one topic's quirks, while capping the hand-written table lists at 25.
- **1 table list per topic (structure):** each topic has its own tables and vocabulary, written once and reused for every design.
- **9–11 tables per table list (judgment):** more than the 8-table maximum design, so 2 designs from one table list can pick different tables.
- **2 designs per topic (plan):** Phase 1's target is 50 designs, and 50 ÷ 25 topics = 2. This is a starting point, not a ceiling: step 3.1 splits whole designs into train, dev, calibration, and test sets, and 50 may prove too few, so the generator makes more with one setting.
- **3–8 tables per design (plan):** at least 3 so questions can join tables; at most 8 so the schema, which goes into every prompt, stays short and queries stay fast.
- **3–5 filled copies (plan; 5 recommended):** more copies make it harder for a wrong query to match by luck; each costs a little execution time per answer during training, and the tables are small, so 5 is cheap.
- **Tens to a few hundred rows per table (plan):** enough rows for filters and totals to mean something, few enough that checking stays fast inside training.
- **About 70% answerable, 30% to decline (plan):** with too many declines, a model can learn to decline everything ("abstention collapse"); Abstain-R1 used about the same split.
- **About 1,000 training questions, 500 minimum (plan, gate G1):** a 300-step run draws 8 questions per step, 2,400 draws, so each question is seen 2 or 3 times.

### How do I read the "from 5 families down to questions" diagram?

Read it top to bottom: each arrow means "each one of these leads to the next level". One path through it, using hotels:

1. **Family:** retail and consumer, one of the 5 big groups.
2. **Topic:** hospitality (hotels), one of that family's 5 topics.
3. **Table list:** the hotel table list, listing 10 tables a hotel database could have.
4. **Design:** an empty hotel database that picks 6 of those tables. It has column headers but no rows yet. Each topic gets 2 designs.
5. **Filled copies (step 1.3):** the same design filled with made-up rows, 3 to 5 times, each with different guests and bookings.
6. **Questions (steps 1.4–1.6):** written for each design. About 70% can be answered from the tables, such as how many bookings were paid by credit card; about 30% should be declined, such as which guests will book again next year, which the data cannot know.

The last 2 levels both hang off a design: its questions are checked against all of its filled copies. Green marks finished steps, orange the next one.

### What exactly is a "table list" for a topic? (first called a "menu")

The nickname "menu" was renamed "table list" on 2026-11-24 because it confused more than it helped; older answers in this log have been updated to match.

A list of every table that makes sense for one topic, with each table's columns and the kind of value each column holds. "1 table list per topic" just means every topic gets its own list, because different topics need different tables: a hotel database needs rooms, guests, and bookings, while a bank database needs accounts, cards, and transactions, so the hotel list is useless for a bank. 25 topics, so 25 lists. Like a restaurant menu, it is not used whole: each database design orders 3 to 8 tables from it. The hospitality table list offers 10 tables (hotels, room types, rooms, guests, bookings, payments, reviews, staff, services, service charges); design `retail_and_consumer__hospitality__01` picked 6 of them (hotels, guests, room types, rooms, bookings, payments, with hotels renamed "properties"). Within a table, the table list says what fills each column: rooms have a room number from 100 to 2400, a floor from 1 to 40, and an accessible yes/no flag. It also lists synonyms, so a table can appear as "bookings" in one design and "reservations" in another. The proper term is a template; the 25 table lists live in `data/generator/templates/`, one file per family.

### What are the 3 candidate models, and why do we need 3?

- **Qwen3.5-2B, the main model.** The original brief named it: the project trains a model of 4B parameters or fewer, because the budget is $30–50 on one GPU, small models are the ones that err most, and a small model also runs on the laptop for the latency test (step 11.3). It is also the riskiest for tooling: it accepts images, mixes 2 kinds of layers, needed a pre-release vLLM when it launched, and its own page warns the 2B size can reason in loops.
- **SmolLM3-3B, the backup and the comparison model.** A conventional text-only model that works in every training tool. If Qwen3.5-2B fails the first paid test run (step 5), the project switches to SmolLM3-3B the same day (decision D5). Otherwise it serves as the control: one run on a different model family shows the result is not a quirk of Qwen.
- **Gemma 4 E2B, the spare.** The brief allowed it as the control too, but it loads 5.1 billion parameters into memory despite the "2B" in its name, so SmolLM3-3B is preferred. Gemma 4 E2B becomes the control only if SmolLM3-3B has to become the main model.

All 3 are Apache 2.0 and load under the pinned versions (step 0.3). See [results/models.md](../results/models.md).

### What has been done so far, and why? (2026-11-24)

A milestone recap at the end of step 1.2; each item has its own entry above or below.

- **Phase 0, setup.** The repo layout (every phase has a home), exact software versions (runs stay comparable, laptop and GPU identical), a check that all 3 candidate models load (incompatibility found for free, before any GPU spend), both outside benchmarks downloaded with pinned fingerprints (standard tests the paper compares against), a Modal account with a $30 cap (the laptop cannot train), and the spend and decision logs (every dollar and choice traceable in the write-up).
- **Why build our own data at all:** no existing dataset has "should decline" questions for training, fresh databases rule out contamination (models cannot have memorized them), and we control everything, including filling each database several times so the checker can catch lucky matches.
- **Step 1.1:** 5 clearly different topic families of 5 topics each, because Phase 9 tests whether the error guarantee survives a change of topic.
- **Step 1.2:** a table list per topic and 50 empty database designs built from them, repeatable from one command, with every column's role recorded for the steps that fill data and write questions.
- **Cleanup:** the benchmark numbers now come from a saved script, so the paper's numbers can be rebuilt; rerunning it caught an addition error (5,538 gold queries, not 7,008).

### Can this be published at the end?

Yes, and the plan already ends in a release (step 13.4): code on GitHub, the trained adapter on Hugging Face, and a write-up. The publishing routes, from easiest to hardest:

- **arXiv preprint:** free and fast, with no peer review. A first-time submitter may need an endorsement from an existing arXiv author.
- **A workshop paper:** the realistic peer-reviewed target. ML and NLP conferences run workshops on trustworthy AI, tables and databases, and student research (ACL's Student Research Workshop is built for this). Workshops accept smaller-scale studies than the main conferences.
- **A main conference (ACL, EMNLP, NeurIPS):** possible but harder. One 2B model, 8 runs, and 2 seeds read as a pilot; reviewers there would want more models, seeds, and baselines.

What makes it publishable: the research found no paper combining RL training, an explicit abstention reward, and risk–coverage evaluation in text-to-SQL, and the guarantee plus the topic-change experiment are new on top of that. Gate G5 re-runs that search before writing, in case someone publishes first. Honest negative results also count; a guarantee that breaks under topic change is itself a finding.

The paperwork is light: every dataset and model used allows research use and publication with attribution. BIRD's share-alike license only matters for redistributing modified BIRD data (decision D7).

### What are the "other 20 templates"?

Every topic needs its own template (table list), and there are 25 topics: 5 families of 5. The healthcare checkpoint built the first 5. The remaining 20 cover the other 4 families: finance (financial services, finance, insurance, venture capital, blockchain), logistics and transport (logistics, ocean shipping, public transportation, aerospace, automotive), retail and consumer (retail, fashion retail, beauty industry, restaurant operations, hospitality), and energy and natural resources (energy, oil and gas, mining operations, water resources, forestry). Each topic belongs to exactly one family, so that is one table list per topic and 5 per family, stored as one file per family (`data/generator/templates/<family>.yaml`). The program and its checks are already done and work for any family; only the table lists are missing. With all 25, the program writes all 50 designs.
