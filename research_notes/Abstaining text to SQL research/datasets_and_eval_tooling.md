# Text-to-SQL Datasets and Evaluation Tooling (survey as of 2026-11-04)

Scope: BIRD, Spider 1.0, Spider 2.0, TrustSQL, EHRSQL — size/content, exact license, unanswerable/ambiguous question coverage, contamination evidence for Qwen and Gemma; plus synthetic SQL data generation prior art and execution-based evaluation tooling. All claims below were verified by fetching the cited page on 2026-11-04 unless marked UNVERIFIED.

## Key question 1: Dataset profiles and exact licenses

### Takeaway
All five datasets are under permissive-ish Creative Commons or OSS licenses: Spider 1.0 and BIRD are CC BY-SA 4.0 (share-alike), TrustSQL and EHRSQL are CC BY 4.0, and the Spider 2.0 repo is MIT. None prohibits training; the share-alike (BY-SA) clause on Spider/BIRD is the only redistribution consideration.

### Cited Findings

**BIRD (BIRD-SQL / BIRD-bench)**
- Size: "over 12,751 unique question-SQL pairs", 95 databases totaling 33.4 GB, 37+ professional domains (blockchain, hockey, healthcare, education, ...) — [bird-bench.github.io](https://bird-bench.github.io/)
- Splits: train has 9,428 examples (a news item mentions a filtered clean subset of 6,601 of 9,428, ~70%); dev has 1,534 examples (per 2025-05-25 news item); test size not stated on the page; a 500-example "Mini-Dev" subset exists — [bird-bench.github.io](https://bird-bench.github.io/)
- License: the site states "we modify the license of our data to CC BY-SA 4.0" — i.e., Creative Commons Attribution-ShareAlike 4.0 — [bird-bench.github.io](https://bird-bench.github.io/). (The phrase "we modify" implies an earlier different license; what it was before is UNVERIFIED — see Gaps.)
- BIRD questions come with "evidence" (external knowledge hints); a Nov 13, 2025 news item says Oracle Evidence will be removed and a new interactive track opened "to better resolve ambiguity" — [bird-bench.github.io](https://bird-bench.github.io/)
- Test submissions are by email per a submission guideline (test set is hidden) — [bird-bench.github.io](https://bird-bench.github.io/)

**Spider 1.0**
- Size: 10,181 questions, 5,693 unique complex SQL queries, 200 databases, 138 domains; split into train/dev/test (split sizes not given on the official page) — [yale-lily.github.io/spider](https://yale-lily.github.io/spider)
- License: "distributed under the CC BY-SA 4.0 license" (Creative Commons Attribution-ShareAlike 4.0) — [yale-lily.github.io/spider](https://yale-lily.github.io/spider)
- Official metrics: exact set match without values, execution accuracy with values, and test-suite accuracy (named as the official metric for Spider, SParC, CoSQL) — [yale-lily.github.io/spider](https://yale-lily.github.io/spider)
- Train/dev/test databases are disjoint by design (tests generalization to unseen schemas) — noted in [arXiv 2608.15389](https://arxiv.org/pdf/2608.15389)

**Spider 2.0**
- Variants and sizes (as of fetch): Spider 2.0-Snow — 547 examples, all on Snowflake (free, queued); Spider 2.0-Lite — 547 examples across BigQuery (214), Snowflake (198), SQLite (135); Spider 2.0-DBT — 68 code-agent tasks on DuckDB/DBT ("repository-level text-to-SQL"). The original Spider2 setting was removed on 2025-05-22 — [github.com/xlang-ai/Spider2](https://github.com/xlang-ai/Spider2)
- License: the repo displays "MIT license" — [github.com/xlang-ai/Spider2](https://github.com/xlang-ai/Spider2). Whether the dataset files (vs. code) carry a separate license was not visible on the README; UNVERIFIED (see Gaps).
- Evaluation: only partial gold answers and some gold SQL are released; most scoring is via an official submission process; gold ground-truth tables are provided for Lite/Snow but results using them must be labeled "oracle tables"; the evaluation suite was fixed 2025-10-29 and the leaderboard refreshed — [github.com/xlang-ai/Spider2](https://github.com/xlang-ai/Spider2)
- Paper: Lei et al., "Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows," [arXiv:2411.07763](https://arxiv.org/abs/2411.07763), ICLR 2025 (Oral) — [github.com/xlang-ai/Spider2](https://github.com/xlang-ai/Spider2)
- On 2025-07-13 spider2-snow.jsonl was updated "to resolve ambiguities" (prior version kept as spider2-snow-0713.jsonl) — [github.com/xlang-ai/Spider2](https://github.com/xlang-ai/Spider2)

**TrustSQL**
- Built from four source datasets: ATIS, Advising, EHRSQL, Spider — [github.com/glee4810/TrustSQL](https://github.com/glee4810/TrustSQL)
- Size: the paper reports 21,356 questions total: 18,888 answerable + 2,468 unanswerable (per-source breakdown in Key question 2) — [arXiv 2403.15879 (HTML v2)](https://arxiv.org/html/2403.15879v2). Note: the GitHub split table sums to ~23,318 (13,958 train / 4,420 valid / 4,940 test) — [github.com/glee4810/TrustSQL](https://github.com/glee4810/TrustSQL). The two totals differ; the discrepancy is unresolved (see Gaps).
- License: CC-BY-4.0 (Creative Commons Attribution 4.0) — [github.com/glee4810/TrustSQL](https://github.com/glee4810/TrustSQL)
- Paper: Lee, Chay, Cho, Choi (2024), "TrustSQL: Benchmarking Text-to-SQL Reliability with Penalty-Based Scoring," [arXiv:2403.15879](https://arxiv.org/abs/2403.15879)

**EHRSQL**
- Content: questions collected from 222 hospital staff (physicians, nurses, insurance review, health records teams); seed questions from a university-hospital poll, templatized and manually linked to two open EHR databases, MIMIC-III and eICU — [github.com/glee4810/EHRSQL](https://github.com/glee4810/EHRSQL); [arXiv:2301.07695](https://arxiv.org/abs/2301.07695)
- Size: ~24,411 question-SQL pairs total across both databases; per database roughly 9.3K train / 1.1K valid / 1.8K test; ~21K released (train+valid), ~3K held back as hidden test — [ar5iv HTML of arXiv 2301.07695](https://ar5iv.labs.arxiv.org/html/2301.07695)
- Databases shipped as preprocessed SQLite files (MIMIC-III 95 MB, eICU 102 MB); building from source requires PhysioNet credentialed access to MIMIC-III v1.4 and eICU-CRD v2.0 — [github.com/glee4810/EHRSQL](https://github.com/glee4810/EHRSQL)
- License: CC-BY-4.0 listed on the repo — [github.com/glee4810/EHRSQL](https://github.com/glee4810/EHRSQL)
- Paper: Lee et al., "EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records," NeurIPS 2022 Datasets and Benchmarks, [arXiv:2301.07695](https://arxiv.org/abs/2301.07695); [NeurIPS proceedings page](https://proceedings.neurips.cc/paper_files/paper/2022/hash/643e347250cf9289e5a2a6c1ed5ee42e-Abstract.html)

### Inferences
- No license among these forbids model training. CC BY-SA 4.0 (Spider 1.0, BIRD) requires attribution and that redistributed adaptations of the *dataset* carry the same license; whether a model trained on it (or its generated SQL) counts as an "adaptation" is legally unsettled — plan to attribute and, if redistributing modified dataset files, keep them BY-SA. CC BY 4.0 (TrustSQL, EHRSQL) only requires attribution.
- EHRSQL's question text/SQL are CC BY 4.0, but the underlying MIMIC/eICU patient data is governed by PhysioNet data use agreements; redistribution of the databases themselves should follow PhysioNet terms even though the repo links preprocessed copies.
- TrustSQL redistributes modified Spider content; Spider is CC BY-SA, while TrustSQL declares CC BY — a potential license-stacking inconsistency to be aware of if re-redistributing (my observation, not from a source).

### Gaps
- BIRD's license *before* the change to CC BY-SA 4.0 is not stated on the current page (UNVERIFIED; commonly said to have been CC BY-NC, but I found no primary source in this pass).
- BIRD hidden test set size: not published on the main page.
- Spider 1.0 official train/dev/test split sizes are not on the Yale page (commonly cited as ~8,659 train / 1,034 dev; UNVERIFIED here — do not quote without checking the data files).
- Whether Spider 2.0's MIT license covers the dataset JSONL files or just the code: not resolvable from the README alone.
- TrustSQL total-count discrepancy (21,356 in paper vs. ~23,318 summed from GitHub split table) unresolved; possibly versioning or counting differences.

## Key question 2: Which datasets contain unanswerable or ambiguous questions, and in what fraction

### Takeaway
Only TrustSQL and EHRSQL deliberately include unanswerable questions: TrustSQL has 2,468 of 21,356 (~11.6%, test-sets only) with a five-type taxonomy, and EHRSQL makes unanswerable questions 33% of its validation and test splits (~1.9K of ~24.4K overall, none in training). BIRD, Spider 1.0, and Spider 2.0 contain no labeled unanswerable questions, though both BIRD and Spider 2.0 have publicly acknowledged ambiguity problems.

### Cited Findings

**TrustSQL**
- 18,888 answerable + 2,468 unanswerable = 21,356 questions; unanswerable questions appear only in the test sets — [arXiv 2403.15879 HTML](https://arxiv.org/html/2403.15879v2)
- Per-source answerable/unanswerable: Spider 8,034/525 (~6.1% infeasible); ATIS 2,079/476 (~18.6%); Advising 2,236/533 (~19.3%); EHRSQL 6,539/934 (~12.5%); overall ~11.6%. (Percentages are computed from the paper's tables, not stated by the paper.) — [arXiv 2403.15879 HTML](https://arxiv.org/html/2403.15879v2)
- Five infeasible-question types, manually annotated: column-surface (hypothetical column resembling an existing one), column-related (hypothetical column that plausibly fits the schema), column-unrelated (column sampled from other databases), non-sql (needs more than SQL, e.g. "draw a chart"), out-of-domain (sampled from MS MARCO, CoQA, QuAC) — [arXiv 2403.15879 HTML](https://arxiv.org/html/2403.15879v2)
- Reliability Score (RS): per question, +1 for correct SQL on answerable; 0 for abstaining on answerable; −c for wrong SQL on answerable; −c for answering an unanswerable; +1 for abstaining on unanswerable. Mean over N samples; penalty c ∈ {0 (lenient), 10 (moderate), N (strict)}. With c=0, RS reduces to accuracy on answerable questions — [arXiv 2403.15879 HTML](https://arxiv.org/html/2403.15879v2)
- Infeasible questions are defined as involving "schema incompatibility or functionalities beyond SQL" — [arXiv abstract 2403.15879](https://arxiv.org/abs/2403.15879)
- The GitHub split table shows validation and test are balanced: equal feasible and infeasible counts per source dataset (e.g., ATIS test: 476 of each) — [github.com/glee4810/TrustSQL](https://github.com/glee4810/TrustSQL)

**EHRSQL**
- Unanswerable questions appear only in validation and test, making up 33% of each of those splits; datasheet reports ~1.9K unanswerable of ~24.4K instances overall (~8%); none in training — [ar5iv HTML of arXiv 2301.07695](https://ar5iv.labs.arxiv.org/html/2301.07695)
- Unanswerable questions are real held-out staff questions from the same hospital poll (not synthetic perturbations) — [arXiv abstract 2301.07695](https://arxiv.org/abs/2301.07695)
- Data JSON includes an `is_impossible` field marking unanswerable questions — [github.com/glee4810/EHRSQL](https://github.com/glee4810/EHRSQL)
- Metrics: F1ans (precision/recall for deciding answerability) and F1exe (execution-correct predictions); reference abstention baseline thresholds T5 output entropy (`abstain_with_entropy.py`, example threshold 0.14923561) — [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2301.07695); [github.com/glee4810/EHRSQL](https://github.com/glee4810/EHRSQL)

**BIRD / Spider 1.0 / Spider 2.0**
- BIRD: no unanswerable questions mentioned anywhere on the official page; ambiguity acknowledged (new "interactive setting" track announced 2025-11-13 "to better resolve ambiguity"; dev set cleaned Sept 2023 and again announced May 2025; site invites "gold error" reports) — [bird-bench.github.io](https://bird-bench.github.io/)
- Spider 1.0: no mention of unanswerable/ambiguous questions on the official page — [yale-lily.github.io/spider](https://yale-lily.github.io/spider)
- Spider 2.0: no unanswerable questions mentioned; data file updated 2025-07-13 "to resolve ambiguities" — [github.com/xlang-ai/Spider2](https://github.com/xlang-ai/Spider2)
- Related data-quality evidence: a VLDB audit reports "pervasive annotation errors" in BIRD Mini-Dev and Spider 2.0-Snow that can distort leaderboard rankings — [VLDB vol19 p931](https://www.vldb.org/pvldb/vol19/p931-jin.pdf); [arXiv 2601.08778](https://arxiv.org/pdf/2601.08778)

### Inferences
- For training an abstaining model: TrustSQL and EHRSQL provide labeled abstention supervision, but note EHRSQL has *no* unanswerable training examples (abstention must come from valid-set tuning or your own synthesis) and TrustSQL's unanswerables are *test-only*, so neither gives large-scale abstention training data out of the box — you will likely need to synthesize infeasible questions (TrustSQL's five-type taxonomy is a ready-made recipe).
- TrustSQL's RS metric is directly reusable as an RL reward shape for a write-or-abstain policy (+1 / 0 / −c structure).
- BIRD/Spider gold-error rates mean some "wrong" answers during RL on these sets are actually gold errors; the VLDB audit is worth consulting before trusting reward signals from their gold SQL.

### Gaps
- TrustSQL: the train-set feasible/infeasible breakdown is not given on GitHub (paper says unanswerables are test-only, which conflicts slightly with the GitHub "balanced validation" statement — needs a direct look at the released files to settle).
- EHRSQL per-split unanswerable *counts* (vs. percentages) are not stated in the paper body I retrieved.

## Key question 3: Contamination of Spider/BIRD in Qwen and Gemma training data

### Takeaway
No published study directly measures Spider or BIRD leakage into Qwen or Gemma pretraining — that specific question is UNVERIFIED in both directions. General evidence strongly suggests Spider (public since 2018, including its gold SQL) is contaminated in web-scale corpora, while BIRD appears less affected; Qwen and Gemma tech reports describe generic benchmark decontamination but neither names Spider or BIRD.

### Cited Findings
- Ranaldi & Ruzzetti et al., "Investigating the Impact of Data Contamination of Large Language Models in Text-to-SQL Translation" ([arXiv:2402.08100](https://arxiv.org/abs/2402.08100)): studied GPT-3.5; introduced a contamination-detection method for GPTs and a new unfamiliar dataset, Termite; found a significant performance drop on Termite vs. Spider, attributed to Spider sample-level contamination. GPT-3.5 only — says nothing about Qwen/Gemma.
- SPENCE, "A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks" ([arXiv 2604.17771](https://arxiv.org/pdf/2604.17771), 2026 preprint): generates syntactic variants of test queries for Spider, SParC, CoSQL, BIRD; reports rank sensitivity (a contamination signal) on the older Spider-family sets but not on BIRD. Preprint — treat as provisional. Whether its model pool includes Qwen/Gemma was not confirmed in this pass.
- Qwen2.5 Technical Report ([arXiv 2412.15115](https://arxiv.org/pdf/2412.15115)): states potentially contaminated data is excluded via n-gram matching when building pre-training and post-training data; n-gram length not confirmed from the search result.
- Qwen2.5-Coder Technical Report ([arXiv 2409.12186](https://arxiv.org/pdf/2409.12186)): 10-gram word-level overlap decontamination; benchmarks named as removed are HumanEval, MBPP, GSM8K, MATH — Spider/BIRD are not in the named list.
- Gemma 3 Technical Report ([arXiv 2503.19786](https://arxiv.org/abs/2503.19786)): states evaluation sets were removed from the pre-training mixture and recitation-risk filtering applied; no matching method, thresholds, or benchmark list given; Gemma 2/1 used the same filtering lineage ([ar5iv 2503.19786](https://ar5iv.labs.arxiv.org/html/2503.19786)).
- Weak anecdotal signal: a fine-tuning project README reports Qwen2.5-Coder base models scoring 77.2% zero-shot on Spider dev with no fine-tuning, and flags possible pretraining inclusion of Spider — [github.com/bonnie-mcconnell/sql-finetune](https://github.com/bonnie-mcconnell/sql-finetune) (hypothesis in a personal project README, not a measurement).
- Broader Qwen caution: "Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination" ([arXiv 2507.10532](https://arxiv.org/html/2507.10532v1)) argues Qwen models' web-scale pretraining leaves them vulnerable to benchmark contamination (demonstrated on math benchmarks, e.g. MATH-500 — not text-to-SQL), and that RL gains on contaminated benchmarks can be illusory.

### Inferences
- Since Qwen2.5-Coder's decontamination list names only code/math benchmarks, Spider/BIRD train and dev sets (publicly hosted on GitHub/HF with gold SQL) were plausibly *not* filtered out of Qwen pretraining. This is an inference from an absence, not evidence of presence.
- For the RL project this matters doubly: (a) held-out evaluation on Spider dev likely overstates true generalization for Qwen-based models; (b) the arXiv 2507.10532 result warns that RL reward improvements measured on a contaminated benchmark can reflect memorization retrieval rather than learned skill. Prefer BIRD (less contamination evidence), Spider 2.0 (2024+, partially hidden gold), hidden-test sets (BIRD test, EHRSQL hidden test), or perturbed/synthetic eval sets for honest measurement.
- "Qwen3.5-2B" is newer than any contamination study found; anything specific to it is UNVERIFIED by definition.

### Gaps
- UNVERIFIED: any direct measurement of Spider or BIRD presence in Qwen (any version) pretraining corpora. Not found.
- UNVERIFIED: any direct measurement for Gemma (any version). Not found.
- UNVERIFIED: whether Qwen3 / Qwen3.5 reports add benchmark-specific decontamination detail for SQL benchmarks (Qwen3 report not fetched in this pass).
- TrustSQL/EHRSQL contamination studies: none found at all (EHRSQL's hidden test and PhysioNet-gated databases make pretraining leakage less likely, but that is an inference).

## Key question 4: Test-suite execution evaluation and the BIRD harness

### Takeaway
Zhong et al. 2020 is real and its `test-suite-sql-eval` repo (Apache-2.0) does exactly what the project needs: it executes predicted vs. gold SQL on *multiple* distilled database instances per schema to approximate semantic equivalence, and it is Spider's official metric. BIRD's official harness (mini_dev repo) instead runs single-database execution accuracy plus efficiency (R-VES) and a lenient result-table Soft-F1; newer 2024–2026 tooling includes FLEX (LLM-as-judge execution evaluation, NAACL 2025).

### Cited Findings

**Test-suite accuracy (Zhong et al. 2020)**
- Paper verified: Zhong, Yu, Klein, "Semantic Evaluation for Text-to-SQL with Distilled Test Suites," EMNLP 2020, [arXiv:2010.02840](https://arxiv.org/abs/2010.02840) — cited on [github.com/taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval)
- Repo verified: [github.com/taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval). License: Apache-2.0. Described as computing "a tighter upper-bound for semantic accuracy efficiently."
- Multiple-instance execution: yes — predicted and gold SQL are executed against a set of distilled database instances per schema ("test suites"); classical-dataset records carry a `testsuite` field listing database paths for denotation comparison — [github.com/taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval)
- Official metric for Spider, SParC, CoSQL; also covers ATIS, Academic, Advising, Geography, IMDB, Restaurants, Scholar, Yelp — [github.com/taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval); [yale-lily.github.io/spider](https://yale-lily.github.io/spider)
- Usage: download test-suite databases into `database/`; run `evaluation.py --gold --pred --db --etype {exec,match,all}` with optional `--plug_value` (for systems that don't predict values) and `--keep_distinct`; `evaluate_classical.py` for the non-Spider datasets; deps are sqlparse and nltk. Databases are at `database/[db_id]/[db_id].sqlite` paths (SQLite; no other engine stated). Gold queries must not be modified because test-suite generation depends on them. Neighbor-query/random-database *generation* code lives in Ruiqi Zhong's separate repo — [github.com/taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval)

**BIRD official evaluation harness**
- Repo: [github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev). README badge links CC BY-SA 4.0.
- Metrics: EX (execution accuracy — core metric; result-comparison mechanics not documented on the README); R-VES (Reward-based Valid Efficiency Score — reward points assigned based on predicted/gold runtime ratio; 3 s timeout per query; best of 5 runs); Soft F1-Score (similarity of the returned result tables, lenient to column order and missing values, computed from per-row matched / predicted-only / gold-only values) — [github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)
- Dialects: SQLite originally; Mini-Dev also ships MySQL and PostgreSQL versions, transpiled from SQLite with sqlglot then refined manually and with GPT-4 Turbo — [github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)
- Workflow: predictions post-processed into `SQL \t----- bird -----\t db_id` format; `sh run_evaluation.sh` computes EX, R-VES, Soft-F1 via `evaluation_ex.py`, `evaluation_ves.py`, `evaluation_f1.py` — [github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)
- BIRD executes on a single database per question (no multi-instance test suites mentioned anywhere in the harness README) — [github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)

**Newer execution-eval tooling (2024–2026)**
- FLEX ("False-Less EXecution"), NAACL 2025 ([arXiv 2409.19014](https://huggingface.co/papers/2409.19014); [ACL anthology version](https://preview.aclanthology.org/moar-dois/2025.naacl-long.228.pdf)): LLM-as-judge evaluation targeting EX's false positives/negatives; branches on whether execution results match (CEQ: check for false positives; CNEQ: judge semantic correctness against the question allowing output-structure/value-representation variation); reported expert agreement — Cohen's kappa improves from 62 to 87.04 in the ACL version (the arXiv abstract says 61 → 78.17; versions disagree, ACL is the peer-reviewed one); re-ranks the BIRD top-10 (e.g., SuperSQL 64.08 under FLEX vs. 57.37 under EX). Code: [github.com/HeegyuKim/FLEX](https://github.com/HeegyuKim/FLEX)
- Spider 2.0's own evaluation suite (submission-based, oracle-table option, fixed 2025-10-29) — [github.com/xlang-ai/Spider2](https://github.com/xlang-ai/Spider2)
- Benchmark-quality auditing: "Pervasive Annotation Errors Break Text-to-SQL Benchmarks and Leaderboards" ([VLDB vol19 p931](https://www.vldb.org/pvldb/vol19/p931-jin.pdf); [arXiv 2601.08778](https://arxiv.org/pdf/2601.08778)) audits BIRD Mini-Dev and Spider 2.0-Snow gold labels.
- SPENCE (contamination probing via syntactic variants, 2026 preprint) doubles as an eval-robustness tool — [arXiv 2604.17771](https://arxiv.org/pdf/2604.17771)

### Inferences
- For an RL reward signal, test-suite execution (multiple DB instances) is markedly stronger than BIRD-style single-DB EX: single-DB EX rewards coincidentally-equal results, which an RL optimizer will exploit. The test-suite repo is Apache-2.0 and SQLite-based, so it can be embedded directly in a reward function; the separate Zhong repo provides the machinery to *generate* new test-suite instances for synthetic databases.
- FLEX is useful for final evaluation but is LLM-judge-based (cost + judge noise), so less suited as a dense RL reward than deterministic test-suite execution.

### Gaps
- How BIRD's EX compares result sets (ordered vs. set semantics, duplicate handling) is not documented in the mini_dev README; would require reading `evaluation_ex.py` source.
- Whether test-suite-sql-eval supports engines other than SQLite: not stated (paths imply SQLite-only).
- No 2024–2026 tool found that generates *distilled test suites for new/synthetic databases* as a maintained package (Zhong's generation repo exists but its current state was not checked).

## Key question 5: Synthetic text-to-SQL data generation prior art

### Takeaway
Two main open resources: Gretel's synthetic_text_to_sql (~106K samples, Apache-2.0, generated by Gretel Navigator with schemas included per-row) and SynSQL-2.5M (2.54M samples over 16,583 fully synthetic databases, Apache-2.0, generated entirely with open-source LLMs via a web-table-to-database schema-generation pipeline). SynSQL's pipeline — database synthesis from TabLib web tables, complexity-controlled SQL generation, SQL-to-question back-translation, execution-majority-voted chain-of-thought — is the state of the art for schema generation and is directly reusable.

### Cited Findings

**Gretel synthetic_text_to_sql**
- Size: ~106K rows (100K train / 5.85K test) — [HF: gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
- Content per row: id, domain (100 distinct verticals, e.g. forestry, defense, marine biology) + description, sql_complexity (8 levels), sql_task_type (4 types), sql_prompt (NL question), sql_context (schema + sample data as CREATE/INSERT statements), sql, sql_explanation — [HF dataset card](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
- License: Apache-2.0 — [HF dataset card](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
- Generation: Gretel Navigator, described as a compound AI system with agent-based execution (code-executing agents so generated SQL is checked by running it) and a proprietary tabular LLM; exact prompts/pipeline internals are not public — [VentureBeat](https://venturebeat.com/ai/gretel-releases-worlds-largest-open-source-text-to-sql-dataset-empowering-businesses-to-unlock-ais-potential); [HF dataset card README](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql/blob/main/README.md)
- Quality: SQL parsability/validity checked with sqlglot and sqlvalidator; LLM-as-judge comparison vs. b-mc2/sql-create-context using GPT-4 — [HF dataset card README](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql/blob/main/README.md)

**SynSQL-2.5M / OmniSQL**
- Size: 2,544,390 samples, each a <database, question, SQL, chain-of-thought> quad, over 16,583 synthetic databases; "generated entirely using open-source LLMs" — [github.com/RUCKBReasoning/OmniSQL](https://github.com/RUCKBReasoning/OmniSQL)
- License: Apache 2.0 (dataset) — [github.com/RUCKBReasoning/OmniSQL](https://github.com/RUCKBReasoning/OmniSQL)
- Paper: [arXiv:2503.02240](https://arxiv.org/abs/2503.02240), VLDB '25
- Schema/database synthesis: sample 0.1% of TabLib (~1.32M web tables) → filter (language, size, duplicate headers, semantic richness) to 19,935 tables → LLM infers a business scenario per table and designs a multi-table database → second LLM pass adds columns and missing keys — [arXiv 2503.02240 HTML](https://arxiv.org/html/2503.02240v2)
- SQL synthesis: LLM writes SQLite queries from schema + sampled advanced functions + sampled column values + target complexity (simple → highly complex) + geometrically sampled column count; post-processing drops non-SELECT, erroring, and timing-out queries; one query kept per SQL template; ~5M generated, ~2.5M retained — [arXiv 2503.02240 HTML](https://arxiv.org/html/2503.02240v2)
- Question synthesis: SQL-to-question back-translation; LLM writes an explanation then a question in one of nine styles (formal, colloquial, conversational, ...); several candidates generated and the one with highest average cosine similarity to the others kept (Sentence Transformers) — [arXiv 2503.02240 HTML](https://arxiv.org/html/2503.02240v2)
- CoT synthesis: 8 samples at temperature 0.8; final solution chosen from the largest group of candidates agreeing on execution results (execution-based majority vote) — [arXiv 2503.02240 HTML](https://arxiv.org/html/2503.02240v2)
- Generator LLMs: Llama3.1 (8B/70B-Instruct), DeepSeek-Coder (6.7B/33B-Instruct, V2-Lite-Instruct), Qwen2.5 (7B/14B/32B/72B-Instruct), Qwen2.5-Coder (7B/14B/32B-Instruct); larger models got more workload — [arXiv 2503.02240 HTML](https://arxiv.org/html/2503.02240v2)
- Quality: GPT-4o and human evaluation — 96% of databases judged realistic, 86% of full samples judged fully correct — [arXiv 2503.02240 HTML](https://arxiv.org/html/2503.02240v2)
- OmniSQL models: 7B/14B/32B built on Qwen2.5-Coder-{7B,14B,32B}-Instruct (7B/14B full fine-tune, 32B LoRA), trained on SynSQL-2.5M plus Spider and BIRD train sets; reported to match or surpass GPT-4o and DeepSeek-V3 across nine benchmarks — [arXiv 2503.02240 HTML](https://arxiv.org/html/2503.02240v2); [github.com/RUCKBReasoning/OmniSQL](https://github.com/RUCKBReasoning/OmniSQL)

**Other prior art**
- SENSE, "Synthesizing Text-to-SQL Data from Weak and Strong LLMs" ([arXiv:2408.03256](https://arxiv.org/abs/2408.03256)): combines synthetic data from strong LLMs with error data from weak LLMs, exploiting "error data supervision through preference learning"; evaluated on Spider and BIRD. (Abstract-level detail only; DPO/executor-feedback mechanics and model sizes not confirmed in this pass.)

### Inferences
- SynSQL's pipeline is the most relevant blueprint for this project: it demonstrates fully open-model schema generation at scale, and its execution-majority-vote CoT selection is essentially a self-consistency reward — adaptable to generating *verifiably answerable* questions. Conversely, deliberately breaking a step (ask about a dropped column, etc., per TrustSQL's taxonomy) yields synthetic *unanswerable* questions with known labels.
- Caveat for a Qwen-based student: SynSQL-2.5M was largely generated *by* Qwen2.5(-Coder) models, and OmniSQL's bases are Qwen2.5-Coder — training Qwen3.5-2B on SynSQL is self-distillation-adjacent, fine for capability but it won't add knowledge diversity beyond the Qwen family, and any Qwen-specific Spider leakage could propagate through the synthetic data (my inference).
- Both big synthetic sets are Apache-2.0: free to filter, remix, and redistribute with attribution; no training restriction.

### Gaps
- Gretel Navigator's exact generation pipeline (models, prompts) is proprietary and undocumented.
- SENSE's released artifacts/licenses were not checked (paper fetched at abstract level only).
- No verified prior art found specifically on synthesizing *unanswerable* text-to-SQL questions at scale beyond TrustSQL's manual annotation (possible novelty for this project).
