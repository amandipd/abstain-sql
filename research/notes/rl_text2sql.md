# RL for Text-to-SQL: Prior Work Catalog (as of 2026-11-04)

Scope: papers/systems that train text-to-SQL models with reinforcement learning (RL), especially execution-based rewards and GRPO-family algorithms (GRPO = Group Relative Policy Optimization, an RL algorithm that scores a group of sampled outputs against each other instead of training a separate value network). Each entry records: what it did, models/sizes, RL algorithm, reward design, abstention handling, risk–coverage reporting, benchmark numbers, and links. Everything below was verified against the linked source on 2026-11-04 unless marked UNVERIFIED.

Glossary used throughout:
- **Execution accuracy (EX)** = fraction of generated SQL queries whose execution result matches the gold query's result.
- **BIRD / Spider** = the two standard text-to-SQL benchmarks; BIRD dev is the commonly reported development split.
- **Abstention** = the model declining to answer ("I don't know" / refusal / clarification request) instead of emitting SQL.
- **Risk–coverage curve** = plot of error rate (risk) among answered questions vs. fraction of questions answered (coverage); the standard selective-prediction evaluation.

## Key Question 1: What exactly do Arctic-Text2SQL-R1, SQL-R1, and Reasoning-SQL do, and what model sizes do they use?

### Takeaway
All three are 2025 GRPO-trained text-to-SQL systems with execution-grounded rewards, at sizes 3B–32B (none below 3B): Arctic-Text2SQL-R1 (Snowflake, 7B/14B/32B) uses the simplest reward (execution match = 1, executable = 0.1), SQL-R1 (3B/7B/14B) uses a composite format/execution/result/length reward, and Reasoning-SQL (14B; other sizes unverified) uses SQL-tailored partial rewards (schema linking, LLM judge, n-gram, syntax). None handles abstention or reports risk–coverage.

### Cited Findings

**Arctic-Text2SQL-R1** (Snowflake AI Research + UMD + UCSD; arXiv:2505.20315, v1 22 May 2025, v2 12 Jan 2026)
- Title: "Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL"; authors Zhewei Yao et al. — [arXiv abs](https://arxiv.org/abs/2505.20315)
- RL algorithm: GRPO, online training; 16 rollouts per prompt, total batch 256, KL penalty beta = 0.001, clip ratio 0.2, temperature 0.8; trained on 64 H100 GPUs ("a few hours" for 7B, up to 2.5 days for 32B) — [arXiv full text v2](https://arxiv.org/html/2505.20315v2)
- Reward: R = 1 if execution results exactly match ground truth; 0.1 if the SQL is syntactically correct and executable; 0 otherwise. Explicitly no partial string-overlap or schema-conformance terms ("lightweight reward signal based solely on execution correctness") — [arXiv full text v2](https://arxiv.org/html/2505.20315v2)
- Model sizes: 7B, 14B, 32B, initialized from OmniSQL supervised checkpoints (Qwen2.5-Coder lineage). No model below 7B — [arXiv full text v2](https://arxiv.org/html/2505.20315v2)
- Results (EX): BIRD dev 68.9 (7B) / 70.1 (14B) / 70.5 (32B); BIRD test 68.5 / 70.0 / 71.8; Spider test 88.8 / 89.4 / 88.7; Spider-DK 81.5 / 79.4 / 80.6 — [arXiv full text v2, Tables 1 and 7](https://arxiv.org/html/2505.20315v2)
- Training data: filtered BIRD-train (8,017), Spider-train (6,972), Spider-dev (985), Gretel-Synth-Filtered (11,811); filtering removed empty-result gold queries and queries running >5 s — [arXiv full text v2, Table 3](https://arxiv.org/html/2505.20315v2)
- Abstention: not mentioned anywhere in the paper. Risk–coverage / selective prediction: not mentioned. Inference-time boost is majority voting over 8 generations (answer selection, not abstention) — [arXiv full text v2](https://arxiv.org/html/2505.20315v2)
- Code: [github.com/snowflakedb/ArcticTraining](https://github.com/snowflakedb/ArcticTraining/); weights: [Snowflake/Arctic-Text2SQL-R1-7B on Hugging Face](https://huggingface.co/Snowflake/Arctic-Text2SQL-R1-7B)
- The 32B model reached the top of the BIRD leaderboard at 71.8% test EX per the paper's own claim — [arXiv abs](https://arxiv.org/abs/2505.20315)

**SQL-R1** (Peixian Ma, Xialie Zhuang, Chengjin Xu, Xuhui Jiang, Ran Chen, Jian Guo; arXiv:2504.08600, v1 11 Apr 2025, v5 9 Oct 2025; accepted NeurIPS 2025)
- Title: "SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning" — [arXiv abs](https://arxiv.org/abs/2504.08600)
- RL algorithm: GRPO (paper notes it "obviates the need for the value model") — [arXiv full text v5](https://arxiv.org/html/2504.08600v5)
- Reward (composite, per Eqs. 2–5): format +1/−1; execution +2 (executable) / 0 (bad format) / −2 (not executable); result +3 (correct result) / 0 / −3 (incorrect result); plus a length-shaping term. Max RL response length 2048 — [arXiv full text v5](https://arxiv.org/html/2504.08600v5)
- Base models: Qwen2.5-Coder-3B/7B/14B-Instruct; additional runs on OmniSQL-7B/14B — [arXiv full text v5](https://arxiv.org/html/2504.08600v5)
- Results (EX): 3B — Spider dev 78.1, Spider test 78.9, BIRD dev 54.6. 7B — Spider dev 87.6, Spider test 88.7, BIRD dev 66.6 (Table 1; but Table 7 and body text say 63.1 — internal inconsistency in the paper). 14B — Spider dev 86.7, Spider test 88.1, BIRD dev 67.1. Headline abstract numbers: 88.6% Spider, 67.1% BIRD — [arXiv full text v5](https://arxiv.org/html/2504.08600v5); [arXiv abs](https://arxiv.org/abs/2504.08600)
- Data: SFT cold start on SynSQL-200K (200k subset of SynSQL-2.5M, 50k per difficulty level); RL on SynSQL-Complex-5K (5k complex pairs). Paper concludes "SFT cold-start training is not universally essential for RL-based NL2SQL models" — [arXiv full text v5](https://arxiv.org/html/2504.08600v5)
- Abstention: none. Risk–coverage: none. Closest mechanism is self-consistency voting over 8 candidates (selection, not refusal) — [arXiv full text v5](https://arxiv.org/html/2504.08600v5)
- Code: [github.com/IDEA-FinAI/SQL-R1](https://github.com/IDEA-FinAI/SQL-R1); community weights: [MPX0222forHF/SQL-R1-7B](https://huggingface.co/MPX0222forHF/SQL-R1-7B)

**Reasoning-SQL** (Mohammadreza Pourreza, Shayan Talaei, Ruoxi Sun, Xingchen Wan, Hailong Li, Azalia Mirhoseini, Amin Saberi, Sercan O. Arik; arXiv:2503.23157, v1 29 Mar 2025, v2 1 Apr 2025; listed as COLM 2025 on Google Research pubs page)
- Title: "Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL" — [arXiv abs](https://arxiv.org/abs/2503.23157); [Google Research pub listing](https://research.google/pubs/reasoning-sql-reinforcement-learning-with-partial-rewards-for-reasoning-enhanced-text-to-sql/)
- RL algorithm: GRPO; multiple candidate queries per input are scored relative to each other to compute advantages — [arXiv abs](https://arxiv.org/abs/2503.23157)
- Reward: partial rewards explicitly designed against reward sparsity — abstract verbatim: "Our reward set includes schema-linking, AI feedback, n-gram similarity, and syntax check" (AI feedback = LLM-as-a-judge scoring) — [arXiv abs](https://arxiv.org/abs/2503.23157)
- Model size: the abstract names only an RL-trained 14B model ("our RL-trained 14B-parameter model significantly outperforms larger proprietary models"); it outperforms o3-mini by 4% and Gemini-1.5-Pro-002 by 3% on BIRD — [arXiv abs](https://arxiv.org/abs/2503.23157). Whether smaller sizes (e.g. 7B) were also trained: UNVERIFIED from the abstract page; the full PDF would need checking.
- Abstention: not mentioned on the abstract page. Risk–coverage: not mentioned. Exact per-size BIRD dev EX numbers: not retrieved (gap below).

### Inferences
- All three treat every question as answerable: the reward space contains no abstain action and no penalty asymmetry between "wrong SQL" and "no SQL" beyond SQL-R1's negative execution/result scores (which still force an attempt).
- The field's reward-design axis runs from minimal (Arctic: execution-only) to dense/composite (Reasoning-SQL partial rewards; SQL-R1 staged scores) — abstention is absent from that axis in all three.

### Gaps
- Reasoning-SQL per-size BIRD dev EX numbers and whether a 7B variant exists — abstract page does not contain them; full PDF not fetched (tool budget).
- Reasoning-SQL official code link — not shown on the arXiv abstract page.

## Key Question 2: What newer RL-for-text-to-SQL work appeared in 2025–2026?

### Takeaway
The space is crowded: at least a dozen GRPO-family systems appeared from mid-2025 through 2026, spanning single-turn execution rewards (CogniSQL-R1-Zero, Think2SQL), self-correction (CSC-SQL, SLM-SQL), multi-turn/agentic RL (SkyRL-SQL, MTIR-SQL, SQL-Trail, Progress-SQL, SERL-SQL), execution-free reward models (Graph-Reward-SQL), clause-level rewards (EXPO-SQL), verified-data RLVR at frontier scale (ReViSQL-BIRD), and reliability-oriented RL (BAR-SQL, Jan 2026 — see Question 3).

### Cited Findings

**CSC-SQL** (Lei Sheng, Shuai-Shuai Xu; arXiv:2505.13271, v1 19 May 2025, v2 30 Jun 2025)
- Self-consistency + self-correction: the two most frequent SQL outputs from parallel sampling go to a merge-revision model; GRPO fine-tunes both the generator and the reviser — [arXiv abs](https://arxiv.org/abs/2505.13271)
- Reward: R = R_EX + 0.1 * R_Format (binary execution match plus think/answer tag check); TRL, lr 3e-6, 6 completions per prompt — [arXiv full text v2](https://arxiv.org/html/2505.13271v2)
- Bases: Qwen2.5-Coder-3B/7B/14B/32B-Instruct, XiYanSQL-QwenCoder-3B/7B/32B, OmniSQL-7B, Llama-3.1-8B, gemma-3-12b, Llama-3.1-70B — [arXiv full text v2](https://arxiv.org/html/2505.13271v2)
- Results: BIRD private test 71.72% (7B), 73.67% (32B); BIRD dev numbers under Question 4 — [arXiv abs](https://arxiv.org/abs/2505.13271)
- No abstention, no risk–coverage. Code: [github.com/CycloneBoy/csc_sql](https://github.com/CycloneBoy/csc_sql); checkpoints under [cycloneboy on Hugging Face](https://huggingface.co/cycloneboy/CscSQL-Grpo-XiYanSQL-QwenCoder-3B-2502)

**SLM-SQL** (Lei Sheng, Shuai-Shuai Xu; arXiv:2507.22478; IJCNLP-AACL 2025)
- Deliberate exploration of small language models (0.5B–1.5B) for text-to-SQL: SFT then GRPO (lr 3e-6, cosine schedule, 6 rollouts), plus a 0.5B merge-revision model at inference — [arXiv PDF](https://arxiv.org/pdf/2507.22478); [HF model card SLM-SQL-Base-1.5B](https://huggingface.co/cycloneboy/SLM-SQL-Base-1.5B/blob/refs%2Fpr%2F1/README.md)
- Bases: Qwen2.5-Coder-0.5B/1.5B-Instruct, Qwen3-0.6B, DeepSeek-Coder-1.3B, Llama-3.2-1B — [HF model card](https://huggingface.co/cycloneboy/SLM-SQL-Base-1B)
- Results under Question 4. Code: [github.com/CycloneBoy/slm_sql](https://github.com/CycloneBoy/slm_sql)

**Think2SQL** (Simone Papicchio, Simone Rossi, Luca Cagliero, Paolo Papotti; arXiv:2504.15077, v1 21 Apr 2025 through v5 4 May 2026; follow-up version accepted at TMLR per arXiv listing comment)
- GRPO within RLVR (RL with verifiable rewards), Open-R1 framework. Reward: R = 0.85 * R_text2SQL + 0.10 * R_FR (format) + 0.05 * R_TCR, where R_text2SQL is either binary execution accuracy or the dense R_QATCH = average of cell precision, cell recall, and tuple cardinality (partial credit for partially-correct result tables). Dense QATCH reward outperforms binary EX — [arXiv full text v5](https://arxiv.org/html/2504.15077v5)
- Models (v5): Qwen2.5-Coder-Instruct at 0.5B, 1.5B, 3B, 7B, 14B — [arXiv full text v5](https://arxiv.org/html/2504.15077v5). The arXiv listing comment says a TMLR follow-up is based on the Qwen3 family, and a mid-2025 v4 abstract claimed a 4B model "competitive with o3" — [arXiv v4 HTML](https://arxiv.org/html/2504.15077v4) (4B claim seen only in search snippet; per-number details UNVERIFIED).
- Headline: "the 14B Qwen-Coder-2.5 surpassing 400B+ models on challenging datasets like BIRD" — [arXiv abs](https://arxiv.org/abs/2504.15077)
- No abstention, no risk–coverage (limitations table notes no self-reflection/self-consistency/ensembling) — [arXiv full text v5](https://arxiv.org/html/2504.15077v5)

**SkyRL-SQL** (NovaSky team, UC Berkeley; released May 2025; blog/release, not an arXiv paper as far as found)
- Multi-turn RL pipeline (built on VeRL + SearchR1): the model probes the database, observes execution feedback, and refines its query over several turns. SkyRL-SQL-7B trained on only ~600 datapoints (653 released) improves base-model EX by up to 9.2% across 5 Spider-family benchmarks, outperforming GPT-4o, o4-mini, and OmniSQL-7B — [NovaSky SkyRL-SQL notion page](https://novasky-ai.notion.site/skyrl-sql)
- Multi-turn vs single-turn: reached a 60% reward score 2.8x faster in training steps, +16% final reward after 35 steps; authors note overconfidence and looping failure modes — [NovaSky SkyRL-SQL notion page](https://novasky-ai.notion.site/skyrl-sql)
- Code: [github.com/NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL) (Apache-2.0). No abstention, no risk–coverage mentioned.

**CogniSQL-R1-Zero** (Kushal Gajjar, Harshit Sikchi, Arpit Singh Gautam, Marc Hammons, Saurabh Jha; arXiv:2507.06013, 8 Jul 2025)
- RL on a 7B backbone with a "lightweight reward signal based on execution correctness and format-tag compliance"; explicitly avoids intermediate supervision and complex reward shaping; trained on 4x A100-40GB. Claims SOTA EX on BIRD (number not on abstract page). Releases 5,024 reasoning traces + 36,356 weakly supervised queries with six reasoning paths each — [arXiv abs](https://arxiv.org/abs/2507.06013). No abstention, no risk–coverage.

**MTIR-SQL** (Zekun Xu et al.; arXiv:2510.25510, 29 Oct 2025)
- "Multi-turn Tool-Integrated Reasoning Reinforcement Learning for Text-to-SQL": execution-aware multi-turn reasoning, feeding DB execution feedback into each step; extends GRPO with trajectory filtering and removal of the KL constraint. 4B-parameter model (base not named on abstract page). Results: 64.4% BIRD dev, 84.6% Spider dev — [arXiv abs](https://arxiv.org/abs/2510.25510). No abstention, no risk–coverage, no code link on the page.

**Graph-Reward-SQL** (Han Weng et al., ByteDance; arXiv:2505.12380; Findings of EMNLP 2025)
- Execution-FREE reward models for RL: GMNScore (graph matching network over SQL query graphs scores candidates without running them, cutting execution latency and GPU memory) plus StepRTM (stepwise reward over CTE subqueries for intermediate supervision). Reports consistently outperforming other reward models on Spider and BIRD — [arXiv HTML](https://arxiv.org/html/2505.12380v3); [EMNLP Findings page](https://preview.aclanthology.org/dashboard/2025.findings-emnlp.694) (exact EX numbers not retrieved)

**Sparks of Tabular Reasoning via Text2SQL RL** (Josefa Lia Stoisser et al.; arXiv:2505.00016; TRL workshop @ ACL 2025)
- Two-stage: synthetic chain-of-thought SFT, then GRPO tying SQL execution accuracy to broader table-reasoning skill; gains on BIRD and CRT-QA — [arXiv HTML v2](https://arxiv.org/html/2505.00016v2); [ACL Anthology](https://aclanthology.org/2025.trl-1.20)

**Reinforcing Code Generation: Improving Text-to-SQL with Execution-Based Learning** (Atharv Kulkarni, Vivek Srikumar; arXiv:2506.06093, Jun 2025)
- Frames text-to-SQL as RL with scalar execution-derived rewards (penalizing failures, rewarding correct answers) within GRPO — [arXiv abs](https://arxiv.org/abs/2506.06093v1) (details beyond abstract UNVERIFIED)

**GradeSQL** (arXiv:2509.01308, Sep 2025) — not RL training but adjacent: outcome reward models for test-time best-of-N over OmniSQL-7B; ORMs reach 68.90% BIRD dev (+5.01 over baseline; execution-based best-of-N 66.04%, majority voting 66.95%) — [arXiv PDF](https://arxiv.org/pdf/2509.01308) (numbers from search snippet of the paper)

**R3-SQL** (arXiv:2604.25325, Apr 2026) — ranking reward + resampling; best pairwise ranker with OmniSQL-7B backbone reports 71.45 EX on BIRD dev — [arXiv PDF](https://arxiv.org/pdf/2604.25325) (search-snippet level; UNVERIFIED detail)

**SQL-Trail** (arXiv:2601.17699, Jan 2026) — multi-turn RL agent with interleaved feedback: iterative database probing, schema exploration, execution-based refinement; compares against single-pass baselines such as SQL-R1 — [arXiv HTML](https://arxiv.org/html/2601.17699v1) (details beyond this UNVERIFIED — not fetched)

**Human-Level Text-to-SQL via RL on Verified Data (ReViSQL-BIRD)** (Yuxuan Zhu, Tengjun Jin, Yoojin Choi, Daniel Kang; arXiv:2603.20004, v1 20 Mar 2026, v4 21 Aug 2026)
- RLVR on BIRD-Platinum, a 2.5k expert-verified subset of BIRD train (errors corrected in 61% of instances); result-based rewards combined with SQL-equivalence verification (noting result-only rewards have "non-trivial false positive rates") plus process rewards encouraging use of external knowledge. Bases: Qwen3-235B and Kimi-K2.6 (frontier scale, not small). ReViSQL-BIRD-K2.6 reaches 92.96% on Arcwise-Plat, described as human-level accuracy — [arXiv abs](https://arxiv.org/abs/2603.20004). No abstention, no risk–coverage on the page.

**Progress-SQL** (Shihao Zhang, Xiaoman Wang, Yuan Liu, Yunshi Lan, Weining Qian; arXiv:2606.06825, 5 Jun 2026)
- Multi-turn RL with progressive rewards "measur[ing] the improvement from the initial SQL to the final SQL": structural alignment via an Oracle-guided Diagnostic Tree, lexical alignment, a progression-latency reward, and an execution-status reward; motivated by final-state-only rewards (e.g. SkyRL-SQL) giving "limited guidance for iterative SQL correction." Improves BIRD/Spider and robustness variants (numbers not on abstract page) — [arXiv abs](https://arxiv.org/abs/2606.06825). Base models/algorithm UNVERIFIED (not on abstract page).

**SERL-SQL** (arXiv:2608.00485, Aug 2026) — "Selective Hindsight Distillation for Text-to-SQL Reinforcement Agentic Learning" — found only as title + PDF link; contents UNVERIFIED — [arXiv PDF](https://arxiv.org/pdf/2608.00485)

**EXPO-SQL** (Findings of ACL 2026) — clause-level policy optimization using execution feedback, arguing query-level rewards give weak learning signals — [ACL Anthology PDF](https://aclanthology.org/2026.findings-acl.1107.pdf) (details beyond this UNVERIFIED)

### Inferences
- The 2025 wave was single-turn GRPO with execution rewards; the late-2025/2026 wave moved to multi-turn agentic RL (SkyRL-SQL, MTIR-SQL, SQL-Trail, Progress-SQL) and reward-model engineering (Graph-Reward-SQL, GradeSQL, R3-SQL, EXPO-SQL). Reliability/abstention entered only in Jan 2026 with BAR-SQL (Question 3).
- OmniSQL (arXiv:2503.02240, SynSQL-2.5M; SFT-only, 7B/14B/32B; ~63.9–64.2 EX BIRD dev greedy for 7B per leaderboard snippets — [OmniSQL paper](https://arxiv.org/html/2503.02240), [HF seeklhy/OmniSQL-7B](https://huggingface.co/seeklhy/OmniSQL-7B)) functions as the de-facto SFT starting checkpoint and data source for much of this RL work (Arctic, SQL-R1, CSC-SQL, GradeSQL, R3-SQL).

### Gaps
- No dedicated survey of RL-for-text-to-SQL was found (searched; the general LLM text-to-SQL survey arXiv:2408.05109 is cited instead).
- Exact EX numbers for Graph-Reward-SQL, Progress-SQL, SQL-Trail, SERL-SQL, EXPO-SQL, CogniSQL-R1-Zero not retrieved within budget — all have links above for follow-up.

## Key Question 3: Do any include an abstention/"I don't know" action in the reward? Do any report risk–coverage?

### Takeaway
Exactly one RL-trained text-to-SQL paper with abstention in the reward was found — BAR-SQL (Jan 2026, Qwen3-1.7B + GRPO, enterprise setting) — and it does NOT report risk–coverage or any selective-prediction metric, nor Spider/BIRD results. Conversely, the papers that DO report risk–coverage/selective-prediction for text-to-SQL (Somov & Tutubalina 2025; Richardson 2026) do post-hoc confidence estimation only, with no RL training. No paper was found combining RL training + an abstention reward + risk–coverage curves, at any model size.

### Cited Findings

**BAR-SQL — "Boundary-Aware NL2SQL: Integrating Reliability through Hybrid Reward and Data Synthesis"** (Songsong Tian, Kongsheng Zhuo, Zhendong Wang, Rong Shen, Shengtao Zhang, Yong Wu; arXiv:2601.10318, 15 Jan 2026)
- The closest prior work to RL + abstention. Policy initialized from Qwen3-1.7B (so under 4B); SFT (full-parameter, lr 5e-5) then GRPO: group size 16, clip 0.2, KL beta 0.01 vs the cold-start reference, lr 1e-6, 48 A800 GPUs, ms-swift v3.9.3 — [arXiv full text](https://arxiv.org/html/2601.10318)
- Reward ("Task-Conditioned Hybrid Reward"): R = 1.5*R_acc + 1.0*R_fmt + 0.9*R_gram + 0.8*R_len (alpha_struct = 0.5). For answerable questions R_acc uses SQL execution/AST + dense result matching. For unanswerable/ambiguous questions the target output is a natural-language refusal or clarification, scored by embedding cosine similarity rho (Qwen3-0.6B-embedding): R = rho if rho >= 0.96; 0.8*rho if 0.90 <= rho < 0.96; 0 otherwise. There is NO explicit negative penalty term for answering an unanswerable question — the penalty is implicit (zero reward) — [arXiv full text](https://arxiv.org/html/2601.10318)
- Data synthesis for the abstention classes: ambiguity injection (precise entities rewritten vague), constraint follow-up (missing mandatory parameters), dimension rejection and metric rejection (schema/metric definitions removed so refusal is correct) — [arXiv full text](https://arxiv.org/html/2601.10318)
- Evaluation: own enterprise benchmark Ent-SQL-Bench (1,262 held-out instances, ~50% answerable / 50% negative-or-interactive). BAR-SQL averages 91.48% vs Claude-4.5-Sonnet 48.02%, GPT-5 46.01%, Qwen3-1.7B-SFT 67.55%. Metrics are execution accuracy and GPT-4o-judged semantic consistency only. It does NOT report risk–coverage curves, selective risk, coverage, AUROC, or any selective-prediction metric, and reports NO Spider or BIRD results — [arXiv full text](https://arxiv.org/html/2601.10318)
- Code: [github.com/TianSongS/BAR-SQL](https://github.com/TianSongS/BAR-SQL)

**Selective prediction / abstention WITHOUT RL training (text-to-SQL):**
- **TrustSQL** (Gyubok Lee et al.; arXiv:2403.15879, v6 Jul 2024): benchmark scoring text-to-SQL reliability with penalty-based scoring — correct SQL on feasible questions and correct abstention on infeasible ones are rewarded; wrong SQL on either is penalized; built over ATIS, Advising, EHRSQL, Spider. Finding: "achieving high scores under severe penalty settings remains highly challenging." Not an RL method — [arXiv abs](https://arxiv.org/abs/2403.15879v4); code [github.com/glee4810/TrustSQL](https://github.com/glee4810/TrustSQL)
- **Confidence Estimation for Error Detection in Text-to-SQL** (Oleg Somov, Elena Tutubalina; arXiv:2501.09527; AAAI 2025): frames abstention as a coverage-vs-risk trade-off ("text-to-SQL with a reject option") with entropy-based selective classifiers; under cross-database/compositional shift the selective classifier has high recall but low precision. Post-hoc; no RL — [arXiv abs](https://arxiv.org/abs/2501.09527)
- **Reliable Text-to-SQL with Adaptive Abstention** (arXiv:2501.10858, Jan 2025): conformal-prediction-style abstention at branching points/schema linking, evaluated with coverage and extra-abstention-rate; no RL — [arXiv PDF](https://arxiv.org/pdf/2501.10858)
- **What Predicts Correctness in Text-to-SQL? A Selective-Prediction Study** (Robert Richardson; arXiv:2607.06799, 7 Jul 2026): evaluation + fine-tuned verifiers (no RL). Metrics: AUROC, expected calibration error, selective risk at answer rate, "abstention frontiers." Black-box signals (self-consistency, executability, schema relevance, log-prob) reach only 0.61–0.68 AUROC; LLM judges 0.72–0.78; a two-provider judge ensemble 0.82 AUROC with ECE 0.03, answering 27% of questions at 24% selective risk on hard multi-table BIRD/Spider items; fine-tuned verifiers hit 0.77–0.79 in-distribution but drop to ~0.66 on unseen schemas — [arXiv abs](https://arxiv.org/abs/2607.06799)

**Abstention-in-RL OUTSIDE text-to-SQL (transferable mechanisms):**
- **Abstain-R1** ("Calibrated Abstention and Post-Refusal Clarification via Verifiable RL"): SFT then GRPO with verifiable rewards; training mix ~30% unanswerable / 70% answerable; base reward 0.3 for explicitly abstaining on unanswerable questions. Not SQL-specific. Source is a literature-review site; arXiv ID not captured — [themoonlight.io review](https://www.themoonlight.io/review/abstain-r1-calibrated-abstention-and-post-refusal-clarification-via-verifiable-rl) (UNVERIFIED against the primary paper)
- **Rewarding Efficient Reasoning Improves Abstention on Underspecified Tasks** (arXiv:2609.20846, Sep 2026): adds an efficiency-oriented reward to GRPO to improve abstention on underspecified tasks in reasoning models; not SQL — [arXiv PDF](https://arxiv.org/pdf/2609.20846)

### Inferences
- **Novelty check answer: the specific combination "RL training + abstention reward + risk–coverage curves on a <=4B model" appears unpublished as of 2026-11-04.** BAR-SQL occupies "RL + abstention + 1.7B" but (a) evaluates only on a private enterprise benchmark with a knowledge-graph schema layer, not Spider/BIRD, (b) scores refusals by text similarity to gold refusal messages rather than treating abstention as a calibrated action with an explicit reward/penalty trade-off, and (c) reports no risk–coverage or selective-prediction analysis. The selective-prediction literature (2501.09527, 2607.06799, TrustSQL) supplies the metrics but never trains with RL.
- TrustSQL's penalty-based score is the natural external yardstick a new RL+abstention method could report against; no RL paper found does so.

### Gaps
- Abstain-R1's primary arXiv record (ID, authors, venue) was not located — only the review-site summary; verify before citing in a paper.
- BAR-SQL's final 42k characters (of 142k) were not read; the appendix could contain extra metrics, though the metrics section read ("Acc_exec and Acc_sem only") makes risk–coverage there unlikely.
- Could not exhaustively rule out an RL+abstention text-to-SQL paper outside arXiv (e.g., industrial blog or under review); searches on multiple phrasings surfaced only BAR-SQL.

## Key Question 4: Which use small models (<=4B)? What execution accuracy do small RL-trained models reach on BIRD dev?

### Takeaway
Several RL-trained models at or below 4B exist, with BIRD dev EX (single-model/greedy-style configs) clustering at roughly 54–67%: SQL-R1-3B 54.6; SLM-SQL 0.5B 56.87 and 1.5B 67.08; CSC-SQL 3B-class models 60.6–65.3 (with voting/revision); MTIR-SQL-4B 64.4; Think2SQL 3B ~0.50 (weighted metric). None of these does abstention or risk–coverage; BAR-SQL (1.7B) is the only small RL model trained for refusal but reports no BIRD numbers.

### Cited Findings
- **SQL-R1 + Qwen2.5-Coder-3B: 54.6 BIRD dev EX** (Spider dev 78.1, Spider test 78.9) — [arXiv full text v5, Table 1](https://arxiv.org/html/2504.08600v5)
- **SLM-SQL: 0.5B reaches 56.87% and 1.5B reaches 67.08% EX on BIRD dev** (Table 5 config includes self-consistency with 64 samples for the 67.08 figure); on BIRD private test 0.5B 61.82% and 1.5B 70.49% — [arXiv PDF](https://arxiv.org/pdf/2507.22478); [HF SLM-SQL-Base-1.5B README](https://huggingface.co/cycloneboy/SLM-SQL-Base-1.5B/blob/refs%2Fpr%2F1/README.md) (dev/test figures from different configurations — flagged by the source summaries; re-check Table 5 before quoting side by side)
- **CSC-SQL, BIRD dev at n=64 samples (Table 1):** Qwen2.5-Coder-3B-Instruct + GRPO: self-consistency 60.60 -> corrective self-consistency 63.34; XiYanSQL-QwenCoder-3B + GRPO: 61.30 -> 65.28; (for scale: 7B + GRPO CSC reaches 69.19) — [arXiv full text v2](https://arxiv.org/html/2505.13271v2)
- **MTIR-SQL (4B, multi-turn tool-integrated GRPO variant): 64.4% BIRD dev, 84.6% Spider dev** — [arXiv abs](https://arxiv.org/abs/2510.25510)
- **Think2SQL (Qwen2.5-Coder bases): BIRD dev weighted-average scores 0.254 (0.5B), 0.442 (1.5B), 0.500 (3B)** — reported from Table 1 of v5; note the table metric is a weighted average over difficulty levels and may be the QATCH-style score rather than plain EX, so treat as not directly comparable to the EX numbers above — [arXiv full text v5](https://arxiv.org/html/2504.15077v5)
- **BAR-SQL (Qwen3-1.7B + SFT + GRPO): no BIRD or Spider results reported at all; 91.48% average on its own Ent-SQL-Bench** — [arXiv full text](https://arxiv.org/html/2601.10318)
- **Arctic-Text2SQL-R1 has no model below 7B** (7B = 68.9 BIRD dev, for reference as the small-model ceiling among RL systems) — [arXiv full text v2](https://arxiv.org/html/2505.20315v2)
- **OmniSQL has no public model below 7B** (sizes 7B/14B/32B) — [HF seeklhy/OmniSQL-7B](https://huggingface.co/seeklhy/OmniSQL-7B)
- Reference points for non-RL small models: XiYanSQL-QwenCoder-3B-2502 without GRPO reaches 58.87 BIRD dev with 64-sample self-consistency (CSC-SQL Table 1) — [arXiv full text v2](https://arxiv.org/html/2505.13271v2)

### Inferences
- A realistic BIRD dev EX band for a well-trained <=4B RL model in 2025–2026 is mid-50s to mid-60s single-pass, high-60s with heavy sampling/self-consistency; ~67% (SLM-SQL 1.5B with n=64 voting) is the best small-model dev figure found.
- Since every small RL model answers 100% of questions, a selective-prediction variant trading ~10–20% coverage for lower risk would have no published baseline to beat at <=4B — the comparison set would have to be constructed from these full-coverage numbers.

### Gaps
- Think2SQL's plain EX (not weighted/QATCH) per small size, and the Qwen3-4B TMLR-version numbers, were not retrieved — check [arXiv v4/v5 HTML](https://arxiv.org/html/2504.15077v4) tables directly.
- CogniSQL-R1-Zero's exact BIRD dev EX (7B anyway, so above the 4B cutoff) is not on the abstract page.
- BIRD leaderboard (bird-bench.github.io) was not fetched directly to enumerate current small-model entries; the leaderboard page is linked here for follow-up: [bird-bench.github.io](https://bird-bench.github.io).
