# How commercial BI copilots handle wrong or uncertain answers in NL-to-SQL (as of Nov 2026)

## Key Question 1: What do each product's official docs say about accuracy safeguards, semantic models, verified queries, or declining to answer?

### Takeaway
All four vendors lean on the same core strategy: a human-curated semantic layer plus a library of human-verified example queries, with the human review happening *before* deployment (curation) and *after* answers (user inspection of SQL) — not via calibrated runtime confidence. Abstention exists but is shallow: Genie asks clarifying questions and says "can't answer based on the provided information"; Cortex Analyst returns suggested alternative questions instead of SQL for ambiguous asks; ThoughtSpot and Hex docs describe no explicit decline behavior at all.

### Cited Findings

**Snowflake Cortex Analyst**
- Semantic models/views are the headline accuracy mechanism: generic text-to-SQL "struggles with schemas alone," and semantic views improve accuracy by providing rich metadata, business logic, and predefined relationships/join paths — [Cortex Analyst docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst)
- Verified Query Repository (VQR): question+SQL pairs stored in the semantic model YAML (`verified_queries`, with optional `verified_at`/`verified_by`); docs say it can "improve the accuracy and trustworthiness of results." Review is manual and author-driven: generate, inspect SQL and results, edit, then "Save as verified query." Snowsight also surfaces "Verified Query Suggestions" from user behavior — [Verified Query Repository docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst/verified-query-repository)
- Declining/ambiguity: the REST API returns `suggestion` content "only... if the user question was ambiguous"; `suggestion` and `sql` are mutually exclusive, so for ambiguous questions it returns a list of answerable alternative questions instead of SQL. Response also carries `warnings` and a `question_category` field — [Cortex Analyst REST API docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst/rest-api)
- "Confidence" exists but is not a probability: the `confidence` object is returned only with SQL content and contains `verified_query_used` (the VQR entry matched, or `null`) — i.e., confidence = provenance, not a calibrated score — [Cortex Analyst REST API docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst/rest-api)
- Scope limitation stated as a known limitation, not a refusal feature: "Cortex Analyst is limited to answering questions that can be resolved with SQL" — [Cortex Analyst docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst)
- Offline evaluation: docs describe running verified queries against Cortex Analyst to measure SQL accuracy, "identify areas for improvement, track regressions over time" — [Cortex Analyst docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst)
- No clarifying-question behavior documented (multi-turn is only follow-up rephrasing) — [Cortex Analyst docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst)

**Databricks AI/BI Genie** (now organized around "Genie Agents" / "Genie One")
- Grounding/governance: "Every answer is grounded in your organization's data and governed through Unity Catalog"; agents are configured with datasets, sample queries, instructions, metrics, business rules, and "trusted assets"; "Genie does not query tables or Unity Catalog functions that are not attached" — [Genie overview](https://docs.databricks.com/aws/en/genie/); [Trusted assets page](https://docs.databricks.com/aws/en/genie/trusted-assets)
- Clarifying questions and declining: Genie "might respond by asking a follow-up clarification question," or explain that it "can't answer based on the provided information"; docs list "Why were sales low this quarter?" as a question Genie cannot answer (no interpretation/recommendations) — [Talk to Genie docs](https://docs.databricks.com/aws/genie/talk-to-genie)
- Trusted-asset labeling at answer time: a Chat-mode response backed by a trusted asset means an editor pre-wrote the query/function; users can expand details — but docs warn "Genie can occasionally match a question to an incorrect function or query" — [Talk to Genie docs](https://docs.databricks.com/aws/genie/talk-to-genie)
- Honest hallucination warning + user verification: "Genie can sometimes respond with hallucinations or errors"; users should expand the Analysis/thinking steps, "Show code" to inspect SQL, and use "Request review" / "Fix it" to flag problems to agent editors; Genie is explicitly non-deterministic — [Talk to Genie docs](https://docs.databricks.com/aws/genie/talk-to-genie)
- Human curation loop: authors review/accept/reject auto-suggested popular workspace queries ("Review Suggested Queries"); agents themselves can be tagged "Certified" or "Deprecated" via a governed tag; "Expect to iterate on your Genie Agent based on testing and usage" — [Trusted assets page](https://docs.databricks.com/aws/en/genie/trusted-assets)
- Built-in benchmark tooling: authors write benchmark question sets with gold SQL; responses auto-rated Good when result sets match (numeric agreement to 4 significant digits counts); empty results/errors rated Bad; per-run accuracy tracked on an Evaluations tab — [Genie benchmarks docs](https://docs.databricks.com/aws/en/genie/benchmarks)

**ThoughtSpot Spotter**
- Architecture claim: Spotter uses LLMs to generate ThoughtSpot's relational search tokens mapped to its semantic layer, not direct text-to-SQL; the launch blog claims that if the tokens are correct, the compiled SQL is deterministic/correct (vendor claim, no independent audit) — [Introducing Spotter blog](https://www.thoughtspot.com/blog/introducing-spotter-ai-analyst)
- Verifiability by non-SQL users: each answer shows a simplified color-coded query (calculations green, groupings blue, filters gray); follow-up answers summarize what changed from the previous step; docs tell users to "always check the search data query" because value-to-column mapping can be wrong (example: "southwest" not counted in "southern region") — [Validate Spotter docs](https://docs.thoughtspot.com/cloud/latest/spotter-validate)
- Human-in-the-loop feedback: vendor materials describe an HITL feedback/coaching system for improving answers; docs have troubleshooting pages for "unexpected answers" — [search results over ThoughtSpot docs/blog](https://www.thoughtspot.com/blog/introducing-spotter-ai-analyst) (UNVERIFIED in detail: the specific feedback-system docs page was not fetched; the `spotter-error` page 404'd)
- The trust-center page makes only general claims ("patented semantics layer, which grounds answers to your data"; "AI should be accurate, transparent, and trustworthy") — no accuracy numbers, no verified-content or decline mechanism described — [ThoughtSpot Trust Center](https://www.thoughtspot.com/trust/enterprise-grade-ai)
- No documented decline/abstain behavior, confidence display, or clarifying-question feature found in the pages fetched — [Validate Spotter docs](https://docs.thoughtspot.com/cloud/latest/spotter-validate)

**Hex (Magic / Notebook agent / Threads)**
- Safeguard = audience + curation, not abstention: the Notebook Agent "is intended to be used by technical users who can audit the SQL and code that the agent suggests"; Threads (for non-technical users) "highly prioritizes endorsed and semantically modeled data" — [Hex AI overview](https://learn.hex.tech/docs/getting-started/ai-overview)
- Semantic models: authored in Hex's Modeling Workbench or synced from Cube, dbt MetricFlow, or Snowflake Semantic Views; a Modeling Agent (Managers/Admins only) generates and edits them — [Hex semantic models docs](https://learn.hex.tech/docs/connect-to-data/semantic-models/intro-to-semantic-models); [Hex AI overview](https://learn.hex.tech/docs/getting-started/ai-overview)
- Context Studio (Admins/Managers): monitor AI usage, review agent reasoning, and adjust workspace rules, semantic models, and endorsements to "improve agent accuracy and trust" — [Hex AI overview](https://learn.hex.tech/docs/getting-started/ai-overview)
- No clarifying questions, declining to answer, or confidence display documented — [Hex AI overview](https://learn.hex.tech/docs/getting-started/ai-overview)

### Inferences
- The industry-standard reliability stack is: (1) curated semantic layer, (2) verified/trusted query library with exact or fuzzy matching, (3) show-the-SQL-and-let-humans-check, (4) author-side eval loops. Runtime uncertainty estimation is essentially absent — "confidence" in the one API that has the word (Snowflake) just reports whether a verified query was reused.
- Abstention is treated as a UX fallback (suggest other questions, ask a clarifying question), not a measured, tunable behavior with published operating points.

### Gaps
- ThoughtSpot's current "verified answers"/coaching feature docs could not be fetched directly (404 on `spotter-error`; feature appears on a Feb 2026 roadmap slide as "Trustable & Verifiable," suggesting it is in flux). UNVERIFIED beyond blog/search snippets.
- Databricks "Genie One" is new (2026) and its quality-tuning sub-pages ("Tune Genie Agent quality") were not fetched in full.

## Key Question 2: Do any publish accuracy numbers or abstention behavior?

### Takeaway
Only Snowflake publishes a number (~90%+ SQL accuracy), and it is from internal evaluation sets, blog-only, and conditional on a good semantic model; none of the four publishes abstention/refusal rates or calibration data.

### Cited Findings
- Snowflake blog: ~90% or higher accuracy "across customer evaluations as well as in our benchmark tests," based on "one of our internal evaluation sets" — [Snowflake blog](https://www.snowflake.com/en/blog/cortex-analyst-ai-self-service-analytics/); engineering blog credits semantic models for "90%+ SQL accuracy on real-world use cases" — [Snowflake engineering blog](https://www.snowflake.com/engineering-blog/snowflake-cortex-analyst-behind-the-scenes)
- Secondary analysis (Atlan) notes Snowflake's reference docs state no quantified accuracy figure (claims live in blog/marketing) and cites a Snowflake BIRD-SQL result of 57% → 78% from adding a semantic model to the identical LLM — [Atlan analysis](https://atlan.com/know/snowflake/cortex-analyst-vs-text-to-sql/) (UNVERIFIED against a primary Snowflake source)
- Databricks publishes no product accuracy number; instead it ships customer-side measurement (per-space benchmark accuracy on the Evaluations tab) and openly warns of hallucinations — [Genie benchmarks docs](https://docs.databricks.com/aws/en/genie/benchmarks); [Talk to Genie docs](https://docs.databricks.com/aws/genie/talk-to-genie)
- ThoughtSpot publishes no accuracy number; its claim is architectural ("if relational search tokens are correct, SQL is 100% accurate" — vendor claim) and release notes attribute accuracy gains to model upgrades (default answer-generation model upgraded to Claude Opus 4.5 in Feb 2026) — [Introducing Spotter blog](https://www.thoughtspot.com/blog/introducing-spotter-ai-analyst); [ThoughtSpot release notes via search](https://docs.thoughtspot.com/cloud/26.4.0.cl/notes-weekly) (release-note detail UNVERIFIED, from search snippet)
- Hex publishes no accuracy numbers in its docs — [Hex AI overview](https://learn.hex.tech/docs/getting-started/ai-overview)
- No vendor documents an abstention *rate*, a refusal threshold, or calibration curves anywhere found in this research. Documented abstention behavior is qualitative only: Cortex Analyst's ambiguous→suggestions path and Genie's clarify/"can't answer" path — [Cortex Analyst REST API docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst/rest-api); [Talk to Genie docs](https://docs.databricks.com/aws/genie/talk-to-genie)

### Inferences
- Vendor accuracy claims are unauditable: internal test sets, no abstention accounting, and no error-cost framing. A customer cannot compare "how often does it answer wrongly vs. decline" across products today.

### Gaps
- No independent third-party benchmark of these four products' accuracy or abstention behavior was found (plausibly none exists publicly, since semantic-model setup is customer-specific).

## Key Question 3: Does any open benchmark today measure abstention/reliability for text-to-SQL in a product-relevant way?

### Takeaway
Academic abstention benchmarks exist (TrustSQL, PRACTIQ, Ambrosia+/ABISS, Query Carefully), but all are built on academic schemas without a semantic layer, verified-query library, or BI-chat setting — none measures abstention under the conditions these products actually operate in, so a product-relevant abstention benchmark + small reference model targets a real, open gap.

### Cited Findings
- TrustSQL (KAIST, arXiv 2403.15879): defines reliability as answering feasible questions and abstaining on infeasible ones, with penalty-based scoring; covers Spider, ATIS, Advising, EHRSQL; finds no method consistently beats a naive always-abstain-style baseline and "achieving high scores under severe penalties requires significant effort" — [TrustSQL](https://arxiv.org/abs/2403.15879)
- PRACTIQ (arXiv 2410.11076, 2024): conversational text-to-SQL dataset with ambiguous and unanswerable questions; reports state-of-the-art systems struggle to handle them — [PRACTIQ](https://arxiv.org/abs/2410.11076)
- Query Carefully (arXiv 2512.21345, late 2025): extends ScienceBenchmark with unanswerable questions over a biomedical DB; column-ambiguity questions were classified at max 0.3 accuracy — [Query Carefully](https://arxiv.org/pdf/2512.21345)
- "To RAG or Not to RAG" (2025): proposes Ambrosia+ with a unified 8-category taxonomy of ambiguous/unanswerable questions plus a multi-turn interactive simulation environment (ABISS) — [paper](https://www.researchgate.net/publication/401528766_To_RAG_or_Not_to_RAG_That_Is_the_Question_Effective_Text-to-SQL_Generation_Under_Ambiguity) (UNVERIFIED: abstract-level only, not fetched in full)
- Earlier work: AmbiQT (3000+ questions each with ≥2 plausible SQL readings), AMBROSIA (scope/attachment ambiguity), TriageSQL (question-intent triage incl. ambiguous/improper), "Know What I Don't Know" (finds ~20% of real-world questions problematic) — [AmbiQT/ambiguity benchmark](https://www.alphaxiv.org/abs/2310.13659); [Know What I Don't Know](https://arxiv.org/html/2212.08902v2)
- None of these uses a semantic layer / verified-query context, and none evaluates commercial BI copilots; vendor-side evaluation is private (Snowflake internal sets) or customer-local (Databricks per-space benchmarks) — [TrustSQL](https://arxiv.org/abs/2403.15879); [Genie benchmarks docs](https://docs.databricks.com/aws/en/genie/benchmarks)

### Inferences
- The gap assessment: abstention benchmarks per se are NOT novel (TrustSQL/PRACTIQ/Ambrosia+ cover feasibility, ambiguity, multi-turn). What is genuinely missing and product-relevant: (1) abstention measured *with* the mitigations vendors actually deploy (semantic model + verified queries in context) — i.e., does curation reduce the need to abstain, and does the model know when it's off the curated path; (2) published, comparable abstention/selective-accuracy operating points (risk-coverage curves) that vendors conspicuously do not provide; (3) a small open reference model demonstrating calibrated abstention, since no product exposes a real confidence signal (Snowflake's `confidence` field is just verified-query provenance).
- Positioning line this supports: commercial BI copilots manage wrongness through human curation before the question and human inspection after the answer; an open benchmark + reference model would fill the unoccupied middle — measured, calibrated behavior at answer time.

### Gaps
- Whether any vendor measures abstention internally is unknowable from public docs.
- Did not exhaustively confirm there is no 2026 industry-led abstention benchmark (one extended search; Ambrosia+/ABISS and Query Carefully were the newest found).
