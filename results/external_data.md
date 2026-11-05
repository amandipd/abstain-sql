# External evaluation data (plan step 0.4)

I downloaded BIRD dev and TrustSQL into `data/external/`, ran every gold query against its database, and counted every question. Everything the plan needs is there, and nothing needs a PhysioNet application. Checked 2026-11-04.

`uv run python data/download_external.py` rebuilds the folder. It fetches about 470 MB and stops if any file's SHA-256 fingerprint (a code that changes if even one byte changes) differs from the one pinned in the script. The folder is gitignored, so this repo never republishes the data.

## What I downloaded

| Dataset | Where it comes from | Download size | License |
|---|---|---|---|
| BIRD dev, original | `dev.zip`, linked from [bird-bench.github.io](https://bird-bench.github.io/); last modified 2024-06-29 | 346 MB | CC BY-SA 4.0 (share-alike) |
| BIRD dev, cleaned 2025-11-06 | [birdsql/bird_sql_dev_20251106](https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106) on Hugging Face | 0.9 MB | CC BY-SA 4.0 (share-alike) |
| TrustSQL | the authors' Google Drive link in [glee4810/TrustSQL](https://github.com/glee4810/TrustSQL) | 121 MB | CC BY 4.0 (attribution only) |

The cleaned BIRD file holds only questions, hints, and gold SQL. It runs on the same 11 databases as the original package, and every one of its 1,534 question IDs matches an original question.

## The PhysioNet question: not needed

The plan expected TrustSQL's hospital-records portion (EHRSQL) to need MIMIC-III and eICU, which sit behind PhysioNet's credentialed-access agreement. It does not. TrustSQL's EHRSQL questions run on one database, `mimic_iv.sqlite` (34 MB), which holds 94 patients. EHRSQL's README says its newest data uses the [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/), a 100-patient sample that PhysioNet lists as open access: "Anyone can access the files" under the Open Database License (ODbL) v1.0. The full MIMIC-IV holds hundreds of thousands of patients, so a 94-patient file is the demo, not the credentialed release.

## BIRD dev, broken down

BIRD dev has 11 databases holding 75 tables, 798 columns, and about 3.9 million rows: 1.5 GB once unpacked. Each of its 1,534 questions belongs to one database, from 64 to 191 per database. None is labeled as one to decline, so BIRD measures coverage and accuracy on answerable questions only. The cleaned version keeps every question on the same database.

| Database | Tables | Rows | Size | Questions |
|---|---|---|---|---|
| california_schools | 3 | 29,941 | 11.1 MB | 89 |
| card_games | 6 | 803,445 | 261.8 MB | 191 |
| codebase_community | 8 | 740,646 | 481.4 MB | 186 |
| debit_card_specializing | 5 | 423,050 | 34.6 MB | 64 |
| european_football_2 | 7 | 222,796 | 597.8 MB | 129 |
| financial | 8 | 1,079,680 | 71.3 MB | 106 |
| formula_1 | 13 | 514,287 | 22.4 MB | 174 |
| student_club | 8 | 42,511 | 2.6 MB | 158 |
| superhero | 10 | 10,614 | 0.2 MB | 129 |
| thrombosis_prediction | 3 | 15,952 | 7.3 MB | 163 |
| toxicology | 4 | 49,813 | 2.7 MB | 145 |
| All 11 | 75 | 3,932,735 | 1,493 MB | 1,534 |

BIRD labels each question simple, moderate, or challenging. The cleaned version relabels 86 questions, every one of them up to challenging: 65 from simple and 21 from moderate.

| Difficulty | Original | Cleaned |
|---|---|---|
| Simple | 925 | 860 |
| Moderate | 464 | 443 |
| Challenging | 145 | 231 |
| All | 1,534 | 1,534 |

1,386 of the original questions (90%) come with a hint in the "evidence" field; 1,377 of the cleaned ones do.

## TrustSQL, broken down

TrustSQL's test set runs on 13 databases: 1 each for ATIS, Advising, and EHRSQL, and 10 for Spider.

| Source | Test databases | Tables | Rows | Size |
|---|---|---|---|---|
| ATIS (flights) | 1 | 25 | 162,241 | 4.4 MB |
| Advising (university courses) | 1 | 12 | 332,596 | 27.3 MB |
| EHRSQL (hospital records) | 1 | 17 | 308,942 | 34.3 MB |
| Spider (mixed topics) | 10 | 44, from 2 to 11 each | 6,052 | 0.3 MB |

ATIS, Advising, and EHRSQL each use their one database in every split. Spider works differently. The download ships 166 Spider databases; training uses 140, and validation and test use 10 each. The paper says TrustSQL built its Spider validation and test sets by splitting Spider's development set in half, since Spider's own test set is not public. No test database appears in training, so the Spider portion tests whether a model handles databases it has never seen.

| Source | Train | Validation | Test |
|---|---|---|---|
| ATIS | 1,114 | 489 | 952 |
| Advising | 1,170 | 533 | 1,066 |
| EHRSQL | 4,674 | 931 | 1,868 |
| Spider | 7,000 | 507 | 1,054 |
| All 4 | 13,958 | 2,460 | 4,940 |

The train and validation files hold answerable questions only. The test set matches the split table in TrustSQL's README exactly: 4,940 questions, half answerable and half that a model should decline.

| Source | Answerable | Should decline | Total |
|---|---|---|---|
| ATIS (flights) | 476 | 476 | 952 |
| Advising (university courses) | 533 | 533 | 1,066 |
| EHRSQL (hospital records) | 934 | 934 | 1,868 |
| Spider (mixed topics) | 527 | 527 | 1,054 |
| All 4 | 2,470 | 2,470 | 4,940 |

The research found 2 TrustSQL totals that disagreed: 21,356 in the paper and about 23,318 from the GitHub table. The standard files sum to 21,358 (13,958 train, 2,460 validation, 4,940 test), within 2 of the paper. The standard train and validation files hold answerable questions only; declines appear only in test. The validation declines that the README's table lists may live in the `*_valid_combined_subset.json` files, which I did not check, since the plan uses only the test set.

TrustSQL labels each answerable test question easy, medium, or hard. EHRSQL uses only easy and hard.

| Source | Easy | Medium | Hard | Total |
|---|---|---|---|---|
| ATIS | 39 | 306 | 131 | 476 |
| Advising | 86 | 364 | 83 | 533 |
| EHRSQL | 122 | 0 | 812 | 934 |
| Spider | 279 | 158 | 90 | 527 |
| All 4 | 526 | 828 | 1,116 | 2,470 |

TrustSQL also tags each answerable question "seen SQL", when its query structure also appears in TrustSQL's training split, or "unseen SQL", when the structure appears only in validation or test. About 3 in 4 answerable test questions are seen in ATIS (363 of 476), Advising (391 of 533), and EHRSQL (696 of 934); all 527 Spider questions count as unseen, because their databases are new. I never train on TrustSQL, so for our model every question is unseen. The tag matters only when comparing against published TrustSQL scores, where a model trained on TrustSQL's training split has an advantage on the seen ones.

The questions to decline come in 5 kinds:

- **column-surface, column-related, column-unrelated:** the question asks about a column the database lacks. TrustSQL splits these into 3 variants named surface, related, and unrelated.
- **ext-know:** the question needs knowledge outside the database, such as "What are the risks of untreated hypertension?"
- **non-sql:** the request is work SQL cannot do, such as "Cluster cities based on their population and geographical location."

| Source | Missing column (3 kinds) | Needs outside knowledge | Not a SQL task | Total |
|---|---|---|---|---|
| ATIS | 285 | 92 | 99 | 476 |
| Advising | 319 | 104 | 110 | 533 |
| EHRSQL | 561 | 182 | 191 | 934 |
| Spider | 300 | 127 | 100 | 527 |
| All 4 | 1,465 | 505 | 500 | 2,470 |

Missing-column questions make up 59% of the declines. Each of the 3 variants holds about a fifth of the declines, the same as ext-know and non-sql, so the 5 kinds are roughly balanced.

## Every gold query runs

I ran all 7,008 gold queries read-only, with a 10-second limit each.

| Set | Gold queries | Ran | Hit the 10 s limit | Errors |
|---|---|---|---|---|
| BIRD original | 1,534 | 1,531 | 3 | 0 |
| BIRD cleaned | 1,534 | 1,527 | 7 | 0 |
| TrustSQL ATIS | 476 | 475 | 1 | 0 |
| TrustSQL Advising | 533 | 533 | 0 | 0 |
| TrustSQL EHRSQL | 934 | 934 | 0 | 0 |
| TrustSQL Spider | 527 | 527 | 0 | 0 |

No query errored. The 11 that hit the limit are slow, not broken.

## What this changes for later steps

**The checker must fix EHRSQL's clock.** 292 of the 934 answerable EHRSQL gold queries say `current_time`, and TrustSQL's scorer (`scoring_utils.py`) replaces it, along with `'now'`, with the fixed moment `'2100-12-31 23:59:00'` before running anything. Without that swap, SQLite reads the real clock: 85 of the 934 gold queries return no rows, and a junk query that returns nothing would match them. With the swap, 0 return empty. Our checker must apply the same swap to gold and predicted SQL alike.

**External evaluation needs a longer time limit.** TrustSQL's scorer allows 60 seconds per query; plan step 2.1 sets 2 to 5 seconds for the training reward. At 5 seconds, at least 11 correct gold queries would time out on the laptop, and the model would be marked wrong on those questions no matter what it wrote. Evaluation on BIRD and TrustSQL should use 60 seconds.

**BIRD's cleaned version changes a lot.** It rewrites 451 of the 1,534 gold queries (29%), 182 question texts, and 381 hints. The published small-model scores the plan compares against were measured on the original. I recommend reporting both: the original for comparison with those papers, the cleaned version as the more accurate number. That is new decision D8 in the plan. The BIRD team also says it will remove the oracle hints (the "evidence" field) in favor of an interactive track, which bears on decision D4.

**TrustSQL's decline kinds differ slightly from the plan's wording.** The plan names TrustSQL's "out-of-domain" type; the files call it "ext-know". The plan's out-of-scope category matches TrustSQL's non-sql and ext-know kinds together, and the plan's ambiguous category has no TrustSQL counterpart.
