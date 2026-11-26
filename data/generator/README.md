# The synthetic data generator (plan Phase 1)

One command builds everything below from the table lists in `templates/`:

    uv run python -m data.generator.build

It is deterministic: the same seed always rebuilds the same files. Everything it writes goes to `data/instances/`, which git ignores.

## Pipeline

| Step | Module | Reads | Writes |
|---|---|---|---|
| 1.2 Designs | `schemas.py` | `families.yaml`, `templates/*.yaml` | `<schema_id>/schema.json`, `schema.sql` |
| 1.3 Filled copies | `fill.py` | `schema.json` | `<schema_id>/copy_1.sqlite` … `copy_5.sqlite`, `copies.json` |
| 1.4 Answerable questions | `questions.py` | `schema.json`, the copies | records with a gold query |
| 1.5 Questions to decline | `declines.py` | `schema.json`, the answerable records | unanswerable, ambiguous, and out-of-scope records |
| 1.6 The mix | `build.py` | all records | `questions.jsonl`, `build_summary.json` |
| 1.7 Prompt format | `prompt.py` | `schema.json`, one record | the chat messages the model sees |

`copies.py` holds the shared helper that runs a query read-only on a filled copy, with a time limit.

## Filled copies

Each design gets 5 copies in its folder, `copy_1.sqlite` to `copy_5.sqlite`. Every copy has exactly the tables and columns in `schema.sql`, filled with different made-up rows:

- Primary keys run 1 to n; every link points at a row that exists. There are no NULLs.
- Every value of a category column appears at least once in every copy whenever the table has at least as many rows as the category has values, so a filter on any listed value finds rows on every copy.
- Numbers stay inside their column's `min` and `max`; dates are `YYYY-MM-DD` inside their range; flags are 0 or 1.
- Tables hold tens to a few hundred rows, and the counts differ between copies.

`copies.json` records each copy's file, seed, and row count per table.

## Question records

`questions.jsonl` holds one JSON object per line. Every record has every field below; fields that do not apply are `null` or `[]`.

| Field | Meaning |
|---|---|
| `question_id` | Unique: `<schema_id>__<prefix><number>`, prefix `a` answerable, `u` unanswerable, `m` ambiguous, `o` out of scope |
| `schema_id`, `family`, `topic` | Which design the question belongs to |
| `kind` | `answerable`, `unanswerable`, `ambiguous`, or `out_of_scope` |
| `question` | The plain-English question the model sees |
| `gold_sql` | The reference query; `null` for every kind except `answerable` |
| `shape` | Answerable: `filter`, `aggregate`, `group_by`, `join`, `order_limit`, `subquery`, or `date_range` |
| `difficulty` | Answerable: `easy`, `medium`, or `hard` |
| `columns_used` | Answerable: the non-key columns the gold query reads, as `{"table", "column"}` |
| `hidden_columns` | Unanswerable: columns removed from the schema the model sees, as `{"table", "column"}` |
| `decline_subtype` | Unanswerable: `hidden_column` or `missing_attribute`; ambiguous: `undefined_metric` or `vague_reference`; out of scope: `action`, `forecast`, `analysis`, or `outside_knowledge` |
| `decline_reason` | One line on why the model should decline; for analysis only, never scored |
| `source_question_id` | Unanswerable questions made by hiding a column: the answerable question they came from |
| `gold_row_counts` | Answerable: rows the gold query returns on each copy, all above 0 |

An answerable record is kept only if its gold query runs without error, inside the time limit, and returns at least one row on every copy. An `ORDER BY ... LIMIT` query is kept only if no tie straddles the cutoff on any copy, so its answer is unambiguous.

## What the model sees

`prompt.py` renders the schema as `CREATE TABLE` statements, minus any `hidden_columns`, followed by the question and the output contract: the model may reason first, then must end with one JSON object, either `{"sql": "...", "abstain": false, "reason": ""}` or `{"sql": null, "abstain": true, "reason": "..."}`. The thinking switch is always set explicitly, through the chat template's `enable_thinking` argument.
