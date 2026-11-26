"""The prompt every model sees, in training and in evaluation (plan step 1.7).

One template, with no randomness: the same schema and record always give the
same messages, so the prompt a model trains on is exactly the prompt it is
scored on. The system message sets the task; the user message holds the schema
as CREATE TABLE statements (minus any hidden columns), the question, and the
output contract. The model may reason first, then must end with one JSON object
(decision D2):

    {"sql": "SELECT name FROM customers WHERE city = 'Austin'", "abstain": false, "reason": ""}
    {"sql": null, "abstain": true, "reason": "the schema has no column for refund dates"}

The accuracy-only baseline (plan, "The reward") gets the same prompt minus the
abstain option: it is told to always answer with SQL, and its JSON object has
only the "sql" key, so a parser should read a missing "abstain" as false.

Passing a system message also replaces SmolLM3-3B's built-in system prompt,
which pushes it toward long reasoning. The thinking switch goes to the chat
template separately, through chat_template_kwargs():

    tokenizer.apply_chat_template(build_messages(schema, record), tokenize=False,
                                  add_generation_prompt=True, **chat_template_kwargs(False))
"""

from __future__ import annotations

from collections.abc import Sequence

from data.generator.schemas import to_sql

KEY_ROLES = {"id", "fk"}

_TASK = (
    "You write SQLite queries that answer questions about a database. "
    "The user gives the database schema as CREATE TABLE statements, then a question. "
)
_ABSTAIN_RULE = (
    "If the question cannot be answered from the schema, is ambiguous, "
    "or is not something a database query can answer, abstain instead of guessing. "
)
_ALWAYS_ANSWER_RULE = "Always answer with a SQL query. "
_FINISH = "You may reason briefly first, then end your reply with exactly one JSON object."

SYSTEM_PROMPT = _TASK + _ABSTAIN_RULE + _FINISH
BASELINE_SYSTEM_PROMPT = _TASK + _ALWAYS_ANSWER_RULE + _FINISH

ANSWER_EXAMPLE = '{"sql": "SELECT name FROM customers WHERE city = \'Austin\'", "abstain": false, "reason": ""}'
ABSTAIN_EXAMPLE = '{"sql": null, "abstain": true, "reason": "the schema has no column for refund dates"}'
BASELINE_EXAMPLE = '{"sql": "SELECT name FROM customers WHERE city = \'Austin\'"}'

OUTPUT_CONTRACT = (
    "End your reply with exactly one JSON object, and write nothing after it.\n"
    "If you answer, the object holds your query:\n"
    f"{ANSWER_EXAMPLE}\n"
    "If you abstain, the object holds a short reason instead:\n"
    f"{ABSTAIN_EXAMPLE}\n"
    "Both examples come from a different database."
)
BASELINE_OUTPUT_CONTRACT = (
    "End your reply with exactly one JSON object, and write nothing after it.\n"
    "The object holds your query:\n"
    f"{BASELINE_EXAMPLE}\n"
    "The example comes from a different database."
)


def render_schema(schema: dict, hidden_columns: Sequence[dict] = ()) -> str:
    """The CREATE TABLE text the model sees, with each {"table", "column"} in hidden_columns removed.

    Raises ValueError if a hidden column does not exist or is a key: hiding a
    primary or foreign key would break the schema rather than remove a fact.
    """
    columns = {(t["name"], c["name"]): c for t in schema["tables"] for c in t["columns"]}
    hidden = set()
    for h in hidden_columns:
        where = (h.get("table"), h.get("column"))
        if where not in columns:
            raise ValueError(f"cannot hide {where[0]}.{where[1]}: no such column")
        if columns[where]["role"] in KEY_ROLES:
            raise ValueError(f"cannot hide {where[0]}.{where[1]}: it is a key")
        hidden.add(where)

    # Build a filtered copy; the caller's schema is never changed.
    shown = {
        **schema,
        "tables": [
            {**t, "columns": [c for c in t["columns"] if (t["name"], c["name"]) not in hidden]}
            for t in schema["tables"]
        ],
    }
    return to_sql(shown)


def build_messages(schema: dict, record: dict, allow_abstain: bool = True) -> list[dict]:
    """The chat messages for one question: a system message, then a user message.

    The user message holds the schema (minus record["hidden_columns"]), the
    question, and the output contract. allow_abstain=False gives the
    accuracy-only baseline, which drops the abstain option.
    """
    schema_text = render_schema(schema, record.get("hidden_columns") or ())
    contract = OUTPUT_CONTRACT if allow_abstain else BASELINE_OUTPUT_CONTRACT
    user = f"Database schema:\n\n{schema_text.rstrip()}\n\nQuestion: {record['question']}\n\n{contract}"
    system = SYSTEM_PROMPT if allow_abstain else BASELINE_SYSTEM_PROMPT
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_template_kwargs(thinking: bool) -> dict:
    """The thinking switch, to pass to tokenizer.apply_chat_template(**kwargs).

    Always set it explicitly, never leave it to a model's default: the defaults
    disagree (SmolLM3-3B thinks by default; Qwen3.5-2B and Gemma 4 E2B do not;
    see results/models.md). All 3 chat templates read `enable_thinking`. A
    non-bool raises TypeError, so None cannot slip through and fall back to a
    default.
    """
    if not isinstance(thinking, bool):
        raise TypeError(f"thinking must be True or False, got {thinking!r}")
    return {"enable_thinking": thinking}
