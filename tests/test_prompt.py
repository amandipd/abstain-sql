"""The prompt format (plan step 1.7) shows the right schema and stays fixed between training and evaluation."""

import copy
import json
import sqlite3

import pytest

from data.generator import prompt
from data.generator import schemas as gen

# A tiny hand-built schema, so the exact-text test below does not depend on the generator.
TINY = {
    "schema_id": "tiny",
    "tables": [
        {
            "name": "stores",
            "primary_key": "store_id",
            "columns": [
                {"name": "store_id", "type": "integer", "role": "id"},
                {"name": "city", "type": "text", "role": "category", "values": ["Austin", "Boston"]},
            ],
        },
        {
            "name": "sales",
            "primary_key": "sale_id",
            "columns": [
                {"name": "sale_id", "type": "integer", "role": "id"},
                {
                    "name": "store_id",
                    "type": "integer",
                    "role": "fk",
                    "references": {"table": "stores", "column": "store_id"},
                },
                {"name": "amount", "type": "real", "role": "number", "min": 1, "max": 500},
                {"name": "sold_on", "type": "date", "role": "date", "min": "2024-01-01", "max": "2024-12-31"},
            ],
        },
    ],
}


@pytest.fixture(scope="module")
def generated():
    return gen.generate(per_topic=1, seed=0)


def columns_where(schema, is_key):
    return [
        {"table": t["name"], "column": c["name"]}
        for t in schema["tables"]
        for c in t["columns"]
        if (c["role"] in prompt.KEY_ROLES) == is_key
    ]


def non_key_columns(schema):
    return columns_where(schema, is_key=False)


def key_columns(schema):
    return columns_where(schema, is_key=True)


def loaded(sql):
    """Load CREATE TABLE text into SQLite and read back every table's columns and links."""
    con = sqlite3.connect(":memory:")
    con.executescript(sql)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    columns = {t: [(r[1], r[2], r[5]) for r in con.execute(f"PRAGMA table_info({t})")] for t in tables}
    links = {t: sorted(r[2:5] for r in con.execute(f"PRAGMA foreign_key_list({t})")) for t in tables}
    con.close()
    return columns, links


def test_hiding_nothing_shows_the_full_schema(generated):
    for s in generated:
        assert prompt.render_schema(s) == gen.to_sql(s), s["schema_id"]


def test_each_hidden_column_disappears_and_nothing_else_changes(generated):
    for s in generated[::10]:  # one schema per family
        full_columns, full_links = loaded(gen.to_sql(s))
        for h in non_key_columns(s):
            columns, links = loaded(prompt.render_schema(s, [h]))
            expected = copy.deepcopy(full_columns)
            expected[h["table"]] = [c for c in expected[h["table"]] if c[0] != h["column"]]
            assert columns == expected, (s["schema_id"], h)
            assert links == full_links, (s["schema_id"], h)


def test_several_hidden_columns_disappear_together(generated):
    s = generated[0]
    hidden = [non_key_columns(s)[0], non_key_columns(s)[-1]]
    columns, _ = loaded(prompt.render_schema(s, hidden))
    for h in hidden:
        assert h["column"] not in [c[0] for c in columns[h["table"]]]
    assert sum(map(len, columns.values())) == sum(len(t["columns"]) for t in s["tables"]) - len(hidden)


def test_rendering_does_not_change_the_schema(generated):
    s = generated[0]
    before = copy.deepcopy(s)
    prompt.render_schema(s, non_key_columns(s)[:3])
    prompt.build_messages(s, {"question": "q", "hidden_columns": non_key_columns(s)[:3]})
    assert s == before


def test_hiding_a_key_raises(generated):
    for s in generated[::10]:
        for h in key_columns(s):
            with pytest.raises(ValueError, match="key"):
                prompt.render_schema(s, [h])


def test_hiding_a_missing_column_raises():
    for h in [
        {"table": "stores", "column": "refund_date"},
        {"table": "refunds", "column": "city"},
        {"table": "sales", "column": "city"},  # exists, but in another table
    ]:
        with pytest.raises(ValueError, match="no such column"):
            prompt.render_schema(TINY, [h])


def test_messages_hold_the_schema_question_and_contract(generated):
    s = generated[0]
    record = {"question": "How many rows are there?", "hidden_columns": []}
    system, user = prompt.build_messages(s, record)
    assert system == {"role": "system", "content": prompt.SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert gen.to_sql(s).rstrip() in user["content"]
    assert "Question: How many rows are there?" in user["content"]
    assert user["content"].endswith(prompt.OUTPUT_CONTRACT)
    assert "abstain" in system["content"]


def test_messages_honor_the_records_hidden_columns():
    record = {"question": "What was the total amount sold?", "hidden_columns": [{"table": "sales", "column": "amount"}]}
    user = prompt.build_messages(TINY, record)[1]["content"]
    assert prompt.render_schema(TINY, record["hidden_columns"]).rstrip() in user
    assert "amount REAL" not in user
    assert "sold_on DATE" in user


def test_missing_or_null_hidden_columns_hide_nothing():
    full = prompt.build_messages(TINY, {"question": "q", "hidden_columns": []})
    assert prompt.build_messages(TINY, {"question": "q", "hidden_columns": None}) == full
    assert prompt.build_messages(TINY, {"question": "q"}) == full


def test_the_exact_prompt_text():
    """Any wording change fails here on purpose: training and evaluation must see the same template."""
    record = {"question": "Which city has the most sales?", "hidden_columns": [{"table": "stores", "column": "city"}]}
    system, user = prompt.build_messages(TINY, record)
    assert system["content"] == (
        "You write SQLite queries that answer questions about a database. "
        "The user gives the database schema as CREATE TABLE statements, then a question. "
        "If the question cannot be answered from the schema, is ambiguous, "
        "or is not something a database query can answer, abstain instead of guessing. "
        "You may reason briefly first, then end your reply with exactly one JSON object."
    )
    assert user["content"] == (
        "Database schema:\n"
        "\n"
        "CREATE TABLE stores (\n"
        "  store_id INTEGER PRIMARY KEY\n"
        ");\n"
        "\n"
        "CREATE TABLE sales (\n"
        "  sale_id INTEGER PRIMARY KEY,\n"
        "  store_id INTEGER,\n"
        "  amount REAL,\n"
        "  sold_on DATE,\n"
        "  FOREIGN KEY (store_id) REFERENCES stores (store_id)\n"
        ");\n"
        "\n"
        "Question: Which city has the most sales?\n"
        "\n"
        "End your reply with exactly one JSON object, and write nothing after it.\n"
        "If you answer, the object holds your query:\n"
        '{"sql": "SELECT name FROM customers WHERE city = \'Austin\'", "abstain": false, "reason": ""}\n'
        "If you abstain, the object holds a short reason instead:\n"
        '{"sql": null, "abstain": true, "reason": "the schema has no column for refund dates"}\n'
        "Both examples come from a different database."
    )


def test_baseline_has_no_abstain_option(generated):
    s = generated[0]
    record = {"question": "How many rows are there?", "hidden_columns": []}
    system, user = prompt.build_messages(s, record, allow_abstain=False)
    assert system["content"] == prompt.BASELINE_SYSTEM_PROMPT
    assert user["content"].endswith(prompt.BASELINE_OUTPUT_CONTRACT)
    for text in (system["content"], user["content"]):
        assert "abstain" not in text.lower()
        assert "null" not in text
    assert "Always answer with a SQL query." in system["content"]
    # Same prompt minus the abstain option: the schema and question are unchanged.
    abstain_user = prompt.build_messages(s, record)[1]["content"]
    assert user["content"].removesuffix(prompt.BASELINE_OUTPUT_CONTRACT) == abstain_user.removesuffix(
        prompt.OUTPUT_CONTRACT
    )


def test_output_is_deterministic(generated):
    for s in generated[::10]:
        record = {"question": "How many rows are there?", "hidden_columns": non_key_columns(s)[:2]}
        for allow in (True, False):
            first = prompt.build_messages(s, record, allow_abstain=allow)
            again = prompt.build_messages(copy.deepcopy(s), copy.deepcopy(record), allow_abstain=allow)
            assert first == again


def contract_objects(contract):
    return [json.loads(line) for line in contract.splitlines() if line.startswith("{")]


def test_contract_examples_are_valid_json():
    answer, abstain = contract_objects(prompt.OUTPUT_CONTRACT)
    assert answer == {"sql": "SELECT name FROM customers WHERE city = 'Austin'", "abstain": False, "reason": ""}
    assert abstain == {"sql": None, "abstain": True, "reason": "the schema has no column for refund dates"}
    (baseline,) = contract_objects(prompt.BASELINE_OUTPUT_CONTRACT)
    assert baseline == {"sql": "SELECT name FROM customers WHERE city = 'Austin'"}


def test_system_prompts_cannot_flip_smollm3s_reasoning_mode():
    # SmolLM3-3B's chat template reads these markers out of the system message and overrides enable_thinking.
    for text in (prompt.SYSTEM_PROMPT, prompt.BASELINE_SYSTEM_PROMPT):
        for marker in ("/think", "/no_think", "/system_override"):
            assert marker not in text


def test_thinking_switch_is_always_explicit():
    assert prompt.chat_template_kwargs(True) == {"enable_thinking": True}
    assert prompt.chat_template_kwargs(False) == {"enable_thinking": False}
    for bad in (None, 0, 1, "false"):
        with pytest.raises(TypeError):
            prompt.chat_template_kwargs(bad)
