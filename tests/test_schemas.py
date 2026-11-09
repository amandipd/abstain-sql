"""The schema generator (plan step 1.2) keeps every rule the plan sets for a schema."""

import copy
import sqlite3

import pytest

from data.generator import schemas as gen


@pytest.fixture(scope="module")
def generated():
    return gen.generate(seed=0)


def test_two_schemas_per_topic(generated):
    families = gen.families_with_templates()
    assert families, "no family has a template yet"
    assert len(generated) == 2 * 5 * len(families)


def test_three_to_eight_tables(generated):
    for s in generated:
        assert gen.MIN_TABLES <= len(s["tables"]) <= gen.MAX_TABLES, s["schema_id"]


def test_text_integer_real_and_date_columns(generated):
    for s in generated:
        assert gen.column_types(s) >= gen.REQUIRED_TYPES, s["schema_id"]


def test_every_link_points_at_an_included_primary_key(generated):
    for s in generated:
        keys = {t["name"]: t["primary_key"] for t in s["tables"]}
        for t in s["tables"]:
            for c in t["columns"]:
                if c["role"] == "fk":
                    ref = c["references"]
                    assert keys.get(ref["table"]) == ref["column"], (s["schema_id"], t["name"], c["name"])


def test_names_are_unique(generated):
    for s in generated:
        names = [t["name"] for t in s["tables"]]
        assert len(names) == len(set(names)), s["schema_id"]
        for t in s["tables"]:
            cols = [c["name"] for c in t["columns"]]
            assert len(cols) == len(set(cols)), (s["schema_id"], t["name"])


def test_sql_creates_every_table_and_link(generated):
    for s in generated:
        con = sqlite3.connect(":memory:")
        con.executescript(gen.to_sql(s))
        created = {r[0] for r in con.execute("select name from sqlite_master where type = 'table'")}
        assert created == {t["name"] for t in s["tables"]}, s["schema_id"]
        for t in s["tables"]:
            links = {(r[2], r[3], r[4]) for r in con.execute(f"pragma foreign_key_list({t['name']})")}
            expected = {
                (c["references"]["table"], c["name"], c["references"]["column"])
                for c in t["columns"] if c["role"] == "fk"
            }
            assert links == expected, (s["schema_id"], t["name"])


def test_schemas_for_one_topic_use_different_tables(generated):
    by_topic = {}
    for s in generated:
        by_topic.setdefault(s["topic"], []).append(frozenset(t["template_key"] for t in s["tables"]))
    for topic, sets in by_topic.items():
        assert len(sets) == len(set(sets)), topic


def test_same_seed_same_schemas_other_seed_different(generated):
    assert gen.generate(seed=0) == generated
    assert gen.generate(seed=1) != generated


def _tiny_template():
    return {"tables": [
        {"key": "a", "names": ["a"], "id": ["a_id"], "columns": [
            {"names": ["label"], "type": "text", "role": "category", "values": ["x", "y"]}]},
        {"key": "b", "names": ["b"], "id": ["b_id"], "columns": [
            {"role": "fk", "ref": "a", "type": "integer"}]},
        {"key": "c", "names": ["c"], "id": ["c_id"], "columns": [
            {"role": "fk", "ref": "b", "type": "integer"}]},
    ]}


def test_a_valid_template_passes():
    gen.validate_template("tiny", _tiny_template())


def test_link_to_a_missing_table_is_rejected():
    t = _tiny_template()
    t["tables"][1]["columns"][0]["ref"] = "missing"
    with pytest.raises(gen.TemplateError, match="unknown table"):
        gen.validate_template("tiny", t)


def test_cycle_of_links_is_rejected():
    t = _tiny_template()
    t["tables"][0]["columns"].append({"role": "fk", "ref": "c", "type": "integer"})
    with pytest.raises(gen.TemplateError, match="cycle"):
        gen.validate_template("tiny", t)


def test_yaml_boolean_category_value_is_rejected():
    t = copy.deepcopy(_tiny_template())
    t["tables"][0]["columns"][0]["values"] = [True, False]  # what YAML makes of unquoted yes/no
    with pytest.raises(gen.TemplateError, match="strings"):
        gen.validate_template("tiny", t)
