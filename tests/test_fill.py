"""The filler (plan step 1.3) keeps every guarantee data/generator/README.md promises."""

import datetime
import random
import sqlite3

import pytest

from data.generator import fill, names
from data.generator import schemas as gen


@pytest.fixture(scope="module")
def filled(tmp_path_factory):
    """One design per topic of 2 families, filled 5 times into a temporary folder."""
    root = tmp_path_factory.mktemp("instances")
    designs = gen.generate(["healthcare", "media_and_entertainment"], per_topic=1, seed=0)
    for s in designs:
        fill.fill_schema(s, root / s["schema_id"], copies=5, seed=0)
    return root, designs


def _rows(path, table):
    con = sqlite3.connect(path)
    try:
        return con.execute(f"select * from {table} order by 1").fetchall()
    finally:
        con.close()


def _copies(root, s):
    return [root / s["schema_id"] / f"copy_{k}.sqlite" for k in range(1, 6)]


def test_every_copy_has_every_table_with_rows(filled):
    root, designs = filled
    for s in designs:
        for path in _copies(root, s):
            for t in s["tables"]:
                rows = _rows(path, t["name"])
                assert 1 <= len(rows) <= 300, (s["schema_id"], t["name"], len(rows))


def test_links_point_at_real_rows_and_nothing_is_null(filled):
    root, designs = filled
    for s in designs:
        for path in _copies(root, s):
            con = sqlite3.connect(path)
            assert con.execute("pragma foreign_key_check").fetchall() == []
            for t in s["tables"]:
                for c in t["columns"]:
                    nulls = con.execute(f"select count(*) from {t['name']} where {c['name']} is null").fetchone()[0]
                    assert nulls == 0, (s["schema_id"], t["name"], c["name"])
            con.close()


def test_every_category_value_appears_in_every_copy(filled):
    root, designs = filled
    for s in designs:
        for path in _copies(root, s):
            con = sqlite3.connect(path)
            for t in s["tables"]:
                n = con.execute(f"select count(*) from {t['name']}").fetchone()[0]
                for c in t["columns"]:
                    if c["role"] == "category" and n >= len(c["values"]):
                        seen = {r[0] for r in con.execute(f"select distinct {c['name']} from {t['name']}")}
                        assert seen == set(c["values"]), (s["schema_id"], t["name"], c["name"])
            con.close()


def test_values_stay_in_their_ranges(filled):
    root, designs = filled
    for s in designs:
        con = sqlite3.connect(_copies(root, s)[0])
        for t in s["tables"]:
            for c in t["columns"]:
                vals = [r[0] for r in con.execute(f"select {c['name']} from {t['name']}")]
                if c["role"] == "number":
                    assert all(c["min"] <= v <= c["max"] for v in vals), (t["name"], c["name"])
                elif c["role"] == "date":
                    assert all(c["min"] <= v <= c["max"] for v in vals), (t["name"], c["name"])
                    assert all(datetime.date.fromisoformat(v) for v in vals)
                elif c["role"] == "flag":
                    assert set(vals) <= {0, 1}
                elif c["role"] == "category":
                    assert set(vals) <= set(c["values"])
        con.close()


def test_made_up_names_never_repeat_within_a_table(filled):
    root, designs = filled
    for s in designs:
        con = sqlite3.connect(_copies(root, s)[0])
        for t in s["tables"]:
            for c in t["columns"]:
                if c["role"] == "entity_name" and "kind" in c:
                    vals = [r[0] for r in con.execute(f"select {c['name']} from {t['name']}")]
                    assert len(vals) == len(set(vals)), (t["name"], c["name"])
        con.close()


def test_the_copies_differ(filled):
    root, designs = filled
    for s in designs:
        first = s["tables"][-1]["name"]
        contents = [tuple(_rows(p, first)) for p in _copies(root, s)]
        assert len(set(contents)) == 5, s["schema_id"]


def test_same_seed_same_data_other_seed_different():
    s = gen.generate(["finance"], per_topic=1, seed=0)[0]
    a = fill.fill_copy(s, random.Random("0:x:copy1"))
    b = fill.fill_copy(s, random.Random("0:x:copy1"))
    c = fill.fill_copy(s, random.Random("1:x:copy1"))
    assert a == b
    assert a != c


def test_every_name_kind_in_the_table_lists_gets_a_name():
    rng = random.Random(0)
    for family in gen.families_with_templates():
        for template in gen.load_templates(family).values():
            for t in template["tables"]:
                for c in t["columns"]:
                    if c["role"] == "entity_name" and "kind" in c:
                        made = names.unique_names(c["kind"], 50, rng)
                        assert len(set(made)) == 50 and all(made), c["kind"]
