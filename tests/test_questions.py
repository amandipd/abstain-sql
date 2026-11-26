"""The answerable-question generator (plan step 1.4) keeps every rule the README sets for a record.

The tests build their own filled copies: a few generated designs plus one small
hand-written design, each filled 5 times with random rows that keep the README's
guarantees. They never read data/instances, which git ignores.
"""

import datetime
import json
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from data.generator import questions as qg
from data.generator import schemas as gen
from data.generator.copies import copy_paths, run_on_copies

RECORD_FIELDS = [
    "question_id",
    "schema_id",
    "family",
    "topic",
    "kind",
    "question",
    "gold_sql",
    "shape",
    "difficulty",
    "columns_used",
    "hidden_columns",
    "decline_subtype",
    "decline_reason",
    "source_question_id",
    "gold_row_counts",
]
FIRST = ["Ava", "Liam", "Mei", "Omar", "Priya", "Lucas", "Hana", "Noah"]
LAST = ["Garcia", "Santos", "Petrov", "Nguyen", "Okafor", "Smith", "Kim", "Rossi"]


def _values(col: dict, n: int, parent_rows: dict[str, int], rng: random.Random) -> list:
    role = col["role"]
    if role == "id":
        return list(range(1, n + 1))
    if role == "fk":
        return [rng.randint(1, parent_rows[col["references"]["table"]]) for _ in range(n)]
    if role == "category":
        out = list(col["values"]) + [rng.choice(col["values"]) for _ in range(n - len(col["values"]))]
        rng.shuffle(out)
        return out
    if role == "number":
        lo, hi = col["min"], col["max"]
        if col["type"] == "integer":
            return [rng.randint(int(lo), int(hi)) for _ in range(n)]
        return [round(rng.uniform(lo, hi), 2) for _ in range(n)]
    if role == "date":
        lo = datetime.date.fromisoformat(col["min"]).toordinal()
        hi = datetime.date.fromisoformat(col["max"]).toordinal()
        return [datetime.date.fromordinal(rng.randint(lo, hi)).isoformat() for _ in range(n)]
    if role == "flag":
        return [rng.randint(0, 1) for _ in range(n)]
    if role == "first_name":
        return [rng.choice(FIRST) for _ in range(n)]
    if role == "last_name":
        return [rng.choice(LAST) for _ in range(n)]
    return [f"{col.get('kind', 'item')} {i}" for i in range(1, n + 1)]  # entity_name, unique


def fill(schema: dict, folder: Path, seed: int = 0) -> list[Path]:
    """5 copies with 25-60 rows a table: every category value present, numbers and dates in range."""
    folder.mkdir(parents=True, exist_ok=True)
    for k, path in enumerate(copy_paths(folder), start=1):
        rng = random.Random(f"{seed}:{schema['schema_id']}:{k}")
        con = sqlite3.connect(path)
        con.executescript(gen.to_sql(schema))
        rows: dict[str, int] = {}
        for t in schema["tables"]:
            widest = max((len(c["values"]) for c in t["columns"] if c["role"] == "category"), default=0)
            n = max(rng.randint(25, 60), widest)
            columns = [_values(c, n, rows, rng) for c in t["columns"]]
            marks = ", ".join("?" for _ in t["columns"])
            con.executemany(f"INSERT INTO {t['name']} VALUES ({marks})", list(zip(*columns, strict=True)))
            rows[t["name"]] = n
        con.commit()
        con.close()
    return copy_paths(folder)


HOTELS = {
    "schema_id": "test__hotels__01",
    "family": "test",
    "topic": "hotels",
    "tables": [
        {
            "name": "hotels",
            "primary_key": "hotel_id",
            "columns": [
                {"name": "hotel_id", "type": "integer", "role": "id"},
                {"name": "name", "type": "text", "role": "entity_name", "kind": "hotel"},
                {"name": "city", "type": "text", "role": "category", "values": ["Austin", "Denver", "Boston"]},
                {"name": "stars", "type": "integer", "role": "number", "min": 1, "max": 5},
            ],
        },
        {
            "name": "bookings",
            "primary_key": "booking_id",
            "columns": [
                {"name": "booking_id", "type": "integer", "role": "id"},
                {
                    "name": "hotel_id",
                    "type": "integer",
                    "role": "fk",
                    "references": {"table": "hotels", "column": "hotel_id"},
                },
                {"name": "channel", "type": "text", "role": "category", "values": ["web", "phone", "agent"]},
                {"name": "total_cost", "type": "real", "role": "number", "min": 50, "max": 5000},
                {"name": "booked_date", "type": "date", "role": "date", "min": "2020-01-01", "max": "2024-12-31"},
            ],
        },
    ],
}


@pytest.fixture(scope="module")
def designs(tmp_path_factory):
    root = tmp_path_factory.mktemp("designs")
    schemas = gen.generate(families=["healthcare", "finance"], per_topic=1, seed=0)
    return [(s, fill(s, root / s["schema_id"])) for s in schemas]


@pytest.fixture(scope="module")
def records(designs):
    return {s["schema_id"]: qg.answerable_questions(s, paths) for s, paths in designs}


def _all(records):
    return [r for rs in records.values() for r in rs]


def test_record_format_matches_readme(designs, records):
    for schema, _ in designs:
        rs = records[schema["schema_id"]]
        assert rs, schema["schema_id"]
        for i, r in enumerate(rs, start=1):
            assert list(r) == RECORD_FIELDS
            assert r["question_id"] == f"{schema['schema_id']}__a{i:03d}"
            assert (r["schema_id"], r["family"], r["topic"]) == (schema["schema_id"], schema["family"], schema["topic"])
            assert r["kind"] == "answerable"
            assert r["shape"] in qg.SHAPES
            assert r["difficulty"] in ("easy", "medium", "hard")
            assert r["hidden_columns"] == []
            assert r["decline_subtype"] is r["decline_reason"] is r["source_question_id"] is None
            assert len(r["gold_row_counts"]) == 5 and min(r["gold_row_counts"]) > 0
            assert r["question"].endswith(("?", ".")) and "SELECT" not in r["question"]
            json.dumps(r)  # plain JSON


def test_twenty_per_design_and_every_shape_appears(records):
    assert all(len(rs) == qg.DEFAULT_TARGET for rs in records.values())
    shapes = Counter(r["shape"] for r in _all(records))
    assert set(shapes) == set(qg.SHAPES), shapes
    assert set(Counter(r["difficulty"] for r in _all(records))) == {"easy", "medium", "hard"}


def test_same_seed_same_records_and_another_seed_differs(designs, records):
    schema, paths = designs[0]
    assert qg.answerable_questions(schema, paths) == records[schema["schema_id"]]
    assert qg.answerable_questions(schema, paths, seed=1) != records[schema["schema_id"]]


def test_gold_queries_return_rows_on_every_copy(designs, records):
    for schema, paths in designs:
        for r in records[schema["schema_id"]]:
            results = run_on_copies(paths, r["gold_sql"])
            assert all(x.status == "ok" for x in results), (r["gold_sql"], results)
            assert [len(x.rows) for x in results] == r["gold_row_counts"]


def test_no_duplicate_sql_or_question_within_a_design(records):
    for rs in records.values():
        assert len({r["gold_sql"] for r in rs}) == len(rs)
        assert len({r["question"] for r in rs}) == len(rs)


def test_no_ties_in_order_limit_answers(designs, records):
    checked = 0
    for schema, paths in designs:
        for r in records[schema["schema_id"]]:
            m = re.search(r" ORDER BY (.+) (DESC|ASC) LIMIT (\d+)$", r["gold_sql"])
            if not m:
                continue
            sql, n = r["gold_sql"], int(m.group(3))
            tie_sql = f"SELECT {m.group(1)} {sql[sql.index(' FROM ') + 1 : sql.rindex(' LIMIT ')]} LIMIT {n + 1}"
            for x in run_on_copies(paths, tie_sql):
                values = [row[0] for row in x.rows]
                assert len(values) == len(set(values)), (r["gold_sql"], values)
            assert r["shape"] == "order_limit" and r["gold_row_counts"] == [n] * 5
            checked += 1
    assert checked


def test_a_tie_at_the_cutoff_is_rejected(tmp_path):
    paths = fill(HOTELS, tmp_path / "ties")
    for path in paths:  # give the top 2 hotels the same, highest stars
        con = sqlite3.connect(path)
        con.execute("UPDATE hotels SET stars = 4 WHERE stars = 5")
        con.execute("UPDATE hotels SET stars = 5 WHERE hotel_id IN (1, 2)")
        con.commit()
        con.close()
    design = qg._Design(HOTELS)
    tables = {t.name: t for t in design.tables}

    def top(n):
        sql = f"SELECT name FROM hotels ORDER BY stars DESC LIMIT {n}"
        tie = f"SELECT stars FROM hotels ORDER BY stars DESC LIMIT {n + 1}"
        return qg._Cand("order_limit", "top", "hotels", sql, "?", [], limit=n, tie_sql=tie)

    assert qg._validate(top(1), paths, tables, {}) is None  # 5, 5: tie at the cutoff
    assert qg._validate(top(2), paths, tables, {}) is None  # 5, 5, 4...: tie inside the top 2
    single = qg._Cand("aggregate", "agg", "hotels", "SELECT MAX(stars) FROM hotels", "?", [])
    assert qg._validate(single, paths, tables, {}) == [1] * 5


def test_columns_used_on_known_queries():
    design = qg._Design(HOTELS)
    joins = qg._join(design, random.Random(0))
    by_template = {c.template: c for c in joins}
    group = by_template["group_count"]
    assert group.sql == (
        "SELECT hotels.city, COUNT(*) FROM bookings JOIN hotels ON bookings.hotel_id = hotels.hotel_id "
        "GROUP BY hotels.city"
    )
    assert qg._record(HOTELS, 1, group, [3] * 5)["columns_used"] == [{"table": "hotels", "column": "city"}]
    agg = by_template["agg"]
    used = qg._record(HOTELS, 1, agg, [1] * 5)["columns_used"]
    assert used[0] == {"table": "bookings", "column": "total_cost"}
    assert used[1]["table"] == "hotels" and used[1]["column"] in ("city", "stars")
    assert {c.difficulty for c in joins if c.template == "group_count"} == {"hard"}
    assert {c.difficulty for c in joins if c.template == "count"} == {"medium"}


def test_columns_used_matches_the_columns_each_query_reads(designs, records):
    for schema, _ in designs:
        tables = {t["name"]: t for t in schema["tables"]}
        for r in records[schema["schema_id"]]:
            tokens = set(re.findall(r"[a-z_][a-z0-9_]*", re.sub(r"'[^']*'", "", r["gold_sql"])))
            read = {
                c["name"]
                for name, t in tables.items()
                if name in tokens
                for c in t["columns"]
                if c["role"] not in ("id", "fk") and c["name"] in tokens
            }
            assert {c["column"] for c in r["columns_used"]} == read, r["gold_sql"]
            assert all(c["table"] in tokens for c in r["columns_used"]), r["gold_sql"]


def test_literals_come_from_the_schema(designs, records):
    for schema, _ in designs:
        values = {v for t in schema["tables"] for c in t["columns"] for v in c.get("values") or []}
        for r in records[schema["schema_id"]]:
            for literal in re.findall(r"'([^']*)'", r["gold_sql"]):
                is_date_part = re.fullmatch(r"%Y|\d{4}(-\d{2}-\d{2})?", literal)
                assert is_date_part or literal in values, (literal, r["gold_sql"])


def test_plain_words():
    assert qg.words("land_area_km2") == "land area (square km)"
    assert qg.words("avg_weight_g") == "average weight (grams)"
    assert qg.words("dissolved_oxygen_mg_l") == "dissolved oxygen (mg/L)"
    assert qg.plural_words("facility type") == "facility types"
    assert qg.plural_words("status") == "statuses"
    assert qg.plural_words("category") == "categories"
    assert qg._nouns("facilities") == ("facility", "facilities", "facility")
    assert qg._nouns("staff") == ("staff member", "staff members", "staff")
    assert qg._nouns("inventory") == ("inventory record", "inventory records", "inventory")
    assert qg._some("page views") == "page views" and qg._some("capacity") == "a capacity"
    assert qg._is("number of tracks") == "is" and qg._is("vesting years") == "are"
    assert qg.fmt_number(1300000, {"name": "population"}) == "1,300,000"
    assert qg.fmt_number(1990, {"name": "founded_year"}) == "1990"
    assert qg.fmt_date("2021-03-01") == "March 1, 2021"


def test_thresholds_are_round_and_inside_the_range():
    for col in [
        {"name": "capacity", "type": "integer", "min": 20, "max": 1500},
        {"name": "population", "type": "integer", "min": 20000, "max": 2500000},
        {"name": "commission", "type": "real", "min": 0.1, "max": 0.2},
        {"name": "model_year", "type": "integer", "min": 2008, "max": 2025},
        {"name": "fee", "type": "real", "min": 0, "max": 5000},
    ]:
        for quantile in (0.3, 0.5, 0.7):
            v = qg.threshold(col, quantile)
            assert col["min"] < v < col["max"], (col, v)
    assert qg.threshold({"name": "capacity", "type": "integer", "min": 20, "max": 1500}, 0.5) == 800


def test_year_cutoffs_are_round_plausible_years():
    def year(lo, hi, name="founded_year"):
        return {"name": name, "type": "integer", "min": lo, "max": hi}

    for quantile in (0.05, 0.3, 0.5, 0.7, 0.95):
        made = qg.threshold(year(-3000, 2020, "year_made"), quantile)
        assert 1000 < made < 2020 and made % 100 == 0, made  # a century from 1000 on, never "less than 10"
        founded = qg.threshold(year(1850, 2022), quantile)
        assert 1850 < founded < 2022 and founded % 10 == 0, founded
        model = qg.threshold(year(2008, 2025, "model_year"), quantile)
        assert 2008 < model < 2025 and model % 5 == 0, model
    # A rate is not a calendar year: it gets an ordinary round number, written with a comma.
    rate = {"name": "carloads_per_year", "type": "integer", "min": 50, "max": 250000}
    assert not qg.is_calendar_year(rate) and qg.fmt_number(4000, rate) == "4,000"


def test_without_removes_exactly_one_condition():
    sql = "SELECT COUNT(*) FROM t WHERE a = 1 AND b > 5"
    assert qg.without(sql, "a = 1") == "SELECT COUNT(*) FROM t WHERE b > 5"
    assert qg.without(sql, "b > 5") == "SELECT COUNT(*) FROM t WHERE a = 1"
    nested = "SELECT COUNT(*) FROM p WHERE pid IN (SELECT pid FROM c WHERE x > 50)"
    assert qg.without(nested, "x > 50") == "SELECT COUNT(*) FROM p WHERE pid IN (SELECT pid FROM c)"
    assert qg.without("SELECT c FROM t GROUP BY c HAVING COUNT(*) > 5", "COUNT(*) > 5") == "SELECT c FROM t GROUP BY c"
    with pytest.raises(ValueError):
        qg.without("SELECT COUNT(*) FROM t WHERE x > 50", "x > 5")  # a prefix of the literal is not the condition


def test_a_condition_that_never_matters_is_rejected(tmp_path):
    paths = fill(HOTELS, tmp_path / "idle")
    design = qg._Design(HOTELS)
    tables = {t.name: t for t in design.tables}
    # Every hotel has at least 1 star, so the filter changes nothing on any copy.
    idle = qg._Cand(
        "aggregate", "distinct", "hotels", "SELECT COUNT(DISTINCT city) FROM hotels WHERE stars >= 1", "?", []
    )
    idle.conditions = ("stars >= 1",)
    assert qg._check(idle, paths, tables, {}) == (None, "condition")
    assert qg._check(idle, paths, tables, {}, conditions=False)[1] == "idle"
    real = qg._Cand("filter", "count", "hotels", "SELECT COUNT(*) FROM hotels WHERE city = 'Austin'", "?", [])
    real.conditions = ("city = 'Austin'",)
    assert qg._check(real, paths, tables, {})[1] == "ok"


def _conditions(sql: str) -> list[str]:
    """Each WHERE predicate and HAVING clause, parsed from the SQL text independently of the generator."""
    out = []
    for m in re.finditer(r" WHERE ", sql):
        depth, i = 0, m.end()
        while i < len(sql) and not (depth == 0 and (sql[i] == ")" or sql.startswith((" GROUP BY ", " ORDER BY "), i))):
            depth += {"(": 1, ")": -1}.get(sql[i], 0)
            i = sql.index("'", i + 1) + 1 if sql[i] == "'" else i + 1
        text = re.sub(r"BETWEEN ('[^']*') AND", r"BETWEEN \1 &AND&", sql[m.end() : i])  # keep BETWEEN whole
        parts, depth, buf = [], 0, ""
        for piece in re.split(r"( AND |[()])", text):
            depth += {"(": 1, ")": -1}.get(piece, 0)
            if piece == " AND " and depth == 0:
                parts.append(buf)
                buf = ""
            else:
                buf += piece
        out += [p.replace("&AND&", "AND") for p in [*parts, buf]]
    if " HAVING " in sql:
        out.append(sql.split(" HAVING ", 1)[1])
    return out


def test_every_condition_changes_the_answer_on_2_copies(designs, records):
    checked = 0
    for schema, paths in designs:
        for r in records[schema["schema_id"]]:
            base = run_on_copies(paths, r["gold_sql"])
            for condition in _conditions(r["gold_sql"]):
                loose = run_on_copies(paths, qg.without(r["gold_sql"], condition))
                changed = sum(qg._answer(a.rows) != qg._answer(b.rows) for a, b in zip(base, loose, strict=True))
                assert changed >= qg.MIN_COPIES_CHANGED, (r["gold_sql"], condition)
                checked += 1
    assert checked
