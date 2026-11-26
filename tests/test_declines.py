"""The decline generator (plan step 1.5) writes README-format records, in the right amounts, that truly decline."""

import re
from collections import Counter

import pytest

from data.generator import declines
from data.generator import schemas as gen
from data.generator.prompt import render_schema
from data.generator.questions import words

README_FIELDS = (
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
)
SHAPES = ("filter", "aggregate", "group_by", "join", "order_limit", "subquery", "date_range")


def fake_answerable(schema: dict, n: int = 20) -> list[dict]:
    """Hand-built answerable records: each asks about one non-key column, filtered by a category value."""
    columns = [(t, c) for t in schema["tables"] for c in t["columns"] if c["role"] not in declines.KEY_ROLES]
    records = []
    for i in range(n):
        t, c = columns[i % len(columns)]
        used = [{"table": t["name"], "column": c["name"]}]
        question = f"What is the average {words(c['name'])} of the {words(t['name'])}"
        cats = [x for x in t["columns"] if x["role"] == "category" and x is not c]
        if cats:
            cat = cats[i % len(cats)]
            used.append({"table": t["name"], "column": cat["name"]})
            question += f" whose {words(cat['name'])} is '{cat['values'][i % len(cat['values'])]}'"
        records.append(
            {
                "question_id": f"{schema['schema_id']}__a{i + 1:03d}",
                "question": f"{question} (number {i + 1})?",
                "shape": SHAPES[i % len(SHAPES)],
                "difficulty": ("easy", "medium", "hard")[i % 3],
                "columns_used": used,
            }
        )
    return records


TINY = {
    "schema_id": "tiny__clinics__01",
    "family": "healthcare",
    "topic": "healthcare",
    "tables": [
        {
            "name": "clinics",
            "template_key": "clinics",
            "primary_key": "clinic_id",
            "columns": [
                {"name": "clinic_id", "type": "integer", "role": "id"},
                {"name": "name", "type": "text", "role": "entity_name", "kind": "clinic"},
                {"name": "clinic_type", "type": "text", "role": "category", "values": ["urgent care", "primary care"]},
                {"name": "exam_rooms", "type": "integer", "role": "number", "min": 2, "max": 40},
            ],
        },
        {
            "name": "patients",
            "template_key": "patients",
            "primary_key": "patient_id",
            "columns": [
                {"name": "patient_id", "type": "integer", "role": "id"},
                {
                    "name": "clinic_id",
                    "type": "integer",
                    "role": "fk",
                    "references": {"table": "clinics", "column": "clinic_id"},
                },
                {"name": "first_name", "type": "text", "role": "first_name"},
                {"name": "last_name", "type": "text", "role": "last_name"},
                {"name": "sex", "type": "text", "role": "category", "values": ["female", "male"]},
                {"name": "birth_date", "type": "date", "role": "date", "min": "1940-01-01", "max": "2010-12-31"},
            ],
        },
        {
            "name": "visits",
            "template_key": "visits",
            "primary_key": "visit_id",
            "columns": [
                {"name": "visit_id", "type": "integer", "role": "id"},
                {
                    "name": "patient_id",
                    "type": "integer",
                    "role": "fk",
                    "references": {"table": "patients", "column": "patient_id"},
                },
                {"name": "visit_date", "type": "date", "role": "date", "min": "2020-01-01", "max": "2024-12-31"},
                {"name": "total_cost", "type": "real", "role": "number", "min": 50, "max": 5000},
                {"name": "wait_minutes", "type": "integer", "role": "number", "min": 0, "max": 180},
            ],
        },
    ],
}


def _with_column(schema: dict, table: str, column: dict) -> dict:
    tables = [{**t, "columns": t["columns"] + [column]} if t["name"] == table else t for t in schema["tables"]]
    return {**schema, "tables": tables}


@pytest.fixture(scope="module")
def designs():
    return gen.generate(seed=0)


@pytest.fixture(scope="module")
def generated(designs):
    out = []
    for s in designs:
        answerable = fake_answerable(s)
        out.append((s, answerable, declines.decline_questions(s, answerable, seed=0)))
    return out


def _all(generated):
    return [r for _, _, records in generated for r in records]


def test_every_record_has_exactly_the_readme_fields(generated):
    for s, answerable, records in generated:
        by_id = {a["question_id"]: a for a in answerable}
        for r in records:
            assert tuple(r) == README_FIELDS, r["question_id"]
            assert (r["schema_id"], r["family"], r["topic"]) == (s["schema_id"], s["family"], s["topic"])
            assert r["kind"] in declines.KINDS
            assert r["decline_subtype"] in declines.SUBTYPES[r["kind"]]
            assert r["gold_sql"] is None
            assert r["columns_used"] == [] and r["gold_row_counts"] == []
            reason = r["decline_reason"]
            assert isinstance(reason, str) and reason.strip() and "\n" not in reason
            assert r["question"].strip() and "\n" not in r["question"]
            if r["decline_subtype"] == "hidden_column":
                source = by_id[r["source_question_id"]]
                assert (r["shape"], r["difficulty"]) == (source["shape"], source["difficulty"])
                assert r["question"] == source["question"]
            else:
                assert r["shape"] is None and r["difficulty"] is None
                assert r["source_question_id"] is None and r["hidden_columns"] == []


def test_ids_number_each_kind_from_001(generated):
    for s, _, records in generated:
        for kind, prefix in declines.PREFIX.items():
            ids = [r["question_id"] for r in records if r["kind"] == kind]
            assert ids == [f"{s['schema_id']}__{prefix}{i:03d}" for i in range(1, len(ids) + 1)]


def test_every_subtype_appears_across_designs(generated):
    designs_with = Counter()
    for _, _, records in generated:
        for subtype in {r["decline_subtype"] for r in records}:
            designs_with[subtype] += 1
    for kind, subtypes in declines.SUBTYPES.items():
        for subtype in subtypes:
            assert designs_with[subtype] >= 0.25 * len(generated), (kind, subtype, designs_with[subtype])


def test_about_043_declines_per_answerable_split_evenly_across_kinds(generated):
    for s, answerable, records in generated:
        assert len(records) == round(0.43 * len(answerable)), s["schema_id"]
        kinds = Counter(r["kind"] for r in records)
        assert max(kinds[k] for k in declines.KINDS) - min(kinds[k] for k in declines.KINDS) <= 1, s["schema_id"]
    total_answerable = sum(len(a) for _, a, _ in generated)
    share = len(_all(generated)) / (len(_all(generated)) + total_answerable)
    assert 0.28 <= share <= 0.32


@pytest.mark.parametrize("n", [0, 1, 3, 7, 12, 40])
def test_amount_follows_the_number_of_answerable_records(designs, n):
    s = designs[0]
    assert len(declines.decline_questions(s, fake_answerable(s, n), seed=0)) == round(0.43 * n)


def test_subtypes_are_spread_within_each_kind(generated):
    counts = Counter(r["decline_subtype"] for r in _all(generated))
    for subtypes in declines.SUBTYPES.values():
        shares = [counts[st] for st in subtypes]
        assert min(shares) >= 0.6 * max(shares), dict(counts)


def test_hidden_columns_are_real_non_key_columns_from_the_source(generated):
    for s, answerable, records in generated:
        by_id = {a["question_id"]: a for a in answerable}
        roles = {(t["name"], c["name"]): c["role"] for t in s["tables"] for c in t["columns"]}
        for r in records:
            if r["decline_subtype"] != "hidden_column":
                continue
            assert len(r["hidden_columns"]) == 1
            hidden = r["hidden_columns"][0]
            where = (hidden["table"], hidden["column"])
            assert where in roles and roles[where] not in declines.KEY_ROLES
            assert hidden in by_id[r["source_question_id"]]["columns_used"]
            shown = render_schema(s, r["hidden_columns"])  # raises if the column is missing or a key
            assert shown != render_schema(s)


def test_a_key_is_never_hidden_even_if_listed():
    rec = {
        "question_id": "tiny__clinics__01__a001",
        "question": "How many visits does each patient have, by total cost?",
        "shape": "join",
        "difficulty": "medium",
        "columns_used": [
            {"table": "visits", "column": "patient_id"},
            {"table": "patients", "column": "patient_id"},
            {"table": "visits", "column": "total_cost"},
        ],
    }
    keys_only = {**rec, "question_id": "tiny__clinics__01__a002", "question": "How many visits are there?"}
    keys_only["columns_used"] = rec["columns_used"][:2]
    answerable = [rec, keys_only] + fake_answerable(TINY, 40)
    for seed in range(5):
        hidden = [r for r in declines.decline_questions(TINY, answerable, seed=seed) if r["hidden_columns"]]
        for r in hidden:
            assert r["source_question_id"] != keys_only["question_id"]
            if r["source_question_id"] == rec["question_id"]:
                assert r["hidden_columns"] == [{"table": "visits", "column": "total_cost"}]


def test_a_name_used_only_for_display_is_not_hidden():
    rec = {
        "question_id": "tiny__clinics__01__a001",
        "question": "Which clinic has the most visits?",
        "shape": "order_limit",
        "difficulty": "hard",
        "columns_used": [{"table": "clinics", "column": "name"}],
    }
    d = declines._Design(TINY, declines.random.Random(0))
    assert declines._hidden_column(d, [rec]) == []
    named = {**rec, "question": "Give the name of the clinic with the most visits."}
    assert [c.hidden for c in declines._hidden_column(d, [named])] == [(("clinics", "name"),)]


def test_missing_attributes_are_truly_absent(generated):
    filler = declines.PHRASE_FILLER | {"whether"}
    checked = 0
    for s, _, records in generated:
        names = [t["name"] for t in s["tables"]] + [c["name"] for t in s["tables"] for c in t["columns"]]
        for r in records:
            if r["decline_subtype"] != "missing_attribute":
                continue
            table_part, attribute = r["decline_reason"].split(" table has no column for ")
            # a flag reads "whether vet clinics are wheelchair accessible"
            attribute = attribute.removeprefix(f"whether {table_part.removeprefix('the ')} ")
            content = [w for w in re.findall(r"[a-z]+", attribute.lower()) if len(w) >= 3 and w not in filler]
            assert content, r["decline_reason"]
            for w in content:
                if len(w) == 3:  # "HOA", "ICU": a whole word of a name, not part of "average"
                    assert not any(w in n.lower().split("_") for n in names), (r["question"], w)
                else:
                    assert not any(w in n.lower() for n in names), (r["question"], w)
            checked += 1
    assert checked > 100


def test_an_attribute_the_design_has_is_never_asked_as_missing():
    with_blood = _with_column(
        TINY, "patients", {"name": "blood_type", "type": "text", "role": "category", "values": ["A", "O"]}
    )
    asked_without, asked_with = False, False
    for seed in range(8):
        for schema, flag in ((TINY, "without"), (with_blood, "with")):
            records = declines.decline_questions(schema, fake_answerable(schema, 60), seed=seed)
            missing = [r for r in records if r["decline_subtype"] == "missing_attribute"]
            hit = any(r["decline_reason"].endswith("no column for blood type") for r in missing)
            if flag == "without":
                asked_without |= hit
            else:
                asked_with |= hit
    assert asked_without  # the attribute is offered when it is truly missing
    assert not asked_with  # and never when the design holds it


def test_short_stems_match_whole_words_only():
    age = declines.Attr("patient age", "number", ("age",))
    assert declines.is_absent(["visits", "average_cost", "stage"], age)
    assert not declines.is_absent(["patients", "patient_age"], age)


def test_same_seed_same_records_and_a_new_seed_changes_them(designs):
    for s in designs[:40]:
        answerable = fake_answerable(s)
        first = declines.decline_questions(s, answerable, seed=0)
        assert declines.decline_questions(s, answerable, seed=0) == first
    s = designs[0]
    answerable = fake_answerable(s)
    texts = {tuple(r["question"] for r in declines.decline_questions(s, answerable, seed=k)) for k in range(5)}
    assert len(texts) > 1


def test_no_two_records_share_a_question(generated):
    for s, answerable, records in generated:
        texts = [" ".join(r["question"].lower().split()) for r in records]
        assert len(texts) == len(set(texts)), s["schema_id"]
        answerable_texts = {" ".join(a["question"].lower().split()) for a in answerable}
        for r in records:
            if r["decline_subtype"] != "hidden_column":
                assert " ".join(r["question"].lower().split()) not in answerable_texts


def test_questions_use_the_designs_table_words(generated):
    subtypes = {"missing_attribute", "undefined_metric", "vague_reference", "action", "forecast", "analysis"}
    for s, _, records in generated:
        heads = {declines.head_noun(t["name"]).lower() for t in s["tables"]}
        heads |= {declines.nouns(t["name"])[1].split(" ")[-1].lower() for t in s["tables"]}
        for r in records:
            if r["decline_subtype"] in subtypes:
                assert any(h in r["question"].lower() for h in heads), r["question"]


def test_no_question_claims_a_trend_or_gap_nobody_checked(generated):
    """'Why do X have a higher Y?' asserts a fact the copies may contradict; questions must stay neutral."""
    claims = re.compile(r"\b(higher|lower|drop|dropped|rise in|keep rising|fell by|than the rest|more than the)\b")
    for r in _all(generated):
        if r["kind"] != "unanswerable":
            assert not claims.search(r["question"]), r["question"]


def test_no_a_or_an_before_a_plural_column_phrase(generated):
    """'a higher total hectares' reads wrong; 'a high capacity (MW)' and 'high total hectares' read right."""
    pattern = re.compile(r"\b(?:a|an) (?:high|low|large|small|recent|old) ([^?.]+)[?.]")
    for r in _all(generated):
        if r["decline_subtype"] != "hidden_column":
            for phrase in pattern.findall(r["question"]):
                assert not declines._is_plural_phrase(phrase), r["question"]
    assert declines._adj("high", "total hectares") == "high total hectares"
    assert declines._adj("high", "capacity (MW)") == "a high capacity (MW)"
    assert declines._adj("old", "opened date") == "an old opened date"


def test_ambiguous_and_vague_questions_point_at_real_columns(generated):
    """A vague cutoff or an undefined 'best' needs a real date or number column to be vague about."""
    for s, _, records in generated:
        col_words = {words(c["name"]) for t in s["tables"] for c in t["columns"]}
        child_words = {declines.nouns(t["name"])[1] for t in s["tables"]}
        for r in records:
            if r["kind"] == "ambiguous":
                assert any(w in r["decline_reason"] for w in col_words | child_words), r["decline_reason"]
