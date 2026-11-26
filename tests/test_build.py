"""The one-command build (plan Phase 1) and its 70/30 mix (step 1.6)."""

import json

import pytest

from data.generator import build


def _rec(kind, i):
    prefix = {"answerable": "a", "unanswerable": "u", "ambiguous": "m", "out_of_scope": "o"}[kind]
    return {"question_id": f"s__{prefix}{i:03d}", "kind": kind}


def test_mix_keeps_every_answerable_and_caps_declines_at_3_per_7():
    answerable = [_rec("answerable", i) for i in range(1, 21)]
    decline = [_rec(k, i) for k in build.DECLINE_KINDS for i in range(1, 11)]
    mixed = build.mix(answerable, decline, seed=0, schema_id="s")
    kinds = [r["kind"] for r in mixed]
    assert kinds.count("answerable") == 20
    assert len(mixed) - 20 == int(20 * 0.3 / 0.7)
    counts = [kinds.count(k) for k in build.DECLINE_KINDS]
    assert max(counts) - min(counts) <= 1  # balanced across the 3 kinds


def test_mix_keeps_all_declines_when_there_are_few():
    answerable = [_rec("answerable", i) for i in range(1, 21)]
    decline = [_rec("ambiguous", 1)]
    assert len(build.mix(answerable, decline, seed=0, schema_id="s")) == 21


@pytest.fixture(scope="module")
def small_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("build")
    summary = build.build(root, families=["finance"], per_topic=1, copies=5, seed=0, target=12, workers=2)
    return root, summary


def test_one_command_writes_everything(small_build):
    root, summary = small_build
    assert summary["designs"] == 10
    records = [json.loads(line) for line in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == summary["questions"] > 0
    assert len({r["question_id"] for r in records}) == len(records)
    for r in records:
        assert (root / r["schema_id"] / "copy_5.sqlite").exists()
        assert (r["gold_sql"] is None) == (r["kind"] != "answerable")


def test_answerable_share_is_at_least_70_percent(small_build):
    _, summary = small_build
    assert summary["answerable_share"] >= 0.7 - 1e-9
