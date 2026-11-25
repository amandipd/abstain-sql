"""data/generator/families.yaml keeps the shape chosen in plan step 1.1: 6 families of 10 distinct topics."""

from pathlib import Path

import yaml

FAMILIES = Path(__file__).resolve().parent.parent / "data" / "generator" / "families.yaml"


def load():
    return yaml.safe_load(FAMILIES.read_text(encoding="utf-8"))


def test_six_families_of_ten_topics():
    families = load()["families"]
    assert len(families) == 6
    for name, family in families.items():
        assert len(family["topics"]) == 10, name


def test_no_topic_in_two_families_or_also_dropped():
    data = load()
    chosen = [t["name"] for f in data["families"].values() for t in f["topics"]]
    assert len(chosen) == len(set(chosen))
    assert not set(chosen) & set(data["dropped"])


def test_every_topic_says_where_it_came_from():
    for family in load()["families"].values():
        for topic in family["topics"]:
            assert topic["source"] in {"gretel", "added"}, topic["name"]
            assert topic["focus"], topic["name"]
