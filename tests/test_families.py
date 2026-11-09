"""data/generator/families.yaml keeps the shape plan step 1.1 chose: 5 families of 5 distinct topics."""

from pathlib import Path

import yaml

FAMILIES = Path(__file__).resolve().parent.parent / "data" / "generator" / "families.yaml"


def load():
    return yaml.safe_load(FAMILIES.read_text(encoding="utf-8"))


def test_five_families_of_five_topics():
    families = load()["families"]
    assert len(families) == 5
    for name, family in families.items():
        assert len(family["topics"]) == 5, name


def test_no_topic_in_two_families_or_also_dropped():
    data = load()
    chosen = [t["gretel_domain"] for f in data["families"].values() for t in f["topics"]]
    assert len(chosen) == len(set(chosen))
    assert not set(chosen) & set(data["dropped"])
