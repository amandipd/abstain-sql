"""Generate database schemas from per-topic templates (plan step 1.2).

This is version 0 of the generator: every schema is sampled from a hand-written
template for its topic, so generation is free, deterministic, and free of
contamination (no model has seen these databases). Templates live in
data/generator/templates/<family>.yaml, one file per family in families.yaml,
with one template per topic.

A schema is a blueprint only: tables, columns, types, and keys, with no rows.
Step 1.3 fills each schema 3-5 times with different data.

Template format
---------------
Each topic has a list of tables. Each table has:

    key      a stable internal name that foreign keys refer to
    names    one or more table names; the generator picks one per schema
    id       the primary-key column name for each entry in names (same length)
    columns  the non-key columns

Every table's first column is its integer primary key, named from `id`. Each
column has a type (text, integer, real, or date) and a role, which later steps
use to fill values (1.3), write questions (1.4), and remove columns to make
questions unanswerable (1.5):

    fk           a link to another table: `ref` names that table's key. The
                 column takes the referenced table's chosen key name unless
                 `names` is given (needed when 2 links point at one table).
    category     one value from `values`
    number       a value between `min` and `max`
    date         a date between `min` and `max` (YYYY-MM-DD)
    flag         0 or 1
    first_name   a person's first name
    last_name    a person's last name
    entity_name  the name of the row's thing; `values` gives a pool to draw
                 from, `kind` a hint for generated names (clinic, company)

`names` lists synonyms; the generator picks one. `optional: true` lets the
generator drop the column. Foreign keys are never optional.

Sampling
--------
Each schema first draws a size from 3 to 8 tables. It starts from a random table
whose links fit in that size, takes that table plus every table it links to,
then grows by adding tables that link into the chosen set, so every schema is
connected and every link points at an included table. A
schema must contain text, integer, real, and date columns, and the schemas for
one topic must use different sets of tables. Each schema's random generator is
seeded from the global seed and the schema's ID, so the same command always
writes the same schemas.

Usage, from the repo root:

    uv run python -m data.generator.schemas
    uv run python -m data.generator.schemas --families healthcare --per-topic 2 --seed 0
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

GENERATOR_VERSION = 0

HERE = Path(__file__).resolve().parent
FAMILIES_FILE = HERE / "families.yaml"
TEMPLATE_DIR = HERE / "templates"
DEFAULT_OUT = HERE.parent / "instances"

MIN_TABLES, MAX_TABLES = 3, 8
OPTIONAL_DROP_RATE = 0.3
MAX_ATTEMPTS = 200
REQUIRED_TYPES = {"text", "integer", "real", "date"}

SQL_TYPES = {"text": "TEXT", "integer": "INTEGER", "real": "REAL", "date": "DATE"}
ROLE_TYPES = {
    "fk": {"integer"},
    "category": {"text"},
    "number": {"integer", "real"},
    "date": {"date"},
    "flag": {"integer"},
    "first_name": {"text"},
    "last_name": {"text"},
    "entity_name": {"text"},
}
# Fields copied from a template column into the schema, for later steps to use.
CARRIED_FIELDS = ("values", "min", "max", "kind")
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


class TemplateError(ValueError):
    """A template breaks the format described in this module's docstring."""


# ---------------------------------------------------------------- loading


def load_families(path: Path = FAMILIES_FILE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["families"]


def families_with_templates() -> list[str]:
    return [f for f in load_families() if (TEMPLATE_DIR / f"{f}.yaml").exists()]


def load_templates(family: str, path: Path | None = None) -> dict[str, dict]:
    path = path or TEMPLATE_DIR / f"{family}.yaml"
    topics = yaml.safe_load(path.read_text(encoding="utf-8"))["topics"]
    for topic, template in topics.items():
        validate_template(topic, template)
    return topics


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _as_list(value) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _iso(value) -> str:
    """YAML reads unquoted dates as date objects; normalize to YYYY-MM-DD."""
    if isinstance(value, datetime.date):
        return value.isoformat()
    return datetime.date.fromisoformat(str(value)).isoformat()


def _refs(table: dict) -> list[str]:
    return [c["ref"] for c in table["columns"] if c["role"] == "fk"]


def validate_template(topic: str, template: dict) -> None:
    tables = template.get("tables") or []
    if len(tables) < MIN_TABLES:
        raise TemplateError(f"{topic}: needs at least {MIN_TABLES} tables, has {len(tables)}")
    keys = [t.get("key") for t in tables]
    if len(keys) != len(set(keys)):
        raise TemplateError(f"{topic}: duplicate table keys")

    for table in tables:
        where = f"{topic}.{table.get('key')}"
        names, ids = _as_list(table.get("names")), _as_list(table.get("id"))
        if not names or len(names) != len(ids):
            raise TemplateError(f"{where}: 'names' and 'id' must be non-empty lists of the same length")
        for name in names + ids:
            if not IDENTIFIER.match(name):
                raise TemplateError(f"{where}: bad identifier {name!r}")
        if not table.get("columns"):
            raise TemplateError(f"{where}: no columns")

        for col in table["columns"]:
            role, typ = col.get("role"), col.get("type")
            label = f"{where}.{(_as_list(col.get('names')) or [col.get('ref')])[0]}"
            if role not in ROLE_TYPES:
                raise TemplateError(f"{label}: unknown role {role!r}")
            if typ not in ROLE_TYPES[role]:
                raise TemplateError(f"{label}: role {role} cannot have type {typ!r}")
            for name in _as_list(col.get("names")):
                if not IDENTIFIER.match(name):
                    raise TemplateError(f"{label}: bad identifier {name!r}")
            if role == "fk":
                if col.get("ref") not in keys:
                    raise TemplateError(f"{label}: links to unknown table {col.get('ref')!r}")
                if col["ref"] == table["key"]:
                    raise TemplateError(f"{label}: a table cannot link to itself")
                if col.get("optional"):
                    raise TemplateError(f"{label}: foreign keys cannot be optional")
            elif not _as_list(col.get("names")):
                raise TemplateError(f"{label}: needs 'names'")
            if role == "category" or (role == "entity_name" and "values" in col):
                values = col.get("values") or []
                if len(values) < 2 or not all(isinstance(v, str) for v in values):
                    raise TemplateError(f"{label}: 'values' must list at least 2 strings (quote yes/no/on/off)")
            if role == "entity_name" and "values" not in col and "kind" not in col:
                raise TemplateError(f"{label}: entity_name needs 'values' or 'kind'")
            if role == "number":
                if not col.get("min", 0) <= col.get("max", -1):
                    raise TemplateError(f"{label}: needs min <= max")
            if role == "date":
                try:
                    lo, hi = _iso(col["min"]), _iso(col["max"])
                except (KeyError, ValueError) as e:
                    raise TemplateError(f"{label}: needs min and max as YYYY-MM-DD") from e
                if lo > hi:
                    raise TemplateError(f"{label}: needs min <= max")

    _creation_order(tables)  # raises on a cycle of links


def _creation_order(tables: list[dict]) -> list[str]:
    """Order table keys so every table comes after the tables it links to."""
    remaining = {t["key"]: set(_refs(t)) for t in tables}
    order: list[str] = []
    while remaining:
        ready = [k for k in remaining if not remaining[k] - set(order)]
        if not ready:
            raise TemplateError(f"cycle of links among {sorted(remaining)}")
        for key in [t["key"] for t in tables if t["key"] in ready]:
            order.append(key)
            del remaining[key]
    return order


# ---------------------------------------------------------------- sampling


def _closure(key: str, by_key: dict[str, dict]) -> set[str]:
    """The table plus every table it links to, directly or indirectly."""
    seen, stack = set(), [key]
    while stack:
        k = stack.pop()
        if k not in seen:
            seen.add(k)
            stack.extend(_refs(by_key[k]))
    return seen


def _max_tables(template: dict) -> int:
    return min(MAX_TABLES, len(template["tables"]))


def _sample_tables(template: dict, rng: random.Random, target: int) -> set[str] | None:
    tables = template["tables"]
    by_key = {t["key"]: t for t in tables}
    order = [t["key"] for t in tables]

    # Start only from tables whose own links fit in the target size. Rejecting
    # oversized starts instead would throw away small targets most often and
    # bias schemas toward 8 tables.
    starts = [k for k in order if len(_closure(k, by_key)) <= target]
    if not starts:
        return None
    chosen = _closure(rng.choice(starts), by_key)
    while len(chosen) < target:
        # Tables that link into the chosen set; what they link to is pulled in too.
        candidates = [k for k in order if k not in chosen and set(_refs(by_key[k])) & chosen]
        rng.shuffle(candidates)
        for key in candidates:
            grown = chosen | _closure(key, by_key)
            if len(grown) <= target:
                chosen = grown
                break
        else:
            break
    return chosen if len(chosen) >= MIN_TABLES else None


def build_schema(
    schema_id: str, family: str, topic: str, template: dict, rng: random.Random, target: int
) -> dict | None:
    """Sample one schema of about `target` tables, or None if this draw breaks a rule (the caller redraws)."""
    chosen = _sample_tables(template, rng, target)
    if chosen is None:
        return None
    by_key = {t["key"]: t for t in template["tables"]}
    order = [k for k in _creation_order(template["tables"]) if k in chosen]

    table_name, key_name = {}, {}
    for key in order:
        i = rng.randrange(len(_as_list(by_key[key]["names"])))
        table_name[key] = _as_list(by_key[key]["names"])[i]
        key_name[key] = _as_list(by_key[key]["id"])[i]
    if len(set(table_name.values())) != len(table_name):
        return None  # 2 tables drew the same synonym

    tables = []
    for key in order:
        columns = [{"name": key_name[key], "type": "integer", "role": "id"}]
        for col in by_key[key]["columns"]:
            if col.get("optional") and rng.random() < OPTIONAL_DROP_RATE:
                continue
            if col["role"] == "fk":
                ref = col["ref"]
                name = rng.choice(_as_list(col["names"])) if col.get("names") else key_name[ref]
                out = {
                    "name": name,
                    "type": "integer",
                    "role": "fk",
                    "references": {"table": table_name[ref], "column": key_name[ref]},
                }
            else:
                out = {"name": rng.choice(_as_list(col["names"])), "type": col["type"], "role": col["role"]}
                for field in CARRIED_FIELDS:
                    if field in col:
                        out[field] = _iso(col[field]) if col["role"] == "date" and field in ("min", "max") else col[field]
            columns.append(out)
        if len({c["name"] for c in columns}) != len(columns):
            return None  # a synonym collided with another column's name
        tables.append({
            "name": table_name[key],
            "template_key": key,
            "primary_key": key_name[key],
            "columns": columns,
        })

    return {
        "schema_id": schema_id,
        "family": family,
        "topic": topic,
        "generator_version": GENERATOR_VERSION,
        "tables": tables,
    }


def column_types(schema: dict) -> set[str]:
    return {c["type"] for t in schema["tables"] for c in t["columns"]}


def generate(families: list[str] | None = None, per_topic: int = 2, seed: int = 0) -> list[dict]:
    all_families = load_families()
    families = families or families_with_templates()
    schemas = []
    for family in families:
        if family not in all_families:
            raise TemplateError(f"unknown family {family!r}")
        templates = load_templates(family)
        topics = [t["gretel_domain"] for t in all_families[family]["topics"]]
        if sorted(templates) != sorted(topics):
            raise TemplateError(f"{family}: templates cover {sorted(templates)}, families.yaml lists {sorted(topics)}")

        for topic in topics:
            used_table_sets: set[frozenset] = set()
            for k in range(1, per_topic + 1):
                schema_id = f"{family}__{slug(topic)}__{k:02d}"
                rng = random.Random(f"{seed}:{schema_id}")
                template = templates[topic]
                target = rng.randint(MIN_TABLES, _max_tables(template))
                for attempt in range(MAX_ATTEMPTS):
                    if attempt and attempt % 25 == 0:
                        target = rng.randint(MIN_TABLES, _max_tables(template))  # this size keeps failing
                    schema = build_schema(schema_id, family, topic, template, rng, target)
                    if schema is None or not REQUIRED_TYPES <= column_types(schema):
                        continue
                    table_set = frozenset(t["template_key"] for t in schema["tables"])
                    if table_set not in used_table_sets:
                        break
                else:
                    raise RuntimeError(f"{schema_id}: no valid schema in {MAX_ATTEMPTS} draws")
                used_table_sets.add(table_set)
                schema["seed"] = seed
                schemas.append(schema)
    return schemas


# ---------------------------------------------------------------- output


def to_sql(schema: dict) -> str:
    """CREATE TABLE statements, in an order where every link points backward."""
    statements = []
    for table in schema["tables"]:
        lines = []
        for col in table["columns"]:
            line = f"  {col['name']} {SQL_TYPES[col['type']]}"
            if col["role"] == "id":
                line += " PRIMARY KEY"
            lines.append(line)
        for col in table["columns"]:
            if col["role"] == "fk":
                ref = col["references"]
                lines.append(f"  FOREIGN KEY ({col['name']}) REFERENCES {ref['table']} ({ref['column']})")
        statements.append(f"CREATE TABLE {table['name']} (\n" + ",\n".join(lines) + "\n);")
    return "\n\n".join(statements) + "\n"


def write(schemas: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for schema in schemas:
        folder = out_dir / schema["schema_id"]
        folder.mkdir(exist_ok=True)
        (folder / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8", newline="\n")
        (folder / "schema.sql").write_text(to_sql(schema), encoding="utf-8", newline="\n")
        index.append({
            "schema_id": schema["schema_id"],
            "family": schema["family"],
            "topic": schema["topic"],
            "tables": [t["name"] for t in schema["tables"]],
        })
    (out_dir / "schemas.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--families", nargs="*", help="families to generate (default: every family with a template)")
    parser.add_argument("--per-topic", type=int, default=2, help="schemas per topic (default: 2)")
    parser.add_argument("--seed", type=int, default=0, help="global seed (default: 0)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default: data/instances)")
    args = parser.parse_args(argv)

    schemas = generate(args.families, args.per_topic, args.seed)
    write(schemas, args.out)

    sizes = Counter(len(s["tables"]) for s in schemas)
    print(f"wrote {len(schemas)} schemas to {args.out}")
    print("tables per schema: " + ", ".join(f"{n}: {sizes[n]}" for n in sorted(sizes)))
    for family, count in Counter(s["family"] for s in schemas).items():
        print(f"  {family}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
