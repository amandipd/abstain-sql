"""Fill each design with made-up rows: 5 copies per design (plan step 1.3).

The checker counts an answer correct only if it matches the gold result on every
copy of a design, so a wrong query cannot pass by matching one copy's data by
luck. Each copy has the same tables and columns, filled with different rows.

What every copy guarantees (data/generator/README.md lists them for the other
steps):

- Primary keys run 1 to n, every link points at a row that exists, and there
  are no NULLs.
- Every value of a category column appears at least once whenever the table has
  at least as many rows as the category has values, so a filter on any listed
  value finds rows in every copy.
- Numbers stay inside the column's min and max; wide ranges (max over 100 times
  min) are drawn on a log scale, as real money and counts are. Dates fall inside
  their range, on the 1st of the month for columns named "...month".
- Tables hold tens to a few hundred rows: tables with no links 15-60, tables one
  link deep 40-160, deeper tables 80-300. Lookup tables drawn from a fixed name
  list are capped by the list's length. Counts differ between copies.
- Category weights and link targets are skewed, and differently per copy, so
  group totals have clear winners and copies disagree on them.

Columns are filled independently, so rows can contradict each other (a hire
date after a visit the person handled). Correctness never depends on realism:
every gold answer comes from running its query on the copies.

Each copy is seeded from the global seed, the design's ID, and the copy number,
so any copy can be rebuilt alone. Run from the repo root after schemas.py:

    uv run python -m data.generator.fill
    uv run python -m data.generator.fill --copies 5 --seed 0
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data.generator import names
from data.generator.schemas import DEFAULT_OUT, to_sql

COPIES = 5
ROW_RANGES = {0: (15, 60), 1: (40, 160), 2: (80, 300)}  # by link depth; deeper tables use 2


def _depths(schema: dict) -> dict[str, int]:
    """How many links deep each table sits; tables come in creation order, parents first."""
    depth: dict[str, int] = {}
    for t in schema["tables"]:
        parents = [c["references"]["table"] for c in t["columns"] if c["role"] == "fk"]
        depth[t["name"]] = 1 + max(depth[p] for p in parents) if parents else 0
    return depth


def _row_count(table: dict, depth: int, rng: random.Random) -> int:
    is_root = not any(c["role"] == "fk" for c in table["columns"])
    pools = [len(c["values"]) for c in table["columns"] if c["role"] == "entity_name" and "values" in c]
    if is_root and pools:
        cap = min(pools)
        return rng.randint(max(3, cap - 3), cap)
    lo, hi = ROW_RANGES[min(depth, 2)]
    n = rng.randint(lo, hi)
    widest = max((len(c["values"]) for c in table["columns"] if c["role"] == "category"), default=0)
    return max(n, widest)


def _skewed_weights(k: int, rng: random.Random) -> list[float]:
    return [rng.gammavariate(0.8, 1.0) + 1e-6 for _ in range(k)]


def _category(values: list[str], n: int, rng: random.Random) -> list[str]:
    out = list(values) if n >= len(values) else []  # every value at least once when there is room
    out += rng.choices(values, weights=_skewed_weights(len(values), rng), k=n - len(out))
    rng.shuffle(out)
    return out


def _number(col: dict, rng: random.Random) -> int | float:
    lo, hi = col["min"], col["max"]
    if lo > 0 and hi / lo >= 100:
        v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    elif lo <= 0 and hi >= 1000:
        # Mostly positive and spread on a log scale, with a few values at or below zero.
        v = rng.uniform(lo, min(hi, 1)) if rng.random() < 0.1 else math.exp(rng.uniform(0, math.log(hi)))
    else:
        v = rng.uniform(lo, hi)
    if col["type"] == "integer":
        return max(int(lo), min(int(hi), round(v)))
    v = round(v, 2) if abs(v) >= 1 else round(v, 4)
    return max(lo, min(hi, v))


def _date(col: dict, rng: random.Random) -> str:
    lo = datetime.date.fromisoformat(col["min"]).toordinal()
    hi = datetime.date.fromisoformat(col["max"]).toordinal()
    d = datetime.date.fromordinal(rng.randint(lo, hi))
    if "month" in col["name"]:
        d = d.replace(day=1)
    return d.isoformat()


def _flags(n: int, rng: random.Random) -> list[int]:
    p = rng.uniform(0.2, 0.8)
    out = [1 if rng.random() < p else 0 for _ in range(n)]
    if n >= 2 and len(set(out)) == 1:
        out[rng.randrange(n)] = 1 - out[0]
    return out


def _column(col: dict, n: int, table_is_root: bool, parent_rows: dict[str, int], rng: random.Random) -> list:
    role = col["role"]
    if role == "id":
        return list(range(1, n + 1))
    if role == "fk":
        k = parent_rows[col["references"]["table"]]
        return rng.choices(range(1, k + 1), weights=_skewed_weights(k, rng), k=n)
    if role == "category":
        return _category(col["values"], n, rng)
    if role == "number":
        return [_number(col, rng) for _ in range(n)]
    if role == "date":
        return [_date(col, rng) for _ in range(n)]
    if role == "flag":
        return _flags(n, rng)
    if role == "first_name":
        return rng.choices(names.FIRST_NAMES, k=n)
    if role == "last_name":
        return rng.choices(names.LAST_NAMES, k=n)
    if role == "entity_name":
        if "values" in col:
            pool = col["values"]
            # A lookup table lists each real name once; a linked table may repeat them (each hotel has a spa).
            return rng.sample(pool, n) if table_is_root and n <= len(pool) else rng.choices(pool, k=n)
        return names.unique_names(col["kind"], n, rng)
    raise ValueError(f"unknown role {role!r} in column {col['name']}")


def fill_copy(schema: dict, rng: random.Random) -> dict[str, list[tuple]]:
    """One copy's rows, table by table, in creation order."""
    depth = _depths(schema)
    rows: dict[str, list[tuple]] = {}
    counts: dict[str, int] = {}
    for t in schema["tables"]:
        n = _row_count(t, depth[t["name"]], rng)
        is_root = depth[t["name"]] == 0
        columns = [_column(c, n, is_root, counts, rng) for c in t["columns"]]
        rows[t["name"]] = list(zip(*columns, strict=True))
        counts[t["name"]] = n
    return rows


def write_copy(schema: dict, rows: dict[str, list[tuple]], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.execute("pragma foreign_keys = on")
        con.executescript(to_sql(schema))
        for t in schema["tables"]:
            marks = ", ".join("?" for _ in t["columns"])
            con.executemany(f"insert into {t['name']} values ({marks})", rows[t["name"]])
        con.commit()
        broken = con.execute("pragma foreign_key_check").fetchall()
        if broken:
            raise RuntimeError(f"{schema['schema_id']}: broken links {broken[:3]}")
    finally:
        con.close()
    tmp.replace(path)


def fill_schema(schema: dict, folder: Path, copies: int = COPIES, seed: int = 0) -> dict:
    """Write copy_1.sqlite ... copy_<copies>.sqlite and copies.json for one design; return the manifest."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("copy_*.sqlite"):
        old.unlink()
    manifest = {"schema_id": schema["schema_id"], "seed": seed, "copies": []}
    for k in range(1, copies + 1):
        copy_seed = f"{seed}:{schema['schema_id']}:copy{k}"
        rows = fill_copy(schema, random.Random(copy_seed))
        path = folder / f"copy_{k}.sqlite"
        write_copy(schema, rows, path)
        manifest["copies"].append({"file": path.name, "seed": copy_seed, "rows": {t: len(r) for t, r in rows.items()}})
    (folder / "copies.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def _fill_one(args: tuple[str, int, int]) -> dict:
    folder, copies, seed = args
    schema = json.loads((Path(folder) / "schema.json").read_text(encoding="utf-8"))
    return fill_schema(schema, Path(folder), copies, seed)


def fill_all(root: Path = DEFAULT_OUT, copies: int = COPIES, seed: int = 0, workers: int = 8) -> list[dict]:
    index = json.loads((root / "schemas.json").read_text(encoding="utf-8"))
    jobs = [(str(root / s["schema_id"]), copies, seed) for s in index]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_fill_one, jobs, chunksize=4))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill every design with made-up rows (plan step 1.3).")
    parser.add_argument("--copies", type=int, default=COPIES, help=f"copies per design (default: {COPIES})")
    parser.add_argument("--seed", type=int, default=0, help="global seed (default: 0)")
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT, help="folder holding the designs")
    args = parser.parse_args(argv)

    if not (args.root / "schemas.json").exists():
        print(f"no designs in {args.root}; run `uv run python -m data.generator.schemas` first", file=sys.stderr)
        return 1
    manifests = fill_all(args.root, args.copies, args.seed)
    rows = [n for m in manifests for c in m["copies"] for n in c["rows"].values()]
    files = sum(len(m["copies"]) for m in manifests)
    print(f"filled {len(manifests)} designs x {args.copies} copies = {files} databases")
    print(f"rows per table: {min(rows)} to {max(rows)}, {sum(rows):,} rows in all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
