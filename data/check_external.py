"""Count and execute everything in the external evaluation data (plan step 0.4).

Produces every number in results/external_data.md and saves them to
results/external_check.json, so the report can be rebuilt from the repo. It
needs data/external/ (run data/download_external.py first).

It counts databases, tables, rows, and questions in BIRD dev (original and
cleaned) and TrustSQL, then runs every answerable gold query read-only with a
10-second limit, on 8 worker processes. TrustSQL queries get the same clock fix
TrustSQL's own scorer applies (scoring_utils.py): `current_time` and `'now'`
become the fixed moment 2100-12-31 23:59:00. EHRSQL also runs once without the
fix, to show what the fix prevents. Timeout counts depend on the machine.

Run from the repo root; it takes a few minutes on the laptop:

    uv run python data/check_external.py
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "data" / "external"
OUT = ROOT / "results" / "external_check.json"

TIMEOUT_S = 10
WORKERS = 8
MAX_ROWS = 10_001
TRUSTSQL_SOURCES = ["atis", "advising", "ehrsql", "spider"]
TRUSTSQL_NOW = "'2100-12-31 23:59:00'"


def load(path: Path) -> list[dict]:
    """Read a JSON list or a JSON-lines file."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


USER_TABLES = "select name from sqlite_master where type = 'table' and name not like 'sqlite_%'"


def db_shape(path: Path) -> dict:
    con = connect_readonly(path)
    tables = [r[0] for r in con.execute(USER_TABLES)]
    columns = sum(len(con.execute(f'pragma table_info("{t}")').fetchall()) for t in tables)
    rows = sum(con.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables)
    con.close()
    return {"tables": len(tables), "columns": columns, "rows": rows, "mb": round(path.stat().st_size / 1e6, 1)}


def pin_clock(sql: str) -> str:
    """TrustSQL's scorer replaces the real clock with a fixed moment, exactly like this."""
    return sql.replace("current_time", TRUSTSQL_NOW).replace("'now'", TRUSTSQL_NOW)


def run_query(job: tuple[str, str]) -> tuple[str, int | str]:
    """Run one query read-only; return ("ok", row count), ("timeout", 0), or ("error", message)."""
    db, sql = job
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    deadline = time.monotonic() + TIMEOUT_S
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        return ("ok", len(con.execute(sql).fetchmany(MAX_ROWS)))
    except sqlite3.OperationalError as e:
        return ("timeout", 0) if "interrupted" in str(e) else ("error", str(e)[:120])
    except Exception as e:
        return ("error", f"{type(e).__name__}: {str(e)[:120]}")
    finally:
        con.close()


def execute(jobs: list[tuple[str, str]], pool: ProcessPoolExecutor) -> dict:
    results = list(pool.map(run_query, jobs, chunksize=8))
    status = Counter(s for s, _ in results)
    return {
        "gold_queries": len(jobs),
        "ran": status["ok"],
        "timed_out": status["timeout"],
        "errors": status["error"],
        "empty_results": sum(1 for s, n in results if s == "ok" and n == 0),
        "top_errors": Counter(m for s, m in results if s == "error").most_common(5),
    }


# ---------------------------------------------------------------- BIRD


def bird_paths() -> tuple[Path, Path, Path]:
    base = EXTERNAL / "bird"
    return base / "dev_20240627" / "dev_databases", base / "dev_20240627" / "dev.json", base / "bird_dev_20251106.json"


def count_bird() -> dict:
    db_dir, orig_path, clean_path = bird_paths()
    orig, clean = load(orig_path), load(clean_path)
    by_id = {r["question_id"]: r for r in orig}
    per_q_orig = Counter(r["db_id"] for r in orig)
    per_q_clean = Counter(r["db_id"] for r in clean)

    databases = {}
    for folder in sorted(p for p in db_dir.iterdir() if p.is_dir()):
        databases[folder.name] = {
            **db_shape(folder / f"{folder.name}.sqlite"),
            "questions_original": per_q_orig[folder.name],
            "questions_cleaned": per_q_clean[folder.name],
        }

    relabeled = Counter(
        f"{by_id[r['question_id']]['difficulty']} -> {r['difficulty']}"
        for r in clean
        if r["difficulty"] != by_id[r["question_id"]]["difficulty"]
    )
    return {
        "databases": databases,
        "totals": {k: sum(d[k] for d in databases.values()) for k in ("tables", "columns", "rows", "mb")},
        "questions": {"original": len(orig), "cleaned": len(clean)},
        "difficulty": {
            "original": dict(Counter(r["difficulty"] for r in orig)),
            "cleaned": dict(Counter(r["difficulty"] for r in clean)),
        },
        "with_hint": {
            "original": sum(bool((r.get("evidence") or "").strip()) for r in orig),
            "cleaned": sum(bool((r.get("evidence") or "").strip()) for r in clean),
        },
        "cleaned_vs_original": {
            "same_question_ids": sorted(by_id) == sorted(r["question_id"] for r in clean),
            "same_database_every_question": all(r["db_id"] == by_id[r["question_id"]]["db_id"] for r in clean),
            "question_text_changed": sum(r["question"] != by_id[r["question_id"]]["question"] for r in clean),
            "hint_changed": sum(r["evidence"] != by_id[r["question_id"]]["evidence"] for r in clean),
            "gold_sql_changed": sum(r["SQL"].strip() != by_id[r["question_id"]]["SQL"].strip() for r in clean),
            "difficulty_relabeled": dict(relabeled),
        },
    }


def bird_jobs(version: str) -> list[tuple[str, str]]:
    db_dir, orig_path, clean_path = bird_paths()
    rows = load(orig_path if version == "original" else clean_path)
    return [(str(db_dir / r["db_id"] / f"{r['db_id']}.sqlite"), r["SQL"]) for r in rows]


# ---------------------------------------------------------------- TrustSQL


def trustsql_dir(source: str) -> Path:
    return EXTERNAL / "trustsql" / "dataset" / source


def trustsql_db(source: str, db_id: str) -> Path:
    d = trustsql_dir(source)
    return d / "database" / db_id / f"{db_id}.sqlite" if source == "spider" else d / f"{db_id}.sqlite"


def is_decline(row: dict) -> bool:
    return row.get("type") == "inf"


def count_trustsql_source(source: str) -> dict:
    d = trustsql_dir(source)
    shipped = (
        sum(1 for p in (d / "database").iterdir() if p.is_dir())
        if source == "spider"
        else sum(1 for _ in d.glob("*.sqlite"))
    )
    splits = {}
    for split in ("train", "valid", "test"):
        rows = load(d / f"{source}_{split}.json")
        splits[split] = {
            "total": len(rows),
            "answerable": sum(not is_decline(r) for r in rows),
            "decline": sum(is_decline(r) for r in rows),
            "databases_used": len({r["db_id"] for r in rows}),
        }

    test = load(d / f"{source}_test.json")
    answerable = [r for r in test if not is_decline(r)]
    test_dbs = sorted({r["db_id"] for r in test})
    shapes = [db_shape(trustsql_db(source, db)) for db in test_dbs]
    train_dbs = {r["db_id"] for r in load(d / f"{source}_train.json")}
    return {
        "databases_shipped": shipped,
        "splits": splits,
        "test_databases": {
            "count": len(shapes),
            "tables": sum(s["tables"] for s in shapes),
            "tables_min_max": [min(s["tables"] for s in shapes), max(s["tables"] for s in shapes)],
            "rows": sum(s["rows"] for s in shapes),
            "mb": round(sum(s["mb"] for s in shapes), 1),
            "also_in_train": len(set(test_dbs) & train_dbs),
        },
        "answerable_difficulty": dict(Counter(r.get("note") for r in answerable)),
        "answerable_seen_or_unseen": dict(Counter(r.get("type") for r in answerable)),
        "decline_kinds": dict(Counter(r.get("note") for r in test if is_decline(r))),
        "answerable_mentioning_current_time": sum("current_time" in r["query"].lower() for r in answerable),
    }


def trustsql_jobs(source: str, fix_clock: bool = True) -> list[tuple[str, str]]:
    rows = [r for r in load(trustsql_dir(source) / f"{source}_test.json") if not is_decline(r)]
    return [(str(trustsql_db(source, r["db_id"])), pin_clock(r["query"]) if fix_clock else r["query"]) for r in rows]


# ---------------------------------------------------------------- main


def main() -> int:
    if not EXTERNAL.exists():
        print("data/external/ is missing; run data/download_external.py first", file=sys.stderr)
        return 1

    started = time.monotonic()
    report = {
        "checked": datetime.date.today().isoformat(),
        "sqlite_version": sqlite3.sqlite_version,
        "timeout_seconds": TIMEOUT_S,
        "bird": count_bird(),
        "trustsql": {s: count_trustsql_source(s) for s in TRUSTSQL_SOURCES},
    }
    con = connect_readonly(trustsql_db("ehrsql", "mimic_iv"))
    report["trustsql"]["ehrsql"]["patients"] = con.execute("select count(*) from patients").fetchone()[0]
    con.close()

    execution = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for version in ("original", "cleaned"):
            print(f"running BIRD {version} gold queries", file=sys.stderr, flush=True)
            execution[f"bird_{version}"] = execute(bird_jobs(version), pool)
        for source in TRUSTSQL_SOURCES:
            print(f"running TrustSQL {source} gold queries", file=sys.stderr, flush=True)
            execution[f"trustsql_{source}"] = execute(trustsql_jobs(source), pool)
        print("running TrustSQL ehrsql again without the clock fix", file=sys.stderr, flush=True)
        execution["trustsql_ehrsql_without_clock_fix"] = execute(trustsql_jobs("ehrsql", fix_clock=False), pool)
    report["execution"] = execution
    report["seconds_taken"] = round(time.monotonic() - started)

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} in {report['seconds_taken']} s", file=sys.stderr)
    for name, e in execution.items():
        print(
            f"  {name:36} {e['ran']:5} ran  {e['timed_out']:3} timed out  "
            f"{e['errors']:3} errors  {e['empty_results']:3} empty",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
