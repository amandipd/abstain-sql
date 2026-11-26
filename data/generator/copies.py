"""Run a query read-only on a filled copy of a design, with a time limit (plan Phase 1).

Phase 1 needs this to validate gold queries; Phase 2's checker builds on it.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

COPIES = 5
DEFAULT_TIMEOUT_S = 2.0
MAX_ROWS = 10_000


@dataclass(frozen=True)
class Result:
    status: str  # "ok", "error", or "timeout"
    rows: list[tuple] | None = None
    error: str | None = None


def copy_paths(schema_dir: Path, copies: int = COPIES) -> list[Path]:
    """The filled copies of one design, copy_1.sqlite to copy_<copies>.sqlite."""
    return [schema_dir / f"copy_{k}.sqlite" for k in range(1, copies + 1)]


def run_readonly(path: Path, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S, max_rows: int = MAX_ROWS) -> Result:
    """Run one statement on a copy opened read-only; never raises."""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        return Result("error", error=f"cannot open {path.name}: {e}")
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        rows = con.execute(sql).fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return Result("error", error=f"more than {max_rows} rows")
        return Result("ok", rows=rows)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            return Result("timeout")
        return Result("error", error=str(e))
    except (sqlite3.Error, ValueError) as e:
        return Result("error", error=f"{type(e).__name__}: {e}")
    finally:
        con.close()


def run_on_copies(paths: list[Path], sql: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> list[Result]:
    return [run_readonly(p, sql, timeout_s) for p in paths]
