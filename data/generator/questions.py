"""Write answerable questions for each design, SQL first (plan step 1.4).

For one design I build candidate gold queries from templates that cover 7 shapes,
then write the English question from the query, not the other way round. The
English comes from fixed phrasings in version 0; version 1 will paraphrase it
with an LLM.

Shapes and difficulty
---------------------
    filter       rows matching a category value, a numeric threshold, or a flag
    aggregate    SUM, AVG, MIN, MAX, or COUNT(DISTINCT) over one table
    group_by     an aggregate per category value, optionally with HAVING
    join         a child table joined to its parent (or grandparent)
    order_limit  the top or bottom N rows, or groups, by a number or a count
    subquery     comparisons against an average or a maximum, IN, and NOT IN
    date_range   rows within a year, between 2 dates, or before or after a date

Difficulty follows the query, not the shape: easy is one table with a filter or
an aggregate; medium is a GROUP BY, ORDER BY ... LIMIT, date range, or one join;
hard is a subquery, a join plus a GROUP BY, or 2 or more joins.

Literal values come from the schema, never from one copy: category values from
the column's `values`, numeric thresholds from inside [min, max] (on a log scale
when the range is wide, as fill.py draws it), and dates from inside the column's
range. So every question means the same thing on every copy. A calendar-year
number column (founded_year) gets a round year as its cutoff: a century, decade,
or 5-year mark, and never one before 1000 when the range starts earlier.

Validation
----------
A candidate is kept only if it runs without error on all 5 copies and returns at
least 1 row on each, and its answer is not a single row of NULL or 0 (a COUNT
that finds nothing). On top of that:

- ORDER BY ... LIMIT N must return exactly N rows on every copy, with no tie
  among the top N+1 values. That is stricter than "no tie at the cutoff": it
  also fixes the row order inside the top N.
- "Which X has the highest Y?" must find exactly 1 row on every copy.
- A query that reports a parent by name after a join or an IN must have names
  that are unique in that table on every copy, so the names identify the rows.
- Every condition must matter: dropping any one WHERE predicate or HAVING clause
  must change the answer on at least 2 of the 5 copies. Otherwise a model that
  ignored the condition would be scored correct.

Each template is drawn several times with fresh literals and cutoffs; a query
whose first version fails a check tries its next version before it is dropped.

`columns_used` lists every non-key column the gold query reads; step 1.5 hides
one of them to make an unanswerable question.

Usage, from the repo root, prints a summary over every design with copies:

    uv run python -m data.generator.questions
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from data.generator.copies import copy_paths, run_on_copies, run_readonly
from data.generator.schemas import DEFAULT_OUT

SHAPES = ("filter", "aggregate", "group_by", "join", "order_limit", "subquery", "date_range")
DEFAULT_TARGET = 20
MAX_FAILURES_PER_SHAPE = 30  # stop trying a shape after this many queries had no passing version
VERSIONS = 6  # versions of each query, with different literals, tried in turn
MIN_COPIES_CHANGED = 2  # dropping any condition must change the answer on at least this many copies

# Every SQLite keyword; an identifier that matches one is double-quoted.
SQLITE_KEYWORDS = set(
    """abort action add after all alter always analyze and as asc attach autoincrement before begin between by
    cascade case cast check collate column commit conflict constraint create cross current current_date current_time
    current_timestamp database default deferrable deferred delete desc detach distinct do drop each else end escape
    except exclude exclusive exists explain fail filter first following for foreign from full generated glob group
    groups having if ignore immediate in index indexed initially inner insert instead intersect into is isnull join
    key last left like limit match materialized natural no not nothing notnull null nulls of offset on or order others
    outer over partition plan pragma preceding primary query raise range recursive references regexp reindex release
    rename replace restrict returning right rollback row rows savepoint select set table temp temporary then ties to
    transaction trigger unbounded union unique update using vacuum values view virtual when where window with
    without""".split()
)
PLAIN_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

# ---------------------------------------------------------------- English helpers

# Abbreviations spelled out or capitalized wherever they appear in a name.
TOKEN_WORDS = {
    "avg": "average",
    "num": "number of",
    "qty": "quantity",
    "pct": "%",
    "usd": "USD",
    "ml": "ML",
    "teu": "TEU",
    "sku": "SKU",
    "skus": "SKUs",
    "atm": "ATM",
    "atms": "ATMs",
    "lp": "LP",
    "lps": "LPs",
    "gl": "GL",
    "rd": "R&D",
    "pt": "PT",
    "vc": "VC",
    "rv": "RV",
    "ceo": "CEO",
    "fda": "FDA",
    "iata": "IATA",
    "msrp": "MSRP",
    "fico": "FICO",
    "apr": "APR",
    "apy": "APY",
    "aum": "AUM",
    "nav": "NAV",
    "tvl": "TVL",
    "mvp": "MVP",
    "qc": "QC",
    "hq": "HQ",
    "co2": "CO2",
    "nox": "NOx",
    "ph": "pH",
    "bsl": "BSL",
    "adl": "ADL",
    "gcs": "GCS",
    "cdl": "CDL",
    "ppa": "PPA",
    "qdia": "QDIA",
    "eu": "EU",
    "inci": "INCI",
    "bpm": "BPM",
    "dbh": "DBH",
    "xray": "X-ray",
    "xrays": "X-rays",
    "id": "ID",
}
# A unit at the end of a column name goes in brackets: cost_usd -> "cost (USD)".
UNITS = {
    "usd": "USD",
    "musd": "million USD",
    "millions": "millions",
    "billions": "billions",
    "thousands": "thousands",
    "pct": "%",
    "km": "km",
    "km2": "square km",
    "m": "meters",
    "m3": "cubic meters",
    "cm": "cm",
    "mm": "mm",
    "kg": "kg",
    "g": "grams",
    "mg": "mg",
    "lb": "lb",
    "ha": "hectares",
    "mw": "MW",
    "mwh": "MWh",
    "kwh": "kWh",
    "kv": "kV",
    "gb": "GB",
    "bpd": "barrels per day",
    "psi": "psi",
    "hpa": "hPa",
    "kmh": "km/h",
    "ppm": "ppm",
    "ppt": "ppt",
    "psu": "PSU",
    "teu": "TEU",
    "min": "minutes",
    "c": "degrees C",
    "sqft": "square feet",
    "t": "tonnes",
    "l": "liters",
}
COMPOUND_UNITS = {"mg_l": "mg/L", "t_ha": "t/ha"}
# Table names whose last word is not a plural count noun.
MEMBER_NOUNS = {"staff", "crew"}
SAME_PLURAL = {"aircraft", "series", "species", "personnel"}
IRREGULAR_SINGULAR = {"errata": "erratum", "skus": "SKU", "atms": "ATM", "lps": "LP", "xrays": "X-ray"}
# Columns a SUM would be meaningless for: rates, scores, positions, calendar years.
NO_SUM_TOKENS = {
    "year",
    "rate",
    "pct",
    "rating",
    "score",
    "age",
    "position",
    "rank",
    "grade",
    "ratio",
    "percent",
    "temperature",
    "temp",
    "ph",
    "latitude",
    "longitude",
    "bpm",
    "stars",
    "elevation",
    "altitude",
    "c",
}

ABOVE = ("above", "over", "greater than")
BELOW = ("below", "under", "less than")
AGG_WORDS = {
    "SUM": ("total",),
    "AVG": ("average",),
    "MIN": ("lowest", "minimum", "smallest"),
    "MAX": ("highest", "maximum", "largest"),
}


def _plain(tokens: list[str]) -> str:
    return " ".join(TOKEN_WORDS.get(t, t) for t in tokens)


def words(name: str) -> str:
    """A column or table name in plain words, with a trailing unit in brackets."""
    for suffix, unit in COMPOUND_UNITS.items():
        if name.endswith("_" + suffix):
            return f"{_plain(name[: -len(suffix) - 1].split('_'))} ({unit})"
    tokens = name.split("_")
    if len(tokens) > 1 and tokens[-1] in UNITS:
        return f"{_plain(tokens[:-1])} ({UNITS[tokens[-1]]})"
    return _plain(tokens)


def flag_words(name: str) -> str:
    """A flag's name for 'the ... flag': is_verified -> 'verified'."""
    return words(name[3:] if name.startswith("is_") else name)


def date_words(name: str) -> str:
    """A date column's name, made a noun: sent_on -> 'sent date', created_at -> 'created date'."""
    tokens = name.split("_")
    if len(tokens) > 1 and tokens[-1] in ("on", "at"):
        return _plain(tokens[:-1]) + " date"
    return words(name)


def plural_words(phrase: str) -> str:
    """Pluralize the last word: 'facility type' -> 'facility types', 'status' -> 'statuses'."""
    head, _, last = phrase.rpartition(" ")
    if last.endswith(("s", "x", "ch", "sh")):
        last += "es"
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last += "s"
    return f"{head} {last}" if head else last


def singular_word(word: str) -> str:
    if word in IRREGULAR_SINGULAR:
        return IRREGULAR_SINGULAR[word]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def article(phrase: str) -> str:
    """'a' or 'an' for a phrase, by its sound."""
    low = phrase.lower()
    if low.startswith(("uni", "use", "usu", "eu", "one", "ur")):
        return "a"
    if low.startswith(("hour", "honor", "honest")) or low[:1] in "aeiou":
        return "an"
    return "a"


def _a(phrase: str) -> str:
    return f"{article(phrase)} {phrase}"


def _reads_plural(phrase: str) -> bool:
    """True when a column phrase reads as plural ('vesting years', 'page views'), not 'number of tracks'."""
    core = phrase.split(" (")[0]
    for joiner in (" of ", " per ", " in "):
        core = core.split(joiner)[0]
    last = core.split(" ")[-1]
    return (
        last.endswith("s")
        and not last.endswith(("ss", "us", "is"))
        and last not in {"news", "gas", "series", "species"}
    )


def _is(phrase: str) -> str:
    return "are" if _reads_plural(phrase) else "is"


def _some(phrase: str) -> str:
    """'a capacity' but 'page views': a plural column phrase takes no article."""
    return phrase if _reads_plural(phrase) else _a(phrase)


def _agg_phrase(agg: str, nw: str, rng: random.Random) -> str:
    """'total capacity', or 'sum of the total hours' where the column name already says total."""
    first = nw.split(" ")[0]
    if agg == "SUM" and first == "total":
        return f"sum of the {nw}"
    if agg == "AVG" and first == "average":
        return f"mean of the {nw}"
    return f"{rng.choice(AGG_WORDS[agg])} {nw}"


def _avg_of(nw: str) -> str:
    """'average age', or 'mean of the average dwell hours' when the column name already says average."""
    return f"mean of the {nw}" if nw.split(" ")[0] == "average" else f"average {nw}"


def _an_avg(nw: str) -> str:
    """'an average age', or just 'an average dwell hours' when the column name already says average."""
    return f"an {nw}" if nw.split(" ")[0] == "average" else f"an average {nw}"


def _say(value: str) -> str:
    """A category value as it appears in a question, quoted so the exact text is clear."""
    return f'"{value}"' if "'" in value else f"'{value}'"


def is_calendar_year(col: dict) -> bool:
    """founded_year or year_built, but not carloads_per_year, which is a rate."""
    tokens = col["name"].split("_")
    return "year" in tokens and "per" not in tokens


def fmt_number(value: int | float, col: dict) -> str:
    if is_calendar_year(col):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.6f}".rstrip("0").rstrip(".")


def fmt_date(iso: str) -> str:
    d = datetime.date.fromisoformat(iso)
    return f"{calendar.month_name[d.month]} {d.day}, {d.year}"


# ---------------------------------------------------------------- SQL helpers


def q(ident: str) -> str:
    """Quote an identifier only if it needs it."""
    if PLAIN_IDENTIFIER.match(ident) and ident not in SQLITE_KEYWORDS:
        return ident
    return '"' + ident.replace('"', '""') + '"'


def sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_num(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _round_sig(x: float, sig: int) -> float:
    return float(f"{x:.{sig}g}")


def year_threshold(col: dict, quantile: float) -> int:
    """A round, plausible year inside (min, max): a century, decade, or 5-year mark.

    A range that reaches back before 1000 (year_made from -3000) uses only its part from 1000 on.
    """
    lo, hi = int(col["min"]), int(col["max"])
    if lo < 1000 < hi - 100:
        lo = 1000
    span = hi - lo
    step = 100 if span >= 400 else 10 if span >= 30 else 5 if span >= 10 else 1
    marks = [y for y in range(lo - lo % step, hi, step) if lo < y < hi]
    if not marks:
        return round(lo + quantile * span)
    return marks[round(quantile * (len(marks) - 1))]


def _spread_point(col: dict, quantile: float) -> float:
    """The value at a quantile of how fill.py spreads the column: on a log scale when the range is wide."""
    lo, hi = col["min"], col["max"]
    if lo > 0 and hi / lo >= 100:
        return math.exp(math.log(lo) + quantile * (math.log(hi) - math.log(lo)))
    if lo <= 0 and hi >= 1000:
        return math.exp(quantile * math.log(hi))
    return lo + quantile * (hi - lo)


def mean_point(col: dict) -> float:
    """The expected mean of the column as fill.py draws it; a group's average sits near it."""
    lo, hi = col["min"], col["max"]
    if lo > 0 and hi / lo >= 100:
        return (hi - lo) / math.log(hi / lo)
    if lo <= 0 and hi >= 1000:
        return 0.9 * (hi - 1) / math.log(hi) + 0.1 * (lo + 1) / 2
    return (lo + hi) / 2


def threshold(col: dict, quantile: float) -> int | float:
    """A round number inside (min, max) near the given quantile of the column."""
    if is_calendar_year(col):
        return year_threshold(col, quantile)
    return nice_number(_spread_point(col, quantile), col)


def nice_number(x: float, col: dict) -> int | float:
    """The roundest number near x that stays strictly inside the column's (min, max)."""
    lo, hi = col["min"], col["max"]
    tolerance = min(0.1 * (hi - lo), 0.3 * abs(x)) if x else 0.1 * (hi - lo)
    is_int = col["type"] == "integer"
    for sig in range(1, 8):
        v = _round_sig(x, sig)
        if is_int:
            v = round(v)
        if lo < v < hi and abs(v - x) <= tolerance:
            return int(v) if is_int or float(v).is_integer() else v
    return round(x) if is_int else round(x, 4)


# ---------------------------------------------------------------- the design


@dataclass
class _Table:
    name: str
    pk: str
    columns: list[dict]
    sing: str  # "facility"
    plural: str  # "facilities"
    prefix: str  # what qualifies a column: "facility" in "facility type"

    def role(self, *roles: str) -> list[dict]:
        return [c for c in self.columns if c["role"] in roles]

    @property
    def name_cols(self) -> list[str]:
        """The column(s) that name a row: an entity name, or a first and last name."""
        entity = self.role("entity_name")
        if entity:
            return [entity[0]["name"]]
        first, last = self.role("first_name"), self.role("last_name")
        if first and last:
            return [first[0]["name"], last[0]["name"]]
        return []

    @property
    def name_word(self) -> str:
        cols = self.name_cols
        return "first and last name" if len(cols) == 2 else words(cols[0])

    @property
    def name_words(self) -> str:
        cols = self.name_cols
        return "first and last names" if len(cols) == 2 else plural_words(words(cols[0]))

    def qual(self, col: dict) -> str:
        """A column named with its table: 'hotel city', or 'facility type' when the name already says it."""
        w = words(col["name"])
        prefix_last = self.prefix.split(" ")[-1]
        if w == self.prefix or w.startswith(self.prefix + " "):
            return w
        if w.split(" ")[0] == prefix_last:
            return f"{self.prefix} {w.split(' ', 1)[1]}" if " " in w else self.prefix
        return f"{self.prefix} {w}"


def _nouns(table_name: str) -> tuple[str, str, str]:
    tokens = table_name.split("_")
    head, last = tokens[:-1], tokens[-1]
    full = _plain(tokens)
    if last in MEMBER_NOUNS:
        return f"{full} member", f"{full} members", full
    if last in SAME_PLURAL:
        return full, full, full
    if last in IRREGULAR_SINGULAR or (last.endswith("s") and not last.endswith(("ss", "us", "is"))):
        sing = " ".join([_plain(head), singular_word(last)]).strip() if head else singular_word(last)
        return sing, full, sing
    return f"{full} record", f"{full} records", full


@dataclass(frozen=True)
class _Link:
    child: _Table
    fk: str
    parent: _Table

    @property
    def on(self) -> str:
        return f"{q(self.child.name)}.{q(self.fk)} = {q(self.parent.name)}.{q(self.parent.pk)}"


class _Design:
    def __init__(self, schema: dict):
        self.tables: list[_Table] = []
        for t in schema["tables"]:
            sing, plural, prefix = _nouns(t["name"])
            self.tables.append(_Table(t["name"], t["primary_key"], t["columns"], sing, plural, prefix))
        by_name = {t.name: t for t in self.tables}
        # Links a question can name without saying which one: a child with 2 links to one parent is skipped.
        self.links: list[_Link] = []
        for t in self.tables:
            fks = t.role("fk")
            targets = Counter(c["references"]["table"] for c in fks)
            for c in fks:
                parent = c["references"]["table"]
                if targets[parent] == 1:
                    self.links.append(_Link(t, c["name"], by_name[parent]))


# ---------------------------------------------------------------- candidates


@dataclass
class _Cand:
    shape: str
    template: str
    table: str
    sql: str
    question: str
    columns: list[tuple[str, str]]
    joins: int = 0
    grouped: bool = False
    nested: bool = False
    limit: int | None = None
    tie_sql: str | None = None
    single_row: bool = False
    unique_names: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()  # each WHERE predicate and HAVING clause, as written in sql

    @property
    def difficulty(self) -> str:
        if self.nested or self.joins >= 2 or (self.joins and self.grouped):
            return "hard"
        if self.shape in ("filter", "aggregate") and not self.joins:
            return "easy"
        return "medium"


@dataclass(frozen=True)
class _Cond:
    sql: str
    has: str  # follows "have": "a facility type of 'hospital'"
    whose: str  # a relative clause: "whose facility type is 'hospital'"
    columns: tuple[tuple[str, str], ...]


def _ref(t: _Table, col: str, qualify: bool) -> str:
    return f"{q(t.name)}.{q(col)}" if qualify else q(col)


def _cat_cond(t: _Table, col: dict, rng: random.Random, qualify: bool) -> _Cond:
    value = rng.choice(col["values"])
    w = words(col["name"])
    return _Cond(
        f"{_ref(t, col['name'], qualify)} = {sql_str(value)}",
        f"{_some(w)} of {_say(value)}",
        f"whose {w} {_is(w)} {_say(value)}",
        ((t.name, col["name"]),),
    )


def _num_cond(t: _Table, col: dict, rng: random.Random, qualify: bool, extreme: bool = False) -> _Cond:
    """A cutoff near the middle of the column, or in its top or bottom tenth when `extreme`."""
    above = rng.random() < 0.6
    if extreme:
        quantile = rng.choice((0.85, 0.9, 0.95)) if above else rng.choice((0.05, 0.1, 0.15))
    else:
        quantile = rng.choice((0.3, 0.5, 0.7))
    value = threshold(col, quantile)
    op, word = (">", rng.choice(ABOVE)) if above else ("<", rng.choice(BELOW))
    w, said = words(col["name"]), fmt_number(value, col)
    return _Cond(
        f"{_ref(t, col['name'], qualify)} {op} {sql_num(value)}",
        f"{_some(w)} {word} {said}",
        f"whose {w} {_is(w)} {word} {said}",
        ((t.name, col["name"]),),
    )


def _flag_cond(t: _Table, col: dict, rng: random.Random, qualify: bool) -> _Cond:
    value = 1 if rng.random() < 0.7 else 0
    truth = "true" if value else "false"
    w = flag_words(col["name"])
    return _Cond(
        f"{_ref(t, col['name'], qualify)} = {value}",
        f"the {w} flag set to {truth}",
        f"whose {w} flag is {truth}",
        ((t.name, col["name"]),),
    )


_COND_BUILDERS = {"category": _cat_cond, "number": _num_cond, "flag": _flag_cond}


def _conds(t: _Table, rng: random.Random, qualify: bool = False, roles=("category", "number", "flag")) -> list[_Cond]:
    return [_COND_BUILDERS[c["role"]](t, c, rng, qualify) for c in t.columns if c["role"] in roles]


def _aggs(col: dict) -> list[str]:
    aggs = ["AVG", "MIN", "MAX"]
    if not set(col["name"].split("_")) & NO_SUM_TOKENS:
        aggs.append("SUM")
    return aggs


def _names_sql(t: _Table, qualify: bool = False) -> str:
    return ", ".join(_ref(t, c, qualify) for c in t.name_cols)


def _name_columns(t: _Table) -> list[tuple[str, str]]:
    return [(t.name, c) for c in t.name_cols]


def _pick(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


# ---------------------------------------------------------------- templates, one function per shape


def _filter(d: _Design, rng: random.Random) -> list[_Cand]:
    out = []
    for t in d.tables:
        T, P = q(t.name), t.plural
        for c in _conds(t, rng):
            question = _pick(
                rng,
                [
                    f"How many {P} have {c.has}?",
                    f"Count the {P} {c.whose}.",
                    f"How many {P} are there {c.whose}?",
                ],
            )
            out.append(
                _Cand(
                    "filter",
                    "count",
                    t.name,
                    f"SELECT COUNT(*) FROM {T} WHERE {c.sql}",
                    question,
                    [*c.columns],
                    conditions=(c.sql,),
                )
            )
        if t.name_cols:
            for c in _conds(t, rng, roles=("category", "number")):
                question = _pick(
                    rng,
                    [
                        f"List the {t.name_words} of {P} {c.whose}.",
                        f"Which {P} have {c.has}? Give their {t.name_words}.",
                        f"What are the {t.name_words} of the {P} {c.whose}?",
                    ],
                )
                sql = f"SELECT {_names_sql(t)} FROM {T} WHERE {c.sql}"
                cols = [*_name_columns(t), *c.columns]
                out.append(_Cand("filter", "list", t.name, sql, question, cols, conditions=(c.sql,)))
        cats, nums = t.role("category"), t.role("number")
        if cats and nums:
            c1 = _cat_cond(t, rng.choice(cats), rng, False)
            c2 = _num_cond(t, rng.choice(nums), rng, False)
            question = _pick(
                rng,
                [
                    f"How many {P} have {c1.has} and {c2.has}?",
                    f"Count the {P} {c1.whose} and {c2.whose.replace('whose ', '', 1)}.",
                ],
            )
            sql = f"SELECT COUNT(*) FROM {T} WHERE {c1.sql} AND {c2.sql}"
            cols = [*c1.columns, *c2.columns]
            out.append(_Cand("filter", "count2", t.name, sql, question, cols, conditions=(c1.sql, c2.sql)))
    return out


def _aggregate(d: _Design, rng: random.Random) -> list[_Cand]:
    out = []
    for t in d.tables:
        T, P = q(t.name), t.plural
        for n in t.role("number"):
            agg = rng.choice(_aggs(n))
            nw = words(n["name"])
            aw = _agg_phrase(agg, nw, rng)
            question = _pick(
                rng,
                [
                    f"What is the {aw} of all {P}?",
                    f"Across all {P}, what is the {aw}?",
                    f"Find the {aw} among {P}.",
                ],
            )
            sql = f"SELECT {agg}({q(n['name'])}) FROM {T}"
            out.append(_Cand("aggregate", "agg", t.name, sql, question, [(t.name, n["name"])]))

            filters = _conds(t, rng, roles=("category", "flag"))
            if filters:
                c = rng.choice(filters)
                agg = rng.choice(_aggs(n))
                aw = _agg_phrase(agg, nw, rng)
                question = _pick(
                    rng,
                    [
                        f"What is the {aw} of {P} {c.whose}?",
                        f"Among {P} {c.whose}, what is the {aw}?",
                        f"Find the {aw} of the {P} that have {c.has}.",
                    ],
                )
                sql = f"SELECT {agg}({q(n['name'])}) FROM {T} WHERE {c.sql}"
                cols = [(t.name, n["name"]), *c.columns]
                out.append(_Cand("aggregate", "agg_where", t.name, sql, question, cols, conditions=(c.sql,)))
        nums = t.role("number")
        for cat in t.role("category"):
            if not nums:
                break
            c = _num_cond(t, rng.choice(nums), rng, False, extreme=True)  # a middle cutoff keeps every value
            cw = plural_words(words(cat["name"]))
            question = _pick(
                rng,
                [
                    f"How many different {cw} appear among {P} {c.whose}?",
                    f"Among {P} {c.whose}, how many distinct {cw} are there?",
                ],
            )
            sql = f"SELECT COUNT(DISTINCT {q(cat['name'])}) FROM {T} WHERE {c.sql}"
            cols = [(t.name, cat["name"]), *c.columns]
            out.append(_Cand("aggregate", "distinct", t.name, sql, question, cols, conditions=(c.sql,)))
    return out


def _group_by(d: _Design, rng: random.Random) -> list[_Cand]:
    out = []
    for t in d.tables:
        T, P = q(t.name), t.plural
        nums = t.role("number")
        for cat in t.role("category"):
            C, cw = q(cat["name"]), words(cat["name"])
            question = _pick(
                rng,
                [
                    f"How many {P} are there for each {cw}?",
                    f"For each {cw}, how many {P} are there?",
                    f"Count the {P} in each {cw}.",
                ],
            )
            sql = f"SELECT {C}, COUNT(*) FROM {T} GROUP BY {C}"
            out.append(_Cand("group_by", "count", t.name, sql, question, [(t.name, cat["name"])], grouped=True))

            k = rng.choice((3, 5, 10, 20))
            question = _pick(
                rng,
                [
                    f"Which {plural_words(cw)} have more than {k} {P}?",
                    f"List the {plural_words(cw)} that have more than {k} {P}.",
                ],
            )
            sql = f"SELECT {C} FROM {T} GROUP BY {C} HAVING COUNT(*) > {k}"
            cols, having = [(t.name, cat["name"])], (f"COUNT(*) > {k}",)
            out.append(_Cand("group_by", "having_count", t.name, sql, question, cols, grouped=True, conditions=having))

            if not nums:
                continue
            n = rng.choice(nums)
            agg = rng.choice(_aggs(n))
            nw = words(n["name"])
            aw = _agg_phrase(agg, nw, rng)
            question = _pick(
                rng,
                [
                    f"What is the {aw} of {P} for each {cw}?",
                    f"For each {cw}, what is the {aw} of {P}?",
                    f"Show the {aw} of {P} by {cw}.",
                ],
            )
            sql = f"SELECT {C}, {agg}({q(n['name'])}) FROM {T} GROUP BY {C}"
            cols = [(t.name, cat["name"]), (t.name, n["name"])]
            out.append(_Cand("group_by", "agg", t.name, sql, question, cols, grouped=True))

            n = rng.choice(nums)
            # Near the column's expected mean, where group averages sit, so the cutoff splits the groups.
            if is_calendar_year(n):
                value = year_threshold(n, rng.choice((0.4, 0.5, 0.6)))
            else:
                value = nice_number(mean_point(n) * rng.choice((0.7, 0.85, 1.0, 1.15)), n)
            nw, said = words(n["name"]), fmt_number(value, n)
            question = _pick(
                rng,
                [
                    f"Among {P}, which {plural_words(cw)} have {_an_avg(nw)} above {said}?",
                    f"Which {plural_words(cw)} of {P} have {_an_avg(nw)} greater than {said}?",
                ],
            )
            having = f"AVG({q(n['name'])}) > {sql_num(value)}"
            sql = f"SELECT {C} FROM {T} GROUP BY {C} HAVING {having}"
            cols = [(t.name, cat["name"]), (t.name, n["name"])]
            out.append(_Cand("group_by", "having_avg", t.name, sql, question, cols, grouped=True, conditions=(having,)))

            c = _num_cond(t, rng.choice(nums), rng, False)
            question = _pick(
                rng,
                [
                    f"For each {cw}, how many {P} have {c.has}?",
                    f"How many {P} {c.whose} are there in each {cw}?",
                ],
            )
            sql = f"SELECT {C}, COUNT(*) FROM {T} WHERE {c.sql} GROUP BY {C}"
            cols = [(t.name, cat["name"]), *c.columns]
            out.append(_Cand("group_by", "count_where", t.name, sql, question, cols, grouped=True, conditions=(c.sql,)))
    return out


def _join(d: _Design, rng: random.Random) -> list[_Cand]:
    out = []
    for link in d.links:
        c, p = link.child, link.parent
        C, Pt, cP, pP = q(c.name), q(p.name), c.plural, p.plural
        base = f"FROM {C} JOIN {Pt} ON {link.on}"
        for pc in _conds(p, rng, qualify=True):
            question = _pick(
                rng,
                [
                    f"How many {cP} are linked to {pP} {pc.whose}?",
                    f"Count the {cP} whose {p.sing} has {pc.has}.",
                    f"How many {cP} belong to {_a(p.sing)} {pc.whose}?",
                ],
            )
            out.append(
                _Cand(
                    "join",
                    "count",
                    c.name,
                    f"SELECT COUNT(*) {base} WHERE {pc.sql}",
                    question,
                    [*pc.columns],
                    1,
                    conditions=(pc.sql,),
                )
            )

        pconds = _conds(p, rng, qualify=True, roles=("category", "number"))
        for n in c.role("number"):
            if not pconds:
                break
            pc = rng.choice(pconds)
            agg = rng.choice(_aggs(n))
            nw = words(n["name"])
            aw = _agg_phrase(agg, nw, rng)
            question = _pick(
                rng,
                [
                    f"What is the {aw} of {cP} whose {p.sing} has {pc.has}?",
                    f"Among {cP} linked to {pP} {pc.whose}, what is the {aw}?",
                ],
            )
            sql = f"SELECT {agg}({C}.{q(n['name'])}) {base} WHERE {pc.sql}"
            cols = [(c.name, n["name"]), *pc.columns]
            out.append(_Cand("join", "agg", c.name, sql, question, cols, 1, conditions=(pc.sql,)))

        if c.name_cols and pconds:
            pc = rng.choice(pconds)
            question = _pick(
                rng,
                [
                    f"List the {c.name_words} of {cP} whose {p.sing} has {pc.has}.",
                    f"What are the {c.name_words} of the {cP} linked to {pP} {pc.whose}?",
                ],
            )
            sql = f"SELECT {_names_sql(c, True)} {base} WHERE {pc.sql}"
            cols = [*_name_columns(c), *pc.columns]
            out.append(_Cand("join", "list", c.name, sql, question, cols, 1, conditions=(pc.sql,)))

        cconds = _conds(c, rng, qualify=True, roles=("category", "number"))
        if cconds and pconds:
            cc, pc = rng.choice(cconds), rng.choice(pconds)
            question = _pick(
                rng,
                [
                    f"How many {cP} have {cc.has} and belong to {_a(p.sing)} {pc.whose}?",
                    f"Count the {cP} {cc.whose} and whose {p.sing} has {pc.has}.",
                ],
            )
            sql = f"SELECT COUNT(*) {base} WHERE {cc.sql} AND {pc.sql}"
            cols = [*cc.columns, *pc.columns]
            out.append(_Cand("join", "count2", c.name, sql, question, cols, 1, conditions=(cc.sql, pc.sql)))

        for cat in p.role("category"):
            ref, pq = f"{Pt}.{q(cat['name'])}", p.qual(cat)
            question = _pick(
                rng,
                [
                    f"For each {pq}, how many {cP} are there?",
                    f"How many {cP} are there for each {pq}?",
                    f"Count the {cP} by {pq}.",
                ],
            )
            sql = f"SELECT {ref}, COUNT(*) {base} GROUP BY {ref}"
            out.append(_Cand("join", "group_count", c.name, sql, question, [(p.name, cat["name"])], 1, grouped=True))

            nums = c.role("number")
            if nums:
                n = rng.choice(nums)
                agg = rng.choice(_aggs(n))
                nw = words(n["name"])
                aw = _agg_phrase(agg, nw, rng)
                question = _pick(
                    rng,
                    [
                        f"For each {pq}, what is the {aw} of {cP}?",
                        f"What is the {aw} of {cP} for each {pq}?",
                    ],
                )
                sql = f"SELECT {ref}, {agg}({C}.{q(n['name'])}) {base} GROUP BY {ref}"
                cols = [(p.name, cat["name"]), (c.name, n["name"])]
                out.append(_Cand("join", "group_agg", c.name, sql, question, cols, 1, grouped=True))

    # Two joins: grandchild -> child -> parent.
    for lower in d.links:
        for upper in d.links:
            if upper.child is not lower.parent:
                continue
            g, c, p = lower.child, lower.parent, upper.parent
            base = f"FROM {q(g.name)} JOIN {q(c.name)} ON {lower.on} JOIN {q(p.name)} ON {upper.on}"
            pconds = _conds(p, rng, qualify=True, roles=("category", "number"))
            if not pconds:
                continue
            pc = rng.choice(pconds)
            question = _pick(
                rng,
                [
                    f"How many {g.plural} are linked to {c.plural} whose {p.sing} has {pc.has}?",
                    f"Count the {g.plural} whose {c.sing} belongs to {_a(p.sing)} {pc.whose}.",
                ],
            )
            out.append(
                _Cand(
                    "join",
                    "chain_count",
                    g.name,
                    f"SELECT COUNT(*) {base} WHERE {pc.sql}",
                    question,
                    [*pc.columns],
                    2,
                    conditions=(pc.sql,),
                )
            )
            nums = g.role("number")
            if nums:
                n, pc = rng.choice(nums), rng.choice(pconds)
                agg = rng.choice(_aggs(n))
                nw = words(n["name"])
                aw = _agg_phrase(agg, nw, rng)
                question = _pick(
                    rng,
                    [
                        f"What is the {aw} of {g.plural} linked to {c.plural} whose {p.sing} has {pc.has}?",
                        f"Among {g.plural} whose {c.sing} belongs to {_a(p.sing)} {pc.whose}, what is the {aw}?",
                    ],
                )
                sql = f"SELECT {agg}({q(g.name)}.{q(n['name'])}) {base} WHERE {pc.sql}"
                cols = [(g.name, n["name"]), *pc.columns]
                out.append(_Cand("join", "chain_agg", g.name, sql, question, cols, 2, conditions=(pc.sql,)))
    return out


def _top_words(desc: bool, rng: random.Random) -> str:
    return rng.choice(("highest", "largest")) if desc else rng.choice(("lowest", "smallest"))


def _order_limit(d: _Design, rng: random.Random) -> list[_Cand]:
    out = []
    for t in d.tables:
        T, P = q(t.name), t.plural
        for n in t.role("number"):
            N, desc = rng.choice((3, 5)), rng.random() < 0.7
            order = f"ORDER BY {q(n['name'])} {'DESC' if desc else 'ASC'}"
            tie = f"SELECT {q(n['name'])} FROM {T} {order} LIMIT {N + 1}"
            nw, top = words(n["name"]), _top_words(desc, rng)
            if t.name_cols:
                question = _pick(
                    rng,
                    [
                        f"Which {N} {P} have the {top} {nw}? Give their {t.name_words}.",
                        f"List the {t.name_words} of the {N} {P} with the {top} {nw}.",
                    ],
                )
                sql = f"SELECT {_names_sql(t)} FROM {T} {order} LIMIT {N}"
                cols = [*_name_columns(t), (t.name, n["name"])]
                out.append(_Cand("order_limit", "top", t.name, sql, question, cols, limit=N, tie_sql=tie))

                cats = t.role("category")
                if cats:
                    c = _cat_cond(t, rng.choice(cats), rng, False)
                    N = rng.choice((2, 3))
                    question = _pick(
                        rng,
                        [
                            f"Among {P} {c.whose}, which {N} have the {top} {nw}? Give their {t.name_words}.",
                            f"List the {t.name_words} of the {N} {P} with the {top} {nw} among those {c.whose}.",
                        ],
                    )
                    sql = f"SELECT {_names_sql(t)} FROM {T} WHERE {c.sql} {order} LIMIT {N}"
                    tie_w = f"SELECT {q(n['name'])} FROM {T} WHERE {c.sql} {order} LIMIT {N + 1}"
                    cols = [*_name_columns(t), (t.name, n["name"]), *c.columns]
                    out.append(
                        _Cand(
                            "order_limit",
                            "top_where",
                            t.name,
                            sql,
                            question,
                            cols,
                            limit=N,
                            tie_sql=tie_w,
                            conditions=(c.sql,),
                        )
                    )
            else:
                question = _pick(
                    rng,
                    [
                        f"What are the {N} {top} {nw} values among {P}?",
                        f"List the {N} {top} values of {nw} in {P}.",
                    ],
                )
                sql = f"SELECT {q(n['name'])} FROM {T} {order} LIMIT {N}"
                out.append(
                    _Cand(
                        "order_limit", "top_values", t.name, sql, question, [(t.name, n["name"])], limit=N, tie_sql=tie
                    )
                )

        for cat in t.role("category"):
            N, cw = rng.choice((1, 2, 3)), words(cat["name"])
            C = q(cat["name"])
            if N == 1:
                question = _pick(rng, [f"Which {cw} has the most {P}?", f"What is the most common {cw} among {P}?"])
            else:
                question = _pick(
                    rng,
                    [
                        f"Which {N} {plural_words(cw)} have the most {P}?",
                        f"List the {N} most common {plural_words(cw)} among {P}.",
                    ],
                )
            tail = f"FROM {T} GROUP BY {C} ORDER BY COUNT(*) DESC LIMIT"
            out.append(
                _Cand(
                    "order_limit",
                    "top_category",
                    t.name,
                    f"SELECT {C} {tail} {N}",
                    question,
                    [(t.name, cat["name"])],
                    grouped=True,
                    limit=N,
                    tie_sql=f"SELECT COUNT(*) {tail} {N + 1}",
                )
            )

    for link in d.links:
        c, p = link.child, link.parent
        if not p.name_cols:
            continue
        Pt = q(p.name)
        base = f"FROM {Pt} JOIN {q(c.name)} ON {link.on} GROUP BY {Pt}.{q(p.pk)}"
        N = rng.choice((1, 3, 5))
        if N == 1:
            question = _pick(
                rng,
                [
                    f"Which {p.sing} has the most {c.plural}? Give its {p.name_word}.",
                    f"What is the {p.name_word} of the {p.sing} with the most {c.plural}?",
                ],
            )
        else:
            question = _pick(
                rng,
                [
                    f"Which {N} {p.plural} have the most {c.plural}? Give their {p.name_words}.",
                    f"List the {p.name_words} of the {N} {p.plural} with the most {c.plural}.",
                ],
            )
        out.append(
            _Cand(
                "order_limit",
                "join_top_count",
                p.name,
                f"SELECT {_names_sql(p, True)} {base} ORDER BY COUNT(*) DESC LIMIT {N}",
                question,
                _name_columns(p),
                joins=1,
                grouped=True,
                limit=N,
                tie_sql=f"SELECT COUNT(*) {base} ORDER BY COUNT(*) DESC LIMIT {N + 1}",
                unique_names=(p.name,),
            )
        )
        sums = [n for n in c.role("number") if "SUM" in _aggs(n)]
        if sums:
            n, N = rng.choice(sums), rng.choice((3, 5))
            expr, nw = f"SUM({q(c.name)}.{q(n['name'])})", words(n["name"])
            total, cP = _agg_phrase("SUM", nw, rng), c.plural
            question = _pick(
                rng,
                [
                    f"Which {N} {p.plural} have the highest {total} across their {cP}? Give their {p.name_words}.",
                    f"List the {p.name_words} of the {N} {p.plural} whose {cP} add up to the highest {total}.",
                ],
            )
            out.append(
                _Cand(
                    "order_limit",
                    "join_top_sum",
                    p.name,
                    f"SELECT {_names_sql(p, True)} {base} ORDER BY {expr} DESC LIMIT {N}",
                    question,
                    [*_name_columns(p), (c.name, n["name"])],
                    joins=1,
                    grouped=True,
                    limit=N,
                    tie_sql=f"SELECT {expr} {base} ORDER BY {expr} DESC LIMIT {N + 1}",
                    unique_names=(p.name,),
                )
            )
    return out


def _subquery(d: _Design, rng: random.Random) -> list[_Cand]:
    out = []
    for t in d.tables:
        T, P = q(t.name), t.plural
        for n in t.role("number"):
            Nc, nw = q(n["name"]), words(n["name"])
            above = rng.random() < 0.7
            op, side = (">", "above") if above else ("<", "below")
            avg = f"(SELECT AVG({Nc}) FROM {T})"
            vs_avg = (f"{Nc} {op} {avg}",)
            question = _pick(
                rng,
                [
                    f"How many {P} have {_some(nw)} {side} the {_avg_of(nw)} of all {P}?",
                    f"Count the {P} whose {nw} {_is(nw)} {side} the average across all {P}.",
                ],
            )
            sql = f"SELECT COUNT(*) FROM {T} WHERE {Nc} {op} {avg}"
            cols = [(t.name, n["name"])]
            out.append(_Cand("subquery", "avg_count", t.name, sql, question, cols, nested=True, conditions=vs_avg))
            if not t.name_cols:
                continue
            question = _pick(
                rng,
                [
                    f"Which {P} have {_some(nw)} {side} the average for all {P}? Give their {t.name_words}.",
                    f"List the {t.name_words} of {P} whose {nw} {_is(nw)} {side} the {_avg_of(nw)} of all {P}.",
                ],
            )
            sql = f"SELECT {_names_sql(t)} FROM {T} WHERE {Nc} {op} {avg}"
            cols = [*_name_columns(t), (t.name, n["name"])]
            out.append(_Cand("subquery", "avg_list", t.name, sql, question, cols, nested=True, conditions=vs_avg))

            agg = "MAX" if rng.random() < 0.6 else "MIN"
            top = _top_words(agg == "MAX", rng)
            question = _pick(
                rng,
                [
                    f"Which {t.sing} has the {top} {nw}? Give its {t.name_word}.",
                    f"What is the {t.name_word} of the {t.sing} with the {top} {nw}?",
                ],
            )
            top_cond = f"{Nc} = (SELECT {agg}({Nc}) FROM {T})"
            sql = f"SELECT {_names_sql(t)} FROM {T} WHERE {top_cond}"
            out.append(
                _Cand(
                    "subquery",
                    "extreme",
                    t.name,
                    sql,
                    question,
                    cols,
                    nested=True,
                    single_row=True,
                    conditions=(top_cond,),
                )
            )

        cats, nums = t.role("category"), t.role("number")
        if cats and nums:
            c, n = _cat_cond(t, rng.choice(cats), rng, False), rng.choice(nums)
            Nc, nw = q(n["name"]), words(n["name"])
            question = _pick(
                rng,
                [
                    f"How many {P} {c.whose} have {_some(nw)} above the {_avg_of(nw)} of all {P}?",
                    f"Among {P} {c.whose}, how many have {_some(nw)} higher than the average across all {P}?",
                ],
            )
            above_avg = f"{Nc} > (SELECT AVG({Nc}) FROM {T})"
            sql = f"SELECT COUNT(*) FROM {T} WHERE {c.sql} AND {above_avg}"
            cols, conds = [*c.columns, (t.name, n["name"])], (c.sql, above_avg)
            out.append(_Cand("subquery", "where_avg", t.name, sql, question, cols, nested=True, conditions=conds))

    for link in d.links:
        c, p = link.child, link.parent
        Pt, C, Fk = q(p.name), q(c.name), q(link.fk)
        cconds = _conds(c, rng, roles=("category", "number"))
        if cconds:
            cc = rng.choice(cconds)
            inner = f"(SELECT {Fk} FROM {C} WHERE {cc.sql})"
            question = _pick(
                rng,
                [
                    f"How many {p.plural} have at least one {c.sing} {cc.whose}?",
                    f"Count the {p.plural} that have {_a(c.sing)} {cc.whose}.",
                ],
            )
            sql = f"SELECT COUNT(*) FROM {Pt} WHERE {q(p.pk)} IN {inner}"
            conds = (f"{q(p.pk)} IN {inner}", cc.sql)
            out.append(
                _Cand("subquery", "in_count", p.name, sql, question, [*cc.columns], nested=True, conditions=conds)
            )
            if p.name_cols:
                cc = rng.choice(cconds)
                inner = f"(SELECT {Fk} FROM {C} WHERE {cc.sql})"
                question = _pick(
                    rng,
                    [
                        f"Which {p.plural} have at least one {c.sing} {cc.whose}? Give their {p.name_words}.",
                        f"List the {p.name_words} of {p.plural} that have {_a(c.sing)} {cc.whose}.",
                    ],
                )
                sql = f"SELECT {_names_sql(p)} FROM {Pt} WHERE {q(p.pk)} IN {inner}"
                cols = [*_name_columns(p), *cc.columns]
                out.append(
                    _Cand(
                        "subquery",
                        "in_list",
                        p.name,
                        sql,
                        question,
                        cols,
                        nested=True,
                        unique_names=(p.name,),
                        conditions=(f"{q(p.pk)} IN {inner}", cc.sql),
                    )
                )
        if p.name_cols:
            question = _pick(
                rng,
                [
                    f"Which {p.plural} have no {c.plural}? Give their {p.name_words}.",
                    f"List the {p.name_words} of {p.plural} without any {c.plural}.",
                ],
            )
            none = f"{q(p.pk)} NOT IN (SELECT {Fk} FROM {C})"
            sql = f"SELECT {_names_sql(p)} FROM {Pt} WHERE {none}"
            out.append(
                _Cand("subquery", "not_in", p.name, sql, question, _name_columns(p), nested=True, conditions=(none,))
            )
    return out


def _full_years(col: dict) -> list[int]:
    lo, hi = datetime.date.fromisoformat(col["min"]), datetime.date.fromisoformat(col["max"])
    first = lo.year if (lo.month, lo.day) == (1, 1) else lo.year + 1
    last = hi.year if (hi.month, hi.day) == (12, 31) else hi.year - 1
    return list(range(first, last + 1))


def _month_window(col: dict, rng: random.Random) -> tuple[str, str] | None:
    """A window of whole months inside the column's range, about a fifth to two fifths of it."""
    lo, hi = datetime.date.fromisoformat(col["min"]), datetime.date.fromisoformat(col["max"])
    start_m = lo.year * 12 + lo.month - 1 + (lo.day > 1)
    end_m = hi.year * 12 + hi.month - 1 - (hi.day < calendar.monthrange(hi.year, hi.month)[1])
    total = end_m - start_m + 1
    if total < 3:
        return None
    length = max(2, round(total * rng.uniform(0.2, 0.4)))
    s = rng.randint(start_m, end_m - length + 1)
    e = s + length - 1
    sy, sm = divmod(s, 12)
    ey, em = divmod(e, 12)
    return (
        datetime.date(sy, sm + 1, 1).isoformat(),
        datetime.date(ey, em + 1, calendar.monthrange(ey, em + 1)[1]).isoformat(),
    )


def _date_range(d: _Design, rng: random.Random) -> list[_Cand]:
    out = []
    for t in d.tables:
        T, P = q(t.name), t.plural
        for dc in t.role("date"):
            D, dw = q(dc["name"]), date_words(dc["name"])
            col = [(t.name, dc["name"])]
            years = _full_years(dc)
            if years and len(years) <= 15:
                y = rng.choice(years)
                where = f"strftime('%Y', {D}) = '{y}'"
                question = _pick(
                    rng,
                    [f"How many {P} have {_some(dw)} in {y}?", f"Count the {P} whose {dw} falls in {y}."],
                )
                out.append(
                    _Cand(
                        "date_range",
                        "year_count",
                        t.name,
                        f"SELECT COUNT(*) FROM {T} WHERE {where}",
                        question,
                        col,
                        conditions=(where,),
                    )
                )
                nums = t.role("number")
                if nums:
                    n, y = rng.choice(nums), rng.choice(years)
                    where = f"strftime('%Y', {D}) = '{y}'"
                    agg = rng.choice(_aggs(n))
                    nw = words(n["name"])
                    aw = _agg_phrase(agg, nw, rng)
                    question = _pick(
                        rng,
                        [
                            f"What is the {aw} of {P} with {_some(dw)} in {y}?",
                            f"For {P} whose {dw} falls in {y}, what is the {aw}?",
                        ],
                    )
                    sql = f"SELECT {agg}({q(n['name'])}) FROM {T} WHERE {where}"
                    cols = [(t.name, n["name"]), *col]
                    out.append(_Cand("date_range", "year_agg", t.name, sql, question, cols, conditions=(where,)))

            window = _month_window(dc, rng)
            if window:
                a, b = window
                where = f"{D} BETWEEN '{a}' AND '{b}'"
                question = _pick(
                    rng,
                    [
                        f"How many {P} have {_some(dw)} between {fmt_date(a)} and {fmt_date(b)}, inclusive?",
                        f"Count the {P} whose {dw} {_is(dw)} from {fmt_date(a)} through {fmt_date(b)}.",
                    ],
                )
                out.append(
                    _Cand(
                        "date_range",
                        "between_count",
                        t.name,
                        f"SELECT COUNT(*) FROM {T} WHERE {where}",
                        question,
                        col,
                        conditions=(where,),
                    )
                )
                if t.name_cols:
                    a, b = _month_window(dc, rng)
                    where = f"{D} BETWEEN '{a}' AND '{b}'"
                    start, end, is_ = fmt_date(a), fmt_date(b), _is(dw)
                    question = _pick(
                        rng,
                        [
                            f"List the {t.name_words} of {P} whose {dw} {is_} between {start} and {end}, inclusive.",
                            f"Which {P} have {_some(dw)} from {start} through {end}? Give their {t.name_words}.",
                        ],
                    )
                    sql = f"SELECT {_names_sql(t)} FROM {T} WHERE {where}"
                    cols = [*_name_columns(t), *col]
                    out.append(_Cand("date_range", "between_list", t.name, sql, question, cols, conditions=(where,)))

            if len(years) >= 2:
                y = rng.choice(years[1:])
                cut = f"{y}-01-01"
                if rng.random() < 0.5:
                    where, said = f"{D} >= '{cut}'", f"on or after {fmt_date(cut)}"
                else:
                    where, said = f"{D} < '{cut}'", f"before {fmt_date(cut)}"
                cats = t.role("category")
                if cats:
                    c = _cat_cond(t, rng.choice(cats), rng, False)
                    question = _pick(
                        rng,
                        [
                            f"How many {P} {c.whose} have {_some(dw)} {said}?",
                            f"Count the {P} {c.whose} and whose {dw} {_is(dw)} {said}.",
                        ],
                    )
                    sql = f"SELECT COUNT(*) FROM {T} WHERE {c.sql} AND {where}"
                    cols, conds = [*c.columns, *col], (c.sql, where)
                    out.append(_Cand("date_range", "cutoff_where", t.name, sql, question, cols, conditions=conds))
                else:
                    question = _pick(
                        rng, [f"How many {P} have {_some(dw)} {said}?", f"Count the {P} whose {dw} {_is(dw)} {said}."]
                    )
                    out.append(
                        _Cand(
                            "date_range",
                            "cutoff_count",
                            t.name,
                            f"SELECT COUNT(*) FROM {T} WHERE {where}",
                            question,
                            col,
                            conditions=(where,),
                        )
                    )
    return out


_SHAPE_BUILDERS = {
    "filter": _filter,
    "aggregate": _aggregate,
    "group_by": _group_by,
    "join": _join,
    "order_limit": _order_limit,
    "subquery": _subquery,
    "date_range": _date_range,
}


def _shape_key(sql: str) -> str:
    """The query with its literals, aggregate, comparison, and direction blanked: what stays fixed across versions."""
    sql = re.sub(r"'[^']*'", "?", sql)
    sql = re.sub(r"(?<![\w.])-?\d+(\.\d+)?(?![\w.])", "#", sql)
    sql = re.sub(r"\b(SUM|AVG|MIN|MAX)\(", "AGG(", sql)
    sql = re.sub(r" [<>] ", " CMP ", sql)
    return re.sub(r" (DESC|ASC) ", " DIR ", sql)


def _slots(design: _Design, schema_id: str, seed: int) -> dict[str, list[list[_Cand]]]:
    """Candidates per shape, each slot holding up to VERSIONS versions with different literals.

    Every template runs VERSIONS times with its own random generator; versions of one query
    (same template, same tables and columns) share a slot, in the order they were drawn.
    """
    slots: dict[str, dict[tuple[str, str], list[_Cand]]] = {shape: {} for shape in SHAPES}
    for k in range(VERSIONS):
        rng = random.Random(f"{seed}:{schema_id}:answerable:{k}")
        for shape in SHAPES:
            for c in _SHAPE_BUILDERS[shape](design, rng):
                versions = slots[shape].setdefault((c.template, _shape_key(c.sql)), [])
                if all(v.sql != c.sql for v in versions):
                    versions.append(c)
    return {shape: list(by_key.values()) for shape, by_key in slots.items()}


def _spread(slots: list[list[_Cand]], rng: random.Random) -> list[list[_Cand]]:
    """Shuffle, then interleave by template so one template does not crowd out the others in a shape."""
    rng.shuffle(slots)
    groups: dict[str, list[list[_Cand]]] = {}
    for slot in slots:
        groups.setdefault(slot[0].template, []).append(slot)
    order = list(groups)
    rng.shuffle(order)
    out = []
    while any(groups.values()):
        for key in order:
            if groups[key]:
                out.append(groups[key].pop(0))
    return out


# ---------------------------------------------------------------- validation


def _trivial(rows: list[tuple]) -> bool:
    """One row holding only NULL or 0: an aggregate over nothing, or a count that found nothing."""
    return len(rows) == 1 and all(v is None or v == 0 for v in rows[0])


def _names_unique(t: _Table, paths: list[Path], cache: dict[str, bool]) -> bool:
    if t.name not in cache:
        cols = _names_sql(t)
        sql = f"SELECT COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT {cols} FROM {q(t.name)})) FROM {q(t.name)}"
        cache[t.name] = all(r.status == "ok" and r.rows[0][0] == 0 for r in run_on_copies(paths, sql))
    return cache[t.name]


def without(sql: str, condition: str) -> str:
    """The query with one WHERE predicate or HAVING clause removed."""
    for old, new in (
        (f" WHERE {condition} AND ", " WHERE "),
        (f" AND {condition}", ""),
        (f" WHERE {condition}", ""),
        (f" HAVING {condition}", ""),
    ):
        # The condition must end where a clause ends, so "x > 5" never matches inside "x > 50".
        pattern = re.escape(old) + ("" if old.endswith(" ") else r"(?=$|[ )])")
        loose, n = re.subn(pattern, lambda _m, new=new: new, sql, count=1)
        if n:
            return loose
    raise ValueError(f"{condition!r} is not a condition of {sql!r}")


def _answer(rows: list[tuple]) -> Counter:
    """A result as a multiset of rows, floats rounded, for comparing two answers."""
    return Counter(tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows)


def _idle_condition(c: _Cand, paths: list[Path], results: list) -> str | None:
    """The first condition whose removal changes the answer on fewer than MIN_COPIES_CHANGED copies, if any.

    A model that ignored such a condition would match the gold answer on most copies and be scored correct.
    """
    for condition in c.conditions:
        loose = run_on_copies(paths, without(c.sql, condition))
        changed = sum(
            lo.status != "ok" or _answer(lo.rows) != _answer(r.rows) for r, lo in zip(results, loose, strict=True)
        )
        if changed < MIN_COPIES_CHANGED:
            return condition
    return None


def _check(
    c: _Cand, paths: list[Path], tables: dict[str, _Table], unique: dict[str, bool], conditions: bool = True
) -> tuple[list[int] | None, str]:
    """Row counts per copy and "ok" if the candidate passes every check, else None and the check it failed.

    With `conditions` off, a candidate with an idle condition still passes, marked "idle", so a run can count them.
    """
    if not all(_names_unique(tables[name], paths, unique) for name in c.unique_names):
        return None, "names"
    results = run_on_copies(paths, c.sql)
    counts = []
    for r in results:
        if r.status != "ok" or not r.rows or _trivial(r.rows):
            return None, "empty"
        if (c.single_row and len(r.rows) != 1) or (c.limit is not None and len(r.rows) != c.limit):
            return None, "row_count"
        counts.append(len(r.rows))
    if c.tie_sql:
        for r in run_on_copies(paths, c.tie_sql):
            values = [row[0] for row in r.rows or []]
            if r.status != "ok" or len(set(values)) != len(values):
                return None, "tie"
    if c.conditions and _idle_condition(c, paths, results):
        return (None, "condition") if conditions else (counts, "idle")
    return counts, "ok"


def _validate(c: _Cand, paths: list[Path], tables: dict[str, _Table], unique: dict[str, bool]) -> list[int] | None:
    """Row counts per copy if the candidate passes every check, else None."""
    return _check(c, paths, tables, unique)[0]


# ---------------------------------------------------------------- the entry point


def _record(schema: dict, number: int, c: _Cand, counts: list[int]) -> dict:
    seen, columns = set(), []
    for table, column in c.columns:
        if (table, column) not in seen:
            seen.add((table, column))
            columns.append({"table": table, "column": column})
    return {
        "question_id": f"{schema['schema_id']}__a{number:03d}",
        "schema_id": schema["schema_id"],
        "family": schema["family"],
        "topic": schema["topic"],
        "kind": "answerable",
        "question": c.question,
        "gold_sql": c.sql,
        "shape": c.shape,
        "difficulty": c.difficulty,
        "columns_used": columns,
        "hidden_columns": [],
        "decline_subtype": None,
        "decline_reason": None,
        "source_question_id": None,
        "gold_row_counts": counts,
    }


def answerable_questions(schema: dict, paths: list[Path], seed: int = 0, target: int = DEFAULT_TARGET) -> list[dict]:
    """Up to `target` validated answerable records for one design, spread across the 7 shapes."""
    return generate(schema, paths, seed, target)[0]


def generate(
    schema: dict, paths: list[Path], seed: int = 0, target: int = DEFAULT_TARGET, check_conditions: bool = True
) -> tuple[list[dict], Counter]:
    """The records, and a count of every version tried by outcome: "ok", or the check it failed.

    With `check_conditions` off, kept records with an idle condition are counted as "idle" instead of rejected.
    """
    rng = random.Random(f"{seed}:{schema['schema_id']}:answerable")
    design = _Design(schema)
    tables = {t.name: t for t in design.tables}
    slots = _slots(design, schema["schema_id"], seed)
    pools = {shape: _spread(slots[shape], rng) for shape in SHAPES}
    order = list(SHAPES)
    rng.shuffle(order)

    kept: list[tuple[_Cand, list[int]]] = []
    seen_sql: set[str] = set()
    seen_question: set[str] = set()
    failures, outcomes = Counter(), Counter()
    unique: dict[str, bool] = {}
    while len(kept) < target and any(pools[s] and failures[s] < MAX_FAILURES_PER_SHAPE for s in order):
        for shape in order:
            # Take the next slot of this shape with a version that passes, if any.
            while len(kept) < target and pools[shape] and failures[shape] < MAX_FAILURES_PER_SHAPE:
                found = None
                for c in pools[shape].pop(0):
                    if c.sql in seen_sql or c.question in seen_question:
                        continue
                    counts, outcome = _check(c, paths, tables, unique, check_conditions)
                    outcomes[outcome] += 1
                    if counts is not None:
                        found = (c, counts)
                        break
                if found is None:
                    failures[shape] += 1
                    continue
                seen_sql.add(found[0].sql)
                seen_question.add(found[0].question)
                kept.append(found)
                break
    records = [_record(schema, i, c, counts) for i, (c, counts) in enumerate(kept, start=1)]
    return records, outcomes


# ---------------------------------------------------------------- summary run


def _slowest_ms(sql: str, paths: list[Path]) -> float:
    worst = 0.0
    for p in paths:
        start = time.perf_counter()
        run_readonly(p, sql)
        worst = max(worst, (time.perf_counter() - start) * 1000)
    return worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write answerable questions for every design and print a summary.")
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT, help="folder holding the designs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--out", type=Path, help="optional JSONL file for the records")
    parser.add_argument(
        "--no-condition-check",
        action="store_true",
        help="keep queries with an idle condition, and count them (to measure the check)",
    )
    args = parser.parse_args(argv)

    records, short, slowest, outcomes = [], [], (0.0, ""), Counter()
    for schema_file in sorted(args.root.glob("*/schema.json")):
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        paths = copy_paths(schema_file.parent)
        if not all(p.exists() for p in paths):
            continue
        found, tried = generate(schema, paths, args.seed, args.target, not args.no_condition_check)
        outcomes += tried
        if len(found) < args.target:
            short.append((schema["schema_id"], len(found)))
        for r in found:
            ms = _slowest_ms(r["gold_sql"], paths)
            if ms > slowest[0]:
                slowest = (ms, r["question_id"])
        records.extend(found)

    print(f"{len(records)} records from {len(records) and len({r['schema_id'] for r in records})} designs")
    print("by shape: " + ", ".join(f"{s} {n}" for s, n in Counter(r["shape"] for r in records).most_common()))
    print("by difficulty: " + ", ".join(f"{s} {n}" for s, n in Counter(r["difficulty"] for r in records).most_common()))
    print(f"designs short of {args.target}: {len(short)} {short[:10]}")
    print(f"slowest gold query: {slowest[0]:.1f} ms ({slowest[1]})")
    print("versions tried, by outcome: " + ", ".join(f"{k} {n}" for k, n in outcomes.most_common()))
    if args.out:
        with args.out.open("w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
