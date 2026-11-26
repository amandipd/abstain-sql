"""Write the questions a model should decline, for each design (plan step 1.5).

The model must learn when not to answer. I write 3 kinds of questions it should
decline, each with a subtype and a one-line reason, following TrustSQL's
taxonomy (results/external_data.md) plus an ambiguous kind TrustSQL lacks:

    unanswerable   hidden_column       an answerable question, shown with a schema that lacks
                                       a column its gold query reads
                   missing_attribute   asks about an attribute the design never had, such as
                                       patients' blood type
    ambiguous      undefined_metric    "Which clinic performs best?" when the table offers 2 or
                                       more measures and nothing says which one "best" means
                   vague_reference     "Which accounts were opened recently?": a cutoff with
                                       no defined value
    out_of_scope   action              a request to do something, not read data: email, delete
                   forecast            the future, which the database cannot hold
                   analysis            clustering, causes, predictive models, written summaries
                   outside_knowledge   real-world facts about values in the database

Every question uses the design's own table and column words, so it reads like a
plausible request to that database rather than an obvious outlier.

Rules
-----
- Hidden columns are never keys. From the source question's columns_used I hide
  the column the question names most clearly; a name column counts only if the
  question names it, because a name used just for display can be swapped for an
  ID and leave the question answerable.
- A missing attribute must be truly absent: none of its stems may appear in any
  table or column name of the design, so it cannot be read or counted from
  another table either.
- Amounts: about 0.43 decline records per answerable record, so declines are
  about 30% of the mix, split evenly across the 3 kinds and spread across
  subtypes. A subtype a design cannot support is skipped and its share goes to
  the kind's other subtypes. No 2 records share a question text, and only a
  hidden_column record may repeat an answerable question's text.

Each design's random generator is seeded from the seed and the design's ID, so
the same input always gives the same records.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from data.generator.questions import article, singular_word, words

DECLINE_RATIO = 0.43  # 0.43 declines per answerable is 30% of the mix
KINDS = ("unanswerable", "ambiguous", "out_of_scope")
PREFIX = {"unanswerable": "u", "ambiguous": "m", "out_of_scope": "o"}
SUBTYPES = {
    "unanswerable": ("hidden_column", "missing_attribute"),
    "ambiguous": ("undefined_metric", "vague_reference"),
    "out_of_scope": ("action", "forecast", "analysis", "outside_knowledge"),
}
KEY_ROLES = frozenset({"id", "fk"})
NAME_ROLES = frozenset({"entity_name", "first_name", "last_name"})

# ---------------------------------------------------------------- English helpers

MEMBER_NOUNS = {"staff", "crew", "personnel", "talent"}
SAME_PLURAL = {"aircraft", "series", "species"}
PIECE_NOUNS = {"equipment", "machinery"}
# Numbers that label a row rather than measure it.
LABEL_TOKENS = {
    "number",
    "year",
    "sequence",
    "round",
    "pick",
    "season",
    "quarter",
    "floor",
    "code",
    "position",
    "rank",
    "level",
    "latitude",
    "longitude",
    "slot",
    "placement",
}
MONEY_TOKENS = {
    "cost",
    "price",
    "fee",
    "amount",
    "revenue",
    "salary",
    "wage",
    "budget",
    "premium",
    "rent",
    "fare",
    "payment",
    "charge",
    "income",
    "spend",
    "earnings",
    "payout",
    "capex",
    "principal",
    "balance",
    "valuation",
    "deposit",
    "royalty",
    "bonus",
    "tip",
    "payroll",
    "receipts",
    "sales",
    "winnings",
}
SIZE_TOKENS = {
    "capacity",
    "area",
    "volume",
    "tonnes",
    "tons",
    "size",
    "weight",
    "length",
    "acreage",
    "acres",
    "hectares",
    "population",
    "depth",
    "diameter",
}
# Plural units: a column ending in one is a measure, not a countable noun.
UNIT_PLURALS = {
    "hours",
    "days",
    "minutes",
    "months",
    "years",
    "seconds",
    "weeks",
    "miles",
    "acres",
    "tonnes",
    "tons",
    "millions",
    "billions",
    "nights",
    "liters",
    "meters",
    "points",
    "stars",
}
# Events whose "best" is really "worst".
NEGATIVE_HEADS = {
    "incident",
    "outage",
    "spill",
    "break",
    "failure",
    "accident",
    "recall",
    "complaint",
    "fault",
    "closure",
    "violation",
    "outbreak",
    "wildfire",
    "drought",
    "delay",
    "injury",
    "defect",
    "warning",
    "correction",
    "erratum",
    "loss",
}
PARTICIPLES = {
    "hire": "hired",
    "join": "joined",
    "open": "opened",
    "close": "closed",
    "launch": "launched",
    "install": "installed",
    "purchase": "purchased",
    "publication": "published",
    "registration": "registered",
    "enrollment": "enrolled",
    "admission": "admitted",
    "admit": "admitted",
    "signing": "signed",
    "build": "built",
    "built": "built",
    "acquisition": "acquired",
    "activation": "activated",
    "completion": "completed",
    "deployment": "deployed",
    "move": "moved",
    "sold": "sold",
    "diagnosis": "diagnosed",
    "inception": "started",
    "start": "started",
}
# Verbs that take no passive: "Which supporters joined recently?", not "were joined".
INTRANSITIVE = {"joined", "subscribed", "occurred", "arrived", "departed", "debuted", "retired", "failed"}
TEAMS = ("finance team", "operations team", "management team", "regional managers", "audit committee")
PERIODS = ("month", "quarter", "year")


def _a(phrase: str) -> str:
    return f"{article(phrase)} {phrase}"


def _of(word: str, phrase: str) -> str:
    """'total' + 'prize pool' -> 'total prize pool', but never 'total total prize pool'."""
    return phrase if phrase.split(" ")[0] == word else f"{word} {phrase}"


def _is_plural_phrase(phrase: str) -> bool:
    """'total hectares' is plural; 'capacity (MW)' and 'status' are not. A bracketed unit is ignored."""
    last = phrase.split(" (")[0].split(" ")[-1]
    return last.endswith("s") and not last.endswith(("ss", "us", "is"))


def _adj(adj: str, phrase: str) -> str:
    """An adjective before a column phrase, with 'a' or 'an' only when the phrase is singular:
    'a high capacity (MW)', 'an old opened date', but 'high total hectares'."""
    return f"{adj} {phrase}" if _is_plural_phrase(phrase) else _a(f"{adj} {phrase}")


def _say(value: str) -> str:
    """A category value quoted as data, the way the answerable questions write it."""
    return f'"{value}"' if "'" in value else f"'{value}'"


def _tokens(name: str) -> list[str]:
    return name.lower().split("_")


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def nouns(table_name: str) -> tuple[str, str]:
    """A table name as a singular and a plural noun: 'funding_awards' -> ('funding award', 'funding awards')."""
    tokens = table_name.split("_")
    head, last = tokens[:-1], tokens[-1]
    full = words(table_name)
    if last in MEMBER_NOUNS:
        return f"{full} member", f"{full} members"
    if last in SAME_PLURAL:
        return full, full
    if last in PIECE_NOUNS:
        return f"piece of {full}", f"pieces of {full}"
    if last.endswith("s") and not last.endswith(("ss", "us", "is")) or last in ("errata", "lps", "skus", "atms"):
        sing = singular_word(last)
        return (f"{words('_'.join(head))} {words(sing)}" if head else words(sing)), full
    return f"{full} record", f"{full} records"


def head_noun(table_name: str) -> str:
    """The singular last word of a table name, used to look up attributes: 'power_plants' -> 'plant'."""
    return singular_word(table_name.split("_")[-1])


def _is_person(table: dict) -> bool:
    return any(c["role"] == "first_name" for c in table["columns"])


def _is_named(table: dict) -> bool:
    return any(c["role"] in NAME_ROLES for c in table["columns"])


def _measures(table: dict) -> list[dict]:
    """Number columns that measure something, not labels such as a jersey number or a year."""
    return [c for c in table["columns"] if c["role"] == "number" and not LABEL_TOKENS & set(_tokens(c["name"]))]


def _year_numbers(table: dict) -> list[dict]:
    return [c for c in table["columns"] if c["role"] == "number" and "year" in _tokens(c["name"])]


def _categories(table: dict) -> list[dict]:
    return [c for c in table["columns"] if c["role"] == "category" and c.get("values")]


def _dates(table: dict) -> list[dict]:
    return [c for c in table["columns"] if c["role"] == "date"]


def _is_money(col: dict) -> bool:
    toks = set(_tokens(col["name"]))
    return bool(toks & MONEY_TOKENS) and not toks & {"pct", "percent", "rate", "ratio", "share", "split"}


def _years(col: dict) -> list[int]:
    """The full years inside a date column's range."""
    lo, hi = int(str(col.get("min", "2020"))[:4]), int(str(col.get("max", "2024"))[:4])
    return list(range(lo + 1, hi)) or [lo]


# ---------------------------------------------------------------- the design


@dataclass(frozen=True)
class _Cand:
    question: str
    reason: str
    group: str  # the table it is about; picks are spread across groups
    hidden: tuple[tuple[str, str], ...] = ()
    source: dict | None = None


class _Design:
    def __init__(self, schema: dict, rng: random.Random):
        self.schema = schema
        self.rng = rng
        self.family = schema.get("family", "")
        self.tables = schema["tables"]
        self.by_name = {t["name"]: t for t in self.tables}
        self.columns = {(t["name"], c["name"]): c for t in self.tables for c in t["columns"]}
        self.children: dict[str, list[dict]] = defaultdict(list)
        for t in self.tables:
            for c in t["columns"]:
                if c["role"] == "fk" and c["references"]["table"] != t["name"]:
                    if t not in self.children[c["references"]["table"]]:
                        self.children[c["references"]["table"]].append(t)
        self.names = [t["name"] for t in self.tables] + [c["name"] for t in self.tables for c in t["columns"]]

    def value(self, col: dict) -> str:
        return self.rng.choice(col["values"])


# ---------------------------------------------------------------- unanswerable: hidden column


def _question_words(question: str) -> set[str]:
    found = set(re.findall(r"[a-z0-9]+", question.lower()))
    return found | {singular_word(w) for w in found}


def _mention(question: str, col: dict) -> int:
    """How clearly the question names this column: 3 fully, 2 partly, 0 not at all."""
    qwords = _question_words(question)
    toks = [t for t in _tokens(col["name"]) if len(t) > 2 and t not in {"date", "per", "the", "and"}]
    hits = [t in qwords or singular_word(t) in qwords for t in toks]
    score = 3 if hits and all(hits) else 2 if any(hits) else 0
    low = question.lower()
    if any(str(v).lower() in low for v in col.get("values") or ()):
        score = 3
    return score


def _hidden_column(d: _Design, answerable: list[dict]) -> list[_Cand]:
    out = []
    for rec in answerable:
        options = []
        for used in rec.get("columns_used") or ():
            col = d.columns.get((used.get("table"), used.get("column")))
            if col is None or col["role"] in KEY_ROLES:
                continue
            score = _mention(rec["question"], col)
            if col["role"] in NAME_ROLES:
                if score < 2:
                    continue  # a display-only name can be replaced by an ID
            else:
                score += 1
            options.append((score, used["table"], col))
        if not options:
            continue
        best = max(s for s, _, _ in options)
        _, table, col = d.rng.choice([o for o in options if o[0] == best])
        sing, _ = nouns(table)
        reason = f"the schema has no column for the {sing}'s {words(col['name'])}"
        out.append(_Cand(rec["question"], reason, table, ((table, col["name"]),), rec))
    return out


# ---------------------------------------------------------------- unanswerable: missing attribute


class Attr(NamedTuple):
    """An attribute a table could plausibly have. kind is text, number, category, or flag.

    For a flag, phrase is a plural verb phrase ("have a swimming pool"). stems are
    the name fragments that would mean the design already holds the attribute; the
    content words of the phrase are added to them.
    """

    phrase: str
    kind: str
    stems: tuple[str, ...] = ()
    values: tuple[str, ...] = ()


def _at(phrase: str, kind: str, stems: tuple[str, ...] = (), values: tuple[str, ...] = ()) -> Attr:
    return Attr(phrase, kind, stems, values)


PHRASE_FILLER = {
    "per",
    "and",
    "the",
    "for",
    "new",
    "has",
    "are",
    "number",
    "name",
    "date",
    "type",
    "level",
    "rate",
    "count",
    "total",
    "annual",
    "average",
    "year",
    "years",
    "time",
    "value",
    "each",
    "their",
    "have",
    "with",
    "from",
    "main",
    "usual",
    "typical",
    "current",
    "known",
    "allow",
    "offer",
    "accept",
    "belong",
    "include",
    "person",
    "minutes",
}

# People.
_EMAIL = _at("email address", "text", ("email", "mail"))
_PHONE = _at("phone number", "text", ("phone", "mobile", "contact"))
_HOME_ADDRESS = _at("home address", "text", ("address", "street", "zip", "postal", "residence"))
_BIRTH = _at("date of birth", "text", ("birth", "dob", "age", "born"))
_LANGUAGE = _at("preferred language", "category", ("language", "lang"), ("English", "Spanish", "French", "Mandarin"))
_EMERGENCY = _at("emergency contact", "text", ("emergency", "contact", "next_of_kin"))
PERSON_GENERIC = (_EMAIL, _PHONE, _HOME_ADDRESS, _BIRTH, _LANGUAGE, _EMERGENCY)

_SALARY = _at("annual salary", "number", ("salary", "wage", "pay", "compensation", "earning", "income"))
_EXPERIENCE = _at("years of experience", "number", ("experience", "seniority", "career"))
_UNION = _at("belong to a labor union", "flag", ("union",))
_LICENSE = _at("license number", "text", ("license", "licence", "cdl", "permit", "credential"))
_VIOLATIONS = _at("number of traffic violations", "number", ("violation", "ticket", "citation", "traffic"))
_FLIGHT_HOURS = _at("total flight hours", "number", ("flight", "hours"))
_MED_SCHOOL = _at("medical school", "text", ("school", "educat", "degree", "university", "college"))
_BOARD = _at("board certification", "text", ("board", "certif", "credential"))
_PRACTICE_YEARS = _at("years in practice", "number", ("experience", "practice", "career", "since"))
_DEGREE = _at("highest degree", "category", ("degree", "educat"), ("bachelor's", "master's", "PhD"))
_HINDEX = _at("h-index", "number", ("index", "citation"))
_BLOOD = _at("blood type", "category", ("blood",), ("A positive", "O negative", "B positive", "AB negative"))
_INSURER = _at("insurance provider", "text", ("insur", "payer", "coverage"))
_ALLERGIES = _at("known allergies", "text", ("allerg",))
_DIET_RESTRICT = _at("dietary restrictions", "text", ("diet", "meal", "food"))
_LOYALTY = _at("loyalty points balance", "number", ("loyalty", "point", "reward"))
_INCOME = _at("annual income", "number", ("income", "salary", "earning", "wage"))
_CONTACT_PREF = _at(
    "preferred contact method", "category", ("contact", "channel", "prefer"), ("email", "phone", "text message", "mail")
)
_CREDIT = _at("credit score", "number", ("credit", "fico", "score"))
_EMPLOYER = _at("employer name", "text", ("employ", "company", "occupation", "job"))
_NET_WORTH = _at("net worth", "number", ("worth", "wealth", "asset"))
_PASSPORT = _at(
    "passport country",
    "category",
    ("passport", "country", "nation", "citizen"),
    ("United States", "Canada", "Mexico", "United Kingdom"),
)
_DIET = _at(
    "dietary preference",
    "category",
    ("diet", "vegetarian", "vegan", "meal", "food"),
    ("vegetarian", "vegan", "halal", "gluten-free"),
)
_DEVICE = _at("device type", "category", ("device", "platform"), ("iPhone", "Android phone", "desktop", "tablet"))
_FAV_PLAYER = _at("favorite player", "text", ("favorite", "favourite", "player"))
_FOLLOWERS = _at("number of social media followers", "number", ("follower", "social", "fans"))
_LIT_AGENT = _at("literary agent", "text", ("agent",))
_HOMETOWN = _at("hometown", "text", ("hometown", "home", "city", "origin", "birth"))
_HEIGHT = _at("height", "number", ("height",))
_JERSEY = _at("jersey number", "number", ("jersey", "shirt"))
_COMMISSION = _at("commission rate", "number", ("commission",))
_CERT_LEVEL = _at(
    "certification level", "category", ("certif", "level", "credential"), ("level 1", "level 2", "level 3")
)
_HOURLY = _at("hourly rate", "number", ("hourly", "rate", "fee", "pay", "wage", "salary"))

PERSON_LEX: dict[str, tuple[Attr, ...]] = {
    **dict.fromkeys(("patient", "resident"), (_BLOOD, _INSURER, _ALLERGIES, _DIET_RESTRICT)),
    **dict.fromkeys(
        (
            "employee",
            "staff",
            "worker",
            "technician",
            "farmhand",
            "associate",
            "personnel",
            "crew",
            "crew_member",
            "mechanic",
            "warden",
            "ranger",
            "responder",
            "operator",
            "health_worker",
        ),
        (_SALARY, _EXPERIENCE, _UNION),
    ),
    **dict.fromkeys(("driver", "courier"), (_LICENSE, _VIOLATIONS, _EXPERIENCE)),
    "pilot": (_FLIGHT_HOURS, _LICENSE, _EXPERIENCE),
    **dict.fromkeys(
        (
            "doctor",
            "physician",
            "surgeon",
            "dentist",
            "vet",
            "veterinarian",
            "therapist",
            "provider",
            "clinician",
            "hygienist",
        ),
        (_MED_SCHOOL, _BOARD, _PRACTICE_YEARS),
    ),
    **dict.fromkeys(("scientist", "researcher"), (_DEGREE, _HINDEX, _EXPERIENCE)),
    **dict.fromkeys(
        ("customer", "client", "shopper", "consumer", "buyer", "ticket_buyer"), (_LOYALTY, _INCOME, _CONTACT_PREF)
    ),
    **dict.fromkeys(
        ("borrower", "applicant", "policyholder", "tenant", "renter", "homeowner"), (_CREDIT, _INCOME, _EMPLOYER)
    ),
    **dict.fromkeys(
        ("investor", "founder", "backer", "supporter", "participant", "beneficiary", "partner", "plan_member"),
        (_NET_WORTH, _INCOME, _EMPLOYER),
    ),
    **dict.fromkeys(("guest", "traveler", "passenger", "rider", "visitor", "park_visitor"), (_PASSPORT, _DIET)),
    **dict.fromkeys(("reader", "subscriber", "user", "member"), (_DEVICE, _INCOME, _CONTACT_PREF)),
    "fan": (_FAV_PLAYER, _DEVICE, _INCOME),
    **dict.fromkeys(("player", "pro_player", "athlete"), (_HEIGHT, _JERSEY, _SALARY)),
    **dict.fromkeys(("author", "writer"), (_LIT_AGENT, _FOLLOWERS, _HOMETOWN)),
    **dict.fromkeys(
        ("songwriter", "journalist", "reporter", "artist", "performer", "creator", "talent"), (_FOLLOWERS, _HOMETOWN)
    ),
    **dict.fromkeys(
        ("agent", "realtor", "broker", "advisor", "adjuster", "underwriter", "loan_officer", "financial_advisor"),
        (_COMMISSION, _LICENSE, _EXPERIENCE),
    ),
    **dict.fromkeys(("coach", "trainer", "instructor", "guide", "tour_guide"), (_CERT_LEVEL, _HOURLY, _EXPERIENCE)),
}

# Organizations and places.
_CEO = _at("CEO's name", "text", ("ceo", "chief", "executive", "president"))
_WEBSITE = _at("website address", "text", ("website", "url", "domain", "web"))
_HEADCOUNT = _at("number of employees", "number", ("employee", "headcount", "staff", "worker", "personnel"))
_FOUNDED = _at("year founded", "number", ("found", "establish", "incorporat", "inception"))
_TICKER = _at("stock ticker symbol", "text", ("ticker", "symbol", "listed", "exchange", "public"))
_HQ_ADDRESS = _at("headquarters address", "text", ("address", "headquarter", "hq", "street"))
_CREDIT_RATING = _at("credit rating", "category", ("credit", "rating"), ("AAA", "AA", "A", "BBB"))
ORG_GENERIC = (_CEO, _WEBSITE, _HEADCOUNT, _FOUNDED, _TICKER, _HQ_ADDRESS, _CREDIT_RATING)

_STREET = _at("street address", "text", ("address", "street", "zip", "postal"))
_PLACE_PHONE = _at("phone number", "text", ("phone", "contact"))
_OPENING = _at("opening hours", "text", ("opening", "open_24", "hours_of"))
_PARKING = _at("number of parking spaces", "number", ("parking", "car_park"))
_MANAGER = _at("manager's name", "text", ("manager", "management", "managed", "director", "head"))
_YEAR_BUILT = _at("year built", "number", ("built", "build", "construct", "opened", "commission", "established"))
_WHEELCHAIR = _at("are wheelchair accessible", "flag", ("wheelchair", "access"))
PLACE_GENERIC = (_STREET, _PLACE_PHONE, _OPENING, _PARKING, _MANAGER, _YEAR_BUILT, _WHEELCHAIR)
SITE_GENERIC = (_STREET, _MANAGER, _YEAR_BUILT)

_POOL = _at("have a swimming pool", "flag", ("pool", "swim"))
_CHECK_IN = _at("check-in time", "text", ("check_in", "checkin", "check"))
_SEATING = _at("seating capacity", "number", ("seat", "capacity", "cover"))
_OUTDOOR = _at("offer outdoor seating", "flag", ("outdoor", "patio"))
_CHEF = _at("head chef", "text", ("chef", "cook"))
_ICU = _at("number of ICU beds", "number", ("icu", "bed"))
_TRAUMA = _at("trauma center level", "category", ("trauma",), ("Level I", "Level II", "Level III"))
_NEW_PATIENTS = _at("accept new patients", "flag", ("new_patient", "accept"))
_ADMISSION = _at("admission price", "number", ("admission", "fee", "price", "ticket", "free"))
_GALLERIES = _at("number of galleries", "number", ("galler",))
_RUNWAYS = _at("number of runways", "number", ("runway",))
_IATA = _at("IATA code", "text", ("iata", "code"))
_BERTHS = _at("number of berths", "number", ("berth",))
_DRAFT = _at("maximum ship draft", "number", ("draft",))
_ENTRANCE_FEE = _at("entrance fee", "number", ("entrance", "fee", "admission", "price", "free"))
_DOGS = _at("allow dogs", "flag", ("dog", "pet"))
_DOCKS = _at("number of loading docks", "number", ("dock", "bay"))
_FLOOR_AREA = _at("floor area", "number", ("area", "sqft", "square", "floor", "size"))
_CHECKOUTS = _at("number of checkout lanes", "number", ("checkout", "lane", "register"))
_ORGANIC = _at("are certified organic", "flag", ("organic", "certif"))
_SAFETY_CERT = _at("safety certification", "text", ("certif", "safety"))
_RAINFALL = _at("average annual rainfall", "number", ("rain", "precip"))
_ELEVATION = _at("elevation", "number", ("elevation", "altitude", "height"))
_MEDIAN_INCOME = _at("median household income", "number", ("income", "household"))
_UNEMPLOYMENT = _at("unemployment rate", "number", ("unemploy", "employ"))
_SOIL_PH = _at("soil pH", "number", ("soil", "ph"))
_IRRIGATED = _at("are irrigated", "flag", ("irrigat",))
_DEPT_HEAD = _at("department head", "text", ("head", "manager", "director", "lead", "chief"))
_DEPT_BUDGET = _at("annual budget", "number", ("budget",))
_STADIUM = _at("home stadium", "text", ("stadium", "venue", "home", "arena"))
_HEAD_COACH = _at("head coach", "text", ("coach",))

# Things.
_WEIGHT = _at("weight", "number", ("weight", "kg", "gram", "lb", "mass"))
_ORIGIN = _at("country of origin", "category", ("origin", "country", "made"), ("China", "Vietnam", "Mexico", "Italy"))
_BARCODE = _at("barcode", "text", ("barcode", "upc", "ean", "sku", "gtin"))
_WARRANTY = _at("warranty length", "number", ("warrant",))
_FABRIC = _at("fabric", "category", ("fabric", "material", "cotton"), ("cotton", "polyester", "wool", "linen"))
_CARE = _at("care instructions", "text", ("care", "wash"))
_SHELF_LIFE = _at("shelf life", "number", ("shelf", "expir", "life"))
_STORAGE_TEMP = _at("storage temperature", "number", ("storage", "temperature", "celsius", "temp"))
_DOSAGE_FORM = _at(
    "dosage form", "category", ("dosage", "dose_form", "product_form"), ("tablet", "capsule", "injection")
)
_ACTIVE = _at("active ingredient", "text", ("ingredient", "active", "substance", "compound"))
_BATTERY = _at("battery life", "number", ("battery",))
_SERIAL = _at("serial number", "text", ("serial",))
_COLOR = _at("color", "category", ("color", "colour", "paint"), ("white", "black", "silver", "red"))
_PLATE = _at("license plate number", "text", ("plate", "registration", "license", "licence", "tail", "vin"))
_FUEL_ECONOMY = _at("fuel economy", "number", ("fuel", "mpg", "efficien", "economy", "consumption"))
_SEATS = _at("number of seats", "number", ("seat", "capacity", "passenger"))
_LAST_SERVICE = _at("last service date", "text", ("service", "maintenance", "repair", "inspect", "overhaul"))
_TONNAGE = _at("gross tonnage", "number", ("tonnage", "gross", "dwt", "ton"))
_CALL_SIGN = _at("call sign", "text", ("call_sign", "callsign", "call"))
_TOP_SPEED = _at("top speed", "number", ("speed", "mph", "kmh"))
_HORSEPOWER = _at("horsepower", "number", ("horsepower", "hp", "power"))
_CALORIES = _at("calorie count", "number", ("calor", "kcal"))
_SPICE = _at("spice level", "category", ("spice", "spicy", "heat"), ("mild", "medium", "hot"))
_ALLERGENS = _at("allergens", "text", ("allerg",))
_PAGES = _at("page count", "number", ("page",))
_ISBN = _at("ISBN", "text", ("isbn",))
_TRACKS = _at("number of tracks", "number", ("track",))
_TEMPO = _at("tempo in beats per minute", "number", ("tempo", "bpm"))
_LENGTH_SECONDS = _at("length in seconds", "number", ("length", "duration", "second", "minute", "runtime"))
_AGE_RATING = _at("age rating", "category", ("rating", "age", "esrb", "pegi"), ("E", "T", "M"))
_RUNTIME = _at("runtime in minutes", "number", ("runtime", "length", "duration", "minute"))
_DIRECTOR = _at("director", "text", ("director", "direct"))
_PROD_BUDGET = _at("production budget", "number", ("budget", "cost"))
_WORD_COUNT = _at("word count", "number", ("word", "length", "char"))
_BYLINE = _at("author's name", "text", ("author", "writer", "byline", "journalist", "reporter"))
_DIMENSIONS = _at("dimensions", "text", ("dimension", "height", "width", "size"))
_INSURED = _at("insured value", "number", ("insur",))
_EXPENSE_RATIO = _at("expense ratio", "number", ("expense", "ratio"))
_FUND_MANAGER = _at("fund manager's name", "text", ("manager",))
_MORNINGSTAR = _at("Morningstar rating", "number", ("morningstar", "rating", "star"))
_DIVIDEND = _at("dividend yield", "number", ("dividend", "yield"))
_MAX_SUPPLY = _at("maximum supply", "number", ("supply", "max"))
_AUDIT_FIRM = _at("audit firm", "text", ("audit",))
_DEDUCTIBLE = _at("deductible", "number", ("deductible",))
_MONTHLY_PREMIUM = _at("monthly premium", "number", ("premium", "price", "cost", "fee"))
_EXPIRY = _at("expiry date", "text", ("expir", "valid", "end"))
_MAX_DISCOUNT = _at("maximum discount", "number", ("discount", "percent", "off", "max"))
_PROJECT_MANAGER = _at("project manager", "text", ("manager", "lead", "owner", "principal"))
_PROJECT_BUDGET = _at("budget", "number", ("budget", "cost", "amount", "fund"))
_AUDIENCE = _at("target audience", "text", ("audience", "target", "age"))
_TICKET_PRICE = _at("ticket price", "number", ("ticket", "price", "fee", "admission"))
_SPONSOR = _at("main sponsor", "text", ("sponsor",))
_GROUP_SIZE = _at("maximum group size", "number", ("group", "size", "max", "capacity"))
_CANCEL_POLICY = _at("cancellation policy", "text", ("cancel", "policy", "refund"))
_ICD = _at("ICD-10 code", "text", ("icd", "code"))
_RECOVERY = _at("typical recovery time", "number", ("recovery", "duration", "days"))
_CONTAGIOUS = _at("are contagious", "flag", ("contagio", "infect", "communicable", "notifiable"))
_SUCCESS = _at("success rate", "number", ("success", "outcome"))
_DURATION = _at("duration in minutes", "number", ("duration", "minute", "hour", "length", "time"))
_LIFESPAN = _at("average lifespan", "number", ("lifespan", "life", "age"))
_CONSERVATION = _at(
    "conservation status",
    "category",
    ("conservation", "status", "iucn", "endanger"),
    ("least concern", "vulnerable", "endangered"),
)
_BREED = _at("breed", "text", ("breed",))
_MICROCHIP = _at("microchip number", "text", ("microchip", "chip"))
_BEDROOMS = _at("number of bedrooms", "number", ("bedroom", "bed", "room"))
_LOT = _at("lot size", "number", ("lot", "acre", "size", "area"))
_HOA = _at("HOA fee", "number", ("hoa", "fee", "association"))
_EXTRA_CHARGE = _at("extra charge", "number", ("charge", "fee", "price", "cost"))
_CONTRACT_LENGTH = _at("contract length", "number", ("length", "term", "duration", "month", "year"))
_EXCLUSIVE = _at("are exclusive", "flag", ("exclusiv",))
_BAND_SIZE = _at("number of members", "number", ("member", "size", "solo"))
_DESIGNER = _at("lead designer", "text", ("designer", "creative", "director"))
_SEND_DAY = _at("send day", "category", ("day", "frequency", "cadence", "schedule"), ("Monday", "Wednesday", "Friday"))
_MAU = _at("monthly active users", "number", ("active", "user", "subscriber", "mau"))
_ELEVATION_GAIN = _at("elevation gain", "number", ("elevation", "climb", "gain"))
_QUORUM = _at("quorum threshold", "number", ("quorum", "threshold"))
_PROPOSER = _at("proposer's wallet address", "text", ("proposer", "author", "creator", "wallet", "address"))
_RIG_DEPTH = _at("maximum water depth", "number", ("depth", "water"))
_CREW_SIZE = _at("crew size", "number", ("crew", "size"))
_FOLLOWER_COUNT = _at("follower count", "number", ("follower",))
_VERIFIED = _at("have a verified badge", "flag", ("verif",))
_OPENING_BALANCE = _at("opening balance", "number", ("balance", "opening"))
_ACCOUNT_OWNER = _at("account owner", "text", ("owner", "manager", "responsible"))
_PARENT_CATEGORY = _at("parent category", "text", ("parent",))
_RARITY = _at("rarity", "category", ("rarity", "rare", "percent", "unlocked"), ("common", "rare", "legendary"))
_TRANSFORMERS = _at("number of transformers", "number", ("transformer",))
_GUEST_PASSES = _at("number of guest passes per month", "number", ("guest", "pass"))
_PLAN_ADMIN = _at("plan administrator", "text", ("administrator", "admin"))
_LOAN_PROVISION = _at("allow participant loans", "flag", ("loan",))
_CARE_COORDINATOR = _at("care coordinator", "text", ("coordinator",))
_NEXT_REVIEW = _at("next review date", "text", ("review", "next"))
_PREPAYMENT = _at("prepayment penalty", "number", ("prepay", "penalty"))
_MIN_CREDIT = _at("minimum credit score", "number", ("credit", "fico", "score"))
_MAX_TERM = _at("maximum loan term", "number", ("term", "maximum", "max"))
_MAX_COVERAGE = _at("maximum coverage amount", "number", ("coverage", "limit", "maximum", "max"))

# Events.
_PAYMENT_METHOD = _at(
    "payment method",
    "category",
    ("payment", "paid", "card", "tender", "method"),
    ("credit card", "cash", "bank transfer", "gift card"),
)
_SALES_REP = _at(
    "sales representative", "text", ("rep", "agent", "salesperson", "associate", "employee", "staff", "seller")
)
_COUPON = _at("coupon code", "text", ("coupon", "promo", "discount", "code"))
_TRACKING = _at("tracking number", "text", ("tracking", "track", "waybill"))
_ROOM = _at("room number", "number", ("room",))
_START_TIME = _at("start time", "text", ("time", "hour", "minute", "start"))
_REFERRAL = _at(
    "referral source",
    "category",
    ("referr", "source", "heard", "channel"),
    ("web search", "friend", "doctor", "advertisement"),
)
_WEATHER = _at(
    "weather condition",
    "category",
    ("weather", "rain", "wind", "precip", "temperature"),
    ("clear", "rain", "snow", "fog"),
)
_SPEED = _at("average speed", "number", ("speed",))
_DEPARTURE = _at("departure time", "text", ("time", "hour", "depart"))
_REPORTER = _at("reporter's name", "text", ("report", "by"))
_INSPECTOR = _at("inspector's name", "text", ("inspector", "auditor", "reviewer", "examiner", "by"))
_TERMINATION = _at("early termination fee", "number", ("termination", "early"))
_RENEWAL = _at("include a renewal option", "flag", ("renew",))
_LIKES = _at("number of likes", "number", ("like", "reaction", "vote", "helpful"))
_REVIEW_LANGUAGE = _at("review language", "category", ("language", "lang"), ("English", "Spanish", "French", "German"))
_ADJUSTER_NAME = _at("adjuster's name", "text", ("adjuster", "by"))
_CONFIRMATION = _at("confirmation number", "text", ("confirm", "reference"))

ENTITY_LEX: dict[str, tuple[Attr, ...]] = {
    "hotel": (_PARKING, _POOL, _CHECK_IN, _STREET),
    "restaurant": (_SEATING, _OUTDOOR, _CHEF, _OPENING),
    "hospital": (_ICU, _TRAUMA, _PARKING, _STREET),
    **dict.fromkeys(
        ("clinic", "vet_clinic", "practice", "health_center", "medical_center"), (_NEW_PATIENTS, _WHEELCHAIR)
    ),
    "museum": (_ADMISSION, _GALLERIES, _OPENING),
    "venue": (_SEATING, _PARKING, _STREET),
    "gym": (_POOL, _OPENING, _PARKING),
    "airport": (_RUNWAYS, _IATA),
    "port": (_BERTHS, _DRAFT),
    **dict.fromkeys(
        ("park", "national_park", "campground", "protected_area", "reserve", "forest"), (_ENTRANCE_FEE, _DOGS)
    ),
    "trail": (_ELEVATION_GAIN, _DOGS),
    **dict.fromkeys(("warehouse", "depot", "yard", "distribution_center", "garage"), (_DOCKS, _FLOOR_AREA)),
    **dict.fromkeys(("store", "shop", "branch", "dealership", "location"), (_FLOOR_AREA, _OPENING, _CHECKOUTS)),
    "farm": (_ORGANIC, _SAFETY_CERT),
    "substation": (_TRANSFORMERS, _YEAR_BUILT),
    **dict.fromkeys(
        ("region", "district", "area", "service_area", "city", "market"), (_RAINFALL, _MEDIAN_INCOME, _UNEMPLOYMENT)
    ),
    **dict.fromkeys(("basin", "tract", "forest_tract", "parcel", "source", "water_source"), (_RAINFALL, _ELEVATION)),
    "field": (_SOIL_PH, _IRRIGATED),
    **dict.fromkeys(("department", "cost_center"), (_DEPT_HEAD, _DEPT_BUDGET)),
    **dict.fromkeys(("team", "club", "sports_team"), (_STADIUM, _HEAD_COACH, _FOUNDED)),
    "league": (_FOUNDED, _WEBSITE),
    **dict.fromkeys(("product", "item", "sku", "store_item", "style"), (_WEIGHT, _ORIGIN, _BARCODE, _WARRANTY)),
    "menu_item": (_CALORIES, _SPICE, _ALLERGENS),
    "garment": (_FABRIC, _CARE, _ORIGIN),
    **dict.fromkeys(("drug", "medication", "vaccine"), (_SHELF_LIFE, _STORAGE_TEMP, _DOSAGE_FORM, _ACTIVE)),
    **dict.fromkeys(("device", "machine"), (_WEIGHT, _BATTERY, _SERIAL, _WARRANTY)),
    **dict.fromkeys(
        ("vehicle", "car", "truck", "van", "bike", "bicycle", "ambulance", "fleet", "railcar"),
        (_COLOR, _PLATE, _FUEL_ECONOMY, _LAST_SERVICE),
    ),
    "locomotive": (_HORSEPOWER, _LAST_SERVICE),
    "aircraft": (_SEATS, _LAST_SERVICE, _COLOR),
    **dict.fromkeys(("vessel", "ship"), (_TONNAGE, _CALL_SIGN, _YEAR_BUILT)),
    **dict.fromkeys(("model", "vehicle_model", "aircraft_model"), (_TOP_SPEED, _FUEL_ECONOMY, _SEATS)),
    "dish": (_CALORIES, _SPICE, _ALLERGENS),
    **dict.fromkeys(("book", "publication"), (_PAGES, _ISBN)),
    "album": (_TRACKS, _DIRECTOR._replace(phrase="record producer", stems=("producer", "produc"))),
    **dict.fromkeys(("song", "track"), (_TEMPO, _LENGTH_SECONDS)),
    **dict.fromkeys(("game", "title", "esports_title"), (_AGE_RATING, _WEBSITE)),
    **dict.fromkeys(
        ("show", "production", "film", "series", "episode", "release", "work"), (_RUNTIME, _DIRECTOR, _PROD_BUDGET)
    ),
    **dict.fromkeys(("article", "story", "newsletter_issue"), (_WORD_COUNT, _BYLINE)),
    "newsletter": (_SEND_DAY, _MAU._replace(phrase="number of subscribers", stems=("subscri",))),
    **dict.fromkeys(("object", "artwork"), (_DIMENSIONS, _INSURED)),
    "fund": (_EXPENSE_RATIO, _FUND_MANAGER, _MORNINGSTAR),
    **dict.fromkeys(("security", "investment_option"), (_DIVIDEND, _MORNINGSTAR)),
    **dict.fromkeys(("token", "asset", "chain", "network"), (_MAX_SUPPLY, _AUDIT_FIRM, _WEBSITE)),
    "coverage_plan": (_DEDUCTIBLE, _MONTHLY_PREMIUM),
    **dict.fromkeys(("membership_plan", "membership_tier"), (_CANCEL_POLICY, _GUEST_PASSES)),
    **dict.fromkeys(("retirement_plan", "employer_plan"), (_PLAN_ADMIN, _LOAN_PROVISION)),
    "care_plan": (_CARE_COORDINATOR, _NEXT_REVIEW),
    "loan_product": (_PREPAYMENT, _MIN_CREDIT, _MAX_TERM),
    "virtual_item": (_RARITY,),
    "plant": (_YEAR_BUILT, _SAFETY_CERT, _MANAGER._replace(phrase="plant manager")),
    **dict.fromkeys(("tier", "reward_tier", "membership_tier", "reward"), (_EXPIRY, _EXTRA_CHARGE)),
    **dict.fromkeys(("promotion", "promo_code"), (_MAX_DISCOUNT, _EXPIRY)),
    **dict.fromkeys(("project", "research_project"), (_PROJECT_MANAGER, _PROJECT_BUDGET)),
    "campaign": (_AUDIENCE, _PROJECT_MANAGER._replace(phrase="campaign manager")),
    **dict.fromkeys(("festival", "competition", "tournament", "concert"), (_TICKET_PRICE, _SPONSOR)),
    **dict.fromkeys(("tour", "excursion", "class"), (_GROUP_SIZE, _CANCEL_POLICY)),
    **dict.fromkeys(("service", "grooming_service"), (_DURATION, _CANCEL_POLICY)),
    "shipping_line": (_CEO, _WEBSITE, _HEADCOUNT, _FOUNDED),
    **dict.fromkeys(("condition", "disease", "disorder"), (_ICD, _RECOVERY, _CONTAGIOUS)),
    **dict.fromkeys(("treatment", "intervention", "service_code"), (_DURATION, _SUCCESS)),
    **dict.fromkeys(("species", "crop", "animal"), (_LIFESPAN, _CONSERVATION, _WEIGHT)),
    "pet": (_BREED, _MICROCHIP, _WEIGHT),
    **dict.fromkeys(("listing", "property"), (_BEDROOMS, _LOT, _HOA, _YEAR_BUILT)),
    "amenity": (_EXTRA_CHARGE,),
    "credential": (_EXPIRY,),
    "achievement": (_RARITY,),
    **dict.fromkeys(("deal", "sponsor_deal", "sponsorship"), (_CONTRACT_LENGTH, _EXCLUSIVE)),
    **dict.fromkeys(("act", "performer", "artist"), (_BAND_SIZE, _HOMETOWN)),
    "collection": (_DESIGNER,),
    "platform": (_MAU, _FOUNDED),
    "proposal": (_QUORUM, _PROPOSER),
    "governance_proposal": (_QUORUM, _PROPOSER),
    "rig": (_RIG_DEPTH, _CREW_SIZE),
    **dict.fromkeys(("category", "product_category"), (_PARENT_CATEGORY,)),
    **dict.fromkeys(("hashtag", "tag"), (_FOLLOWER_COUNT,)),
    # Events and records.
    **dict.fromkeys(
        ("order", "sale", "purchase", "booking", "reservation", "rental"), (_PAYMENT_METHOD, _SALES_REP, _COUPON)
    ),
    **dict.fromkeys(("transaction", "payment", "invoice", "bill"), (_PAYMENT_METHOD, _CONFIRMATION)),
    **dict.fromkeys(
        ("shipment", "delivery", "package", "parcel", "consignment", "waybill", "pickup"), (_TRACKING, _WEIGHT)
    ),
    **dict.fromkeys(("visit", "appointment", "encounter", "admission", "stay"), (_ROOM, _START_TIME, _REFERRAL)),
    **dict.fromkeys(("session", "class_session"), (_ROOM, _START_TIME)),
    **dict.fromkeys(("trip", "ride", "flight", "voyage", "sailing", "dispatch"), (_WEATHER, _SPEED, _DEPARTURE)),
    **dict.fromkeys(
        ("incident", "outage", "spill", "break", "failure", "accident", "fault", "closure", "outbreak", "wildfire"),
        (_START_TIME, _REPORTER, _WEATHER),
    ),
    **dict.fromkeys(
        ("inspection", "audit", "test", "assessment", "screening", "check", "exam", "survey", "calibration"),
        (_INSPECTOR, _DURATION),
    ),
    **dict.fromkeys(("loan", "lease", "contract", "agreement", "policy", "grant", "award"), (_TERMINATION, _RENEWAL)),
    "claim": (_ADJUSTER_NAME,),
    **dict.fromkeys(("review", "comment", "post", "reply", "rating"), (_LIKES, _REVIEW_LANGUAGE)),
}
# The same word means different things in different families.
FAMILY_LEX: dict[tuple[str, str], tuple[Attr, ...]] = {
    ("finance", "account"): (_OPENING_BALANCE, _ACCOUNT_OWNER),
    ("media_and_entertainment", "account"): (_FOLLOWER_COUNT, _VERIFIED),
    ("finance", "instrument"): (_DIVIDEND, _MORNINGSTAR),
    ("healthcare", "instrument"): (_SERIAL, _LAST_SERVICE, _WARRANTY),
    ("media_and_entertainment", "event"): (_TICKET_PRICE, _SPONSOR),
    ("finance", "product"): (_DEDUCTIBLE, _MAX_COVERAGE),
    ("retail_and_consumer", "trip"): (_GROUP_SIZE, _CANCEL_POLICY),
    ("energy_and_natural_resources", "station"): (_YEAR_BUILT, _MANAGER._replace(phrase="site manager")),
    ("logistics_and_transport", "station"): (_STREET, _WHEELCHAIR),
    ("healthcare", "station"): (_STREET, _PLACE_PHONE, _MANAGER._replace(phrase="station chief")),
}
ORG_HEADS = {
    "company",
    "firm",
    "operator",
    "carrier",
    "manufacturer",
    "maker",
    "producer",
    "distributor",
    "developer",
    "publisher",
    "automaker",
    "shipper",
    "contractor",
    "brokerage",
    "agency",
    "employer",
    "sponsor",
    "advertiser",
    "retailer",
    "merchant",
    "startup",
    "railroad",
    "airline",
    "utility",
    "studio",
    "label",
    "brand",
    "outlet",
    "house",
    "stockist",
    "seller",
    "lp",
    "LP",
    "manager",
    "dealer",
    "vendor",
    "supplier",
    "provider",
    "family",
    "partner",
    "sender",
    "buyer",
    "owner",
    "customer",
}
# Places the public visits get opening hours and parking; industrial sites do not.
PLACE_HEADS = {
    "clinic",
    "hospital",
    "practice",
    "home",
    "center",
    "facility",
    "store",
    "shop",
    "restaurant",
    "hotel",
    "gym",
    "museum",
    "venue",
    "branch",
    "office",
    "garage",
    "campground",
    "park",
    "location",
    "dealership",
    "club",
}
SITE_HEADS = {
    "lab",
    "laboratory",
    "depot",
    "warehouse",
    "yard",
    "port",
    "airport",
    "station",
    "base",
    "stop",
    "plant",
    "mill",
    "mine",
    "site",
    "farm",
    "locker",
    "substation",
}


def _singular_name(name: str) -> str:
    """'power_plants' -> 'power_plant': the name with its last word made singular."""
    return "_".join(name.split("_")[:-1] + [head_noun(name)])


def _attrs_for(table: dict, family: str) -> tuple[list[Attr], list[Attr]]:
    """The attributes worth asking about for one table: (specific ones, generic ones)."""
    full, head = _singular_name(table["name"]), head_noun(table["name"])
    # The template key tells apart tables that share a name: "units" can be ambulances or departments.
    template = _singular_name(table.get("template_key") or table["name"])
    lex = PERSON_LEX if _is_person(table) else ENTITY_LEX
    specific: tuple[Attr, ...] = ()
    matched = None
    for key in dict.fromkeys((full, template, head)):  # the most specific key wins: loan_product before product
        specific = FAMILY_LEX.get((family, key)) or lex.get(key) or ()
        if specific:
            matched = key
            break
    generic: tuple[Attr, ...] = ()
    if _is_person(table):
        generic = PERSON_GENERIC
    elif matched in (None, head) and _is_named(table):
        for heads, attrs in ((ORG_HEADS, ORG_GENERIC), (PLACE_HEADS, PLACE_GENERIC), (SITE_HEADS, SITE_GENERIC)):
            if head in heads:
                generic = attrs
                break
    return list(specific), [a for a in generic if a not in specific]


def attr_stems(attr: Attr) -> tuple[str, ...]:
    """The stems that would mean a design already holds this attribute: the listed ones plus the phrase's words.

    3-letter words count too, as whole words: a loan_fee column rules out "early termination fee".
    """
    auto = [w for w in re.findall(r"[a-z]+", attr.phrase.lower()) if len(w) >= 3 and w not in PHRASE_FILLER]
    return tuple(dict.fromkeys(s.lower() for s in (*attr.stems, *auto)))


def _stem_in(stem: str, name: str) -> bool:
    if "_" not in stem and len(stem) <= 3:  # short stems must match a whole word: "age", not "average"
        toks = _tokens(name)
        return stem in toks or stem + "s" in toks
    return stem in name.lower()


def is_absent(schema_names: list[str], attr: Attr) -> bool:
    """True if no table or column name in the design matches any of the attribute's stems."""
    return not any(_stem_in(s, n) for s in attr_stems(attr) for n in schema_names)


def _attribute_questions(table: dict, attr: Attr, d: _Design) -> list[str]:
    sing, plural = nouns(table["name"])
    poss = "their" if _is_person(table) else "its"
    cats = _categories(table)
    cat = d.rng.choice(cats) if cats else None
    p = attr.phrase
    if attr.kind == "flag":
        qs = [f"Which {plural} {p}?", f"How many {plural} {p}?"]
        if cat:
            qs.append(f"Which {plural} with {words(cat['name'])} {_say(d.value(cat))} {p}?")
    elif attr.kind == "number":
        qs = [
            f"Which {sing} has the highest {p}?",
            f"What is the {_of('average', p)} of the {plural}?",
            f"List the {plural} in order of {p}, highest first.",
        ]
        if cat:
            qs.append(f"What is the {_of('average', p)} of the {plural} for each {words(cat['name'])}?")
    elif attr.kind == "category":
        v = d.rng.choice(attr.values)
        qs = [
            f"How many {plural} have {v} as their {p}?",
            f"List the {plural} whose {p} is {v}.",
            f"How many {plural} are there for each {p}?",
        ]
    else:
        be = "are" if _is_plural_phrase(p) else "is"
        qs = [f"What {be} the {p} of each {sing}?", f"List every {sing} with {poss} {p}."]
        if cat:
            qs.append(f"Show the {p} of every {sing} whose {words(cat['name'])} is {_say(d.value(cat))}.")
    return qs


def _missing_attribute(d: _Design) -> tuple[list[_Cand], list[_Cand]]:
    specific, generic = [], []
    for t in d.tables:
        _, plural = nouns(t["name"])
        for bucket, attrs in zip((specific, generic), _attrs_for(t, d.family), strict=True):
            for attr in attrs:
                if not is_absent(d.names, attr):
                    continue
                if attr.kind == "flag":
                    reason = f"the {words(t['name'])} table has no column for whether {plural} {attr.phrase}"
                else:
                    reason = f"the {words(t['name'])} table has no column for {attr.phrase}"
                for q in _attribute_questions(t, attr, d):
                    bucket.append(_Cand(q, reason, t["name"]))
    return specific, generic


# ---------------------------------------------------------------- ambiguous


def _measure_phrases(d: _Design, table: dict) -> list[str]:
    poss = "their" if _is_person(table) else "its"
    phrases = [f"the highest {words(c['name'])}" for c in _measures(table)]
    for child in d.children.get(table["name"], []):
        _, child_plural = nouns(child["name"])
        phrases.append(f"the most {child_plural}")
        for c in _measures(child)[:1]:
            phrases.append(f"the highest {_of('total', words(c['name']))} across {poss} {child_plural}")
    return phrases


def _undefined_metric(d: _Design) -> list[_Cand]:
    out = []
    for t in d.tables:
        sing, plural = nouns(t["name"])
        measures = _measure_phrases(d, t)
        if len(measures) >= 2:
            if _is_person(t):
                templates = [
                    (f"Who are our best {plural}?", "best"),
                    (f"Who is our top {sing}?", "top"),
                    (f"Which {plural} are the most valuable to us?", "most valuable"),
                ]
            elif _is_named(t):
                templates = [
                    (f"Which {sing} performs best?", "best"),
                    (f"Which are our most successful {plural}?", "most successful"),
                    (f"What are our top {plural}?", "top"),
                ]
            elif head_noun(t["name"]) in NEGATIVE_HEADS:
                templates = [
                    (f"Which {sing} was the worst?", "worst"),
                    (f"What were the most serious {plural}?", "most serious"),
                ]
            else:
                templates = [
                    (f"Which were the best {plural}?", "best"),
                    (f"Which {sing} was the most successful?", "most successful"),
                ]
            for question, adj in templates:
                m1, m2 = d.rng.sample(measures, 2)
                reason = f"'{adj}' is not defined: it could mean {m1} or {m2}"
                out.append(_Cand(question, reason, t["name"]))
        # A parent judged by its children: "Which district has the best facilities?"
        for child in d.children.get(t["name"], []):
            child_measures = [f"the highest {words(c['name'])}" for c in _measures(child)]
            if len(child_measures) >= 2:
                _, child_plural = nouns(child["name"])
                m1, m2 = d.rng.sample(child_measures, 2)
                out.append(
                    _Cand(
                        f"Which {sing} has the best {child_plural}?",
                        f"'best' {child_plural} is not defined: it could mean {m1} or {m2}",
                        t["name"],
                    )
                )
    return out


def _participle(col_name: str, table_name: str) -> str | None:
    """'opened_date' -> 'were opened', so 'Which facilities were opened recently?'; None if no clean verb."""
    toks = [t for t in _tokens(col_name) if t not in {"date", "on", "at", "day", "month", "year"}]
    if len(toks) != 1:
        return None
    tok = toks[0]
    head = head_noun(table_name)
    if tok[:4] == head[:4]:
        return None  # "Which sales were sold recently?" reads badly
    verb = PARTICIPLES.get(tok) or (tok if tok.endswith("ed") and len(tok) > 4 else None)
    if verb is None:
        return None
    return verb if verb in INTRANSITIVE else f"were {verb}"


def _vague_reference(d: _Design) -> list[_Cand]:
    out = []
    for t in d.tables:
        sing, plural = nouns(t["name"])
        for c in _dates(t) + _year_numbers(t):
            cw = words(c["name"])
            if "birth" in c["name"]:  # a recent birth date means a young person
                if _is_person(t):
                    for adj in ("young", "elderly"):
                        reason = f"'{adj}' has no defined age cutoff for the {cw}"
                        out.append(_Cand(f"Which {plural} are {adj}?", reason, t["name"]))
                continue
            verb = _participle(c["name"], t["name"])
            reason = f"'recently' has no defined cutoff for the {cw}"
            old_reason = f"'a long time ago' has no defined cutoff for the {cw}"
            if verb:
                out.append(_Cand(f"Which {plural} {verb} recently?", reason, t["name"]))
                out.append(_Cand(f"Show the {plural} that {verb} a long time ago.", old_reason, t["name"]))
            else:
                out.append(_Cand(f"Which {plural} have {_adj('recent', cw)}?", reason, t["name"]))
                old = f"'old' has no defined cutoff for the {cw}"
                out.append(_Cand(f"List the {plural} with {_adj('old', cw)}.", old, t["name"]))
        for c in _measures(t):
            toks = _tokens(c["name"])
            cw = words(c["name"])
            last = toks[-1]
            countable = len(toks) == 1 and last.endswith("s") and not last.endswith("ss") and last not in UNIT_PLURALS
            if countable:
                pairs = [
                    (f"Which {plural} have many {cw}?", "many"),
                    (f"Show the {plural} with only a few {cw}.", "a few"),
                ]
            elif set(toks) & SIZE_TOKENS:
                pairs = [
                    (f"Show the large {plural}.", "large"),
                    (f"Which {plural} have {_adj('large', cw)}?", "large"),
                    (f"List the {plural} with {_adj('small', cw)}.", "small"),
                ]
            else:
                pairs = [
                    (f"Which {plural} have {_adj('high', cw)}?", "high"),
                    (f"List the {plural} with {_adj('low', cw)}.", "low"),
                ]
            for question, adj in pairs:
                out.append(_Cand(question, f"'{adj}' has no defined threshold for the {cw}", t["name"]))
    return out


# ---------------------------------------------------------------- out of scope

_MESSAGE = "it asks to send a message, which a SQL query cannot do"
_WRITE = "it asks to change the data, and the model may only run read-only SELECT queries"
_EXPORT = "it asks to export and share a file, which a SQL query cannot do"
_ALERT = "it asks to set up an ongoing alert, which a single SQL query cannot do"


def _action(d: _Design) -> list[_Cand]:
    out = []
    cancellable = {"booking", "reservation", "order", "appointment", "subscription", "membership", "rental", "ticket"}
    for t in d.tables:
        sing, plural = nouns(t["name"])
        team = d.rng.choice(TEAMS)
        cands = [
            (f"Export the {words(t['name'])} table to a spreadsheet and share it with the {team}.", _EXPORT),
            (f"Set up an alert that emails me whenever a new {sing} is added.", _ALERT),
        ]
        for cat in _categories(t):
            cw, v = words(cat["name"]), _say(d.value(cat))
            cands += [
                (f"Email the list of {plural} with {cw} {v} to the {d.rng.choice(TEAMS)}.", _MESSAGE),
                (f"Delete every {sing} whose {cw} is {v}.", _WRITE),
                (f"Mark all {plural} with {cw} {v} as reviewed.", _WRITE),
            ]
            if _is_person(t):
                cands.append((f"Send a text message to every {sing} whose {cw} is {v}.", _MESSAGE))
            if head_noun(t["name"]) in cancellable:
                cands.append((f"Cancel all {plural} whose {cw} is {v}.", _WRITE))
        for c in _measures(t):
            if _is_money(c):
                cands.append((f"Raise the {words(c['name'])} of every {sing} by 5 percent.", _WRITE))
        if _is_person(t):
            cands.append((f"Send a thank-you email to all of our {plural}.", _MESSAGE))
        out += [_Cand(q, r, t["name"]) for q, r in cands]
    return out


_FUTURE = "it asks about the future, and the database records only what has already happened"


def _forecast(d: _Design) -> list[_Cand]:
    out = []
    for t in d.tables:
        sing, plural = nouns(t["name"])
        period = d.rng.choice(PERIODS)
        cands = []
        for c in _measures(t):
            cw = words(c["name"])
            cands += [
                f"What will the {_of('total', cw)} of our {plural} be next {d.rng.choice(PERIODS)}?",
                f"Predict the {_of('average', cw)} of our {plural} for next year.",
            ]
            for dc in _dates(t)[:1]:
                year = _years(dc)[-1] + 1  # neutral: it assumes no trend the data may not show
                cands.append(f"Will the {_of('average', cw)} of our {plural} rise or fall after {year}?")
        if _dates(t):
            if _is_named(t):
                cands.append(f"How many new {plural} will we have next {period}?")
            else:
                cands.append(f"How many {plural} will there be next {period}?")
        for cat in _categories(t):
            cands.append(
                f"How many {plural} with {words(cat['name'])} {_say(d.value(cat))} should we expect "
                f"next {d.rng.choice(PERIODS)}?"
            )
        out += [_Cand(q, _FUTURE, t["name"]) for q in cands]
    return out


_CLUSTER = "clustering needs a statistical algorithm, not a SQL query"
_WHY = "it asks for causes, and the data shows what happened, not why"
_MODEL = "it asks to build a predictive model, which a SQL query cannot do"
_SUMMARY = "it asks for a written interpretation, not a query result"


def _analysis(d: _Design) -> list[_Cand]:
    out = []
    for t in d.tables:
        sing, plural = nouns(t["name"])
        ms, cats = _measures(t), _categories(t)
        cands = [(f"Write a short summary of the main trends in our {plural}.", _SUMMARY)]
        if len(ms) >= 2:
            m1, m2 = d.rng.sample(ms, 2)
            cands.append(
                (
                    f"Group the {plural} into clusters based on their {words(m1['name'])} and {words(m2['name'])}.",
                    _CLUSTER,
                )
            )
        elif ms and cats:
            cands.append(
                (
                    f"Cluster the {plural} by {words(ms[0]['name'])} and {words(cats[0]['name'])}, "
                    "and describe each cluster.",
                    _CLUSTER,
                )
            )
        # Causal questions stay neutral: they never claim a rise, a drop, or a gap nobody checked on the copies.
        for dc in _dates(t)[:1]:
            year = d.rng.choice(_years(dc))
            cands += [
                (f"What explains the year-to-year pattern in the number of {plural}?", _WHY),
                (f"What events in {year} shaped the number of {plural} recorded that year?", _WHY),
            ]
        for c in ms:
            cw = words(c["name"])
            cands.append((f"What factors drive the {cw} of our {plural}?", _WHY))
            for cat in cats[:2]:
                cat_w = words(cat["name"])
                cands.append((f"Why does the {cw} of our {plural} vary from one {cat_w} to another?", _WHY))
        for cat in cats:
            cands.append((f"Build a model that predicts the {words(cat['name'])} of a new {sing}.", _MODEL))
        out += [_Cand(q, r, t["name"]) for q, r in cands]
    return out


def _knowledge_reason(topic: str) -> str:
    return f"it needs real-world knowledge about {topic} that the database does not hold"


class Rule(NamedTuple):
    """Category columns whose values carry real-world meaning, and questions about that meaning.

    test gets the column's name and its words. A template or topic gets {v} (a value),
    {av} (the value with a or an), {cw} (the column in words), and {plural} (the table's rows);
    topic names what the reason says the question needs knowledge about.
    """

    test: Callable[[str, list[str]], bool]
    templates: tuple[str, ...]
    topic: str = "{v}"
    family: str | None = None


ORG_COLUMNS = {
    "make",
    "manufacturer",
    "chassis_make",
    "insurer",
    "regulator",
    "testing_body",
    "fuel_supplier",
    "auditor",
    "audit_firm",
    "research_firm",
    "awarding_body",
}
JOB_COLUMNS = {
    "position",
    "job_role",
    "job_title",
    "profession",
    "primary_craft",
    "craft",
    "crew_role",
    "crew_position",
    "seat_position",
    "job",
    "job_craft",
}
KNOWLEDGE_RULES = (
    Rule(
        lambda n, t: bool({"country", "nationality", "citizenship"} & set(t)) or n == "flag_state",
        (
            "What is the official language of each {cw} listed for our {plural}?",
            "What is the population of each {cw} that appears among our {plural}?",
            "Which currency is used in each {cw} listed for our {plural}?",
        ),
        "each {cw}",
    ),
    Rule(
        lambda n, t: "state" in t,
        (
            "What is the capital of each {cw} listed for our {plural}?",
            "What is the sales tax rate in each {cw} listed for our {plural}?",
        ),
        "each {cw}",
    ),
    Rule(
        lambda n, t: "city" in t or n == "destination",
        (
            "What is the population of each {cw} listed for our {plural}?",
            "Which time zone is each {cw} of our {plural} in?",
        ),
        "each {cw}",
    ),
    Rule(
        lambda n, t: "commodity" in t,
        ("What is today's world market price of {v}?", "Which countries produce the most {v}?"),
    ),
    Rule(
        lambda n, t: "fuel" in t and "supplier" not in t,
        ("What are the main environmental impacts of {v} power?",),
        "{v} power",
    ),
    Rule(
        lambda n, t: bool({"species", "organism", "pest"} & set(t)),
        ("What is the native range of the {cw} {v}?", "What is the typical lifespan of the {cw} {v}?"),
    ),
    Rule(
        lambda n, t: n in {"condition_name", "diagnosis", "disease", "condition"},
        ("What are the usual symptoms of {v}?", "How is {v} usually treated?"),
    ),
    Rule(lambda n, t: n == "test_name", ("What is a normal result for a {v} test?",), "{v} tests"),
    Rule(lambda n, t: "symptom" in t, ("What usually causes {v}?",)),
    Rule(lambda n, t: n == "analyte", ("What health risks does {v} in drinking water pose?",), "{v} in drinking water"),
    Rule(lambda n, t: n == "drug_class", ("What are the common side effects of {v} drugs?",), "{v} drugs"),
    Rule(
        lambda n, t: "genre" in t,
        ("Who are considered the pioneers of {v}?", "Where did {v} originate as a genre?"),
        "the {v} genre",
        "media_and_entertainment",
    ),
    Rule(lambda n, t: "cuisine" in t, ("Which dishes is {v} cuisine best known for?",), "{v} cuisine"),
    Rule(lambda n, t: "language" in t, ("How many people around the world speak {v}?",), "the {v} language"),
    Rule(
        lambda n, t: "currency" in t,
        ("What is today's exchange rate from {v} to US dollars?",),
        "today's exchange rate for {v}",
    ),
    Rule(lambda n, t: n in {"exchange", "listed_exchange"}, ("What are the trading hours of the {v}?",), "the {v}"),
    Rule(lambda n, t: n in ORG_COLUMNS, ("Where is {v} headquartered?", "Who currently leads {v}?")),
    Rule(
        lambda n, t: n in JOB_COLUMNS,
        ("What qualifications does {av} usually need?", "What does {av} typically earn in the wider job market?"),
        "the {v} profession",
    ),
    Rule(
        lambda n, t: (
            bool({"technology", "method", "consensus", "powertrain", "mechanism"} & set(t)) and "payment" not in t
        ),
        ("How does the {v} {cw} work?",),
        "the {v} {cw}",
    ),
    Rule(
        lambda n, t: bool({"material", "fabric"} & set(t)),
        ("What is {v} made from?", "How long does {v} usually last?"),
    ),
    Rule(lambda n, t: "soil" in t, ("Which crops grow best in {v} soil?",), "{v} soil"),
)
SKIP_VALUES = {"none", "other", "unlisted", "multi-genre", "mixed", "unknown", "native"}


def _outside_knowledge(d: _Design) -> tuple[list[_Cand], list[_Cand]]:
    specific, generic = [], []
    for t in d.tables:
        sing, plural = nouns(t["name"])
        for cat in _categories(t):
            n, toks = cat["name"], _tokens(cat["name"])
            for rule in KNOWLEDGE_RULES:
                if not rule.test(n, toks) or (rule.family and rule.family != d.family):
                    continue
                values = [v for v in cat["values"] if v.lower() not in SKIP_VALUES]
                for tpl in rule.templates if values else ():
                    v = d.rng.choice(values)
                    fields = {"v": v, "av": _a(v), "cw": words(n), "plural": plural}
                    q = tpl.format(**fields)
                    reason = _knowledge_reason(rule.topic.format(**fields))
                    specific.append(_Cand(q[0].upper() + q[1:], reason, t["name"]))
                break
        for c in _measures(t):
            cw = words(c["name"])
            generic.append(
                _Cand(
                    f"How does the {_of('average', cw)} of our {plural} compare with the industry average?",
                    "it needs an industry benchmark, which the database does not hold",
                    t["name"],
                )
            )
            if _is_money(c):
                generic += [
                    _Cand(
                        f"What is the {_of('total', cw)} of all {plural} in euros at today's exchange rate?",
                        "it needs today's exchange rate, which the database does not hold",
                        t["name"],
                    ),
                    _Cand(
                        f"What is each {sing}'s {cw} in today's dollars, adjusted for inflation?",
                        "it needs inflation figures, which the database does not hold",
                        t["name"],
                    ),
                ]
        for c in _dates(t):
            cw = words(c["name"])
            generic.append(
                _Cand(
                    f"Which {plural} had their {cw} fall on a public holiday?",
                    "it needs a calendar of public holidays, which the database does not hold",
                    t["name"],
                )
            )
            if not _is_named(t):
                generic.append(
                    _Cand(
                        f"What was the weather like on the {cw} of each {sing}?",
                        "it needs weather records, which the database does not hold",
                        t["name"],
                    )
                )
    return specific, generic


# ---------------------------------------------------------------- choosing and writing records


def _spread(cands: list[_Cand], rng: random.Random) -> list[_Cand]:
    """Shuffle, then interleave by table, so consecutive picks come from different tables."""
    groups: dict[str, list[_Cand]] = defaultdict(list)
    for c in cands:
        groups[c.group].append(c)
    order = list(groups)
    rng.shuffle(order)
    for g in order:
        rng.shuffle(groups[g])
    out = []
    while any(groups[g] for g in order):
        for g in order:
            if groups[g]:
                out.append(groups[g].pop())
    return out


def _pools(d: _Design, answerable: list[dict]) -> dict[str, list[_Cand]]:
    """Every candidate per subtype, best first: specific attributes and facts before generic ones."""
    rng = d.rng
    attr_specific, attr_generic = _missing_attribute(d)
    know_specific, know_generic = _outside_knowledge(d)
    return {
        "hidden_column": _spread(_hidden_column(d, answerable), rng),
        "missing_attribute": _spread(attr_specific, rng) + _spread(attr_generic, rng),
        "undefined_metric": _spread(_undefined_metric(d), rng),
        "vague_reference": _spread(_vague_reference(d), rng),
        "action": _spread(_action(d), rng),
        "forecast": _spread(_forecast(d), rng),
        "analysis": _spread(_analysis(d), rng),
        "outside_knowledge": _spread(know_specific, rng) + _spread(know_generic, rng),
    }


def decline_count(n_answerable: int) -> int:
    """How many declines to write for a design with this many answerable questions."""
    return round(DECLINE_RATIO * n_answerable)


def _kind_quotas(total: int, rng: random.Random) -> dict[str, int]:
    quotas = dict.fromkeys(KINDS, total // 3)
    for kind in rng.sample(KINDS, total % 3):
        quotas[kind] += 1
    return quotas


def _record(schema: dict, kind: str, number: int, subtype: str, c: _Cand) -> dict:
    src = c.source
    return {
        "question_id": f"{schema['schema_id']}__{PREFIX[kind]}{number:03d}",
        "schema_id": schema["schema_id"],
        "family": schema["family"],
        "topic": schema["topic"],
        "kind": kind,
        "question": c.question,
        "gold_sql": None,
        "shape": src.get("shape") if src else None,
        "difficulty": src.get("difficulty") if src else None,
        "columns_used": [],
        "hidden_columns": [{"table": t, "column": col} for t, col in c.hidden],
        "decline_subtype": subtype,
        "decline_reason": c.reason,
        "source_question_id": src["question_id"] if src else None,
        "gold_row_counts": [],
    }


def decline_questions(schema: dict, answerable: list[dict], seed: int = 0) -> list[dict]:
    """The decline records for one design, about 0.43 per answerable record, in the README's record format."""
    rng = random.Random(f"{seed}:{schema['schema_id']}:declines")
    d = _Design(schema, rng)
    pools = {k: list(reversed(v)) for k, v in _pools(d, answerable).items()}  # pop() takes the best first
    answer_texts = {_norm(r["question"]) for r in answerable}
    decline_texts: set[str] = set()
    chosen: dict[str, list[_Cand]] = {st: [] for subs in SUBTYPES.values() for st in subs}

    def take(subtype: str) -> bool:
        pool = pools[subtype]
        while pool:
            c = pool.pop()
            key = _norm(c.question)
            if key in decline_texts or (subtype != "hidden_column" and key in answer_texts):
                continue
            decline_texts.add(key)
            chosen[subtype].append(c)
            return True
        return False

    def fill(subtypes: list[str], short: int) -> int:
        while short and any(pools[st] for st in subtypes):
            for st in subtypes:
                if short and take(st):
                    short -= 1
        return short

    left = 0
    for kind, quota in _kind_quotas(decline_count(len(answerable)), rng).items():
        order = rng.sample(SUBTYPES[kind], len(SUBTYPES[kind]))
        short = 0
        for i in range(quota):
            if not take(order[i % len(order)]):
                short += 1
        left += fill(order, short)  # a subtype that ran dry passes its share to the kind's others
    fill([st for kind in KINDS for st in SUBTYPES[kind]], left)  # then to any kind

    records = []
    for kind in KINDS:
        n = 0
        for subtype in SUBTYPES[kind]:
            for c in chosen[subtype]:
                n += 1
                records.append(_record(schema, kind, n, subtype, c))
    return records
