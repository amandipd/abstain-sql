"""Build all of plan Phase 1 with one command, and mix the questions (step 1.6).

Runs every step in order: the designs (schemas.py), the filled copies (fill.py),
the answerable questions (questions.py), and the questions to decline
(declines.py). Then it mixes them about 70% answerable to 30% decline, design by
design, and writes data/instances/questions.jsonl plus build_summary.json.

Why 70/30: with too many questions to decline, a model can learn to decline
everything (abstention collapse); the plan keeps answerable questions at 70% or
more. Each design keeps all its answerable questions and at most 3 declines for
every 7 of them, taken evenly across the 3 decline kinds. Which topics go to
training, tuning, calibration, and test is decided later, in step 3.1.

Run from the repo root (a few minutes):

    uv run python -m data.generator.build
    uv run python -m data.generator.build --families healthcare --per-topic 2
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data.generator import copies as cp
from data.generator import declines, fill, questions
from data.generator import schemas as gen

ANSWERABLE_SHARE = 0.7
DECLINE_KINDS = ("unanswerable", "ambiguous", "out_of_scope")


def mix(answerable: list[dict], decline: list[dict], seed: int, schema_id: str) -> list[dict]:
    """All answerable records plus at most 3 declines per 7 answerable, balanced across the 3 kinds."""
    cap = int(len(answerable) * (1 - ANSWERABLE_SHARE) / ANSWERABLE_SHARE)
    rng = random.Random(f"{seed}:{schema_id}:mix")
    by_kind = {k: [r for r in decline if r["kind"] == k] for k in DECLINE_KINDS}
    for pool in by_kind.values():
        rng.shuffle(pool)
    order = list(DECLINE_KINDS)
    rng.shuffle(order)  # a different kind goes first in each design, so none is always cut short
    chosen: list[dict] = []
    while len(chosen) < cap and any(by_kind.values()):
        for k in order:  # one of each kind in turn, so no kind crowds out the others
            if by_kind[k] and len(chosen) < cap:
                chosen.append(by_kind[k].pop())
    return sorted(answerable + chosen, key=lambda r: r["question_id"])


def _questions_for(args: tuple[str, int, int]) -> list[dict]:
    folder, seed, target = args
    schema = json.loads((Path(folder) / "schema.json").read_text(encoding="utf-8"))
    paths = cp.copy_paths(Path(folder))
    answerable = questions.answerable_questions(schema, paths, seed=seed, target=target)
    decline = declines.decline_questions(schema, answerable, seed=seed)
    return mix(answerable, decline, seed, schema["schema_id"])


def summarize(records: list[dict], designs: int) -> dict:
    kinds = Counter(r["kind"] for r in records)
    answerable = [r for r in records if r["kind"] == "answerable"]
    return {
        "designs": designs,
        "questions": len(records),
        "by_kind": dict(kinds),
        "answerable_share": round(kinds["answerable"] / len(records), 3) if records else 0,
        "by_decline_subtype": dict(Counter(r["decline_subtype"] for r in records if r["kind"] != "answerable")),
        "answerable_by_shape": dict(Counter(r["shape"] for r in answerable)),
        "answerable_by_difficulty": dict(Counter(r["difficulty"] for r in answerable)),
        "by_family": dict(Counter(r["family"] for r in records)),
        "designs_with_fewer_than_10_answerable": sum(
            1 for n in Counter(r["schema_id"] for r in answerable).values() if n < 10
        ),
    }


def build(
    root: Path = gen.DEFAULT_OUT,
    families: list[str] | None = None,
    per_topic: int = gen.DEFAULT_PER_TOPIC,
    copies: int = fill.COPIES,
    seed: int = 0,
    target: int = 20,
    workers: int = 8,
) -> dict:
    # 1.2 Designs. Remove design folders a previous, larger build left behind.
    designs = gen.generate(families, per_topic, seed)
    keep = {s["schema_id"] for s in designs}
    if root.exists():
        for folder in root.iterdir():
            if folder.is_dir() and (folder / "schema.json").exists() and folder.name not in keep:
                shutil.rmtree(folder)
    gen.write(designs, root)
    print(f"1.2  {len(designs)} designs", file=sys.stderr, flush=True)

    # 1.3 Filled copies.
    fill.fill_all(root, copies, seed, workers)
    print(f"1.3  {len(designs) * copies} filled copies", file=sys.stderr, flush=True)

    # 1.4-1.6 Questions, declines, and the mix, design by design.
    jobs = [(str(root / s["schema_id"]), seed, target) for s in designs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        per_design = list(pool.map(_questions_for, jobs, chunksize=2))
    records = [r for design in per_design for r in design]
    print(f"1.4-1.6  {len(records)} questions", file=sys.stderr, flush=True)

    with (root / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(records, len(designs)) | {"seed": seed, "copies": copies, "per_topic": per_topic}
    (root / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build all of plan Phase 1 with one command.")
    parser.add_argument("--families", nargs="*", help="families to build (default: all)")
    parser.add_argument("--per-topic", type=int, default=gen.DEFAULT_PER_TOPIC, help="designs per topic")
    parser.add_argument("--copies", type=int, default=fill.COPIES, help="filled copies per design")
    parser.add_argument("--seed", type=int, default=0, help="global seed")
    parser.add_argument("--target", type=int, default=20, help="answerable questions to aim for per design")
    parser.add_argument("--root", type=Path, default=gen.DEFAULT_OUT, help="output folder")
    args = parser.parse_args(argv)

    summary = build(args.root, args.families, args.per_topic, args.copies, args.seed, args.target)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
