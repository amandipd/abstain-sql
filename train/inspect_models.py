"""Check the 3 candidate models against the pinned transformers (plan step 0.3).

Downloads each model's config and tokenizer (never the weights), proves they
load, and records: the architecture name vLLM keys on, the vocabulary size,
the license, the chat template, and how thinking mode turns on and off.

Writes results/models.md. Run it from the repo root:

    uv run python train/inspect_models.py
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

MODELS = [
    "Qwen/Qwen3.5-2B",
    "HuggingFaceTB/SmolLM3-3B",
    "google/gemma-4-E2B-it",
]

OUT = Path(__file__).resolve().parent.parent / "results" / "models.md"

# One question, asked three ways, to show each model's chat template in use.
SAMPLE = [{"role": "user", "content": "How many customers live in Austin?"}]


def license_of(repo: str) -> str:
    """Read the license from the model card's metadata header."""
    try:
        from huggingface_hub import model_info

        info = model_info(repo)
        card = info.card_data or {}
        name = card.get("license") or "not stated"
        link = card.get("license_link")
        return f"{name} ({link})" if link else str(name)
    except Exception as e:  # gated repos can refuse even metadata
        return f"could not read: {type(e).__name__}: {e}"


def thinking_probe(tok, repo: str) -> dict[str, str]:
    """Render the sample chat with thinking on, off, and unspecified."""
    out: dict[str, str] = {}
    for label, kwargs in [
        ("default", {}),
        ("thinking on", {"enable_thinking": True}),
        ("thinking off", {"enable_thinking": False}),
    ]:
        try:
            out[label] = tok.apply_chat_template(
                SAMPLE, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except Exception as e:
            out[label] = f"REFUSED: {type(e).__name__}: {e}"
    return out


def inspect(repo: str) -> list[str]:
    lines = [f"## {repo}", ""]
    try:
        config = AutoConfig.from_pretrained(repo)
        tok = AutoTokenizer.from_pretrained(repo)
    except Exception as e:
        lines += [
            f"**FAILED TO LOAD** with transformers {transformers.__version__}:",
            "",
            f"    {type(e).__name__}: {e}",
            "",
        ]
        return lines

    arch = ", ".join(getattr(config, "architectures", None) or ["not stated"])
    # Multimodal configs nest the language model's settings one level down.
    text_cfg = getattr(config, "text_config", None) or config
    vocab = getattr(text_cfg, "vocab_size", "not stated")
    ctx = getattr(text_cfg, "max_position_embeddings", "not stated")

    lines += [
        f"- Architecture (the name vLLM keys on): `{arch}`",
        f"- Model type: `{config.model_type}`",
        f"- Vocabulary size: {vocab}",
        f"- Max context (positions): {ctx}",
        f"- License: {license_of(repo)}",
        f"- Tokenizer class: `{type(tok).__name__}`, {len(tok)} tokens incl. added ones",
        "",
        "Chat template on one sample question, with the thinking switch:",
        "",
    ]
    for label, text in thinking_probe(tok, repo).items():
        lines += [f"**{label}:**", "", "```"]
        lines += text.splitlines() or [text]
        lines += ["```", ""]
    return lines


def main() -> int:
    stamp = datetime.date.today().isoformat()
    lines = [
        "# Model checks (plan step 0.3)",
        "",
        f"Generated {stamp} by `train/inspect_models.py` with "
        f"transformers {transformers.__version__}. Configs and tokenizers only; "
        "no weights were downloaded.",
        "",
    ]
    failures = 0
    for repo in MODELS:
        block = inspect(repo)
        failures += any("FAILED TO LOAD" in line for line in block)
        lines += block
        print(f"checked {repo}", file=sys.stderr)

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({failures} of {len(MODELS)} failed)", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
