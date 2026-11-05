# abstain-sql

Small text-to-SQL models answer every question and get a third to half of them wrong: the best RL-trained models at 4B parameters or smaller reach about 54–67% execution accuracy on BIRD dev (execution accuracy means the generated SQL counts as correct only if running it returns the same result as the reference query). A model that wrong loses its users' trust one confident answer at a time. This project trains a small open model to do the thing none of them does: say "I don't know."

## How

The method is GRPO, a reinforcement-learning algorithm that samples a group of candidate answers per question and reinforces the ones that score best. The reward is three numbers: +1 for SQL whose execution result is correct, 0 for abstaining, −c for a wrong answer. Sweeping the penalty c traces the trade-off between answering more and erring less. A conformal calibration step — choosing a confidence cutoff on held-out data so a stated error target provably holds — then turns the trained behavior into a guarantee: error ≤ α on the queries the model answers.

## Status

Research phase complete; verdict: go. As of 2026-11-04, no published work combines RL training, an explicit abstention reward, and risk–coverage evaluation (how many queries get answered vs. how often those answers are wrong) in text-to-SQL at any model size. Three near-misses each hold two of the three pieces; the full catalog and the differentiators are in the novelty table.

What to do next lives in [plan.md](plan.md): 14 phases with checkboxes, 5 gates that must pass before money moves, the reward table, open decisions with defaults, and the proposed repo structure. Why each step is there lives in [research/REPORT.md](research/REPORT.md), in 3 parts: the flags that changed the plan and the go/no-go verdict (Part 1), the novelty table with one row per paper (Part 2), and the cost model, where the full 8-run matrix is estimated at $16–36 against a $30–50 budget (Part 3). The raw research notes behind it sit in [research/notes/](research/notes/).

Phase 0 is under way. The repo follows the layout in plan.md, every dependency is pinned, all 3 candidate models load under those pins, and the external evaluation data downloads and runs (steps 0.1 to 0.4). The code folders hold stubs that name the plan step each file implements. Training has not started, and no money has been spent.

## Setup

I pinned every dependency in one `pyproject.toml` and one `uv.lock`, so the laptop and the cloud GPU run identical versions of all 214 packages. The training stack is TRL 1.17.1, vLLM 0.32.0, torch 2.13.0, transformers 5.19.0, and PEFT 0.21.2. vLLM 0.32.0 is the newest release TRL 1.17.1 accepts; it fixes torch at 2.13.0 and caps transformers below 5.20, and its own dependencies cap numpy below 2.4.

```bash
uv sync                          # laptop: data generation, checker, calibration, analysis
uv sync --frozen --extra train   # cloud GPU: adds TRL, vLLM, torch, transformers, PEFT
uv run pytest
uv run python data/download_external.py   # BIRD dev and TrustSQL, about 470 MB
```

The external data is not in git. The download script fetches it from the official sources and stops if any file's fingerprint differs from the one pinned in the script, since a changed upstream file would make results incomparable.

The GPU image needs glibc 2.31 or newer, the system C library that Ubuntu 20.04 and later ship. llguidance, which vLLM uses to constrain output to a format, publishes no Linux build for older systems.
