# abstain-sql

Small text-to-SQL models answer every question and get a third to half of them wrong: the best RL-trained models at 4B parameters or smaller reach about 54–67% execution accuracy on BIRD dev (execution accuracy means the generated SQL counts as correct only if running it returns the same result as the reference query). A model that wrong loses its users' trust one confident answer at a time. This project trains a small open model to do the thing none of them does: say "I don't know."

## How

The method is GRPO, a reinforcement-learning algorithm that samples a group of candidate answers per question and reinforces the ones that score best. The reward is three numbers: +1 for SQL whose execution result is correct, 0 for abstaining, −c for a wrong answer. Sweeping the penalty c traces the trade-off between answering more and erring less. A conformal calibration step — choosing a confidence cutoff on held-out data so a stated error target provably holds — then turns the trained behavior into a guarantee: error ≤ α on the queries the model answers.

## Status

Research phase complete; verdict: go. As of 2026-11-04, no published work combines RL training, an explicit abstention reward, and risk–coverage evaluation (how many queries get answered vs. how often those answers are wrong) in text-to-SQL at any model size. Three near-misses each hold two of the three pieces; the full catalog and the differentiators are in the novelty table.

What to do next lives in [plan.md](plan.md): 14 phases with checkboxes, 5 gates that must pass before money moves, the reward table, open decisions with defaults, and the proposed repo structure. Why each step is there lives in [research/REPORT.md](research/REPORT.md), in 3 parts: the flags that changed the plan and the go/no-go verdict (Part 1), the novelty table with one row per paper (Part 2), and the cost model, where the full 8-run matrix is estimated at $16–36 against a $30–50 budget (Part 3). The raw research notes behind it sit in [research/notes/](research/notes/).

Phase 0 is complete. The repo follows the layout in plan.md, every dependency is pinned, all 3 candidate models load under those pins, the external evaluation data downloads and runs, a Modal account is open with a $30 monthly cap, and the spend and decision logs have started (steps 0.1 to 0.6). The code folders hold stubs that name the plan step each file implements. Training has not started, and no money has been spent.

## Setup

I pinned every dependency in one `pyproject.toml` and one `uv.lock`, so the laptop and the cloud GPU run identical versions of all 215 packages. The training stack is TRL 1.17.1, vLLM 0.32.0, torch 2.13.0, transformers 5.19.0, and PEFT 0.21.2. vLLM 0.32.0 is the newest release TRL 1.17.1 accepts; it fixes torch at 2.13.0 and caps transformers below 5.20, and its own dependencies cap numpy below 2.4.

```bash
uv sync                          # laptop: data generation, checker, calibration, analysis
uv sync --frozen --extra train   # cloud GPU: adds TRL, vLLM, torch, transformers, PEFT
uv run pytest
uv run ruff check . && uv run ruff format .   # style check, then auto-format
uv run python data/download_external.py   # BIRD dev and TrustSQL, about 470 MB
uv run python data/check_external.py      # recount and rerun them; a few minutes
```

The external data is not in git. The download script fetches it from the official sources and stops if any file's fingerprint differs from the one pinned in the script, since a changed upstream file would make results incomparable. The check script then recomputes every number in `results/external_data.md` and saves them to `results/external_check.json`.

`.gitattributes` stores every text file with Unix line endings, so files written on the Windows laptop and the Linux GPU box stay identical.

The GPU image needs glibc 2.31 or newer, the system C library that Ubuntu 20.04 and later ship. llguidance, which vLLM uses to constrain output to a format, publishes no Linux build for older systems.

## How the synthetic data is built

The training data comes from one hierarchy. Each level is built from the one above it; the right-hand column says which plan step builds it and whether it exists yet.

```
6 families                                        step 1.1   done
│   healthcare, finance, logistics and transport, retail
│   and consumer, energy and natural resources, media
│   and entertainment
│
└──► 10 topics per family = 60 topics             step 1.1   done
     │   e.g. healthcare → hospitals, public health, dental
     │   care, veterinary care, medical devices, and 5 more
     │
     └──► 1 table list per topic = 60 lists       step 1.2   done
          │   each lists 9-12 tables that fit the topic and
          │   allows at least 20 different designs
          │
          └──► 8 designs per topic = 480          step 1.2   done
               │   each picks 3-8 linked tables from its table list;
               │   empty tables: columns and links, no rows
               │
               ├──► 5 filled copies per design    step 1.3   next
               │       = 2,400 SQLite databases;
               │       same tables, different made-up rows
               │
               └──► questions per design          steps 1.4-1.6
                    ├──► answerable, about 70%:
                    │       question + gold SQL, correct only
                    │       if it matches on every filled copy
                    └──► should decline, about 30%:
                            unanswerable, ambiguous, out-of-scope
```

One path through it: the healthcare family contains the public health topic; its table list offers 10 tables; design `healthcare__public_health__02` picks 5 of them (districts, facilities, funding awards, staff, screenings); step 1.3 fills that design 5 times; step 1.4 writes its questions. Step 3.1 then splits whole topics, never single designs or questions, into train, dev, calibration, and test sets, so the model is always tested on topics it never trained on.

## Codebase map

Files marked *stub* hold only a docstring naming the plan step that fills them in. Everything else runs today.

```
abstain-sql/
  README.md                  # this file
  plan.md                    # 14 phases with checkboxes, gates, reward table, open decisions
  pyproject.toml             # pinned dependencies; the "train" extra adds the GPU stack
  uv.lock                    # exact versions of all 215 packages, laptop and GPU alike
  .gitattributes             # Unix line endings for every text file
  research/
    REPORT.md                # findings, novelty table, cost model
    notes/                   # 7 raw research notes behind the report
  data/
    download_external.py     # fetches BIRD dev and TrustSQL, checks pinned fingerprints
    check_external.py        # recounts both and reruns every gold query; writes the numbers
    generator/               # synthetic schemas, data, questions, declines (Phase 1)
      families.yaml          # 6 schema families of 10 topics each (step 1.1)
      templates/             # one file per family: a table list for each topic (step 1.2)
      schemas.py             # samples 3-8 linked tables per schema; writes 480 schemas (step 1.2)
    splits/                  # empty: train / dev / calibration / test manifests (Phase 3)
    external/                # gitignored: BIRD dev and TrustSQL, rebuilt by the script above
    instances/               # gitignored: generated schemas, then their SQLite instances (Phase 1)
  checker/
    execute.py               # stub: read-only sandboxed SQL execution (step 2.1)
    testsuite.py             # stub: multi-instance correctness oracle (steps 2.2-2.4)
  train/
    inspect_models.py        # checks the 3 candidate models; writes results/models.md
    reward.py                # stub: the abstention reward (step 4.1)
    grpo_train.py            # stub: GRPO training with vLLM and LoRA (step 4.3)
    configs/                 # empty: one YAML per training run (step 4.3)
  infra/                     # empty: Modal runner (step 4.6)
  calib/
    signals.py               # stub: confidence signals (step 8.1)
    ltt.py                   # stub: Learn-then-Test certification (steps 8.2-8.3)
  eval/
    generate.py              # stub: one pass of model outputs per checkpoint (step 7.4)
    metrics.py               # stub: accuracy, coverage, selective risk, AURC, RS(c) (step 11.1)
    baselines.py             # stub: prompted, self-consistency, frontier baselines (Phase 10)
    bootstrap.py             # stub: resampling for confidence intervals (step 12.1)
    local_gguf.py            # stub: laptop latency harness (step 11.3)
  tests/
    test_imports.py          # every CPU-side module imports
    test_families.py         # families.yaml keeps 6 families of 10 distinct topics
    test_schemas.py          # every generated schema keeps the plan's rules; bad templates fail
  results/
    models.md                # the 3 models: architecture, vocabulary, license, thinking switch
    external_data.md         # BIRD and TrustSQL breakdown, gold-query runs, checker rules
    external_check.json      # the raw numbers behind external_data.md, from check_external.py
    spend.md                 # every dollar against the $30-50 budget; caps
    decisions.md             # open decisions D1-D8, plus choices already settled
  runs/                      # gitignored: checkpoints and logs (Phase 5 on)
```
