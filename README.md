# abstain-sql

Small text-to-SQL models answer every question and get a third to half of them wrong: the best RL-trained models at 4B parameters or smaller reach about 54–67% execution accuracy on BIRD dev (execution accuracy means the generated SQL counts as correct only if running it returns the same result as the reference query). A model that wrong loses its users' trust one confident answer at a time. This project trains a small open model to do the thing none of them does: say "I don't know."

## How

The method is GRPO, a reinforcement-learning algorithm that samples a group of candidate answers per question and reinforces the ones that score best. The reward is three numbers: +1 for SQL whose execution result is correct, 0 for abstaining, −c for a wrong answer. Sweeping the penalty c traces the trade-off between answering more and erring less. A conformal calibration step — choosing a confidence cutoff on held-out data so a stated error target provably holds — then turns the trained behavior into a guarantee: error ≤ α on the queries the model answers.

## Status

Research phase complete; verdict: go. As of 2026-11-04, no published work combines RL training, an explicit abstention reward, and risk–coverage evaluation (how many queries get answered vs. how often those answers are wrong) in text-to-SQL at any model size. Three near-misses each hold two of the three pieces; the full catalog and the differentiators are in the novelty table.

What to do next lives in [plan.md](plan.md): 14 phases with checkboxes, 5 gates that must pass before money moves, the reward table, open decisions with defaults, and the proposed repo structure. Why each step is there lives in [research/REPORT.md](research/REPORT.md), in 3 parts: the flags that changed the plan and the go/no-go verdict (Part 1), the novelty table with one row per paper (Part 2), and the cost model, where the full 8-run matrix is estimated at $20–36 against a $30–50 budget (Part 3). The raw research notes behind it sit in [research/notes/](research/notes/).

Training has not started, and no money has been spent.
