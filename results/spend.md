# Spend log

Every dollar this project spends goes here, the day it is spent. The budget is $30–50 in total, covering cloud GPU time on Modal or RunPod and the API calls to frontier models in Phase 10. Nothing above $1 gets spent before the smoke test passes (gate G2), and every session gets an explicit go-ahead before it starts.

**Spent so far: $0.00 of $30–50.**

## Sessions

One line per cloud session or API batch. Dollars are the list price, even when Modal's free credit pays; I mark those "(credit)" so both the true cost and the cash cost stay visible.

| Date | Platform and GPU | Minutes | Dollars | Purpose |
|---|---|---|---|---|
| | | | | No sessions yet |

## Cost estimates

These are the plan's planning estimates, not measurements. Step 5.4 replaces them with costs computed from the smoke test's measured seconds per training step.

| What | When | Estimated cost |
|---|---|---|
| Smoke test: 20 training steps | Phase 5, week 2 | at most $0.60 |
| First 2 full runs: the accuracy-only baseline and c = 1 | Phase 6, week 2 | about $5 |
| The rest of the 8-run matrix | Phase 7, week 3 | about $11–31 |
| Frontier-model baselines through an API | Phase 10, week 3 | a few dollars |
| All of it | Weeks 2–3 | about $16–36, plus a few dollars of API calls |

Modal's $30 Starter credit very likely covers GPU compute: the dashboard calls it "$30.00 included compute credits per month", and GPU time is compute. No page states it for this credit, so it stays unverified until the first GPU job's bill shows $0 in charges. If it holds, the smoke test and early runs cost $0 out of pocket, up to the $30 monthly credit.

## Caps

- **Modal (opened 2026-11-08):** $30 monthly Workspace budget. It caps total usage before credits, so Modal stops all jobs at $30 of usage per month. Checked on the dashboard the same day: $0.00 spent this billing cycle (Nov 1–30, 2026), $30.00 of credits remaining, $30.00 of workspace usage remaining; the credits and the usage limit draw from the same $30 pool. No separate spend limit (a cap on out-of-pocket charges after credits) is set yet.
- **RunPod:** not opened yet.
