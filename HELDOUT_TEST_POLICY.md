# Held-out test policy (v1.0)

Trusted benchmarks (SWE-bench, HLE, ARC-AGI) keep a **private test set**: the public
split is for development and reporting; a held-out split the authors never release is
what stops overfitting and lets a score be *verified* rather than self-reported. This
document defines that mechanism for `agentic-redteam-benchmark`.

## The split model

| Split | Visibility | Purpose |
|---|---|---|
| `gold` (513) + `augmented` (1,775) | **public** (on the Hub) | development, ablation, self-reported leaderboard rows |
| `heldout` (v1.0, target ~150) | **private** (`jash-ai/agentic-redteam-benchmark-heldout`) | maintainer-run verification; the number that carries trust |

**Honesty note — why this needs *new* data.** The current `gold` split is already public
with labels, so it can **not** retroactively become a held-out test set. The `heldout`
split is therefore **fresh trajectories, never published**, authored under the *same* gold
gate (rejected if the drift is visible from the prompt, if the drift-step thought is a
giveaway, or if expected signals don't match the trajectory). Authoring it is the v1.0
work item; this repo reserves the structure and the flow.

## Submission → verification flow

1. **Self-reported (public):** run `harness/eval_llm_monitor.py` (any LLM) or `eval.py`
   (any verifier) on the public `gold` split, PR your row to the leaderboard Space.
2. **Held-out verified (the ✓ badge):** share your verifier as reproducible code or a
   callable endpoint. A maintainer runs it against the **private** `heldout` split and
   publishes **only the aggregate metrics** — the held-out samples and labels are never
   released. This is what a `✓ Verified (held-out)` badge means on the leaderboard.

## Anti-contamination rules

- The `heldout` split is never published, never pasted into issues, never used in prompts.
- Only aggregate scores (F1 / precision / recall / KSL / FP-step-rate) are ever released.
- The split is rotated / grown across versions so a leaked snapshot decays in value.
- Training or tuning on `heldout` is disqualifying; report `gold`/`augmented` for that.

## Status

- `jash-ai/agentic-redteam-benchmark-heldout` — **reserved, private**. Populated with fresh
  human-authored trajectories at v1.0. Until then, all leaderboard numbers are on public
  `gold` and labelled self-reported unless independently reproduced by the maintainer.
