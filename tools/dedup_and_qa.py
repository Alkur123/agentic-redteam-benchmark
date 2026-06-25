#!/usr/bin/env python3
"""dedup_and_qa.py — quality gate for the AUGMENTED tier.

Run AFTER expand_openai.py. Operates ONLY on machine-augmented samples
(provenance.tier == "augmented"); GOLD seeds are never modified or deleted.

Two passes:
  1. NEAR-DUPLICATE FILTER — token-set Jaccard over (declared_goal + all action.raw
     + all thoughts). An augmented sample is dropped if it is >= --threshold similar
     to any GOLD sample or to an already-kept augmented sample. This kills the
     "templated tell" that reviewers spot in mass-generated sets.
  2. SEMANTIC QA — beyond the schema validator:
       - benign_mirror variants must be CONTINUE with no hard_block signal;
       - attack samples must be KILL_SESSION/WARN;
       - rationale must be a real paragraph (>= 60 chars);
       - declared_goal must not be byte-identical to its seed.
     Failures are dropped (or only reported with --report-only).

Usage:
    python tools/dedup_and_qa.py                 # apply (delete drops)
    python tools/dedup_and_qa.py --report-only   # report, delete nothing
    python tools/dedup_and_qa.py --threshold 0.82
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SAMPLES = _HERE.parent / "v0.1" / "samples"

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(sample: dict) -> set[str]:
    parts = [sample.get("declared_goal", "")]
    for step in sample.get("trajectory", []) or []:
        parts.append(step.get("thought", ""))
        act = step.get("action", {}) or {}
        parts.append(str(act.get("raw", "")))
    text = " ".join(parts).lower()
    return set(_WORD.findall(text))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _load(fp: Path) -> dict | None:
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _semantic_issues(s: dict, seed_goal: str | None) -> list[str]:
    out = []
    gt = s.get("ground_truth", {}) or {}
    decision = gt.get("expected_decision_at_drift")
    signals = gt.get("drift_signals_expected", []) or []
    variant = (s.get("provenance", {}) or {}).get("variant")
    if variant == "benign_mirror":
        if decision != "CONTINUE":
            out.append(f"benign_mirror but decision={decision}")
        if any(str(x).startswith("hard_block") for x in signals):
            out.append("benign_mirror carries a hard_block signal")
    else:
        if decision not in ("KILL_SESSION", "WARN"):
            out.append(f"attack sample but decision={decision}")
    if len(str(gt.get("rationale", ""))) < 60:
        out.append("rationale too short (<60 chars)")
    if seed_goal and s.get("declared_goal", "").strip() == seed_goal.strip():
        out.append("declared_goal byte-identical to seed")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.85, help="Jaccard dup threshold")
    ap.add_argument("--report-only", action="store_true", help="do not delete, just report")
    args = ap.parse_args()

    files = sorted(_SAMPLES.rglob("art-*.json"))
    gold: list[tuple[Path, dict, set]] = []
    aug: list[tuple[Path, dict, set]] = []
    gold_goal_by_id: dict[str, str] = {}
    for fp in files:
        s = _load(fp)
        if not s:
            continue
        toks = _tokens(s)
        if "provenance" in s and (s["provenance"] or {}).get("tier") == "augmented":
            aug.append((fp, s, toks))
        else:
            gold.append((fp, s, toks))
            if s.get("id"):
                gold_goal_by_id[s["id"]] = s.get("declared_goal", "")

    print(f"Gold: {len(gold)}  Augmented: {len(aug)}")

    kept_tokens: list[set] = [t for _, _, t in gold]  # dups checked against gold + kept aug
    drop_dup: list[Path] = []
    drop_qa: list[tuple[Path, list[str]]] = []
    kept = 0
    qa_counts: Counter = Counter()

    # Deterministic order: by id.
    aug.sort(key=lambda x: x[1].get("id", ""))
    for fp, s, toks in aug:
        seed_id = (s.get("provenance", {}) or {}).get("seed_id")
        issues = _semantic_issues(s, gold_goal_by_id.get(seed_id))
        if issues:
            drop_qa.append((fp, issues))
            for i in issues:
                qa_counts[i.split(" but ")[0].split(" (")[0]] += 1
            continue
        is_dup = any(_jaccard(toks, kt) >= args.threshold for kt in kept_tokens)
        if is_dup:
            drop_dup.append(fp)
            continue
        kept_tokens.append(toks)
        kept += 1

    print(f"\nAugmented kept: {kept}")
    print(f"Dropped as near-duplicate: {len(drop_dup)}")
    print(f"Dropped on QA: {len(drop_qa)}")
    if qa_counts:
        print("  QA breakdown:", dict(qa_counts))
    final_total = len(gold) + kept
    print(f"\nProjected final corpus: {final_total} "
          f"({len(gold)} gold + {kept} augmented)")

    if args.report_only:
        print("\n--report-only: nothing deleted.")
        return 0

    for fp in drop_dup:
        fp.unlink()
    for fp, _ in drop_qa:
        fp.unlink()
    print(f"\nDeleted {len(drop_dup) + len(drop_qa)} augmented files.")
    print("Re-run: python tools/validate_samples.py v0.1/samples/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
