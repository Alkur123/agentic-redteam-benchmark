#!/usr/bin/env python3
"""corpus_stats.py — tier-aware distribution report for the released dataset.

Splits the corpus into GOLD (hand-authored, no provenance block) and AUGMENTED
(provenance.tier == "augmented") and prints the numbers the dataset card and
paper P3 quote: per-category gold/aug/total, difficulty / language / runtime
shares, attack-vs-benign balance, drift-step stats, hard-block fraction, tool
diversity. Also emits a ready-to-paste markdown block (--md).

Pure stdlib.

Usage:
    python tools/corpus_stats.py
    python tools/corpus_stats.py --md      # markdown tables for the card
    python tools/corpus_stats.py --json stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SAMPLES = _HERE.parent / "v0.1" / "samples"

CAT_ORDER = ["tool_call_abuse", "goal_shift", "plan_exec_divergence",
             "agent_coercion", "capability_laundering"]


def _stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load():
    gold, aug = [], []
    for fp in sorted(_SAMPLES.rglob("art-*.json")):
        try:
            s = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if (s.get("provenance") or {}).get("tier") == "augmented":
            aug.append(s)
        else:
            gold.append(s)
    return gold, aug


def report(samples):
    n = len(samples)
    cat = Counter(s.get("category") for s in samples)
    diff = Counter(s.get("difficulty") for s in samples)
    lang = Counter(s.get("language") for s in samples)
    runtime = Counter(s.get("agent_runtime") for s in samples)
    decision = Counter((s.get("ground_truth", {}) or {}).get("expected_decision_at_drift") for s in samples)
    drift = [(s.get("ground_truth", {}) or {}).get("drift_step") for s in samples]
    drift = [d for d in drift if isinstance(d, int)]
    hardblock = sum(
        1 for s in samples
        if any(str(x).startswith("hard_block")
               for x in (s.get("ground_truth", {}) or {}).get("drift_signals_expected", []) or [])
    )
    tools = set()
    for s in samples:
        for st in s.get("trajectory", []) or []:
            nm = (st.get("action", {}) or {}).get("name")
            if nm:
                tools.add(nm)
    return {
        "n": n, "cat": cat, "diff": diff, "lang": lang, "runtime": runtime,
        "decision": decision, "drift_mean": (sum(drift) / len(drift)) if drift else 0,
        "drift_std": _stdev(drift), "drift_min": min(drift) if drift else 0,
        "drift_max": max(drift) if drift else 0, "hardblock": hardblock, "tools": len(tools),
    }


def _pct(c: Counter, total: int):
    return {k: (v, 100 * v / total if total else 0) for k, v in c.most_common()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="emit markdown tables")
    ap.add_argument("--json", type=Path, help="write stats json")
    args = ap.parse_args()

    gold, aug = load()
    alls = gold + aug
    rg, ra, rall = report(gold), report(aug), report(alls)
    total = rall["n"]

    benign = rall["decision"].get("CONTINUE", 0)
    attack = total - benign

    if args.json:
        args.json.write_text(json.dumps({
            "total": total, "gold": rg["n"], "augmented": ra["n"],
            "by_category": {c: {"gold": rg["cat"].get(c, 0), "aug": ra["cat"].get(c, 0),
                                "total": rall["cat"].get(c, 0)} for c in CAT_ORDER},
            "difficulty": dict(rall["diff"]), "language": dict(rall["lang"]),
            "runtime": dict(rall["runtime"]), "decision": dict(rall["decision"]),
            "attack": attack, "benign": benign,
            "drift_mean": rall["drift_mean"], "drift_std": rall["drift_std"],
            "drift_range": [rall["drift_min"], rall["drift_max"]],
            "hardblock": rall["hardblock"], "hardblock_frac": rall["hardblock"] / total if total else 0,
            "tool_diversity": rall["tools"],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    print(f"TOTAL {total}  =  gold {rg['n']}  +  augmented {ra['n']}")
    print(f"attack {attack} ({100*attack/total:.1f}%) | benign/CONTINUE {benign} ({100*benign/total:.1f}%)")
    print(f"drift_step mean {rall['drift_mean']:.2f} std {rall['drift_std']:.2f} range [{rall['drift_min']},{rall['drift_max']}]")
    print(f"hard_block fraction {rall['hardblock']/total:.3f}  | tool diversity {rall['tools']} unique tools")

    if args.md:
        print("\n### Category (gold / augmented / total)\n")
        print("| Category | Gold | Augmented | Total |")
        print("|---|--:|--:|--:|")
        for c in CAT_ORDER:
            print(f"| `{c}` | {rg['cat'].get(c,0)} | {ra['cat'].get(c,0)} | {rall['cat'].get(c,0)} |")
        print(f"| **Total** | **{rg['n']}** | **{ra['n']}** | **{total}** |")

        def block(title, c):
            print(f"\n### {title}\n")
            print("| Value | Count | % |")
            print("|---|--:|--:|")
            for k, (v, p) in _pct(c, total).items():
                print(f"| `{k}` | {v} | {p:.1f}% |")

        block("Difficulty", rall["diff"])
        block("Language", rall["lang"])
        block("Agent runtime", rall["runtime"])
        block("Expected decision", rall["decision"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
