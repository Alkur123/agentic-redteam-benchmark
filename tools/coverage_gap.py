"""coverage_gap.py — report what's authored vs the v1.0 (1000-sample) plan.

Phase B authoring is manual; this tool makes it *targeted*. It reads every sample under
the samples tree and reports, against the BENCHMARK_DESIGN.md targets, exactly how many
more samples each category / subcategory / difficulty / runtime / language bucket needs —
plus the §5 balance metrics (drift-step distribution, hard-block fraction, tool diversity,
IAA-subset size). The headline is the per-subcategory GAP: the to-author worklist.

Targets are transcribed from BENCHMARK_DESIGN.md (§2.4 composition, §3.2 difficulty,
§3.3 language, §3.4 runtime, §5 balance metrics). If you change the design, change it
there first, then update the constants below.

Pure standard library (CI needs no venv). Mirrors tools/validate_samples.py conventions.

Usage:
    python tools/coverage_gap.py                      # defaults to ../v0.1/samples
    python tools/coverage_gap.py v0.1/samples/
    python tools/coverage_gap.py v0.1/samples/ --json coverage.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Windows consoles default to cp1252 and choke on the marker glyphs below.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

# ── targets (BENCHMARK_DESIGN.md) ─────────────────────────────────────────────

TOTAL_TARGET = 1000

# §2.4 — per-category 200; subcategory counts uneven on purpose (incident prevalence).
SUBCAT_TARGETS = {
    "tool_call_abuse":       {"filesystem_exfil": 60, "network_exfil": 60,
                              "credential_read": 50, "sudo_escalation": 30},
    "goal_shift":            {"topic_pivot": 55, "identity_swap": 40,
                              "persona_assumption": 55, "helpfulness_override": 50},
    "plan_exec_divergence":  {"step_skip": 50, "step_inject": 55,
                              "step_reorder": 40, "plan_abandonment": 55},
    "agent_coercion":        {"indirect_prompt": 60, "output_poisoning": 55,
                              "role_confusion": 45, "authority_spoofing": 40},
    "capability_laundering": {"benign_tool_misuse": 60, "multi_step_concealment": 55,
                              "scope_creep": 50, "side_channel": 35},
}
CATEGORY_TARGET = {cat: sum(subs.values()) for cat, subs in SUBCAT_TARGETS.items()}  # all 200

# §3.2 difficulty share (of 1000); per-category target = share * 200.
DIFFICULTY_SHARE = {"easy": 0.25, "medium": 0.50, "hard": 0.25}
# §3.3 language share (v1.0).
LANGUAGE_SHARE = {"en": 0.85, "hi-en": 0.10, "hi": 0.05}
# §3.4 agent_runtime share.
RUNTIME_SHARE = {"langgraph_v0.2": 0.35, "claude_code_v1": 0.25,
                 "crewai_v0.3": 0.15, "autogen_v0.3": 0.10, "generic": 0.15}

# §5 balance metrics.
IAA_SUBSET_TARGET = 100
HARDBLOCK_FRAC_RANGE = (0.20, 0.30)
TOOL_DIVERSITY_MIN = 80
DRIFT_MEAN_RANGE = (5.0, 7.0)
DRIFT_STDEV_RANGE = (1.0, 2.5)
SHARE_TOL_PP = 5.0  # ±5 percentage-point tolerance on share targets


# ── loading ───────────────────────────────────────────────────────────────────

def load_samples(root: Path) -> tuple:
    """Return (samples, skipped) — samples is a list of parsed dicts."""
    files = sorted(root.rglob("art-*.json")) if root.is_dir() else [root]
    samples, skipped = [], []
    for fp in files:
        try:
            samples.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            skipped.append((fp, str(e)))
    return samples, skipped


# ── helpers ───────────────────────────────────────────────────────────────────

def stdev(xs: list) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def mark(ok: bool) -> str:
    return "OK " if ok else "GAP"


def share_line(name: str, have: int, total: int, target_share: float) -> str:
    obs = (have / total) if total else 0.0
    tgt_ct = round(target_share * TOTAL_TARGET)
    gap = max(0, tgt_ct - have)
    within = abs(obs - target_share) * 100 <= SHARE_TOL_PP
    return (f"  {name:<16} have={have:<5d} share={obs*100:5.1f}%  "
            f"target={target_share*100:4.0f}% ({tgt_ct})  to_author={gap:<4d} "
            f"[{mark(within)} ±{SHARE_TOL_PP:.0f}pp]")


# ── report ─────────────────────────────────────────────────────────────────────

def build_report(samples: list) -> dict:
    cat_counts = Counter()
    subcat_counts = defaultdict(Counter)            # cat -> subcat -> n
    diff_counts = Counter()
    lang_counts = Counter()
    runtime_counts = Counter()
    decision_counts = Counter()
    drift_steps = []
    hardblock = 0
    iaa = 0
    tool_names = set()
    unknown = []                                    # (id, reason)

    for s in samples:
        sid = s.get("id", "?")
        cat = s.get("category", "?")
        sub = s.get("subcategory", "?")
        cat_counts[cat] += 1
        subcat_counts[cat][sub] += 1
        if cat not in SUBCAT_TARGETS:
            unknown.append((sid, f"unknown category {cat!r}"))
        elif sub not in SUBCAT_TARGETS[cat]:
            unknown.append((sid, f"unknown subcategory {sub!r} for {cat}"))

        diff_counts[s.get("difficulty", "?")] += 1
        lang_counts[s.get("language", "?")] += 1
        runtime_counts[s.get("agent_runtime", "?")] += 1

        gt = s.get("ground_truth", {}) or {}
        decision_counts[gt.get("expected_decision_at_drift", "?")] += 1
        ds = gt.get("drift_step")
        if isinstance(ds, int):
            drift_steps.append(ds)
        if any(str(sig).startswith("hard_block") for sig in gt.get("drift_signals_expected", []) or []):
            hardblock += 1

        if (s.get("metadata", {}) or {}).get("iaa_subset") is True:
            iaa += 1

        for step in s.get("trajectory", []) or []:
            name = (step.get("action", {}) or {}).get("name")
            if name:
                tool_names.add(name)

    return {
        "n": len(samples),
        "cat_counts": cat_counts, "subcat_counts": subcat_counts,
        "diff_counts": diff_counts, "lang_counts": lang_counts,
        "runtime_counts": runtime_counts, "decision_counts": decision_counts,
        "drift_steps": drift_steps, "hardblock": hardblock, "iaa": iaa,
        "tool_names": tool_names, "unknown": unknown,
    }


def print_report(r: dict) -> None:
    n = r["n"]
    bar = "=" * 78
    print(bar)
    print("  COVERAGE-GAP REPORT — authored vs the v1.0 (1000-sample) plan")
    print("  targets: BENCHMARK_DESIGN.md §2.4 / §3.2 / §3.3 / §3.4 / §5")
    print(bar)
    print(f"  total authored : {n} / {TOTAL_TARGET}    to_author = {max(0, TOTAL_TARGET - n)}")

    # ── per category + subcategory (the worklist) ──
    print("\n" + "-" * 78)
    print("  PER-CATEGORY / SUBCATEGORY  (have → target = to_author)")
    print("-" * 78)
    total_gap = 0
    for cat in SUBCAT_TARGETS:
        chave = r["cat_counts"].get(cat, 0)
        ctgt = CATEGORY_TARGET[cat]
        cgap = max(0, ctgt - chave)
        print(f"\n  {cat:<24} {chave:>4d} / {ctgt:<4d}   to_author={cgap:<4d} [{mark(chave >= ctgt)}]")
        for sub, stgt in SUBCAT_TARGETS[cat].items():
            shave = r["subcat_counts"][cat].get(sub, 0)
            sgap = max(0, stgt - shave)
            total_gap += sgap
            print(f"      {sub:<24} {shave:>4d} / {stgt:<4d}   to_author={sgap:<4d} [{mark(shave >= stgt)}]")
        # surface any off-vocab subcategories actually present
        for sub, cnt in r["subcat_counts"][cat].items():
            if cat in SUBCAT_TARGETS and sub not in SUBCAT_TARGETS[cat]:
                print(f"      {sub:<24} {cnt:>4d}  <-- OFF-PLAN subcategory (not in §2.4)")
    print(f"\n  Σ subcategory to_author = {total_gap}   (the manual Phase-B worklist)")

    # ── difficulty ──
    print("\n" + "-" * 78)
    print("  DIFFICULTY  (§3.2 — 25 / 50 / 25, ±5pp)")
    print("-" * 78)
    for d in ("easy", "medium", "hard"):
        print(share_line(d, r["diff_counts"].get(d, 0), n, DIFFICULTY_SHARE[d]))
    off = [d for d in r["diff_counts"] if d not in DIFFICULTY_SHARE]
    for d in off:
        print(f"  {d:<16} have={r['diff_counts'][d]:<5d}  <-- OFF-PLAN difficulty")

    # ── language ──
    print("\n" + "-" * 78)
    print("  LANGUAGE  (§3.3 — 85 / 10 / 5)")
    print("-" * 78)
    for lg in ("en", "hi-en", "hi"):
        print(share_line(lg, r["lang_counts"].get(lg, 0), n, LANGUAGE_SHARE[lg]))
    for lg in [x for x in r["lang_counts"] if x not in LANGUAGE_SHARE]:
        print(f"  {lg:<16} have={r['lang_counts'][lg]:<5d}  <-- OFF-PLAN language")

    # ── runtime ──
    print("\n" + "-" * 78)
    print("  AGENT_RUNTIME  (§3.4)")
    print("-" * 78)
    for rt in RUNTIME_SHARE:
        print(share_line(rt, r["runtime_counts"].get(rt, 0), n, RUNTIME_SHARE[rt]))
    for rt in [x for x in r["runtime_counts"] if x not in RUNTIME_SHARE]:
        print(f"  {rt:<16} have={r['runtime_counts'][rt]:<5d}  <-- OFF-PLAN runtime")

    # ── §5 balance metrics ──
    print("\n" + "-" * 78)
    print("  §5 BALANCE METRICS")
    print("-" * 78)
    ds = r["drift_steps"]
    if ds:
        m, sd = sum(ds) / len(ds), stdev(ds)
        m_ok = DRIFT_MEAN_RANGE[0] <= m <= DRIFT_MEAN_RANGE[1]
        sd_ok = DRIFT_STDEV_RANGE[0] <= sd <= DRIFT_STDEV_RANGE[1]
        print(f"  drift_step mean   {m:5.2f}  target {DRIFT_MEAN_RANGE}   [{mark(m_ok)}]")
        print(f"  drift_step stdev  {sd:5.2f}  target {DRIFT_STDEV_RANGE}   [{mark(sd_ok)}]")
    else:
        print("  drift_step        (none parsed)")
    hb = (r["hardblock"] / n) if n else 0.0
    hb_ok = HARDBLOCK_FRAC_RANGE[0] <= hb <= HARDBLOCK_FRAC_RANGE[1]
    print(f"  hard-block frac   {hb*100:5.1f}%  target {int(HARDBLOCK_FRAC_RANGE[0]*100)}-"
          f"{int(HARDBLOCK_FRAC_RANGE[1]*100)}%   [{mark(hb_ok)}]")
    td = len(r["tool_names"])
    print(f"  tool diversity    {td:<5d}  target >= {TOOL_DIVERSITY_MIN}   [{mark(td >= TOOL_DIVERSITY_MIN)}]")
    print(f"  IAA subset        {r['iaa']:<5d}  target {IAA_SUBSET_TARGET}   "
          f"[{mark(r['iaa'] >= IAA_SUBSET_TARGET)}]   (to_reserve={max(0, IAA_SUBSET_TARGET - r['iaa'])})")
    dec = r["decision_counts"]
    print("  decisions         " + ", ".join(f"{k}={dec[k]}" for k in sorted(dec)))

    if r["unknown"]:
        print("\n" + "-" * 78)
        print(f"  ANOMALIES ({len(r['unknown'])}) — off-vocabulary category/subcategory")
        print("-" * 78)
        for sid, why in r["unknown"][:25]:
            print(f"    {sid}: {why}")
        if len(r["unknown"]) > 25:
            print(f"    ... +{len(r['unknown']) - 25} more")
    print("\n" + bar)


def to_json(r: dict) -> dict:
    n = r["n"]
    return {
        "total": {"have": n, "target": TOTAL_TARGET, "to_author": max(0, TOTAL_TARGET - n)},
        "categories": {
            cat: {
                "have": r["cat_counts"].get(cat, 0),
                "target": CATEGORY_TARGET[cat],
                "to_author": max(0, CATEGORY_TARGET[cat] - r["cat_counts"].get(cat, 0)),
                "subcategories": {
                    sub: {"have": r["subcat_counts"][cat].get(sub, 0), "target": tgt,
                          "to_author": max(0, tgt - r["subcat_counts"][cat].get(sub, 0))}
                    for sub, tgt in SUBCAT_TARGETS[cat].items()
                },
            }
            for cat in SUBCAT_TARGETS
        },
        "difficulty": dict(r["diff_counts"]),
        "language": dict(r["lang_counts"]),
        "agent_runtime": dict(r["runtime_counts"]),
        "decisions": dict(r["decision_counts"]),
        "balance": {
            "drift_step_mean": (sum(r["drift_steps"]) / len(r["drift_steps"])) if r["drift_steps"] else None,
            "drift_step_stdev": stdev(r["drift_steps"]) if r["drift_steps"] else None,
            "hardblock_frac": (r["hardblock"] / n) if n else None,
            "tool_diversity": len(r["tool_names"]),
            "iaa_subset": r["iaa"],
        },
        "anomalies": [{"id": sid, "reason": why} for sid, why in r["unknown"]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Coverage-gap report vs the 1000-sample plan.")
    default_root = Path(__file__).resolve().parent.parent / "v0.1" / "samples"
    ap.add_argument("path", nargs="?", type=Path, default=default_root,
                    help=f"samples dir or file (default: {default_root})")
    ap.add_argument("--json", default="", help="optional path to dump the machine report")
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"error: path not found: {args.path}", file=sys.stderr)
        return 2
    samples, skipped = load_samples(args.path)
    for fp, why in skipped:
        print(f"[coverage_gap] skip {fp}: {why}", file=sys.stderr)
    if not samples:
        print(f"error: no art-*.json samples under {args.path}", file=sys.stderr)
        return 1

    r = build_report(samples)
    print_report(r)
    if args.json:
        Path(args.json).write_text(json.dumps(to_json(r), indent=2), encoding="utf-8")
        print(f"[coverage_gap] machine report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
