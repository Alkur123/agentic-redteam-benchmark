"""Validate v0.1 samples against the schema in SCHEMA.md.

Exits 0 if every JSON file under the target path validates. Exits 1 on the
first invalid file in CI mode; collects all errors in non-CI mode.

Usage:
    python tools/validate_samples.py v0.1/samples/
    python tools/validate_samples.py v0.1/samples/tool_call_abuse/
    python tools/validate_samples.py --strict v0.1/samples/   # stop on first error

Implemented in pure stdlib so CI does not need a venv.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CATEGORIES = {
    "tool_call_abuse": {"filesystem_exfil", "network_exfil", "credential_read", "sudo_escalation"},
    "goal_shift": {"topic_pivot", "identity_swap", "persona_assumption", "helpfulness_override"},
    "plan_exec_divergence": {"step_skip", "step_inject", "step_reorder", "plan_abandonment"},
    "agent_coercion": {"indirect_prompt", "output_poisoning", "role_confusion", "authority_spoofing"},
    "capability_laundering": {"benign_tool_misuse", "multi_step_concealment", "scope_creep", "side_channel"},
}

CATEGORY_PREFIX = {
    "tool_call_abuse": "tca",
    "goal_shift": "gs",
    "plan_exec_divergence": "ped",
    "agent_coercion": "ac",
    "capability_laundering": "cl",
}

ACTION_CLASSES = {"READ", "WRITE", "NETWORK", "COMPUTE", "COMMUNICATE", "SYSTEM"}
ACTION_TYPES = {"tool_call", "message", "final_answer"}
DIFFICULTY = {"easy", "medium", "hard"}
DECISIONS = {"KILL_SESSION", "WARN", "CONTINUE"}
SIGNALS = {
    "s1_cosine", "s2_class", "s3_surprise", "s4_plan_match", "s5_paraphrase",
    "hard_block_pii", "hard_block_capability_laundering",
}

ID_RE = re.compile(r"^art-(tca|gs|ped|ac|cl)-\d{3,4}$")


def _check(cond: bool, msg: str, errs: list[str]) -> None:
    if not cond:
        errs.append(msg)


def validate_sample(path: Path) -> list[str]:
    errs: list[str] = []
    try:
        sample = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]

    for field in [
        "id", "version", "category", "subcategory", "language", "difficulty",
        "declared_goal", "declared_plan", "agent_runtime", "trajectory",
        "ground_truth", "metadata",
    ]:
        _check(field in sample, f"missing field: {field}", errs)
    if errs:
        return errs

    _check(bool(ID_RE.match(sample["id"])), f"id must match art-<prefix>-NNN, got {sample['id']!r}", errs)
    _check(sample["version"] == "0.1.0", f"version must be '0.1.0', got {sample['version']!r}", errs)

    cat = sample["category"]
    _check(cat in CATEGORIES, f"unknown category {cat!r}", errs)
    if cat in CATEGORIES:
        _check(
            sample["subcategory"] in CATEGORIES[cat],
            f"subcategory {sample['subcategory']!r} not in {sorted(CATEGORIES[cat])} for {cat}",
            errs,
        )
        prefix = CATEGORY_PREFIX[cat]
        _check(
            sample["id"].startswith(f"art-{prefix}-"),
            f"id prefix mismatch: expected art-{prefix}-, got {sample['id']!r}",
            errs,
        )

    _check(sample["difficulty"] in DIFFICULTY, f"difficulty must be one of {DIFFICULTY}", errs)
    _check(isinstance(sample["declared_plan"], list) and 3 <= len(sample["declared_plan"]) <= 7,
           "declared_plan must be a 3-7 item list", errs)

    traj = sample["trajectory"]
    _check(isinstance(traj, list) and 6 <= len(traj) <= 12,
           f"trajectory must be a 6-12 step list, got {len(traj) if isinstance(traj, list) else type(traj).__name__}", errs)

    if isinstance(traj, list):
        for i, step in enumerate(traj):
            ctx = f"trajectory[{i}]"
            for f in ("index", "thought", "action", "observation"):
                _check(f in step, f"{ctx}: missing {f}", errs)
            _check(step.get("index") == i, f"{ctx}: index mismatch (got {step.get('index')})", errs)
            act = step.get("action", {})
            for f in ("type", "name", "args", "class", "raw"):
                _check(f in act, f"{ctx}.action: missing {f}", errs)
            _check(act.get("type") in ACTION_TYPES, f"{ctx}.action.type invalid: {act.get('type')!r}", errs)
            _check(act.get("class") in ACTION_CLASSES, f"{ctx}.action.class invalid: {act.get('class')!r}", errs)
            _check(isinstance(act.get("args"), dict), f"{ctx}.action.args must be object", errs)

    gt = sample["ground_truth"]
    for f in ("drift_step", "expected_decision_at_drift", "drift_signals_expected", "rationale"):
        _check(f in gt, f"ground_truth: missing {f}", errs)
    drift = gt.get("drift_step")
    if isinstance(traj, list) and isinstance(drift, int):
        _check(4 <= drift <= len(traj) - 1, f"drift_step {drift} must be in [4, {len(traj)-1}]", errs)
    _check(gt.get("expected_decision_at_drift") in DECISIONS,
           f"expected_decision_at_drift must be one of {DECISIONS}", errs)
    signals = gt.get("drift_signals_expected", [])
    _check(isinstance(signals, list) and len(signals) >= 1,
           "drift_signals_expected must be a non-empty list", errs)
    for s in signals or []:
        _check(s in SIGNALS, f"unknown drift signal {s!r}", errs)

    meta = sample["metadata"]
    for f in ("author", "reviewed_by", "iaa_subset", "created_at"):
        _check(f in meta, f"metadata: missing {f}", errs)
    _check(isinstance(meta.get("reviewed_by"), list), "metadata.reviewed_by must be list", errs)
    _check(isinstance(meta.get("iaa_subset"), bool), "metadata.iaa_subset must be bool", errs)

    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="file or directory to validate")
    ap.add_argument("--strict", action="store_true", help="exit on first invalid file")
    args = ap.parse_args()

    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.rglob("*.json"))

    total = len(files)
    bad = 0
    ids: set[str] = set()
    for fp in files:
        errs = validate_sample(fp)
        try:
            sid = json.loads(fp.read_text(encoding="utf-8")).get("id")
            if sid in ids:
                errs.append(f"duplicate id across files: {sid}")
            elif sid:
                ids.add(sid)
        except Exception:
            pass

        if errs:
            bad += 1
            print(f"FAIL {fp}")
            for e in errs:
                print(f"  - {e}")
            if args.strict:
                return 1

    if bad:
        print(f"\n{bad}/{total} samples failed validation.")
        return 1
    print(f"{total}/{total} samples valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
