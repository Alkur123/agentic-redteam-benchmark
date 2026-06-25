#!/usr/bin/env python3
"""push_to_hf.py — build the HF dataset payload and publish it.

Builds two parquet splits (gold + augmented) from v0.1/samples, assembles the
card + license + citation + schema + raw JSON tree into a staging folder, and
(optionally) uploads to the Hugging Face Hub as a dataset repo.

Uses huggingface_hub + pyarrow only (no `datasets` dependency).

Local build / validation (no network, no token):
    python tools/push_to_hf.py --dry-run

Publish (needs `huggingface-cli login` first, or HF_TOKEN in env):
    python tools/push_to_hf.py --repo-id aegis-ai/agentic-redteam-benchmark

The dataset card front-matter declares the gold/augmented splits, so:
    load_dataset("aegis-ai/agentic-redteam-benchmark", split="gold")
    load_dataset("aegis-ai/agentic-redteam-benchmark", split="augmented")
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent
_SAMPLES = _BENCH / "v0.1" / "samples"


def _rows(samples_root: Path):
    gold, aug = [], []
    for fp in sorted(samples_root.rglob("art-*.json")):
        try:
            s = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        prov = s.get("provenance") or {}
        tier = prov.get("tier", "gold")
        gt = s.get("ground_truth", {}) or {}
        row = {
            "id": s.get("id"),
            "category": s.get("category"),
            "subcategory": s.get("subcategory"),
            "language": s.get("language"),
            "difficulty": s.get("difficulty"),
            "declared_goal": s.get("declared_goal"),
            "declared_plan": list(s.get("declared_plan", []) or []),
            "agent_runtime": s.get("agent_runtime"),
            # each trajectory step JSON-encoded -> sequence<string> (matches card schema)
            "trajectory": [json.dumps(st, ensure_ascii=False) for st in s.get("trajectory", []) or []],
            "ground_truth": json.dumps(gt, ensure_ascii=False),
            "metadata": json.dumps(s.get("metadata", {}), ensure_ascii=False),
            "tier": tier,
            "provenance": json.dumps(prov, ensure_ascii=False),
            "seed_id": prov.get("seed_id"),
            "variant": prov.get("variant"),
            "drift_step": gt.get("drift_step"),
            "expected_decision": gt.get("expected_decision_at_drift"),
        }
        (aug if tier == "augmented" else gold).append(row)
    return gold, aug


def _write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        # still emit an empty, schema'd file so the split exists
        rows = []
    cols = ["id", "category", "subcategory", "language", "difficulty", "declared_goal",
            "declared_plan", "agent_runtime", "trajectory", "ground_truth", "metadata",
            "tier", "provenance", "seed_id", "variant", "drift_step", "expected_decision"]
    table = pa.table({c: [r.get(c) for r in rows] for c in cols})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def build(stage: Path) -> tuple[int, int]:
    gold, aug = _rows(_SAMPLES)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    _write_parquet(gold, stage / "data" / "gold.parquet")
    _write_parquet(aug, stage / "data" / "augmented.parquet")

    # Card -> README.md (the HF dataset card).
    card = _BENCH / "HF_DATASET_CARD.md"
    shutil.copy(card, stage / "README.md")

    # Companion docs / licenses / citation.
    for name in ("CITATION.cff", "LICENSE", "LICENSE-DATA", "LICENSE-CODE",
                 "SCHEMA.md", "LEADERBOARD.md"):
        src = _BENCH / name
        if src.exists():
            shutil.copy(src, stage / name)

    # Raw JSON tree (so consumers can use the native multi-file form too).
    shutil.copytree(_SAMPLES, stage / "v0.1" / "samples")
    return len(gold), len(aug)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="aegis-ai/agentic-redteam-benchmark")
    ap.add_argument("--stage", type=Path, default=_BENCH / "build" / "hf")
    ap.add_argument("--dry-run", action="store_true", help="build locally, do not upload")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    args = ap.parse_args()

    n_gold, n_aug = build(args.stage)
    print(f"Built staging at {args.stage}")
    print(f"  gold split:      {n_gold}")
    print(f"  augmented split: {n_aug}")
    print(f"  total:           {n_gold + n_aug}")
    print(f"  parquet: data/gold.parquet, data/augmented.parquet")
    print(f"  card: README.md  (+ CITATION.cff, LICENSE*, SCHEMA.md, LEADERBOARD.md, v0.1/samples/)")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded. Inspect the staging folder above.")
        print(f"Publish for real:\n  huggingface-cli login\n  python tools/push_to_hf.py --repo-id {args.repo_id}")
        return 0

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 1

    print(f"\nCreating/locating dataset repo: {args.repo_id}")
    create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    api = HfApi()
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.stage),
        commit_message=f"Publish agentic-redteam-benchmark: {n_gold} gold + {n_aug} augmented",
    )
    print(f"\nDONE. https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
