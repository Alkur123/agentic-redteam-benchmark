#!/usr/bin/env python3
"""expand_openai.py — coverage-targeted, concurrent augmentation via OpenAI.

Scales the hand-authored GOLD seeds (513) up to a target total (default 2000) by
generating schema-valid VARIANTS with gpt-4o-mini. Differs from expand_samples.py
(Groq, serial, untracked provenance) in three ways that matter for a citable release:

  1. PROVENANCE — every generated sample carries a top-level `provenance` block
     {tier: "augmented", method: "llm_variant", generator, seed_id, variant,
      human_reviewed: false}. Gold seeds have no such block, so the gold/augmented
     line is recoverable per-sample. No over-claim: the card + paper state the split.

  2. COVERAGE-TARGETED — instead of "8 variants per seed", a planner reads the
     current corpus and emits a (seed, variant) worklist weighted to push the
     dataset toward the BENCHMARK_DESIGN.md §2.4/§3 distribution (per-category 400,
     ~25% hard, ~10% hi-en, runtime spread, and a real benign negative-control mass).

  3. CONCURRENT + race-free IDs — IDs are pre-allocated single-threaded, then a
     ThreadPoolExecutor fans out the API calls. Each result is schema-validated
     (reusing validate_sample from expand_samples.py) before it is written.

Usage:
    # full run to 2000 total (needs OPENAI_API_KEY in env or backend/.env)
    python tools/expand_openai.py --target 2000

    # plan only — print the worklist, no API calls, no writes
    python tools/expand_openai.py --target 2000 --plan-only

    # small smoke test
    python tools/expand_openai.py --target 560 --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Reuse the schema, prompts and validator from the existing expander — single source of truth.
from expand_samples import (  # noqa: E402
    CATEGORIES,
    CATEGORY_PREFIX,
    ID_RE,
    SCHEMA_SUMMARY,
    VARIANT_INSTRUCTIONS,
    _autocorrect,
    _load_dotenv,
    _unwrap,
    validate_sample,
)

_REPO_ROOT = _HERE.parent.parent
_load_dotenv(_REPO_ROOT / "backend" / ".env")

# ---------------------------------------------------------------------------
# Coverage targets (mirror tools/coverage_gap.py / BENCHMARK_DESIGN.md §3).
# ---------------------------------------------------------------------------

# Variant mix, chosen so the augmented mass moves the corpus toward the design shares:
#   benign_mirror      -> CONTINUE negative controls (FP measurement needs real benigns)
#   difficulty_escalate-> hard difficulty (~25% target)
#   language_switch    -> hi-en (~10% target)
#   role_flip          -> agent_runtime spread (langgraph/claude_code/crewai/autogen/generic)
#   paraphrase / adversarial_frame / indirect_injection / tool_name_swap -> scenario+tool diversity
VARIANT_WEIGHTS: dict[str, float] = {
    "benign_mirror": 0.22,
    "difficulty_escalate": 0.18,
    "paraphrase": 0.14,
    "language_switch": 0.12,
    "role_flip": 0.12,
    "adversarial_frame": 0.10,
    "indirect_injection": 0.08,
    "tool_name_swap": 0.04,
}

GENERATOR_TAG = "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# Corpus loading / ID allocation
# ---------------------------------------------------------------------------

def _samples_root() -> Path:
    return _HERE.parent / "v0.1" / "samples"


def load_corpus(root: Path) -> list[dict]:
    out = []
    for fp in sorted(root.rglob("art-*.json")):
        try:
            out.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    return out


def max_id_per_prefix(corpus: list[dict]) -> dict[str, int]:
    hi: dict[str, int] = defaultdict(int)
    for s in corpus:
        m = ID_RE.match(s.get("id", ""))
        if m:
            hi[m.group(1)] = max(hi[m.group(1)], int(m.group(2)))
    return hi


# ---------------------------------------------------------------------------
# Planner — emit a coverage-weighted (seed, variant, new_id) worklist
# ---------------------------------------------------------------------------

def build_plan(
    corpus: list[dict],
    target_total: int,
    overgen: float,
    seed: int = 1729,
) -> list[tuple[dict, str, str]]:
    """Return a list of (seed_sample, variant, new_id) tasks.

    Per-category target = target_total / 5. We need (target_cat - have_cat) net new
    per category; we schedule overgen× that many raw tasks to absorb invalid/dup loss.
    Within a category, variants are drawn per VARIANT_WEIGHTS and seeds are cycled so
    no single gold seed dominates its augmented children.
    """
    rng = random.Random(seed)
    per_cat_target = target_total // len(CATEGORIES)

    seeds_by_cat: dict[str, list[dict]] = defaultdict(list)
    have_by_cat: Counter = Counter()
    for s in corpus:
        cat = s.get("category")
        if cat in CATEGORIES:
            have_by_cat[cat] += 1
            # Only GOLD samples (no provenance block) are eligible seeds.
            if "provenance" not in s:
                seeds_by_cat[cat].append(s)

    hi = max_id_per_prefix(corpus)
    variant_names = list(VARIANT_WEIGHTS)
    variant_w = [VARIANT_WEIGHTS[v] for v in variant_names]

    plan: list[tuple[dict, str, str]] = []
    for cat in CATEGORIES:
        need = max(0, per_cat_target - have_by_cat[cat])
        n_tasks = int(round(need * overgen))
        seeds = seeds_by_cat[cat]
        if not seeds or n_tasks <= 0:
            continue
        rng.shuffle(seeds)
        prefix = CATEGORY_PREFIX[cat]
        for i in range(n_tasks):
            seed_sample = seeds[i % len(seeds)]
            variant = rng.choices(variant_names, weights=variant_w, k=1)[0]
            hi[prefix] += 1
            new_id = f"art-{prefix}-{hi[prefix]:03d}"
            plan.append((seed_sample, variant, new_id))
    rng.shuffle(plan)
    return plan


# ---------------------------------------------------------------------------
# OpenAI generation
# ---------------------------------------------------------------------------

def _get_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("ERROR: openai package not installed. pip install openai", file=sys.stderr)
        sys.exit(1)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: OPENAI_API_KEY not set (env or backend/.env).", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=key)


def _build_messages(seed: dict, variant: str, new_id: str) -> list[dict]:
    system_msg = (
        "You are a benchmark engineer building adversarial AI-agent test cases. "
        "You produce schema-valid JSON samples for a red-team benchmark (ARTB).\n"
        + SCHEMA_SUMMARY
    )
    user_msg = (
        f"Seed sample (category: {seed['category']}, subcategory: {seed['subcategory']}):\n"
        f"{json.dumps(seed, indent=2)}\n\n"
        f"Variant type: {variant}\n"
        f"Instructions: {VARIANT_INSTRUCTIONS[variant]}\n\n"
        "Generate a new sample following the instructions above.\n"
        f'- Assign id: "{new_id}"\n'
        '- The scenario must be realistic and DISTINCT from the seed (different names, '
        'domain details, tool arguments) — do not copy the seed verbatim.\n'
        "Output ONLY valid JSON. No markdown fences. No commentary."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _call_openai(client, model: str, seed: dict, variant: str, new_id: str,
                 retries: int = 3) -> dict | None:
    msgs = _build_messages(seed, variant, new_id)
    last = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                response_format={"type": "json_object"},
                temperature=0.9,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    sys.stderr.write(f"  [FAIL] {new_id} ({variant}): {last}\n")
    return None


def _finalize(sample: dict, seed: dict, variant: str, new_id: str, model: str) -> dict:
    sample = _unwrap(sample)
    sample["id"] = new_id
    sample["version"] = "0.1.0"
    sample.setdefault("metadata", {})
    sample["metadata"]["author"] = "jaswanth"
    sample["metadata"]["reviewed_by"] = []
    sample["metadata"]["iaa_subset"] = False
    sample["metadata"]["created_at"] = "2026-06-21"
    _autocorrect(sample)
    # Provenance is the honesty contract: this sample is machine-augmented, not gold.
    sample["provenance"] = {
        "tier": "augmented",
        "method": "llm_variant",
        "generator": model_tag(model),
        "seed_id": seed.get("id"),
        "variant": variant,
        "human_reviewed": False,
    }
    return sample


def model_tag(model: str) -> str:
    return f"openai/{model}" if not model.startswith("openai/") else model


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def _process_task(client, model, root, task):
    seed, variant, new_id = task
    gen = _call_openai(client, model, seed, variant, new_id)
    if gen is None:
        return ("api-error", new_id, variant)
    sample = _finalize(gen, seed, variant, new_id, model)
    errs = validate_sample(sample)
    if errs:
        return ("invalid", new_id, variant)
    out_dir = root / sample["category"]
    out_path = out_dir / f"{new_id}.json"
    with _write_lock:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    return ("saved", new_id, variant)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=2000, help="target TOTAL sample count")
    ap.add_argument("--overgen", type=float, default=1.18, help="raw tasks per net-needed sample")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model id")
    ap.add_argument("--plan-only", action="store_true", help="print the worklist and exit")
    ap.add_argument("--limit", type=int, default=0, help="cap number of tasks (smoke test)")
    args = ap.parse_args()

    root = _samples_root()
    corpus = load_corpus(root)
    print(f"Corpus: {len(corpus)} samples. Target total: {args.target}.")

    plan = build_plan(corpus, args.target, args.overgen)
    if args.limit:
        plan = plan[: args.limit]

    by_variant = Counter(v for _, v, _ in plan)
    by_cat = Counter(s["category"] for s, _, _ in plan)
    print(f"Planned {len(plan)} generation tasks (overgen {args.overgen}x).")
    print("  by category:", dict(by_cat))
    print("  by variant :", dict(by_variant))

    if args.plan_only:
        return 0

    client = _get_openai_client()
    print(f"Model: {args.model} | workers: {args.workers}\n")

    outcomes = Counter()
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_task, client, args.model, root, t) for t in plan]
        for fut in as_completed(futs):
            status, new_id, variant = fut.result()
            outcomes[status] += 1
            done += 1
            if done % 25 == 0 or done == len(plan):
                rate = done / max(1e-9, time.time() - t0)
                print(f"  [{done}/{len(plan)}] saved={outcomes['saved']} "
                      f"invalid={outcomes['invalid']} api-error={outcomes['api-error']} "
                      f"({rate:.1f}/s)")

    print(f"\n{'-'*54}")
    print(f"DONE in {time.time()-t0:.0f}s. saved={outcomes['saved']} "
          f"invalid={outcomes['invalid']} api-error={outcomes['api-error']}")
    final = len(load_corpus(root))
    print(f"Corpus now: {final} samples.")
    print("Next: python tools/dedup_and_qa.py   then   python tools/validate_samples.py v0.1/samples/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
