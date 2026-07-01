#!/usr/bin/env python3
"""
eval_llm_monitor.py — score ANY LLM as a trajectory monitor on the benchmark.

The model is prompted (per sample) to emit a CONTINUE/WARN/KILL_SESSION decision for
every trajectory step; we take its first fire and score with the SAME tested metric
as eval.py (`compute_metrics`, catch-at-drift corrected). One command, any model:

    pip install litellm datasets
    export OPENAI_API_KEY=...           # or ANTHROPIC_API_KEY, etc.
    python harness/eval_llm_monitor.py --model gpt-4o-mini
    python harness/eval_llm_monitor.py --model anthropic/claude-sonnet-5 --workers 8
    python harness/eval_llm_monitor.py --model ollama/llama3.1 --limit 50

Prints a ready-to-paste leaderboard row and (optionally) a results.json entry you can
PR to the leaderboard Space.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so `from eval import ...` works
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # monitor_prompt

from eval import load_samples, SAMPLES_DIR, SampleResult, compute_metrics  # noqa: E402
from monitor_prompt import build_monitor_prompt, parse_decisions, steps_from_sample  # noqa: E402


def call_model(model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    import litellm
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=4,                           # backoff on transient 429/5xx
    )
    return resp.choices[0].message.content or ""


def score_one(model: str, sample: dict, max_tokens: int, temperature: float):
    """Returns (SampleResult, error_or_None). A hard failure is surfaced as an error
    (NOT silently scored as a no-fire) so rate-limit drops can't inflate the score."""
    steps = steps_from_sample(sample)
    gt = sample["ground_truth"]
    err = None
    try:
        text = call_model(model, build_monitor_prompt(
            sample["declared_goal"], sample.get("declared_plan", []), steps), max_tokens, temperature)
        decisions, fire = parse_decisions(text, len(steps))
    except Exception as e:                       # a failed call = the monitor said nothing
        err = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"  [WARN] {sample['id']}: {err}", file=sys.stderr)
        decisions, fire = ["CONTINUE"] * len(steps), None
    return SampleResult(
        sample_id=sample["id"],
        category=sample["category"],
        difficulty=sample.get("difficulty", "medium"),
        expected_drift_step=gt["drift_step"],
        expected_decision=gt["expected_decision_at_drift"],
        predicted_kill_step=fire,
        predicted_decisions=decisions,
        expected_signals=gt.get("drift_signals_expected", []) or [],
    ), err


def main() -> None:
    ap = argparse.ArgumentParser(description="Score an LLM as a trajectory monitor.")
    ap.add_argument("--model", required=True, help="litellm model id (e.g. gpt-4o-mini, anthropic/claude-sonnet-5, ollama/llama3.1)")
    ap.add_argument("--tier", default="gold", choices=["gold", "augmented", "all"])
    ap.add_argument("--limit", type=int, default=0, help="score only the first N samples")
    ap.add_argument("--workers", type=int, default=8, help="concurrent model calls")
    ap.add_argument("--kill-window", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--samples-dir", default=None)
    ap.add_argument("--output", default=None, help="write full results JSON here")
    args = ap.parse_args()

    samples_dir = pathlib.Path(args.samples_dir) if args.samples_dir else SAMPLES_DIR
    samples = load_samples(samples_dir, tier=args.tier)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print(f"ERROR: no samples found in {samples_dir} (tier={args.tier})", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {args.model} on {len(samples)} samples (tier={args.tier}) with {args.workers} workers...")
    t0 = time.perf_counter()
    results: list[SampleResult] = [None] * len(samples)  # type: ignore
    errs: list[tuple[str, str]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(score_one, args.model, s, args.max_tokens, args.temperature): i
                for i, s in enumerate(samples)}
        done = 0
        for fut in cf.as_completed(futs):
            i = futs[fut]
            results[i], err = fut.result()
            if err:
                errs.append((samples[i]["id"], err))
            done += 1
            if done % 25 == 0 or done == len(samples):
                print(f"  {done}/{len(samples)}  ({len(errs)} errors)")
    elapsed = time.perf_counter() - t0

    n_err = len(errs)
    if n_err:
        print(f"\n[WARN] {n_err}/{len(samples)} samples errored (e.g. {errs[0][1]}).")
        if n_err / len(samples) > 0.02:
            print("[WARN] >2% errored — this row is UNRELIABLE (rate limits / budget?). "
                  "Re-run with fewer --workers before submitting.")

    m = compute_metrics(results, kill_window=args.kill_window)
    print("\n" + "=" * 60)
    print(f"Model         : {args.model}")
    print(f"Split         : {args.tier}  ({len(samples)} samples, {elapsed:.1f}s)")
    print(f"F1 (catch)    : {m['f1']:.4f}")
    print(f"Precision     : {m['precision']:.4f}")
    print(f"Recall        : {m['recall']:.4f}")
    print(f"KSL           : {m['ksl']}")
    print(f"FP step rate  : {m['fp_step_rate']:.4f}")
    print(f"TP/FP/FN/TN   : {m['tp']} / {m['fp']} / {m['fn']} / {m.get('tn', 0)}")
    print("=" * 60)

    row = {
        "method": f"{args.model} (LLM monitor)", "split": args.tier,
        "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
        "ksl": (f"+{m['ksl']}" if isinstance(m["ksl"], (int, float)) else "—"),
        "fp_step_rate": m["fp_step_rate"], "submitter": "you", "verified": False,
    }
    print("\nLeaderboard row (add to the Space's results.json):")
    print(json.dumps(row))

    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(
            {"model": args.model, "tier": args.tier, "elapsed_sec": round(elapsed, 1),
             "n_samples": len(samples), "n_errors": n_err, "metrics": m, "row": row}, indent=2),
            encoding="utf-8")
        print(f"\nFull results -> {args.output}")


if __name__ == "__main__":
    main()
