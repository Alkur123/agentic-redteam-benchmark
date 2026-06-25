# tools/

Authoring + measurement tooling for the benchmark. All scripts are **Python standard library only**
(no venv needed) unless noted, and mirror each other's CLI conventions.

| Tool | Deps | What it does |
|---|---|---|
| `validate_samples.py` | stdlib | Schema validator. The CI gate — exits non-zero on any invalid sample. |
| `coverage_gap.py` | stdlib | Coverage vs the v1.0 (1000-sample) plan: the per-subcategory **to-author worklist** + balance flags. |
| `iaa_kappa.py` | stdlib | Inter-annotator agreement (Cohen's κ) + the reviewer-B blind-template emitter. |
| `expand_samples.py` | **GROQ_API_KEY** | LLM seed-expansion (Phase B): 8 variant types per seed. |
| `phase_b_prompt.txt` | — | The canonical Phase-B generation prompt (committed for reproducibility). |

The baseline evaluation harness is `../eval.py` (random / cosine / gpt4 / ring12).

---

## validate_samples.py — schema gate

```bash
python tools/validate_samples.py v0.1/samples/            # validate all
python tools/validate_samples.py --strict v0.1/samples/   # stop on first error (CI mode)
```

Exit 0 = every sample valid. Checks ids, category/subcategory vocab, trajectory length (6–12),
`4 ≤ drift_step ≤ len−1`, action classes, signal vocab, and metadata fields. See `../SCHEMA.md`.

---

## coverage_gap.py — what's authored vs the 1000-plan

```bash
python tools/coverage_gap.py                    # report against ../v0.1/samples
python tools/coverage_gap.py v0.1/samples/ --json coverage.json
```

Reads every sample and reports, against the `../BENCHMARK_DESIGN.md` targets (§2.4 category/subcategory,
§3.2 difficulty, §3.3 language, §3.4 runtime, §5 balance), exactly how many more samples each bucket
needs — the **manual Phase-B worklist** — plus balance metrics (drift-step mean/stdev, hard-block
fraction, tool diversity, IAA-subset size) and any off-plan category/subcategory/runtime values. Always
exits 0 on a readable tree (it is a report, not a gate). `--json` dumps a machine report.

---

## iaa_kappa.py — inter-annotator agreement (the legitimacy number)

The §4.3 protocol in two steps:

```bash
# 1. emit the blind review template from the iaa_subset=true samples (no ground_truth/category leak):
python tools/iaa_kappa.py --emit-blind --out blind_review_B.json
#    -> hand blind_review_B.json to the second reviewer; they fill each `review` block.

# 2. compute Cohen's kappa of reviewer B vs each sample's authored ground_truth (reviewer A):
python tools/iaa_kappa.py --compute --reviewer-b blind_review_B.json --json iaa.json
```

Reports drift-step κ (tolerance `--tol`, default ±1), per-signal + macro signal κ, mean Jaccard,
decision κ, and a per-subcategory drift-κ breakdown when a gate fails — against the §4.3 gates
(κ_drift ≥ 0.70, κ_signals ≥ 0.60). It also detects the **high-agreement/low-κ paradox** (when the
drift-step distribution is too peaked, p_o ≈ p_e → κ ≈ 0 despite high raw agreement) and tells you to
flatten the drift-step distribution rather than chase the metric. `--selftest` proves the κ math on
hand-computed cases (runs in CI).

---

## expand_samples.py — Phase-B LLM expansion

```bash
python tools/expand_samples.py v0.1/samples/tool_call_abuse/art-tca-001.json   # one seed, 8 variants
python tools/expand_samples.py v0.1/samples/ --variants paraphrase,language_switch
python tools/expand_samples.py v0.1/samples/ --dry-run
```

Requires `GROQ_API_KEY` (env or `backend/.env`). Generated variants still pass through human review +
`validate_samples.py` before they join the scored set (see `../AUTHORING_GUIDE.md`).
