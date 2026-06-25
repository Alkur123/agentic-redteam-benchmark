# Sample schema — v0.1

Every sample is a single JSON file under `v0.1/samples/<category>/`, named `art-<cat-prefix>-NNN.json`.

Category prefixes: `tca` (tool_call_abuse), `gs` (goal_shift), `ped` (plan_exec_divergence), `ac` (agent_coercion), `cl` (capability_laundering).

---

## Top-level fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string | yes | Globally unique, format `art-<prefix>-NNN`. Lowercase. |
| `version` | string | yes | Semver of the schema (currently `0.1.0`). |
| `category` | enum | yes | One of `tool_call_abuse`, `goal_shift`, `plan_exec_divergence`, `agent_coercion`, `capability_laundering`. |
| `subcategory` | string | yes | One of the subcategories listed in [README.md](README.md). |
| `language` | string | yes | ISO 639-1 code. `en`, `hi-en` (Hinglish), `hi` for v0.1. |
| `difficulty` | enum | yes | `easy`, `medium`, `hard`. |
| `declared_goal` | string | yes | The user-stated goal, one sentence. |
| `declared_plan` | string[] | yes | Step list as the agent would announce it. 3–7 entries. |
| `agent_runtime` | string | yes | e.g. `langgraph_v0.2`, `claude_code_v1`, `crewai_v0.3`, `generic`. |
| `trajectory` | object[] | yes | Ordered list of step objects (see below). Length 6–12. |
| `ground_truth` | object | yes | See below. |
| `metadata` | object | yes | See below. |

---

## `trajectory[i]` step object

| Field | Type | Required | Notes |
|---|---|---|---|
| `index` | int | yes | Zero-based. Must equal array position. |
| `thought` | string | yes | What the agent "thinks" before acting. 1–2 sentences. |
| `action` | object | yes | See below. |
| `observation` | string | yes | What the tool / environment returned. Can be truncated with `...`. |

### `action` object

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | enum | yes | `tool_call`, `message`, `final_answer`. |
| `name` | string | yes | Tool name, e.g. `fs.read`, `http.get`, `db.query`. For `message`, the recipient agent id. |
| `args` | object | yes | Free-form. For tool calls, keys are parameter names. |
| `class` | enum | yes | One of `READ`, `WRITE`, `NETWORK`, `COMPUTE`, `COMMUNICATE`, `SYSTEM`. |
| `raw` | string | yes | Human-readable single-line rendering of the call (used by S3 surprise scoring). |

---

## `ground_truth` object

| Field | Type | Required | Notes |
|---|---|---|---|
| `drift_step` | int | yes | Index of the first step where the trajectory diverges from the declared goal. Must satisfy `4 ≤ drift_step ≤ len(trajectory) − 1`. |
| `expected_decision_at_drift` | enum | yes | `KILL_SESSION`, `WARN`, `CONTINUE`. For attack samples it's always `KILL_SESSION` or `WARN`. |
| `drift_signals_expected` | string[] | yes | Subset of `["s1_cosine", "s2_class", "s3_surprise", "s4_plan_match", "s5_paraphrase", "hard_block_pii", "hard_block_capability_laundering"]`. |
| `rationale` | string | yes | One paragraph human-readable explanation. |

---

## `metadata` object

| Field | Type | Required | Notes |
|---|---|---|---|
| `author` | string | yes | GitHub handle. |
| `reviewed_by` | string[] | yes | List of reviewer handles. Empty list for unreviewed. |
| `iaa_subset` | bool | yes | `true` if this sample is in the inter-annotator-agreement scoring subset. |
| `created_at` | string | yes | ISO-8601 date. |

---

## Worked example

See `v0.1/samples/tool_call_abuse/art-tca-001.json` for a complete, schema-valid sample.

---

## Validation

```bash
python tools/validate_samples.py v0.1/samples/
```

Exit code 0 = all valid. Exit code 1 = at least one violation; offending paths printed.

CI runs the validator on every PR — see `.github/workflows/ci.yml`.
