# Example differentiation run (live OpenAI)

A real run against the OpenAI API, captured so reviewers do not have to
spend tokens to see the pipeline work end-to-end. The full journal is
checked in next to this file as `journal.json`.

For a controlled, zero-cost demonstration of the rollback feedback loop
(cross-check → diagnose → re-specialize), see
[`../example_run_rollback/`](../example_run_rollback/).

## Run metadata

| Field | Value |
|---|---|
| Started | 2026-04-17T09:21:55Z |
| Finished | 2026-04-17T09:26:46Z |
| Wall clock | ~4m 51s |
| Domain | `code_quality_analysis` |
| Planning / baseline model | `gpt-4o-mini` |
| Execution / specialized model | `gpt-4o` |
| LLM calls | 42 (22 × gpt-4o-mini, 20 × gpt-4o) |
| Total tokens | 40,909 |
| State transitions | `undifferentiated → sensing → differentiating → validating → specialized` |
| Final state | `SPECIALIZED` (no rollbacks on this particular run) |
| Rollback budget | 3 attempts; 0 consumed |
| Cross-check disagreements logged | 3 |

## Token usage by phase

| Phase | Calls | Tokens |
|---|---:|---:|
| sensing | 1 | 1,218 |
| planning | 1 | 1,546 |
| validation_baseline (20-sample benchmark) | 20 | 17,248 |
| validation_specialized (20-sample benchmark) | 20 | 20,897 |
| **Total** | **42** | **40,909** |

## Benchmark outcome

| Metric | Baseline (undifferentiated) | Specialized | Δ |
|---|---:|---:|---:|
| Precision | 0.000 | 0.667 | +0.667 |
| Recall | 0.000 | 0.933 | +0.933 |
| F1 | 0.000 | 0.778 | +0.778 |
| Specificity | 0.000 | 0.300 | +0.300 |

The baseline score is zero because the undifferentiated prompt does not
ask for structured JSON — the parser falls back to an empty category
set for every sample, so precision and recall are both undefined and
report as 0. That is the signal the differentiation pipeline is there
to fix, and the delta above is what it actually buys you. The specialized
run picks up 14 of 15 buggy samples (recall 0.933) while keeping
false-positive rate manageable (specificity 0.300 — the 5 clean
adversarial samples are the expensive ones).

## Phase-by-phase narrative

Taken straight from `journal.json`, not a summary:

1. **Sensing** (1 LLM call, 1,218 tokens). The agent asked `gpt-4o-mini`
   to name the domain and produced 5 review strategies, a 5-category
   taxonomy, 5 tool categories, and 5 expert insights. Captured as a
   `PHASE_RESULT` event.
2. **Capability generation** (post-sensing, pre-planning). The agent
   proposed one novel capability on top of the six hand-authored ones.
   On this run the sandbox admitted it (`ast_scan` clean, subprocess
   smoke-test passed) but it did not survive prompt composition because
   the planning phase did not select it — the A/B is therefore against
   the registered capability set, not a drifted one.
3. **Planning** (1 LLM call, 1,546 tokens). Selected 6 capabilities
   (`structural_analysis`, `logic_correctness`, `security_analysis`,
   `performance_analysis`, `style_consistency`, `severity_ranking`) and
   5 review passes with an F1 threshold of 0.6 and minimum precision /
   recall of 0.5.
4. **Specialization**. Composed a 3,178-character system prompt from
   the capability fragments plus domain insights. 12 `CAPABILITY_ADDED`
   events logged (each capability is recorded both when planning picks
   it and when specialization wires it into the pipeline — the audit
   trail has both sides of the decision).
5. **Validation**. 40 reviews, 20 baseline + 20 specialized. Every
   review is an `LLM_CALL` event with model, prompt hash, and token
   count. Metrics logged as two `METRIC_MEASUREMENT` events (baseline
   and specialized). Comparison logged as a `PHASE_RESULT`.
6. **Cross-check**. Three disagreements surfaced (see below) — logged
   as `DECISION` events. F1 passed both guards, so graduation to
   `SPECIALIZED` succeeded without consuming rollback budget.

## What the cross-check caught

After the specialized benchmark run, `ValidationPhase.cross_check_verdicts`
replayed each verdict through the deterministic static checks and logged
three disagreements as `DECISION` events. Exactly the signals the
rollback diagnoser would have consumed if F1 had fallen short:

| Sample | Disagreement | Evidence |
|---|---|---|
| `smell_03` | LLM flagged `structure` but AST metrics look clean | `max_fn_length=18`, `max_nesting_depth=1` — under the 20-line / depth-2 heuristics |
| `clean_03` | LLM flagged `structure` but AST metrics look clean | `max_fn_length=14`, `max_nesting_depth=2` — short and shallow |
| `clean_03` | Pattern scanner caught a security issue the LLM missed | `Use of eval() — potential code injection` (the sample uses `eval()` safely with `__builtins__={}`, so this is a regex false positive the LLM correctly ignored) |

These are the kinds of discrepancies an agent monitoring itself should
surface rather than paper over. On this particular run F1 still met the
graduation guard, so the disagreements are informational rather than
actionable. The companion rollback demo shows what happens when the
diagnoser does fire.

## How to reproduce

```bash
export OPENAI_API_KEY=sk-…
.venv/bin/stem-agent differentiate --domain code_quality_analysis
```

Roughly twenty cents per run at current OpenAI pricing. The journal
lands under `evolution_journals/` by default and can be re-pretty-printed
with `.venv/bin/stem-agent journal --last`.
