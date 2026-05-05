# plan.md; 21-Task Audit Closeout

This plan groups the 21 tasks into 4 commit batches in execution order. Two
tasks (13, 11) move out of their nominal batch because dependencies force it;
the moves are flagged inline. Each task lists target files, line anchors, hard
dependencies, and notes on the failure mode if executed out of order.

## Repo invariants observed

- 142 tests in 9 files; ruff clean; `.venv/bin/python` is the interpreter.
- `style_consistency` is registered in `build_default_registry()` but missing
  from `CAPABILITY_FRAGMENTS` (`prompt_library.py:107-113`). Sixth capability
  in writeup is partly cosmetic.
- `core/agent.py:17` imports `JsonStorageAdapter` directly; task 3 target.
- `_guard_improvement` (`state_machine.py:117-125`) uses strict `>` and ignores
  specificity; task 2 target.
- `total_tokens` is on `EvolutionJournal` (line 195); no `token_budget_cap`
  config field yet.
- Cross-check is wired through `phases/validation.py`; static-tool injection is
  wired through `make_llm_review_fn(use_tools=True)`. **No generated-capability
  validators run anywhere at review time**; task 8 fixes that.
- Existing journals at `evolution_journals/journal_*.json` for `core/priors.py`
  to read.

## Commit 1; Truth-up batch (no API spend)

Backdate: `GIT_AUTHOR_DATE`/`GIT_COMMITTER_DATE = 2026-04-20T15:21:24`.

**Re-ordering note.** Task 13 is in the spec's "substantive rebuild" but must
execute first: tasks 4, 9, 14 all consume the partition function. Task 6
(drop logic_04) reshapes the corpus before the partition is computed, so the
sequence is 6 → 13 → everything that touches the partition. I move 13 into
commit 1 and lift 6 to the very front.

### Task 6; Drop `logic_04`
- `src/stem_agent/evaluation/fixtures/code_samples.py`; delete the
  `LOGIC_04_INTEGER_OVERFLOW` sample (lines 77-89), drop it from
  `get_benchmark_corpus()` (line 606).
- `tests/test_integration.py:303-339`; `assert len(corpus) == 20` → `19`,
  `len(buggy) == 15` → `14`.
- `tests/conftest.py:367-380`; the `"average"` substring response becomes dead
  but harmless; can leave for now.
- **Depends on:** none. **Why first:** partition (task 13) is sized to the new
  corpus.

### Task 13; Deterministic 11/4/4 corpus partition
- `code_samples.py`; append `partition(corpus, seed) -> tuple[validation,
  holdout, probe]`. Sort by `sample_id`; assign each via stable hash of
  `(sample_id, seed)`; lowest-11 → validation, next-4 → holdout, last-4 →
  probe. Pairwise disjoint by construction.
- `security_audit_samples.py`; `partition_overlapping(corpus, seed) -> dict`
  with `overlapping=True`. 8 samples can't cleanly do 11/4/4; document the
  overlap.
- `tests/test_partition.py`; new file, ~12 tests. Determinism, disjointness,
  overlap-flag.
- **Depends on:** task 6. **Consumed by:** tasks 4, 9, 14.

### Task 1; JSON-disciplined baseline
- `prompt_library.py:24-26`; append `OUTPUT_FORMAT_FRAGMENT` to
  `UNDIFFERENTIATED_PROMPT`. Don't touch `BASE_REVIEW_PROMPT`.
- `validation.py:106`; verify the new prompt flows into baseline via
  `agent_config.baseline_prompt` (already does via `SpecializedAgentConfig`
  default).
- `tests/test_baseline_prompt.py`; new file, ~4 tests asserting `## Output
  Format` and `is_clean` keys in baseline prompt; baseline parses
  end-to-end via `parse_review_response`.
- `scripts/generate_rollback_demo.py:38-39`; the scripted baseline branch
  must be updated to return JSON instead of free-form text. Patch
  `ScriptedLLM.generate` so the baseline branch returns `_review(None)`.
  The scripted journal numbers will shift; regenerate after task 1 lands.
- **Depends on:** none for code. Affects task 19 (writeup numbers) and task 21
  (example_run README narrative).

### Task 2; δ ≥ 0.05 + specificity guard + token-budget guard
- `state_machine.py:117-125`; replace `_guard_improvement` body. Read
  `specialized_specificity`, `baseline_specificity`; pass only when
  `(specialized_f1 ≥ baseline_f1 + 0.05) and (specialized_specificity ≥
  baseline_specificity − 0.05)`.
- `state_machine.py:75-105`; append `("token_budget_under_cap",
  _guard_token_budget)` to `VALIDATING → SPECIALIZED`. Only fires if
  `ctx["token_budget_cap"]` is not None.
- `core/agent.py:144-147`; extend `guard_context` with
  `specialized_specificity`, `baseline_specificity`, `total_tokens`,
  `token_budget_cap`.
- `phases/validation.py:130-145`; populate
  `context["baseline_specificity"]`, `context["specialized_specificity"]`.
- `core/config.py`; add `token_budget_cap: int | None = None`.
- `tests/test_state_machine.py`; new tests for delta boundary, specificity
  drop, token cap.
- `tests/test_state_machine_properties.py:23-30`; append the new context
  keys to `_PERMISSIVE_CONTEXT`.
- **Depends on:** task 1 (so baseline isn't a parser-zero making delta trivial).

### Task 3; Move adapter import out of `core/`
- `core/agent.py:17`; delete the `JsonStorageAdapter` import.
- `core/agent.py:49-71`; `__init__` accepts `storage: StoragePort`.
- `cli.py:48-49`, `109-111`; every `StemAgent(...)` callsite injects
  `storage = JsonStorageAdapter(config.journal_dir)`.
- `tests/conftest.py`; add `InMemoryStorage` fixture so tests don't need the
  filesystem.
- `tests/test_architecture.py`; new file. Walks `ast.Import`/`ast.ImportFrom`
  over every `.py` in `src/stem_agent/core/` and `src/stem_agent/phases/`.
  Asserts no module name starts with `stem_agent.adapters`.
- **Depends on:** none. **Why early:** subsequent commits add modules to
  `core/`/`phases/`; easier to enforce the rule before adding code that
  has to satisfy it.

### Task 4; `--seeds`, `--store-prompts`, `--max-rollbacks`; `replay` command
- `cli.py:26-56`; extend `differentiate` with `--seeds N`,
  `--store-prompts`, `--max-rollbacks`. When `--seeds > 1`, build a fresh
  `StemAgent` per iteration with `corpus = partition(...)[0]`, save journal
  as `docs/example_run/seeds/journal_seed{i}.json`.
- `core/config.py`; add `seed: int = 0`.
- `adapters/openai_adapter.py:115-152`; pass `seed=self._config.seed` and
  `temperature=0` to chat completions.
- `cli.py`; new `@app.command()` `replay(prompt_hash: str, prompts_dir: str)`.
- The `prompt_hash` storage filename uses the same 16-char prefix as
  `EvolutionJournal.hash_prompt`.
- `tests/test_cli.py`; new file, ~10 tests with `typer.testing.CliRunner`
  against a stub `StemAgent`.
- **Depends on:** task 13 (partition), task 3 (storage injection).

### Task 5; `make eval-ablation` target
- `Makefile`; add `eval-ablation` target.
- `cli.py`; new `eval_ablation` command.
- `evaluation/ablation.py`; new module, ~120 lines. Function
  `run_ablation(...)` returns `{(arm, domain): metrics}`.
- `tests/test_ablation.py`; new file, ~5 tests including wall-clock
  assertion.
- **Depends on:** task 13 (partition), task 8 (uses dispatcher in
  with-capgen arm). **Move into commit 2** so dispatcher is available.

### Task 7; Test count target ≥256 with wall time <10s
- Bookkeeping target. By the end of commit 1, count ≈ 142 + (12 partition + 4
  architecture + 10 cli + 6 guard + 4 baseline-prompt) = **178**. Commit 2
  brings it past 256.

## Commit 2; Substantive rebuild (no API spend)

Backdate: `2026-04-20T18:26:38`.

### Task 8; `ReviewDispatcher`
- `capabilities/dispatcher.py`; new module, ~140 lines.
  - `@dataclass(frozen=True) GeneratedCheckFinding(name, description, hits)`.
  - `ReviewDispatcher(registry, *, timeout_per_check=0.5)`. Compiles each
    `validator_code` once via `compile(...)` and execs into a fresh dict.
  - `run(self, code: str) -> list[GeneratedCheckFinding]`. Sets
    `signal.SIGALRM` for `timeout_per_check`. Try/except for `Exception,
    TimeoutError`; quarantines on bad return type/exception/timeout.
- `evaluation/benchmark.py:234-292`; `make_llm_review_fn` accepts
  `dispatcher: ReviewDispatcher | None = None`. Tool block becomes
  `format_tool_findings(...) + format_dispatcher_findings(dispatcher.run(code))`.
- `core/agent.py:201-217`; `review` builds a `ReviewDispatcher(self._registry)`
  and folds its output into `tool_block`.
- `phases/validation.py:78-95`; only specialized arm gets the dispatcher;
  baseline stays untooled.
- `tests/test_dispatcher.py`; new file, ~12 tests.
- **Depends on:** tasks 1, 3.

### Task 5 (deferred from commit 1)
Moved here because it uses task 8's dispatcher.

### Task 9; Empirical-holdout gate
- `phases/capability_generation.py:179-209`; between sandbox pass and
  `registry.register(capability)`, run a 4-sample A/B against the holdout
  slice. Admit only if with-proposal arm is strictly better.
- When `partition.overlapping == True` (security domain), skip the empirical
  gate and log a `DECISION` event.
- `core/agent.py`; plumb partition into context after sensing.
- `tests/test_capability_generation.py`; extend with 4 new tests.
- **Depends on:** tasks 8, 13, 6.

### Task 10; Probe-grounded sensing
- `phases/sensing.py:45-77`; replace `SENSING_PROMPT_TEMPLATE` with one that
  embeds 4 probe sample bodies (truncated to 600 chars each) and per-sample
  static-tool summaries.
- `phases/sensing.py:19-43`; `DomainKnowledge` adds `tool_fit: dict[str,
  int]`.
- `phases/planning.py:54-89`; `PLANNING_PROMPT_TEMPLATE` adds `tool fit
  hits: {tool_fit}` line; phase code biases capability ordering.
- `tests/conftest.py`; `_make_sensing_response` adds `tool_fit` key.
- `tests/test_phases.py`; extend with 5 new tests.
- **Depends on:** task 13, task 9.

### Task 11; Cross-run priors
- `core/priors.py`; new module. `weight_capabilities(domain: str, storage:
  StoragePort) -> dict[str, float]`. Loads journals via storage, filters by
  domain, computes Laplace-smoothed `(success+1)/(selected+2)`.
- `phases/planning.py:117-130`; call `weight_capabilities(domain,
  context["storage"])`; reorder candidates and use as tiebreaker.
- `core/agent.py`; populate `self._context["storage"] = self._storage`.
- `tests/test_priors.py`; new file, ~10 tests.
- **Depends on:** task 3 (StoragePort injection).

### Task 12; K=3 splicing
- `phases/specialization.py:74-100`; read `context["rollback_history"]:
  list[dict]` instead of flat list. Splice 3 most recent in full; older
  collapse to one-line summary.
- `core/agent.py:175-181`; append `{attempt_idx, adjustments, summary}` to
  `context["rollback_history"]`.
- `phases/validation.py:267-328`; add sibling `diagnose_summary`.
- `tests/test_specialization_splicing.py`; new file, ~6 tests including
  bounded-prompt-length over a 6-attempt rollback chain.
- **Depends on:** tasks 2, 3.

## Commit 3; Live evaluation (API spend)

Plain `git commit`. **Confirm OPENAI_API_KEY and ~$25 budget with user
before starting.**

### Task 14; Multi-seed live runs
- 3 seeds × 2 domains. Outputs: `docs/example_run/seeds/cq/{0,1,2}.json` and
  `docs/example_run/seeds/sec/{0,1,2}.json`.
- **Depends on:** tasks 1, 2, 3, 4, 8, 9, 10, 11, 12, 13.

### Task 15; Bootstrap CIs
- `evaluation/bootstrap.py`; new module. `bootstrap_metric`,
  `pool_seed_verdicts`, `headline_table`.
- `cli.py`; new `report --seeds-dir <dir>` command.
- `tests/test_bootstrap.py`; new file, ~12 tests with `n_resamples=50` for
  speed.
- **Depends on:** task 14 for artifacts; code can land before.

### Task 16; Cross-domain 2×2
- `scripts/cross_domain_eval.py`; new script. Pulls specialized prompts
  from `cq/0.json` and `sec/0.json`, runs against the opposite corpus.
- **Depends on:** tasks 14, 8.

### Task 17; Cohen's κ second-rater pass
- `scripts/adjudicate.py`; new script. `gpt-4o-mini` strict label-only.
- `evaluation/fixtures/adjudicator_labels_{cq,sec}.json`; output artifacts.
- `evaluation/kappa.py`; new module. `cohen_kappa`, `kappa_with_ci`.
- `tests/test_kappa.py`; new file, ~6 tests.
- **Depends on:** none (independent of seeds).

### Task 18; Total budget cap $25
- `evaluation/cost.py`; new module. `PRICE_TABLE`, `estimate_cost`,
  `assert_under_cap`.
- `cli.py:differentiate`; pre-iteration spend check.
- `tests/test_cost.py`; new file, ~8 tests.
- **Depends on:** none.

## Commit 4; Writeup update

Plain `git commit`.

### Task 19; Update `docs/writeup.tex`
- Table 1 (lines 113-128): `Logic bugs & 5` → `4`; total `20` → `19`.
- Table 2 (lines 145-161): replace with 3-seed mean ± 95% CI from
  `bootstrap.headline_table` output. Add security-domain rows.
- New subsection; Cross-domain 2×2 from task 16.
- New subsection; Cohen's κ values + disputed counts from task 17.
- New line; "Live spend across all batches" total.
- Line 97; replace existing capability-generation anecdote with whatever
  seed-0 capability_generation actually produced. (See Contradictions §1.)
- Line 172; keep "off-diagonals beat on-diagonals" only if cross-domain
  replicates it; otherwise drop.
- Resolve any `??` cross-references.
- Line 130; update test count to actual final number.
- Re-render: `cd docs && pdflatex writeup.tex && pdflatex writeup.tex`.

### Task 20; Update `README.md`
- Line 11; replace `0.000 → 0.778` with new headline `mean [lo, hi]`.
- Lines 115-119; corpus distribution.
- Line 123; test count.

### Task 21; Update `docs/example_run/README.md`
- Add `docs/example_run/seeds/README.md` summarising all 6 seed runs.
- Shorten `docs/example_run/README.md` to a "single-seed historical reference"
  pointer.

## Critical path

```
6 → 13 → 1 → 2 → 3 → 4 → 8 → 9 → 10 → 11 → 12 → 14 → 15 → 16 → 19 → 20 → 21
```

Parallelizable side paths: task 5 after task 8; task 17 any time before
task 19; task 18 any time before task 14.

## Genuine contradictions / spec gaps

1. **The "code_smell_detection rejected for forbidden import" anecdote does
   not currently exist in `docs/writeup.tex`.** Line 97 currently says the
   proposal "cleared both layers but planning did not select it." Decision:
   replace the existing anecdote with whatever seed-0 produced.

2. **Task 1's empirical claim.** "Fresh baseline run no longer gets F1=0.000"
   is a model-behaviour assertion. Test as `precision > 0 OR true_positives
   > 0`, not `f1 > 0`.

3. **Task 9 + small security corpus.** 8 samples can't accommodate a
   disjoint 4-sample holdout. Decision: skip the empirical gate when
   `partition.overlapping == True` and log a DECISION event.

4. **Task 1 affects scripted rollback demo.** With JSON-disciplined baseline,
   the demo's free-form baseline narrative is wrong. Patch `ScriptedLLM` to
   return JSON for baseline; regenerate journal.

5. **`signal.SIGALRM` is POSIX-only.** macOS dev box and Linux CI are fine;
   document the platform constraint in dispatcher module docstring.

6. **OpenAI `seed` parameter is best-effort.** Document in writeup that
   reproducibility is best-effort, not guaranteed.

7. **Task 4 prompt-hash filename.** `EvolutionJournal.hash_prompt` truncates
   to 16 chars; storage filename uses the same prefix.

## Test-writing strategy to hit ≥256

Current 142 + new tests below = **257** by end of commit 2.

| Module | Tests |
|---|---:|
| test_partition.py | 12 |
| test_architecture.py | 4 |
| test_cli.py | 10 |
| test_ablation.py | 5 |
| test_state_machine.py extensions | 6 |
| test_baseline_prompt.py | 4 |
| test_dispatcher.py | 12 |
| test_capability_generation.py extensions | 4 |
| test_phases.py extensions | 5 |
| test_priors.py | 10 |
| test_specialization_splicing.py | 6 |
| test_bootstrap.py | 12 |
| test_kappa.py | 6 |
| test_cost.py | 8 |
| test_state_machine_properties.py +2 | 2 |
| test_phases.py validation +4 | 4 |
| test_integration.py +5 | 5 |
| **Subtotal** | **115** |

142 + 115 = **257**. Wall-time: Hypothesis at 200 examples ≈ 0.4s; new tests
mostly sub-millisecond pure-function tests + ~10 in-process Typer CLI
invocations (~5–10ms each). Estimated total: ~4.1s. Comfortably under 10s.

## Live-batch spend estimate

| Item | Est. |
|---|---:|
| Task 14: 3 cq @ ~$0.20 + 3 sec @ ~$0.10 | $0.90 |
| Task 16: cross-domain 19 calls @ gpt-4o | $0.40 |
| Task 17: 27 strict-label gpt-4o-mini calls | $0.01 |
| Task 9 holdout pre-checks | absorbed |
| **Total** | **~$1.31** |

Even with prompt growth from probe-grounded sensing and K=3 splicing on
retries, upper bound is ~$1.50. Comfortably under $25.

Wall-clock: cq ~5min × 3 + sec ~2.5min × 3 = ~22.5 min for task 14, plus
~5 min for tasks 16+17 = **~30 min total live**.
