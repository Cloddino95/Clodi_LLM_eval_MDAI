# MDR LLM evaluation refactor: implementation report

Date completed: 2026-08-13

Documentation and legacy-path cleanup: 2026-09-07

## Outcome

Phases 0 through 15 of the supplied implementation plan are complete. The
project now has one YAML-driven Python workflow that is reproducible,
resumable, auditable, and explicit about smoke-test, development, and final
phases. Inference, orchestration, validation, provenance, persistence, and
analysis remain separate responsibilities.

The definitive final-data experiment was not run. The final data files are
deliberately absent, and `configs/final.yaml` deliberately contains null freeze
values so the tracked template fails closed until an independent final-study
review. The actual run will use a populated, Git-ignored
`configs/final_frozen.local.yaml`, whose exact content is recorded in the run
manifest.

## Phase record

| Phase | Implemented result                                                                                                                                                                                                   |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | Audited the original Bash-to-inference-to-analysis flow, duplicated settings, path assumptions, output writes, metadata, and reusable scientific logic.                                                              |
| 1     | Established`src/`, one shared inference implementation, project/config-relative paths, a strict typed YAML contract, and thin CLI entry points.                                                                    |
| 2     | Split inference descriptions from expert labels, made phase declarations mandatory, blocked label columns in inference data, and added contamination checks.                                                         |
| 3     | Added unique run IDs, non-overwriting run directories, manifest state transitions, structured logging, hashes, package/hardware/environment provenance, and secret filtering.                                        |
| 4     | Added append-and-flush JSONL evidence, rebuildable CSV, deterministic generation keys, compatibility-checked resume, and missing-work-only execution.                                                                |
| 5     | Added configuration, dataset, checkpoint/cache, tokenizer, context-window, disk, CUDA/GPU-memory, and built-in generation preflight checks.                                                                          |
| 6     | Preserved raw output while recording parse, schema, repair, truncation, citation, contradiction, and extraneous-text flags separately.                                                                               |
| 7     | Added exact expected/actual generation-grid validation with missing, duplicate, and unexpected key detection.                                                                                                        |
| 8     | Added the Python model-loop orchestrator and isolated one checkpoint per worker process with explicit interruption, model-failure, integrity-failure, and study-failure handling.                                    |
| 9     | Made`run_study.py` authoritative; retained `sweep_models.sh` only as a parameter-free compatibility wrapper.                                                                                                     |
| 10    | Added validated smoke, development, and final YAML files, exact checkpoint revisions, and separate development description/label files.                                                                              |
| 11    | Refactored analysis into`src.analysis` plus an independently runnable, run-aware CLI. Labels are joined only in analysis.                                                                                          |
| 12    | Required complete model grids, froze JSONL/CSV evidence by hash, and conditionally invoked analysis after verification.                                                                                              |
| 13    | Added a fail-closed final gate for prompt, data, labels, parser, models/revisions, generation/repeat/seed design, analysis specification, and Git commit.                                                            |
| 14    | Added component and end-to-end coverage for normal execution, interruption/resume, collisions, malformed output, execution/integrity failures, phase safety, foreign working directories, and final dry-run refusal. |
| 15    | Replaced configuration duplication with config-driven checkpoint download, documented the actual architecture and operational procedures, and marked legacy paths.                                                   |

## Files created or substantially refactored

- Top-level: `run_study.py`, `README.md`, and `.gitignore`; explanatory records
  and the final-run checklist are grouped under `Descriptive_Docs/`.
- Central configuration: `configs/smoke_test.yaml`,
  `configs/development.yaml`, `configs/final.yaml`, and `configs/README.md`.
- Analysis contract: `analysis_specifications/primary_analysis_v1.yaml` and its
  README.
- Core modules: `src/config.py`, `src/paths.py`, `src/datasets.py`,
  `src/prompt.py`, `src/inference.py`, `src/results.py`, `src/integrity.py`,
  `src/preflight.py`, `src/provenance.py`, `src/run_state.py`, `src/resume.py`,
  `src/model_worker.py`, `src/orchestrator.py`, `src/analysis.py`, and
  `src/final_gate.py`.
- Data contracts: phase-specific smoke/development description and label CSVs,
  plus the sealed-final placeholder README.
- Entry points: `03_evaluation/run_eval.py`,
  `05_data_analysis/analyze_results.py`, and
  `01_general_information_and_requirements/scripts/download_models.py` are
  configuration-driven; `02_testing/diagnose_prompt.py` is phase guarded and
  requires an exact checkpoint revision.
- Tests: 14 test modules under `tests/`, covering component contracts and full
  workflow behavior.
- Folder READMEs were updated for setup, diagnostics, compatibility entry
  points, final-data handling, and independent analysis.

The following compatibility paths remain deprecated:

- `03_evaluation/sweep_models.sh`: thin wrapper only, not a configuration source.
- `03_evaluation/prompt.py`: compatibility import of canonical `src.prompt`.

On 2026-09-07, the unused `04_results/` directory and legacy combined
`data/example_devices.csv` were removed. The test that proves combined inputs
are rejected now creates a temporary invalid fixture. Documentation was updated
to refer only to `runs/<run_id>/` and phase-specific separated datasets.

## Important design decisions

- The YAML file is the only source of study-level model, seed, sampling, path,
  failure-policy, and analysis settings. Python supplies schema validation, not
  competing experimental values.
- The inference loader returns only device ID and description. It rejects any
  ground-truth-like column. A separate label loader is called by validation and
  analysis, never by shared inference.
- One model runs per child process so GPU resources are released between
  checkpoints. The mock engine is permitted only for `smoke_test`.
- Every attempted generation is immediately appended to JSONL and flushed.
  The CSV is derived, never the sole evidence source.
- Malformed model text is scientific evidence, not an execution failure. Raw
  text is retained and issue flags are added without silently coercing labels.
- Evidence immutability is enforced portably with stored SHA-256 hashes plus a
  write guard. Analysis verifies those hashes before use; it does not depend on
  filesystem permission semantics.
- Analysis is descriptive because no confirmatory inferential method was
  specified. It reports explicit denominators for all-generation and
  schema-valid accuracy and repeat consistency.
- The parser freeze hashes the complete canonical `src/prompt.py`, covering the
  prompt, accepted labels, parser, schema checks, and issue-flag rules.
- `src.final_gate` reports candidate identifiers but never edits the final YAML.
  An independent reviewer must copy approved values into `final_freeze`.

## Verification performed

Final automated suite:

```text
python -m unittest discover -s tests -v
Ran 55 tests in 4.079s
OK
```

Syntax/import compilation:

```text
python -m compileall -q src tests run_study.py 02_testing/diagnose_prompt.py \
  03_evaluation/run_eval.py 05_data_analysis/analyze_results.py \
  01_general_information_and_requirements/scripts/download_models.py
exit code 0
```

Process-level end-to-end smoke test:

```text
python run_study.py --config configs/smoke_test.yaml
runs/smoke_20260813_154644_194f
status: completed
generations: 4
```

The smoke run successfully traversed configuration validation, preflight,
worker execution, incremental JSONL/CSV persistence, generation-grid integrity,
raw-evidence freezing, automatic analysis, manifest completion, and logging.
The independently rerunnable analysis command also succeeded on the frozen run:

```text
python 05_data_analysis/analyze_results.py \
  --run runs/smoke_20260813_154644_194f
```

The mock model's 0.5 accuracy is expected: it deterministically emits one class
for two synthetic records with different reference labels. Smoke accuracy is
not a model-performance result.

Development workflow tests exercised real pipeline code with an injected
deterministic test engine. They did not claim to validate vLLM or a checkpoint.
Final-mode dry-run tests confirmed that missing freeze fields are rejected
before a run directory is created.

## Completion-criteria audit

- [X] One top-level Python command executes a configured study.
- [X] Shared inference serves worker, development, and final configurations.
- [X] Smoke, development, and final phases are enforced.
- [X] Prompt diagnostics and development tooling reject final data.
- [X] Inference cannot load expert ground truth.
- [X] Paths are independent of the terminal working directory.
- [X] Runs have unique identities and existing evidence is not overwritten.
- [X] Resume executes only missing generations after compatibility validation.
- [X] Every run has a manifest, structured log, and traceable artifacts.
- [X] Dataset, label, prompt, parser, models, revisions, code, packages, and
  environment identifiers are captured where available.
- [X] Preflight occurs before full checkpoint execution.
- [X] Raw responses survive parse/schema failures unchanged.
- [X] Execution failures and model-output issues are distinct.
- [X] Completion requires zero missing, duplicate, or unexpected keys.
- [X] Analysis is independently rerunnable and never generates responses.
- [X] Automatic analysis occurs only after verified inference and evidence freeze.
- [X] Final scientific state has a fail-closed, auditable freeze gate.
- [X] Bash is no longer an independent configuration source.
- [X] Synthetic smoke and deterministic development end-to-end tests pass
  without using final data.

## Limitations, risks, and follow-up

1. This Windows host has no vLLM installation, CUDA-visible GPU, downloaded
   checkpoints, or WSL2 runtime. The actual vLLM loading/preflight path and the
   three pinned open-weight checkpoints therefore remain to be validated on the
   intended Linux/WSL2 GPU host.
2. The sealed final `devices.csv` and `reference_labels.csv` are intentionally
   absent. They must be provided separately, ID-reconciled, contamination
   checked, hashed, and reviewed without prompt tuning.
3. The dependency file uses compatibility ranges because vLLM, PyTorch, CUDA,
   and drivers must be selected together for the target GPU host. After the
   environment is validated, archive an exact environment lock/container and
   retain the manifest's runtime package and hardware capture.
4. The primary analysis specification intentionally contains descriptive
   metrics only. Any inferential statistical plan must be scientifically
   approved and frozen before final execution, not added after observing final
   responses.

Implementation confidence: the software architecture and tested non-GPU paths
are complete. The project is ready for GPU-host development validation, but the
real final study is not yet ready to run until the limitations above are
resolved and independently reviewed.
