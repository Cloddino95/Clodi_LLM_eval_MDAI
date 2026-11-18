# The refactored code, explained in plain language

The code is organized by responsibility so that no component secretly performs
two scientifically different jobs.

- `configs/*.yaml` are the experiment forms: they name the phase, data, exact
  model versions, repeats, seeds, output location, and analysis plan.
- `run_study.py` is the coordinator. It validates the form, creates a protected
  run, starts one model worker at a time, freezes complete evidence, and then
  requests analysis.
- `src/config.py` rejects incomplete or surprising configuration fields.
- `src/datasets.py` reads descriptions and expert labels through different
  functions. The inference object has no place to store a correct answer.
- `src/prompt.py` holds the one canonical natural-language prompt and checks
  the structure of model answers without changing the original text.
- `src/inference.py` is the only model-running implementation. vLLM turns the
  chat messages into local GPU generation. A fake deterministic engine exists
  only for smoke tests.
- `src/model_worker.py` loads one model in one process, runs preflight, writes
  only missing responses, and validates the expected response grid.
- `src/results.py` appends canonical JSONL records durably, rebuilds the readable
  CSV, and freezes file hashes after completion.
- `src/run_state.py` creates unique folders, maintains `manifest.json`, and
  writes `run.log`.
- `src/resume.py` prevents results made with different scientific settings from
  being mixed in one run.
- `src/integrity.py` proves that every expected model/device/repeat combination
  appears exactly once—even when a response is malformed.
- `src/final_gate.py` compares final artifacts with audited fingerprints and
  stops if anything has changed or is missing.
- `src/analysis.py` reads frozen evidence and separately stored expert labels.
  It creates descriptive summary tables and never calls a model.

The central flow is:

```text
YAML → validation → run folder → one model worker → incremental raw answers
     → grid check → evidence hash freeze → label join → derived analysis
```

Changing a model, revision, temperature, repeat count, seeds, token limit, or
data path is therefore a YAML change. Changing the actual instructions,
classification categories, or answer structure is a `src/prompt.py` change.
Changing which measures are prespecified is an
`analysis_specifications/*.yaml` change. These distinctions prevent two places
from silently disagreeing about the same experiment.

See [README.md](../README.md) for exact commands, configuration examples, folder
layouts, final-freeze instructions, and troubleshooting. See
[FINAL_RUN_CHECKLIST.md](FINAL_RUN_CHECKLIST.md) when preparing the definitive
experiment.
