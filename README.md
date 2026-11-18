# Clodi LLM Evaluation MDAI

This private, portable blueprint evaluates locally executed, open-weight
instruction-tuned language models on software-product classification under EU
MDR 2017/745 and, where relevant, IVDR. It is research infrastructure, not a
clinical or regulatory decision system.

The repository preserves the original version-1 study instrument, original
three-model design, example development data, and 1,024-token generation
limit. Later engineering improvements that do not change those scientific
defaults have been incorporated: optional local model paths and multi-GPU
settings, selected-model execution across resumable jobs, vLLM-aligned
tokenizer preflight, stronger prompt diagnostics, and additional automated
tests. See [BLUEPRINT_NOTES.md](BLUEPRINT_NOTES.md) for the exact boundary.

The pipeline is designed to be reproducible, resumable, auditable, and safe
against accidental use of the final holdout set. Device descriptions are sent
only to a model loaded locally through vLLM. Reference labels are stored in a
separate file and are loaded only by validation and analysis code.

## Architecture

```text
configs/*.yaml
      |
      v
run_study.py (coordination only)
      |
      +-- configuration/path/phase validation
      +-- run identity, manifest, and logging
      +-- one subprocess per configured model
      |       +-- optional selection of models for staged/HPC jobs
      |       |
      |       +-- model/cache/GPU preflight
      |       +-- src.inference shared generation code
      |       +-- incremental JSONL persistence and resume
      |       +-- per-model grid-integrity validation
      |
      +-- hash-freeze raw evidence
      +-- src.analysis (only after verified inference)
```

There is one inference implementation: [src/inference.py](src/inference.py).
Both the model worker and compatibility entry points use it. Statistical
analysis is separate in [src/analysis.py](src/analysis.py) and never generates
model output.

## Repository map

| Path | Responsibility |
|---|---|
| `run_study.py` | Authoritative top-level command for new runs and resumes. |
| `configs/` | Sole source of study-level settings: smoke, development, final. |
| `src/config.py` | Strict YAML schema and configuration-relative paths. |
| `src/datasets.py` | Isolated dataset/label loading, hashes, phase checks, contamination checks. |
| `src/prompt.py` | Canonical prompt and non-destructive output parser/validation flags. |
| `src/inference.py` | Shared vLLM and deterministic smoke-engine adapters. |
| `src/preflight.py` | Dataset, model asset, tokenizer, context, GPU, and smoke-generation checks. |
| `src/model_worker.py` | Executes exactly one model in its own process. |
| `src/results.py` | Append-only JSONL, derived CSV rebuild, evidence freeze and verification. |
| `src/integrity.py` | Expected/actual/duplicate/missing generation-grid validation. |
| `src/run_state.py` | Run IDs, directories, manifest transitions, and logging. |
| `src/resume.py` | Scientific compatibility checks for resume. |
| `src/provenance.py` | Hashes and runtime package/CUDA/GPU/Git metadata. |
| `src/final_gate.py` | Fail-closed confirmatory freeze verification. |
| `src/analysis.py` | Independently rerunnable descriptive analysis. |
| `03_evaluation/run_eval.py` | Thin one-model compatibility worker; not a settings source. |
| `03_evaluation/sweep_models.sh` | Deprecated thin wrapper around `run_study.py`. |
| `02_testing/diagnose_prompt.py` | Development-only prompt ablation diagnostic; rejects final data. |
| `05_data_analysis/analyze_results.py` | Thin independent analysis CLI. |
| `tests/` | Component and end-to-end workflow tests. |
| `Descriptive_Docs/` | Plain-language guides, implementation record, and final-run checklist. |
| `BLUEPRINT_NOTES.md` | What the blueprint preserves, adopts, and intentionally excludes. |

New results always use `runs/<run_id>/`. Operational datasets always keep
descriptions and expert labels in different files; combined input files are
rejected by the inference loader.

## Requirements and installation

Production inference requires Linux (or a validated WSL2 Linux environment),
Python, vLLM, a compatible NVIDIA/CUDA stack, enough GPU memory and disk space,
and locally cached model checkpoints. The deterministic smoke test needs no GPU
or vLLM.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r 01_general_information_and_requirements/requirements.txt
```

On Windows PowerShell, activation is normally:

```powershell
.venv\Scripts\Activate.ps1
```

Install vLLM in its supported Linux/CUDA environment. Record the exact working
environment in the final project lock/archival record; the run manifest also
queries installed versions programmatically.

## Configuration contract

All study settings live in YAML:

```yaml
phase: development              # smoke_test | development | final
study:
  name: study_name
  run_label: development
dataset:
  path: ../path/devices.csv
  phase: development
  name: dataset_version
  development_paths: []         # required comparison inputs in final mode
reference_labels:
  path: ../path/reference_labels.csv
models:
  - repository: Qwen/Qwen2.5-7B-Instruct
    revision: a09a35458c702b33eeacc393d103063234e8bc28
    local_path: null            # optional local model folder
    quantization: null
    max_model_len: 8192
    gpu_memory_utilization: 0.9
    tensor_parallel_size: 1     # optional multi-GPU model parallelism
    max_num_seqs: null          # optional vLLM batching limit
    gdn_prefill_backend: null   # optional: flashinfer | triton | cutedsl
generation:
  repeats: 5
  temperature: 0.0
  seeds: [2001, 2002, 2003, 2004, 2005]
  max_tokens: 1024
outputs:
  root: ../runs
analysis:
  enabled: true
  specification: ../analysis_specifications/primary_analysis_v1.yaml
failure_policy:
  continue_on_model_failure: false
final_freeze: {}                # mandatory audited hashes in final mode
```

Unknown fields, missing values, invalid phases, invalid seed counts, and phase
mismatches fail before inference. Paths are resolved relative to the YAML file,
so commands behave the same from any terminal directory.

The four optional model-execution fields preserve the original behavior when
omitted. `local_path` is useful when a cluster stores reviewed model snapshots
in shared folders. The repository and exact revision remain mandatory for
provenance. For a final study, separately verify that the local folder really
contains the approved snapshot; folder presence alone is not cryptographic
proof of the declared revision.

## Experimental phases

### `smoke_test`

Uses synthetic descriptions and a deterministic mock response. It proves that
configuration, subprocess orchestration, incremental saving, integrity checks,
evidence freezing, analysis, and manifests work. It does not evaluate an LLM.

```bash
python run_study.py --config configs/smoke_test.yaml
```

### `development`

Used for prompt diagnostics, model compatibility tests, parameter selection,
and non-confirmatory runs. Download the exact configured model snapshots once:

```bash
python 01_general_information_and_requirements/scripts/download_models.py \
  --config configs/development.yaml
python run_study.py --config configs/development.yaml
```

Meta Llama may require accepting its licence and authenticating with Hugging
Face before the online download step. Inference itself is forced into Hugging
Face/Transformers offline mode.

#### Development-only prompt diagnostic

The diagnostic remains separate from study execution and rejects final-phase
data. It now refuses to overwrite existing results, preserves full raw output
in a companion `.raw.jsonl`, records prompt/model/revision/token provenance,
and accepts optional GPU-parallelism settings. Its default output limit remains
the blueprint's original 1,024 tokens.

```bash
python 02_testing/diagnose_prompt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --devices 01_general_information_and_requirements/data/development/devices.csv \
  --phase development \
  --dataset-phase development \
  --out prompt_diagnostic_v1.csv
```

### `final`

Final mode accepts only a dataset explicitly declared `phase: final`, requires
development datasets for deterministic overlap/near-duplicate checks, requires
exact 40-character model commit hashes, and verifies the complete final freeze.
Prompt diagnostics reject final-phase datasets.

Do not use final descriptions for debugging or prompt development.

## Dataset and label isolation

The inference file must contain exactly:

```csv
device_id,description
```

The separate expert-label file must contain exactly:

```csv
device_id,ground_truth
```

Inference objects contain no ground-truth field. Analysis joins labels only
after generation grids have passed integrity validation and raw evidence has
been frozen. Duplicate IDs, empty values, unexpected columns, or differing
device-ID sets are rejected.

## Starting and resuming runs

Start a new run with an automatically generated unique ID:

```bash
python run_study.py --config configs/development.yaml
```

Or request a meaningful unused ID:

```bash
python run_study.py --config configs/development.yaml --run-id development_01
```

Existing destinations are never overwritten. Resume an interrupted run:

```bash
python run_study.py --config configs/development.yaml --resume development_01
```

Resume verifies the dataset and label hashes, prompt hash, model definitions,
and every generation parameter. It calculates expected minus completed
generation keys and runs only missing work. A frozen/completed run cannot be
resumed; derived analysis can instead be rerun independently.

For clusters where different models require different GPU allocations, run a
selected subset while retaining one logical run:

```bash
python run_study.py --config configs/development.yaml \
  --run-id development_01 \
  --models meta-llama/Llama-3.1-8B-Instruct

python run_study.py --config configs/development.yaml \
  --resume development_01 \
  --models Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3
```

After the first command the run remains `partial`. Evidence is frozen and
analysis starts only when every model listed in the YAML has completed. Run
these jobs sequentially; multiple jobs must not write to the same run directory
at the same time.

## Run IDs and directory structure

Automatic IDs combine the YAML label, UTC timestamp, and random suffix, for
example `development_20260813_154212_a1b2`.

```text
runs/<run_id>/
|-- manifest.json
|-- run.log
|-- raw/
|   |-- responses.jsonl
|   |-- answers.csv
|   `-- evidence_manifest.json
`-- analysis/
    |-- model_summary.csv
    |-- device_model_consistency.csv
    |-- analysis_metadata.json
    `-- figures/
```

`responses.jsonl` is canonical evidence. It retains raw text even when JSON is
malformed, labels are invalid, citations are absent, or output is truncated.
`answers.csv` is a deterministic inspection view rebuilt from JSONL.

Results are flushed and `fsync`-ed in incremental batches. After every model
has exactly the expected `(model, revision, device, repeat)` keys—with no
duplicates, missing keys, or unexpected keys—the raw files are hash-frozen.
Subsequent `ResultStore` writes are rejected and analysis rechecks the hashes.

## Manifest, logging, and provenance

`manifest.json` is created before model work and evolves through `created`,
`running`, `partial`, `failed`, and `completed`. It records:

- run identity, timestamps, phase and events;
- configuration, dataset, label and prompt hashes;
- exact repositories, model revisions, quantization and generation settings;
- analysis specification and final-freeze identifiers;
- Python/package versions, CUDA version, GPU/driver information and Git commit;
- per-model preflight, smoke-generation, integrity, failure and completion data;
- raw-evidence hashes and derived-analysis locations.

`run.log` records major lifecycle events and full tracebacks for unexpected
failures. Model-output defects are not execution failures: they remain rows in
the raw evidence with flags such as `parse_success`, `schema_valid`,
`classification_valid`, `citation_present`, and `response_truncated`.

The failure policy determines whether a fatal failure in one model stops later
models. Study-level validation always stops. Interruptions remain `partial` and
resumable. Integrity failures can never be marked successful.

## Preflight checks

Before full inference the pipeline checks:

- YAML schema, phase, required files and output semantics;
- inference/label columns, IDs, hashes and final contamination;
- exact model revision in the local Hugging Face cache, or the presence and
  basic structure of an explicitly configured `local_path`;
- tokenizer and chat template loaded through vLLM, matching inference behavior;
- maximum prompt plus output length against the context window;
- CUDA visibility and a conservative weights-only memory estimate;
- one built-in synthetic generation that never consumes a final observation.

The memory check is an estimate, not a guarantee; runtime/KV-cache allocations
can still trigger CUDA OOM.

## Independent analysis

When `analysis.enabled: true` (as in all three supplied configurations), the
orchestrator invokes analysis automatically after inference is complete,
integrity-verified, and hash-frozen. No separate analysis command is required
for a normal successful run.

Analysis never invokes inference. The following command is available when a
researcher needs to regenerate derived tables from an already completed run,
recover from an analysis-only problem, or verify the analysis independently:

```bash
python 05_data_analysis/analyze_results.py --run <run_id>
```

For a non-default runs directory:

```bash
python 05_data_analysis/analyze_results.py \
  --run <run_id> --runs-root D:/study/runs
```

`--run` may also be the explicit run-directory path. The primary specification
reports parse/schema-valid rates, accuracy over all generations and separately
over valid outputs, repeat consistency, citation flags, truncation, JSON repair,
and extraneous-text rates. These are descriptive measures; unapproved
inferential methods are deliberately absent.

The automatic and manual paths call the same `src.analysis` implementation.

## Final-study freeze procedure

Before the real confirmatory experiment:

1. Finish all prompt/model/parameter development.
2. Place sealed `devices.csv` and `reference_labels.csv` in the final-data
   location without exposing descriptions to prompt-development tools.
3. Confirm expert adjudication and exact matching IDs.
4. Review and commit the complete tracked codebase; a Git commit is mandatory.
   Confirm that `git status --short` prints nothing.
5. Review the tracked `configs/final.yaml` template, exact model commits, seeds,
   and analysis specification.
6. Copy the template to the Git-ignored local run configuration:

   ```bash
   cp configs/final.yaml configs/final_frozen.local.yaml
   ```

   In PowerShell, use
   `Copy-Item configs/final.yaml configs/final_frozen.local.yaml`.
7. Print current identifiers without editing the configuration:

   ```bash
   python -m src.final_gate --config configs/final_frozen.local.yaml
   ```

8. Independently review those values and copy them into `final_freeze` in the
   local configuration. Leave the tracked template null and fail-closed. This
   avoids the impossible self-reference of committing a file containing its
   own new Git commit ID; the run manifest records the exact local config hash.
9. Run the complete test suite and smoke test again.
10. Download/verify the final configured checkpoints while online, then isolate
   inference from the network.
11. Confirm `git status --short` is still empty, then start the final run:

   ```bash
   python run_study.py --config configs/final_frozen.local.yaml
   ```

Any change to the prompt, parser module, dataset, labels, model definitions,
generation design, analysis specification, or Git commit causes final mode to
stop. Intentional changes must return to development and produce a new audited
freeze.

The final CSVs and populated local configuration are Git-ignored by default to
avoid accidental publication. See
`Descriptive_Docs/FINAL_RUN_CHECKLIST.md` for the complete plain-language
procedure, including secure archiving and interruption recovery.

## Tests

Run everything:

```bash
python -m unittest discover -s tests -v
```

The expanded suite uses deterministic test engines and does not require a GPU. The real
vLLM/CUDA path must additionally be smoke-tested on the intended GPU host.
Tests do not run as part of `run_study.py`; run them before commits and pull
requests, after code/configuration changes, and immediately before the final
batch. GitHub Actions also runs them automatically on pushes to `main` and on
pull requests. Per-run preflight and integrity checks are separate protections
and do run automatically inside the pipeline.

## Troubleshooting

- **Model not in cache:** run the configuration-driven download command online,
  then retry inference offline.
- **Gated model error:** accept the licence and run `hf auth login` before the
  download step.
- **No CUDA GPU visible:** verify Linux/WSL GPU passthrough, drivers, PyTorch and
  `nvidia-smi`.
- **CUDA OOM:** select an approved quantized checkpoint or reduce approved
  memory/context settings during development; never tune after final results.
- **Missing chat template:** choose a compatible instruction checkpoint or
  resolve compatibility during development.
- **Context-window failure:** shorten inputs or revise approved limits during
  development.
- **Run directory exists:** choose a new `--run-id` or use `--resume` only for a
  compatible interrupted run.
- **Resume incompatible:** do not mix evidence; start a new run.
- **Frozen evidence error:** inspect hashes and preserve the original run. Never
  edit raw evidence to make validation pass.
- **Final freeze mismatch:** return to development for intentional changes or
  restore the audited artifact.

## Privacy boundary

Inference enables `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, and disables Hub
telemetry before importing vLLM. The model-download command is the only
intentionally online project command and reads no study data. For a stronger
operational guarantee, run inference on a network-isolated host or enforce an
OS/network firewall; library environment variables are not a universal network
sandbox.

## Collaborative development

Use a short-lived branch and pull request for each logical change. GitHub
Actions runs the CPU-only automated suite on `main` and on pull requests; real
vLLM/CUDA checks remain local. The pull-request template explicitly identifies
changes that could invalidate the scientific freeze. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete workflow.
