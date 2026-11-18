# Final confirmatory run checklist

This checklist is the operational bridge between a finished development
workflow and the one definitive run on the sealed expert-labelled dataset. It
does not replace the study protocol or independent scientific review.

## What must be available

1. A Linux or validated WSL2 environment with a compatible NVIDIA GPU, CUDA,
   PyTorch, vLLM, and enough disk/GPU memory.
2. The three approved instruction-model snapshots cached locally at the exact
   revisions listed in the chosen YAML configuration.
3. `devices.csv`, containing exactly `device_id,description`.
4. `reference_labels.csv`, containing exactly `device_id,ground_truth` and the
   same unique device IDs as the device file.
5. A finalized prompt/parser, analysis specification, model list, generation
   parameters, and committed codebase.

The reference-label file is needed during preflight and later analysis, but it
is never placed in the object passed to the model. The model sees only the ID
and description.

## 1. Finish development before opening the final data

Use `configs/development.yaml` to verify model compatibility, prompt wording,
output format, temperature, repeat count, seeds, context length, and the
analysis plan. Once the final descriptions have been opened, do not use them
to repair or tune the prompt. A scientifically necessary change means returning
to development and creating a new freeze before another final attempt.

## 2. Validate the code and basic workflow

From the repository root:

```bash
python -m unittest discover -s tests -v
python run_study.py --config configs/smoke_test.yaml
```

The unit tests check software rules and edge cases. The smoke run checks that
the components work together. Neither command measures a real LLM.

## 3. Verify and cache the approved model snapshots

While the approved model-download machine is online:

```bash
python 01_general_information_and_requirements/scripts/download_models.py \
  --config configs/final.yaml
```

This downloads or verifies only the model artifacts. It does not read the
device or label CSVs. Resolve model licences and Hugging Face authentication
before this point. The actual inference process later forces Hugging Face and
Transformers into offline mode.

## 4. Place the sealed CSVs locally

Put the two authoritative files at:

```text
01_general_information_and_requirements/data/final/devices.csv
01_general_information_and_requirements/data/final/reference_labels.csv
```

Both filenames are ignored by Git, so a normal commit or push will not publish
them. Keep the authoritative copies in the study's approved secure storage and
control access according to the applicable data-governance requirements.

Do not add extra columns. If the source export contains other information,
make reviewed run-ready extracts with only the required columns before the
freeze. Do not change the pipeline's input contract during the final stage just
to accommodate an unexpected export.

## 5. Commit the finished study code

Review and commit all tracked code, prompt, configuration-template, and
analysis-specification changes. Then verify the working tree is clean:

```bash
git status --short
git rev-parse HEAD
```

The first command should print nothing. The second prints the exact code commit
that the final freeze will record.

## 6. Create the local frozen configuration

Keep `configs/final.yaml` as the tracked fail-closed template. Make a local
copy in the same directory:

```bash
cp configs/final.yaml configs/final_frozen.local.yaml
```

PowerShell equivalent:

```powershell
Copy-Item configs/final.yaml configs/final_frozen.local.yaml
```

The local filename is ignored by Git. This avoids a self-reference problem:
committing a file that contains the current Git commit would itself create a
new commit ID. The exact local configuration is still hashed and copied into
the run manifest for auditability.

## 7. Generate and independently review the freeze values

With the final files in place, print the current identifiers:

```bash
python -m src.final_gate --config configs/final_frozen.local.yaml
```

The command prints SHA-256 fingerprints for the prompt, device data, expert
labels, analysis specification, prompt/parser source, model configuration, and
generation design, plus the current Git commit. It never edits a YAML file.

An independent reviewer should verify the final inputs and approve those
identifiers. Copy the approved values into `final_freeze` in
`configs/final_frozen.local.yaml`. Run the same command again: the displayed
current values should be identical to the populated values.

Run `git status --short` once more. It should still print nothing because the
final data and local frozen configuration are intentionally ignored. If any
tracked file changed, stop, review the change, commit it if appropriate, and
generate a new freeze.

## 8. Run the confirmatory experiment

On the validated GPU host, with the model snapshots already cached:

```bash
python run_study.py --config configs/final_frozen.local.yaml
```

The program performs configuration, dataset, overlap, checkpoint, tokenizer,
context-window, GPU, and synthetic-generation preflight checks before the full
model loop. It runs one model process at a time, saves every response
incrementally, verifies that every model/device/repeat combination occurs
exactly once, freezes the raw evidence by hash, and then automatically performs
the configured analysis.

If the process is genuinely interrupted before evidence is frozen, resume only
with the same local configuration and the printed run ID:

```bash
python run_study.py --config configs/final_frozen.local.yaml --resume <run_id>
```

Do not use resume to mix changed settings or to append to a completed run; the
pipeline rejects both.

## 9. Review and archive the run

The authoritative run is under `runs/<run_id>/`:

```text
manifest.json                         run status and provenance
run.log                               execution history and failures
raw/responses.jsonl                   canonical raw model evidence
raw/answers.csv                       readable derived answer sheet
raw/evidence_manifest.json            hashes proving raw-evidence integrity
analysis/model_summary.csv            per-model descriptive measures
analysis/device_model_consistency.csv per-device repeat consistency
analysis/analysis_metadata.json       analysis provenance
```

Archive the run directory, the exact environment/container lock, the populated
local configuration, the sealed input files or their approved archive
references, and the independent review record according to the study protocol.
The generated `runs/` directory is ignored by Git; GitHub is not the evidence
archive.

Analysis normally runs automatically. To regenerate only the derived tables
from an already completed and hash-verified run, without invoking an LLM:

```bash
python 05_data_analysis/analyze_results.py --run <run_id>
```
