# Blueprint scope and provenance

This repository starts from the cleaned MDR LLM evaluation blueprint at commit
`78de195` of the collaborative `mdr-llm-eval` project. It is an independent
private copy for preserving and improving the general-purpose methodology.

## Preserved scientific design

- Canonical prompt version `mdr_samd_risk_classification_v1`.
- Original MDR/IVDR wording and structured output contract.
- The three original candidate checkpoints: Llama 3.1 8B Instruct, Qwen 2.5
  7B Instruct, and Mistral 7B Instruct v0.3.
- Five repeats at temperature zero with the original seed design.
- A 1,024-token maximum output in development and final templates.
- Separate device-description and expert-label files.
- Smoke, development, and fail-closed final phases.
- Append-only raw evidence, compatible resume, grid-integrity validation,
  evidence hashing, automatic descriptive analysis, and final freeze checks.

## Later engineering improvements adopted

These changes retain the original behavior when their optional settings are
omitted:

- `run_study.py --models` for completing one logical run across sequential
  jobs with different GPU requirements.
- Optional per-model `local_path`, `tensor_parallel_size`, `max_num_seqs`, and
  `gdn_prefill_backend` settings.
- Local-folder preflight checks for model configuration and weight files.
- Tokenizer loading through vLLM so preflight matches inference, including
  Mistral-style tokenizer formats.
- Robust prompt-token counting for both plain token lists and mappings with
  `input_ids`.
- Prompt diagnostics that never overwrite previous output, preserve raw text,
  and record prompt/model/revision/token provenance.
- Automated tests for selected-model resume, local paths, tokenizer behavior,
  and cross-platform path handling.

## Study-specific changes intentionally not adopted

- Prompt versions 2 and 3 or their MDR-only methodological changes.
- The HPI eight-model roster and HPI absolute filesystem paths.
- HPI Slurm account, partition, user, log, and energy-monitoring scripts.
- HPI-specific exact dependency versions and CUDA build choices.
- EUDAMED development/final data paths, data hashes, or study-specific run IDs.
- The 4,096-token generation change.

Those elements belong to the collaborative research implementation, where the
principal researcher controls the scientific methodology and computing
environment. This blueprint remains a portable reference implementation.

## Reproducibility caution for local model folders

The preflight checks that a configured local folder exists and contains a model
configuration plus weight files. It does not prove that those bytes correspond
to the declared Hugging Face revision. A final study should additionally retain
snapshot metadata or approved file hashes and include them in the archival
record.
