# Study configurations

These YAML files are the sole source of study-level experimental settings.

- `smoke_test.yaml` uses synthetic data and a deterministic mock engine. It
  validates software plumbing and does not measure an LLM.
- `development.yaml` uses the example development data and the three pinned
  candidate model checkpoints. Prompt diagnostics and parameter development
  belong here.
- `final.yaml` is the tracked, fail-closed final-study template. It cannot run
  while the sealed data are absent and the `final_freeze` values are null.
  After all code is committed, copy it to the Git-ignored
  `final_frozen.local.yaml`, insert the independently audited freeze values,
  and use that local file for the confirmatory run.

Relative paths are resolved from the YAML file, not the terminal directory.
Unknown fields and invalid types are rejected.

Optional per-model execution fields are `local_path`, `tensor_parallel_size`,
`max_num_seqs`, and `gdn_prefill_backend`. Omitting them preserves the original
single-GPU Hugging Face-cache behavior. Use `run_study.py --models ...` only for
sequential staged jobs that contribute to the same protected run.

Change model repositories/revisions and generation parameters such as
temperature, repeat count, seeds, and maximum output tokens here—not in the
Python orchestration code. The number of seeds must equal `repeats`. Prompt
wording, category definitions, and output rules live in `src/prompt.py`; metric
definitions live in the selected file under `analysis_specifications/`.
