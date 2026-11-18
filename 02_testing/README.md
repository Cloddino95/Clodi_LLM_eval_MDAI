# 02 — Development-only prompt diagnostics

`diagnose_prompt.py` runs leave-one-rule-out and cumulative prompt probes on a
small smoke/development subset. It is a prompt-development tool, not a study
result, and it refuses `dataset-phase=final` before importing vLLM.

Example:

```bash
python 02_testing/diagnose_prompt.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --devices 01_general_information_and_requirements/data/development/devices.csv \
  --phase development \
  --dataset-phase development \
  --num-devices 5 \
  --out 02_testing/prompt_diagnostics_v1.csv
```

Use the same exact cached checkpoint intended for development. Do not use final
descriptions, and rerun diagnostics whenever the canonical prompt changes. The
command refuses to overwrite its CSV or companion `.raw.jsonl`; each diagnostic
therefore needs a new `--out`. The CSV records prompt/model/revision, finish
reason and token counts, while JSONL preserves complete raw output.

Optional `--tensor-parallel-size`, `--max-num-seqs`, and
`--gdn-prefill-backend` arguments support compatible GPU environments without
changing the prompt or default single-GPU behavior.
