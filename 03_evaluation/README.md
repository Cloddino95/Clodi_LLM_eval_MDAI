# 03 — Evaluation compatibility entry points

The authoritative command is now:

```bash
python run_study.py --config configs/development.yaml
```

For sequential jobs with different GPU requirements, add `--models` with one
or more repository names from the configuration. Resume the same run ID for
the remaining models; evidence is frozen and analysis begins only after all
configured models have completed.

`run_eval.py` is a thin worker wrapper for schedulers that need to execute one
configured model inside an existing protected run:

```bash
python 03_evaluation/run_eval.py \
  --config configs/development.yaml \
  --run-id <existing-run-id> \
  --model-index 0
```

It defines no models, seeds, paths, or sampling settings. `sweep_models.sh` is
deprecated and merely forwards arguments to `run_study.py`. `prompt.py` is a
compatibility import of the canonical `src.prompt`; it contains no duplicate
prompt text.

All actual inference lives in `src/inference.py`, while `src/model_worker.py`
keeps one model per process so GPU memory is released between checkpoints.
