# Contributing

This research repository uses short-lived branches and pull requests so that
scientifically material changes remain reviewable.

## Workflow

1. Update local `main` before starting work.
2. Create a descriptive branch, for example `feature/add-statistical-checks` or
   `docs/clarify-final-freeze`.
3. Keep one logical change per branch and commit.
4. Run the complete automated test suite.
5. Push the branch and open a pull request into `main`.
6. Record what changed, why, validation performed, and whether the change
   affects the prompt, parser, datasets, model selection, generation settings,
   analysis specification, or final-study freeze.
7. Merge only after review and passing checks. Prefer squash merge for a clear
   main-branch history.

## Validation

Run from the repository root:

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow repeats the CPU-only test suite. Local vLLM/CUDA
validation remains necessary for changes affecting real model execution.

## Experimental safeguards

- Never commit model checkpoints, tokens, credentials, generated `runs/`, or
  unapproved final-study data.
- Keep the tracked `configs/final.yaml` fail-closed. Populate the Git-ignored
  `configs/final_frozen.local.yaml` only after the reviewed code commit exists.
- Do not combine inference descriptions and expert labels in one input file.
- Never use final descriptions for prompt development or diagnostics.
- Treat changes to frozen final-study artifacts as a return to development,
  followed by a new independent freeze review.
- Preserve raw experimental evidence; do not edit a completed run in place.

## Adding a collaborator

After a collaborator accepts the repository invitation, give them write access.
They should work on branches and open pull requests rather than committing
directly to `main`.
