# 05 — Independent analysis

Analysis is separate from inference and consumes one integrity-verified,
hash-frozen run:

```bash
python 05_data_analysis/analyze_results.py --run <run-id>
```

Use `--runs-root <path>` when outputs are not under the project `runs/`
directory, or pass the explicit run-directory path as `--run`.

The CLI calls `src.analysis`, joins the separately stored expert labels only at
analysis time, verifies label and raw-evidence hashes, and writes derived files
under the run's `analysis/` folder. It never imports vLLM or regenerates model
responses.

The method is frozen in
`analysis_specifications/primary_analysis_v1.yaml`. Accuracy is reported with
two explicit denominators: all expected generations (primary) and schema-valid
outputs (secondary). Consistency is modal agreement among schema-valid outputs,
with an additional all-generation share so malformed outputs are not hidden.
