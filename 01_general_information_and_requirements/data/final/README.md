# Sealed final-data location

Do not place prompt-development or smoke-test data here.

Immediately before the confirmatory run, an authorized researcher must place
the frozen files below in this directory:

- `devices.csv` with only `device_id,description`;
- `reference_labels.csv` with only `device_id,ground_truth`.

The final-mode gate will reject the run until these files exist and their
SHA-256 hashes match the audited values in the populated local final
configuration.

These two CSV filenames are ignored by Git. They should not be committed to
the GitHub repository. The tracked `configs/final.yaml` is a template; follow
the [final-run checklist](../../../Descriptive_Docs/FINAL_RUN_CHECKLIST.md) to
create the ignored `configs/final_frozen.local.yaml` used for the confirmatory
run.
