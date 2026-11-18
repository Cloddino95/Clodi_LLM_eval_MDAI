# What this project does — plain-language guide

This project tests whether downloadable AI language models can classify
software products under European medical-device rules. It is a research tool,
not a replacement for a regulatory expert.

The researcher provides two separate spreadsheets:

1. device IDs and descriptions, which the AI may see;
2. expert-approved answers, which the AI never sees and which are used only
   after the AI has finished.

The program loads each chosen AI model on the local machine, asks it the same
fixed question for every description, repeats the question to measure
consistency, and saves every answer. Even badly formatted answers are retained
as evidence. Only after the expected number of answers has been checked does
the program lock their digital fingerprints and compare them with expert
labels.

There are three modes:

- **Smoke test:** synthetic examples and a fake deterministic model prove that
  the software plumbing works. It says nothing about AI quality.
- **Development:** real local models and non-final examples are used to test
  compatibility and refine the study before results are frozen.
- **Final:** the sealed holdout data can run only when the prompt, data, labels,
  models, parameters, parser, analysis plan, and exact code version match
  previously reviewed digital fingerprints.

The smoke YAML is runnable now. The development YAML is operational once the
listed model snapshots and a suitable Linux/WSL2 NVIDIA GPU environment are
available. The tracked final YAML is deliberately an incomplete template: the
real data and independently reviewed fingerprints must be supplied in an
ignored local frozen copy before final execution.

Every run receives a unique folder containing a manifest, log, original model
answers, an easier-to-read answer sheet, and derived analysis. Interrupted runs
can continue only missing work if their scientific settings have not changed.
Completed evidence cannot be appended to or silently overwritten.

The model files are downloaded once from Hugging Face at exact saved versions.
During inference, vLLM runs those files locally. The code places Hugging Face
and Transformers in offline mode before vLLM loads. For the strongest privacy
assurance, the inference computer should also be network-isolated or protected
by an operating-system/network firewall.

Start with [FINAL_RUN_CHECKLIST.md](FINAL_RUN_CHECKLIST.md) for the confirmatory
procedure or the technical [README.md](../README.md) when you are ready to
install, test, run, resume, inspect, or reproduce the workflow.
