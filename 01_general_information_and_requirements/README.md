# 01 — Requirements and data contracts

Install Python dependencies from the project root:

```bash
pip install -r 01_general_information_and_requirements/requirements.txt
```

The deterministic `configs/smoke_test.yaml` workflow does not require vLLM or
a GPU. Real development/final inference requires a validated Linux/WSL2 CUDA
environment and locally cached checkpoints.

Alternatively, a model entry may provide `local_path` for a reviewed snapshot
stored on shared or scratch storage. Preflight checks the folder structure and
weights; retain separate metadata or hashes proving that its bytes match the
declared revision before a final study.

Download exact model commits from a YAML configuration—the configuration is the
source of truth:

```bash
python 01_general_information_and_requirements/scripts/download_models.py \
  --config configs/development.yaml
```

Data are separated by phase and role:

```text
data/
|-- smoke_test/
|   |-- devices.csv
|   `-- reference_labels.csv
|-- development/
|   |-- devices.csv
|   `-- reference_labels.csv
`-- final/
    |-- devices.csv             # supplied only when sealed/frozen
    `-- reference_labels.csv    # supplied only when sealed/frozen
```

Inference CSVs contain exactly `device_id,description`. Label CSVs contain
exactly `device_id,ground_truth`. Combined description-and-label files are
deliberately unsupported: keeping the files separate prevents the inference
component from seeing the expert answer.

The two final CSVs are ignored by Git by default. This reduces the risk of
publishing sealed or sensitive study data to GitHub. Store and transfer the
authoritative final files through the study's approved secure data location.
