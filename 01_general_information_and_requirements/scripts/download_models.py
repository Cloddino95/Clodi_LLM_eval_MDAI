#!/usr/bin/env python
"""Download every non-mock model at the exact revision in a study YAML file.

This is the only intentionally online project command. It reads no device or
reference-label files and performs no inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)

    from huggingface_hub import snapshot_download

    downloaded = 0
    for model in config.models:
        if model.repository.startswith("mock://"):
            print(f"Skipping synthetic model {model.repository}")
            continue
        print(f"Downloading {model.repository}@{model.revision} ...")
        resolved = snapshot_download(
            repo_id=model.repository,
            revision=model.revision,
        )
        print(f"  cached at: {resolved}")
        downloaded += 1
    print(f"Downloaded/verified {downloaded} model checkpoint(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
