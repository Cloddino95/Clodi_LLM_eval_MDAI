"""Execute one configured model inside an existing protected study run.

This compatibility entry point is useful for schedulers that launch one model
per job. It does not define scientific settings or create output locations;
those come from the YAML configuration and the run manifest created by
``run_study.py``. Normal users should invoke ``run_study.py`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.model_worker import execute_model  # noqa: E402
from src.run_state import open_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True, help="Existing run created by run_study.py")
    parser.add_argument("--model-index", required=True, type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if not 0 <= args.model_index < len(config.models):
        parser.error("--model-index is outside the configured model list")
    paths = open_run(config.outputs.root, args.run_id)
    execute_model(config, paths, args.model_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
