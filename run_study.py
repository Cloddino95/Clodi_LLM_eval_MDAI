"""Top-level command for reproducible MDR LLM study execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.orchestrator import run_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--run-id", help="Optional explicit ID for a new run; collisions are rejected")
    identity.add_argument("--resume", metavar="RUN_ID", help="Resume an existing compatible run")
    parser.add_argument("--models", nargs="+", metavar="REPOSITORY",
                        help="Run only these models from the config; the run is completed and analysed once all models are done")
    args = parser.parse_args()
    paths = run_study(args.config, run_id=args.run_id, resume_run_id=args.resume, models=args.models)
    print(f"Run directory: {paths.root}")
    print(f"Manifest: {paths.manifest}")
    return 0 if load_status(paths.manifest) == "completed" else 1


def load_status(manifest_path: Path) -> str:
    import json

    return str(json.loads(manifest_path.read_text(encoding="utf-8"))["status"])


if __name__ == "__main__":
    raise SystemExit(main())

