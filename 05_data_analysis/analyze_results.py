"""Run the independently rerunnable analysis for one protected study run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import analyze_run, resolve_run_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run ID or explicit run-directory path")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs",
        help="Directory containing run IDs (ignored when --run is a directory)",
    )
    args = parser.parse_args()
    paths = resolve_run_paths(args.run, args.runs_root)
    outputs = analyze_run(paths)
    print(f"Analyzed run: {paths.root}")
    print(f"Model summary: {outputs.model_summary}")
    print(f"Consistency table: {outputs.device_model_consistency}")
    print(f"Analysis metadata: {outputs.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
