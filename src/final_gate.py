"""Fail-closed confirmatory-run freeze validation and audit helpers.
--> compares final artifacts with audited fingerprints and stops if anything has changed or is missing."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import StudyConfig, load_config
from .datasets import sha256_file
from .paths import PROJECT_ROOT
from .preflight import DatasetPreflight, dataset_preflight
from .prompt import SYSTEM_PROMPT
from .provenance import canonical_json_hash, git_commit, sha256_text


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class FinalFreezeError(RuntimeError):
    """Raised when confirmatory execution is not completely frozen."""


def output_parser_sha256() -> str:
    """Hash the complete canonical prompt/parser module implementing output rules."""
    from . import prompt

    return sha256_file(Path(prompt.__file__).resolve())


def current_final_freeze_values(
    config: StudyConfig,
    checked: DatasetPreflight,
) -> dict[str, str | None]:
    specification = config.analysis.specification
    return {
        "prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "dataset_sha256": checked.dataset_sha256,
        "reference_labels_sha256": checked.reference_labels_sha256,
        "analysis_specification_sha256": (
            sha256_file(specification) if specification and specification.is_file() else None
        ),
        "output_parser_sha256": output_parser_sha256(),
        "models_sha256": canonical_json_hash([asdict(model) for model in config.models]),
        "generation_sha256": canonical_json_hash(asdict(config.generation)),
        "code_git_commit": git_commit(PROJECT_ROOT),
    }


def final_freeze_preflight(config: StudyConfig, checked: DatasetPreflight) -> dict[str, Any]:
    """Verify every confirmatory artifact against explicitly audited identifiers."""
    if config.phase != "final":
        return {"required": False, "verified": True}
    if not config.analysis.enabled or config.analysis.specification is None:
        raise FinalFreezeError("Final mode requires an enabled primary analysis specification")
    invalid_revisions = [
        f"{model.repository}@{model.revision}"
        for model in config.models
        if not _COMMIT_RE.fullmatch(model.revision)
    ]
    if invalid_revisions:
        raise FinalFreezeError(
            "Final model revisions must be exact 40-character commit hashes: "
            + ", ".join(invalid_revisions)
        )
    expected = asdict(config.final_freeze)
    missing = sorted(key for key, value in expected.items() if not value)
    if missing:
        raise FinalFreezeError(
            "Final freeze is incomplete; set: " + ", ".join(missing)
        )
    current = current_final_freeze_values(config, checked)
    unavailable = sorted(key for key, value in current.items() if value is None)
    if unavailable:
        raise FinalFreezeError(
            "Final freeze cannot be verified in this environment; unavailable: "
            + ", ".join(unavailable)
        )
    mismatches = [
        f"{key}: frozen={expected[key]!r}, current={current[key]!r}"
        for key in expected
        if expected[key] != current[key]
    ]
    if mismatches:
        raise FinalFreezeError(
            "Final freeze mismatch. Return to development for any intentional change:\n- "
            + "\n- ".join(mismatches)
        )
    return {"required": True, "verified": True, "values": current}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print current final-freeze identifiers for independent review; never edits YAML."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.phase != "final":
        parser.error("--config must declare phase: final")
    checked = dataset_preflight(config)
    print(json.dumps(current_final_freeze_values(config, checked), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
