"""Resume compatibility and missing-work calculation.
--> prevents results made with different scientific settings from
being mixed in one run."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from .config import StudyConfig
from .prompt import SYSTEM_PROMPT
from .provenance import sha256_text
from .run_state import RunStateError


def validate_resume_compatibility(
    manifest: dict[str, Any],
    config: StudyConfig,
    *,
    dataset_sha256: str,
    reference_labels_sha256: str | None = None,
) -> None:
    """Reject mixing scientifically incompatible generations in one run."""
    requested_generation = json.loads(json.dumps(asdict(config.generation)))
    comparisons = {
        "phase": (manifest.get("phase"), config.phase),
        "dataset hash": (manifest.get("dataset", {}).get("sha256"), dataset_sha256),
        "reference-label hash": (
            manifest.get("reference_labels", {}).get("sha256"),
            reference_labels_sha256,
        ),
        "prompt hash": (manifest.get("prompt", {}).get("sha256"), sha256_text(SYSTEM_PROMPT)),
        "models": (manifest.get("models"), [asdict(model) for model in config.models]),
        "generation parameters": (manifest.get("generation"), requested_generation),
    }
    mismatches = [
        f"{name}: original={original!r}, requested={requested!r}"
        for name, (original, requested) in comparisons.items()
        if original != requested
    ]
    if mismatches:
        raise RunStateError("Resume configuration is incompatible:\n- " + "\n- ".join(mismatches))
