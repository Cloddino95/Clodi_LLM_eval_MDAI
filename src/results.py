"""Append-only raw evidence and deterministic structured-answer rebuilding.
--> appends canonical JSONL records durably, rebuilds the readable CSV, and freezes file hashes after completion."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import GenerationConfig, ModelConfig
from .inference import DeviceInput
from .datasets import sha256_file
from .run_state import RunPaths, RunStateError, load_manifest, update_manifest


ANSWER_COLUMNS = [
    "model", "model_revision", "device_id", "run_index", "seed",
    "classification", "rationale", "applicable_regulation",
    "applicable_provision", "missing_information", "parse_ok", "parse_success",
    "schema_valid", "parse_error", "classification_valid", "regulation_valid",
    "required_fields_present", "rationale_present", "applicable_provision_present",
    "missing_information_valid_type", "citation_present",
    "no_external_resources_statement_present", "response_truncated", "json_repaired",
    "extraneous_text_present", "validation_issues", "prompt_tokens",
    "completion_tokens", "latency_seconds", "finish_reason",
]


@dataclass(frozen=True, order=True)
class GenerationKey:
    model: str
    model_revision: str
    device_id: str
    repeat_index: int


def generation_key(record: dict[str, Any]) -> GenerationKey:
    return GenerationKey(
        str(record["model"]),
        str(record["model_revision"]),
        str(record["device_id"]),
        int(record["run_index"]),
    )


def expected_generation_keys(
    models: Iterable[ModelConfig],
    devices: Iterable[DeviceInput],
    generation: GenerationConfig,
) -> set[GenerationKey]:
    return {
        GenerationKey(model.repository, model.revision, device.device_id, repeat_index)
        for model in models
        for device in devices
        for repeat_index in range(generation.repeats)
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunStateError(f"Malformed raw evidence at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise RunStateError(f"Non-object raw evidence at {path}:{line_number}")
            records.append(record)
    return records


def completed_generation_keys(path: Path) -> set[GenerationKey]:
    return {generation_key(record) for record in read_jsonl(path)}


def missing_generation_keys(
    expected: set[GenerationKey], completed: set[GenerationKey]
) -> set[GenerationKey]:
    return expected - completed


def rebuild_answers_csv(responses_path: Path, answers_path: Path) -> None:
    """Transactionally derive the clean CSV from canonical JSONL evidence."""
    records = read_jsonl(responses_path)
    temporary = answers_path.with_name(f".{answers_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANSWER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in ANSWER_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(answers_path)


class ResultStore:
    """Append raw records durably while rejecting duplicate generation keys."""

    def __init__(self, paths: RunPaths):
        self.paths = paths
        if load_manifest(paths).get("raw_evidence", {}).get("frozen"):
            raise RunStateError(f"Raw evidence is frozen; refusing further writes: {paths.root}")
        self.completed = completed_generation_keys(paths.responses)
        if paths.responses.exists():
            rebuild_answers_csv(paths.responses, paths.answers)

    def append(self, record: dict[str, Any]) -> GenerationKey:
        key = generation_key(record)
        if key in self.completed:
            raise RunStateError(f"Refusing duplicate generation evidence: {key}")
        serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.paths.responses.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        self.completed.add(key)
        return key

    def append_batch(self, records: Iterable[dict[str, Any]]) -> list[GenerationKey]:
        keys = [self.append(record) for record in records]
        rebuild_answers_csv(self.paths.responses, self.paths.answers)
        return keys


def freeze_raw_evidence(paths: RunPaths) -> dict[str, Any]:
    """Hash-lock canonical evidence after all model grids pass integrity checks.

    Immutability is enforced by the run state and verified hashes rather than
    platform-specific file permissions, which are unreliable across Windows,
    WSL, shared filesystems, and archival copies.
    """
    manifest = load_manifest(paths)
    existing = manifest.get("raw_evidence", {})
    if existing.get("frozen"):
        verify_frozen_raw_evidence(paths, manifest)
        return dict(existing)
    if not paths.responses.is_file():
        raise RunStateError(f"Cannot freeze missing raw evidence: {paths.responses}")
    rebuild_answers_csv(paths.responses, paths.answers)
    records = read_jsonl(paths.responses)
    if not records:
        raise RunStateError("Cannot freeze an empty raw evidence file")
    payload = {
        "frozen": True,
        "responses_sha256": sha256_file(paths.responses),
        "answers_sha256": sha256_file(paths.answers),
        "generation_count": len(records),
    }
    temporary = paths.evidence_manifest.with_name(f".{paths.evidence_manifest.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(paths.evidence_manifest)
    update_manifest(
        paths,
        event="raw_evidence_frozen",
        event_details=payload,
        updates={"raw_evidence": payload},
    )
    return payload


def verify_frozen_raw_evidence(
    paths: RunPaths, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Fail if canonical evidence differs from the hashes frozen in the manifest."""
    state = (manifest or load_manifest(paths)).get("raw_evidence", {})
    if not state.get("frozen"):
        raise RunStateError("Raw evidence has not been frozen")
    checks = {
        "responses_sha256": sha256_file(paths.responses),
        "answers_sha256": sha256_file(paths.answers),
        "generation_count": len(read_jsonl(paths.responses)),
    }
    mismatches = [
        f"{key}: frozen={state.get(key)!r}, current={value!r}"
        for key, value in checks.items()
        if state.get(key) != value
    ]
    if mismatches:
        raise RunStateError("Frozen raw evidence integrity failure:\n- " + "\n- ".join(mismatches))
    return dict(state)
