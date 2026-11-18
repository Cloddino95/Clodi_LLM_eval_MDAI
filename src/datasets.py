"""Dataset loading, phase isolation, hashing, and contamination checks.
--> reads descriptions and expert labels through different
functions. The inference object has no place to store a correct answer."""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from .config import ALLOWED_PHASES
from .inference import DeviceInput


class DatasetError(ValueError):
    """Raised when a dataset violates the experimental data contract."""


@dataclass(frozen=True)
class ReferenceLabel:
    device_id: str
    ground_truth: str


@dataclass(frozen=True)
class ContaminationFinding:
    kind: str
    development_device_id: str
    final_device_id: str
    similarity: float | None = None


def sha256_file(path: str | Path) -> str:
    """Hash the exact bytes stored in a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_phase_compatibility(run_phase: str, dataset_phase: str) -> None:
    """Require an explicitly matching phase; final data is never downgraded."""
    if run_phase not in ALLOWED_PHASES:
        raise DatasetError(f"Unknown run phase: {run_phase}")
    if dataset_phase not in ALLOWED_PHASES:
        raise DatasetError(f"Unknown dataset phase: {dataset_phase}")
    if run_phase != dataset_phase:
        raise DatasetError(
            f"Run phase {run_phase!r} cannot consume a {dataset_phase!r} dataset"
        )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise DatasetError(f"Dataset does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"CSV has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def load_inference_dataset(path: str | Path) -> list[DeviceInput]:
    """Load the inference-only contract; labels and extra columns are rejected."""
    source = Path(path).resolve()
    columns, rows = _read_csv(source)
    expected = ["device_id", "description"]
    if set(columns) != set(expected) or len(columns) != len(expected):
        raise DatasetError(
            f"Inference dataset must contain only {expected}; found {columns}. "
            "Store ground_truth in the separate reference-label file."
        )
    if not rows:
        raise DatasetError(f"Inference dataset is empty: {source}")
    devices: list[DeviceInput] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        device_id = (row.get("device_id") or "").strip()
        description = (row.get("description") or "").strip()
        if not device_id:
            raise DatasetError(f"Missing device_id at row {row_number} in {source}")
        if device_id in seen:
            raise DatasetError(f"Duplicate device_id {device_id!r} in {source}")
        if not description:
            raise DatasetError(f"Missing description for {device_id!r} in {source}")
        seen.add(device_id)
        devices.append(DeviceInput(device_id, description))
    return devices


def load_reference_labels(path: str | Path) -> list[ReferenceLabel]:
    source = Path(path).resolve()
    columns, rows = _read_csv(source)
    expected = ["device_id", "ground_truth"]
    if set(columns) != set(expected) or len(columns) != len(expected):
        raise DatasetError(f"Reference labels must contain only {expected}; found {columns}")
    labels: list[ReferenceLabel] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        device_id = (row.get("device_id") or "").strip()
        ground_truth = (row.get("ground_truth") or "").strip()
        if not device_id or not ground_truth:
            raise DatasetError(f"Blank reference-label value at row {row_number} in {source}")
        if device_id in seen:
            raise DatasetError(f"Duplicate reference device_id {device_id!r} in {source}")
        seen.add(device_id)
        labels.append(ReferenceLabel(device_id, ground_truth))
    return labels


_NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_description(text: str) -> str:
    """NFKC + casefold + punctuation removal + whitespace collapse."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _NON_WORD_RE.sub(" ", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def contamination_findings(
    development: Iterable[DeviceInput],
    final: Iterable[DeviceInput],
    *,
    near_duplicate_threshold: float = 0.90,
) -> list[ContaminationFinding]:
    """Report deterministic ID, exact, normalized, and near-description overlap."""
    if not 0 <= near_duplicate_threshold <= 1:
        raise DatasetError("near_duplicate_threshold must be between 0 and 1")
    development_items = list(development)
    final_items = list(final)
    findings: list[ContaminationFinding] = []
    for dev in development_items:
        for holdout in final_items:
            if dev.device_id == holdout.device_id:
                findings.append(ContaminationFinding("device_id", dev.device_id, holdout.device_id, 1.0))
            if dev.description == holdout.description:
                findings.append(ContaminationFinding("exact_description", dev.device_id, holdout.device_id, 1.0))
                continue
            dev_normalized = normalize_description(dev.description)
            final_normalized = normalize_description(holdout.description)
            if dev_normalized == final_normalized:
                findings.append(ContaminationFinding("normalized_description", dev.device_id, holdout.device_id, 1.0))
                continue
            similarity = SequenceMatcher(None, dev_normalized, final_normalized, autojunk=False).ratio()
            if similarity >= near_duplicate_threshold:
                findings.append(
                    ContaminationFinding("near_description", dev.device_id, holdout.device_id, similarity)
                )
    return findings

