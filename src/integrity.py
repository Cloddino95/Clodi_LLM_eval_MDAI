"""Experimental-grid completeness and duplicate validation.
-->  proves that every expected model/device/repeat combination appears exactly once—even when a response is malformed."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import GenerationConfig, ModelConfig
from .inference import DeviceInput
from .results import GenerationKey, generation_key, read_jsonl


class IntegrityError(RuntimeError):
    """Raised when raw evidence does not form the expected experimental grid."""


@dataclass(frozen=True)
class IntegrityReport:
    model: str
    model_revision: str
    expected_count: int
    actual_count: int
    unique_count: int
    duplicate_keys: tuple[GenerationKey, ...]
    missing_keys: tuple[GenerationKey, ...]
    unexpected_keys: tuple[GenerationKey, ...]
    completed: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["duplicate_keys"] = [asdict(key) for key in self.duplicate_keys]
        result["missing_keys"] = [asdict(key) for key in self.missing_keys]
        result["unexpected_keys"] = [asdict(key) for key in self.unexpected_keys]
        return result


def validate_model_integrity(
    responses_path: Path,
    model: ModelConfig,
    devices: Iterable[DeviceInput],
    generation: GenerationConfig,
    *,
    raise_on_failure: bool = False,
) -> IntegrityReport:
    expected = {
        GenerationKey(model.repository, model.revision, device.device_id, repeat_index)
        for device in devices
        for repeat_index in range(generation.repeats)
    }
    all_records = read_jsonl(responses_path)
    model_records = [
        record for record in all_records
        if record.get("model") == model.repository
        and record.get("model_revision") == model.revision
    ]
    keys = [generation_key(record) for record in model_records]
    counts = Counter(keys)
    actual = set(keys)
    duplicates = tuple(sorted(key for key, count in counts.items() if count > 1))
    missing = tuple(sorted(expected - actual))
    unexpected = tuple(sorted(actual - expected))
    completed = not duplicates and not missing and not unexpected and len(actual) == len(expected)
    report = IntegrityReport(
        model=model.repository,
        model_revision=model.revision,
        expected_count=len(expected),
        actual_count=len(keys),
        unique_count=len(actual),
        duplicate_keys=duplicates,
        missing_keys=missing,
        unexpected_keys=unexpected,
        completed=completed,
    )
    if raise_on_failure and not completed:
        raise IntegrityError(
            f"Integrity failed for {model.repository}@{model.revision}: "
            f"expected={report.expected_count}, actual={report.actual_count}, "
            f"unique={report.unique_count}, duplicates={len(duplicates)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return report

