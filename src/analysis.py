"""Independent descriptive analysis of a completed protected study run.
--> reads frozen evidence and separately stored expert labels. It creates descriptive summary tables and never calls a model."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .datasets import load_reference_labels, sha256_file
from .paths import PROJECT_ROOT
from .results import read_jsonl, verify_frozen_raw_evidence
from .run_state import RunPaths, RunStateError, load_manifest, paths_for_run


ANALYSIS_SPECIFICATION_VERSION = "primary_analysis_v1"


class AnalysisError(RuntimeError):
    """Raised when a run cannot be analyzed without violating its contract."""


@dataclass(frozen=True)
class AnalysisOutputs:
    model_summary: Path
    device_model_consistency: Path
    metadata: Path


def resolve_run_paths(run: str | Path, runs_root: str | Path | None = None) -> RunPaths:
    """Resolve either an explicit run directory or a run ID under runs_root."""
    candidate = Path(run).expanduser()
    if candidate.is_dir():
        run_root = candidate.resolve()
        return paths_for_run(run_root.parent, run_root.name)
    root = Path(runs_root).expanduser().resolve() if runs_root else PROJECT_ROOT / "runs"
    paths = paths_for_run(root, str(run))
    if not paths.root.is_dir():
        raise AnalysisError(f"Run does not exist: {paths.root}")
    return paths


def load_analysis_specification(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise AnalysisError(f"Analysis specification does not exist: {source}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency/environment specific
        raise AnalysisError("PyYAML is required to load the analysis specification") from exc
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AnalysisError(f"Invalid analysis specification YAML: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AnalysisError("Analysis specification must be a mapping")
    if value.get("version") != ANALYSIS_SPECIFICATION_VERSION:
        raise AnalysisError(
            f"Unsupported analysis specification version: {value.get('version')!r}"
        )
    return value


def _analysis_ready(manifest: Mapping[str, Any]) -> None:
    models = manifest.get("models")
    model_runs = manifest.get("model_runs")
    if not isinstance(models, list) or not models:
        raise AnalysisError("Manifest contains no configured models")
    if not isinstance(model_runs, Mapping):
        raise AnalysisError("Manifest contains no model-run status")
    incomplete = []
    for model in models:
        key = f"{model.get('repository')}@{model.get('revision')}"
        entry = model_runs.get(key, {})
        integrity = entry.get("integrity", {}) if isinstance(entry, Mapping) else {}
        if entry.get("status") != "completed" or not integrity.get("completed"):
            incomplete.append(key)
    if incomplete:
        raise AnalysisError(
            "Analysis requires integrity-verified model runs; incomplete=" + ", ".join(incomplete)
        )


def _joined_records(paths: RunPaths, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    reference = manifest.get("reference_labels", {})
    reference_path = reference.get("path") if isinstance(reference, Mapping) else None
    if not reference_path:
        raise AnalysisError("Manifest does not identify a reference-label file")
    labels_path = Path(str(reference_path)).resolve()
    expected_hash = reference.get("sha256")
    if expected_hash and sha256_file(labels_path) != expected_hash:
        raise AnalysisError("Reference-label file hash no longer matches the run manifest")
    labels = {item.device_id: item.ground_truth for item in load_reference_labels(labels_path)}
    records = read_jsonl(paths.responses)
    if not records:
        raise AnalysisError(f"Run contains no raw responses: {paths.responses}")
    response_ids = {str(record.get("device_id", "")) for record in records}
    if response_ids != set(labels):
        raise AnalysisError(
            "Response and reference-label device IDs differ; "
            f"missing_labels={sorted(response_ids - set(labels))}, "
            f"unused_labels={sorted(set(labels) - response_ids)}"
        )
    joined = []
    for record in records:
        item = dict(record)
        item["ground_truth"] = labels[str(record["device_id"])]
        item["correct"] = (
            bool(record.get("schema_valid"))
            and str(record.get("classification", "")) == item["ground_truth"]
        )
        joined.append(item)
    return joined


def _rate(items: Iterable[Mapping[str, Any]], predicate) -> float | None:
    values = list(items)
    return sum(1 for item in values if predicate(item)) / len(values) if values else None


def build_model_summary(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record.get("model", "")), str(record.get("model_revision", "")))
        groups.setdefault(key, []).append(record)
    rows = []
    for (model, revision), group in sorted(groups.items()):
        valid = [item for item in group if bool(item.get("schema_valid"))]
        correct_all = sum(1 for item in group if bool(item.get("correct")))
        correct_valid = sum(1 for item in valid if bool(item.get("correct")))
        rows.append({
            "model": model,
            "model_revision": revision,
            "n_generations": len(group),
            "n_schema_valid": len(valid),
            "parse_rate": _rate(group, lambda item: bool(item.get("parse_success"))),
            "schema_valid_rate": _rate(group, lambda item: bool(item.get("schema_valid"))),
            "accuracy_all_generations": correct_all / len(group),
            "accuracy_valid_outputs": correct_valid / len(valid) if valid else None,
            "citation_present_rate": _rate(group, lambda item: bool(item.get("citation_present"))),
            "no_external_resources_statement_rate": _rate(
                group, lambda item: bool(item.get("no_external_resources_statement_present"))
            ),
            "citation_contradiction_rate": _rate(
                group,
                lambda item: bool(item.get("citation_present"))
                and bool(item.get("no_external_resources_statement_present")),
            ),
            "response_truncation_rate": _rate(
                group, lambda item: bool(item.get("response_truncated"))
            ),
            "json_repair_rate": _rate(group, lambda item: bool(item.get("json_repaired"))),
            "extraneous_text_rate": _rate(
                group, lambda item: bool(item.get("extraneous_text_present"))
            ),
        })
    return rows


def build_device_model_consistency(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            str(record.get("model", "")),
            str(record.get("model_revision", "")),
            str(record.get("device_id", "")),
        )
        groups.setdefault(key, []).append(record)
    rows = []
    for (model, revision, device_id), group in sorted(groups.items()):
        valid_labels = [
            str(item.get("classification", ""))
            for item in group
            if bool(item.get("schema_valid"))
        ]
        counts = Counter(valid_labels)
        modal_label, modal_count = counts.most_common(1)[0] if counts else ("", 0)
        rows.append({
            "device_id": device_id,
            "model": model,
            "model_revision": revision,
            "ground_truth": str(group[0].get("ground_truth", "")),
            "n_generations": len(group),
            "n_schema_valid": len(valid_labels),
            "modal_classification": modal_label,
            "modal_share_valid_outputs": modal_count / len(valid_labels) if valid_labels else None,
            "modal_share_all_generations": modal_count / len(group),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"Refusing to write empty analysis table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(paths: RunPaths) -> AnalysisOutputs:
    """Analyze one run without importing or invoking inference code."""
    manifest = load_manifest(paths)
    _analysis_ready(manifest)
    try:
        verify_frozen_raw_evidence(paths, manifest)
    except RunStateError as exc:
        raise AnalysisError(str(exc)) from exc
    analysis_config = manifest.get("analysis", {})
    specification_path = analysis_config.get("specification")
    if not specification_path:
        raise AnalysisError("Manifest does not identify an analysis specification")
    load_analysis_specification(specification_path)
    records = _joined_records(paths, manifest)
    summary = build_model_summary(records)
    consistency = build_device_model_consistency(records)
    paths.analysis.mkdir(parents=True, exist_ok=True)
    summary_path = paths.analysis / "model_summary.csv"
    consistency_path = paths.analysis / "device_model_consistency.csv"
    metadata_path = paths.analysis / "analysis_metadata.json"
    _write_csv(summary_path, summary)
    _write_csv(consistency_path, consistency)
    metadata = {
        "analysis_specification": str(Path(specification_path).resolve()),
        "analysis_specification_sha256": sha256_file(specification_path),
        "raw_responses_sha256": sha256_file(paths.responses),
        "reference_labels_sha256": sha256_file(manifest["reference_labels"]["path"]),
        "n_generations": len(records),
        "outputs": {
            "model_summary": str(summary_path),
            "device_model_consistency": str(consistency_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return AnalysisOutputs(summary_path, consistency_path, metadata_path)
