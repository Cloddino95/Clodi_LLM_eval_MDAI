"""Run identity, directories, manifests, status transitions, and logging.
--> creates unique folders, maintains `manifest.json`, and writes `run.log`"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import StudyConfig
from .datasets import sha256_file
from .paths import PROJECT_ROOT
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT
from .provenance import collect_environment, sha256_text


MANIFEST_SCHEMA_VERSION = 1
ALLOWED_STATUSES = {"created", "running", "partial", "failed", "completed"}
_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_-]+")


class RunStateError(RuntimeError):
    """Raised when run lifecycle invariants are violated."""


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    log: Path
    raw: Path
    responses: Path
    answers: Path
    evidence_manifest: Path
    analysis: Path
    figures: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_run_id(label: str, *, now: datetime | None = None) -> str:
    safe_label = _SAFE_LABEL_RE.sub("_", label.strip()).strip("_") or "run"
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    return f"{safe_label}_{timestamp}_{secrets.token_hex(2)}"


def paths_for_run(root: Path, run_id: str) -> RunPaths:
    run_root = root / run_id
    raw = run_root / "raw"
    analysis = run_root / "analysis"
    return RunPaths(
        root=run_root,
        manifest=run_root / "manifest.json",
        log=run_root / "run.log",
        raw=raw,
        responses=raw / "responses.jsonl",
        answers=raw / "answers.csv",
        evidence_manifest=raw / "evidence_manifest.json",
        analysis=analysis,
        figures=analysis / "figures",
    )


def open_run(root: Path, run_id: str) -> RunPaths:
    paths = paths_for_run(root, run_id)
    if not paths.root.is_dir():
        raise RunStateError(f"Run does not exist: {paths.root}")
    if not paths.manifest.is_file():
        raise RunStateError(f"Run has no manifest and cannot be resumed: {paths.root}")
    return paths


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def initial_manifest(
    config: StudyConfig,
    run_id: str,
    *,
    dataset_sha256: str,
    reference_labels_sha256: str | None,
) -> dict[str, Any]:
    started = utc_now()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "study_name": config.study_name,
        "phase": config.phase,
        "status": "created",
        "start_timestamp_utc": started,
        "completion_timestamp_utc": None,
        "config": {
            "path": str(config.source),
            "sha256": sha256_file(config.source),
        },
        "dataset": {
            "name": config.dataset.name,
            "path": str(config.dataset.path),
            "phase": config.dataset.phase,
            "sha256": dataset_sha256,
        },
        "reference_labels": {
            "path": str(config.reference_labels.path) if config.reference_labels.path else None,
            "sha256": reference_labels_sha256,
        },
        "prompt": {
            "version": PROMPT_VERSION,
            "sha256": sha256_text(SYSTEM_PROMPT),
        },
        "models": [asdict(model) for model in config.models],
        "generation": asdict(config.generation),
        "analysis": {
            "enabled": config.analysis.enabled,
            "specification": str(config.analysis.specification) if config.analysis.specification else None,
        },
        "final_freeze": asdict(config.final_freeze),
        "environment": collect_environment(PROJECT_ROOT),
        "model_runs": {},
        "events": [{"timestamp_utc": started, "event": "run_created"}],
    }


def create_run(
    config: StudyConfig,
    *,
    dataset_sha256: str,
    reference_labels_sha256: str | None,
    run_id: str | None = None,
) -> RunPaths:
    run_id = run_id or generate_run_id(config.run_label)
    paths = paths_for_run(config.outputs.root, run_id)
    try:
        paths.root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RunStateError(f"Run destination already exists; refusing overwrite: {paths.root}") from exc
    paths.raw.mkdir()
    paths.analysis.mkdir()
    paths.figures.mkdir()
    _atomic_json_write(
        paths.manifest,
        initial_manifest(
            config,
            run_id,
            dataset_sha256=dataset_sha256,
            reference_labels_sha256=reference_labels_sha256,
        ),
    )
    return paths


def load_manifest(paths: RunPaths) -> dict[str, Any]:
    if not paths.manifest.is_file():
        raise RunStateError(f"Missing run manifest: {paths.manifest}")
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def update_manifest(
    paths: RunPaths,
    *,
    status: str | None = None,
    event: str | None = None,
    event_details: dict[str, Any] | None = None,
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(paths)
    if status is not None:
        if status not in ALLOWED_STATUSES:
            raise RunStateError(f"Invalid run status: {status}")
        manifest["status"] = status
        if status == "completed":
            manifest["completion_timestamp_utc"] = utc_now()
    if updates:
        manifest.update(updates)
    if event:
        manifest.setdefault("events", []).append({
            "timestamp_utc": utc_now(),
            "event": event,
            "details": event_details or {},
        })
    _atomic_json_write(paths.manifest, manifest)
    return manifest


def update_model_run(
    paths: RunPaths,
    model_key: str,
    *,
    status: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(paths)
    entry = manifest.setdefault("model_runs", {}).setdefault(model_key, {})
    entry.update(details or {})
    entry["status"] = status
    entry["updated_at_utc"] = utc_now()
    _atomic_json_write(paths.manifest, manifest)
    return manifest


def setup_run_logger(
    paths: RunPaths,
    *,
    console: bool = True,
    component: str = "study",
) -> logging.Logger:
    logger = logging.getLogger(f"mdr_llm_eval.{paths.root.name}.{component}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(paths.log, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def close_run_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
