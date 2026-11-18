"""Study-level coordination without duplicating inference or analysis logic."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

from .config import StudyConfig, load_config
from .model_worker import model_key
from .preflight import configuration_preflight, dataset_preflight
from .resume import validate_resume_compatibility
from .run_state import (
    RunPaths,
    close_run_logger,
    create_run,
    load_manifest,
    open_run,
    setup_run_logger,
    update_manifest,
)
from .paths import PROJECT_ROOT
from .results import freeze_raw_evidence
from .analysis import analyze_run
from .final_gate import final_freeze_preflight
from .run_state import RunStateError


WorkerRunner = Callable[[StudyConfig, RunPaths, int], int]


def subprocess_worker(config: StudyConfig, paths: RunPaths, model_index: int) -> int:
    command = [
        sys.executable,
        "-m",
        "src.model_worker",
        "--config",
        str(config.source),
        "--run-id",
        paths.root.name,
        "--model-index",
        str(model_index),
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def run_study(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    resume_run_id: str | None = None,
    worker_runner: WorkerRunner = subprocess_worker,
    models: list[str] | None = None,
) -> RunPaths:
    if run_id and resume_run_id:
        raise ValueError("Choose either a new --run-id or --resume, not both")
    config = load_config(config_path)
    selected: set[int] | None = None
    if models:
        repositories = [m.repository for m in config.models]
        unknown = sorted(set(models) - set(repositories))
        if unknown:
            raise ValueError(f"--models not in the configuration: {unknown}")
        selected = {i for i, r in enumerate(repositories) if r in models}
    configuration_preflight(config, resume_run_id=resume_run_id)
    checked = dataset_preflight(config)
    freeze_audit = final_freeze_preflight(config, checked)

    if resume_run_id:
        paths = open_run(config.outputs.root, resume_run_id)
        resume_manifest = load_manifest(paths)
        if resume_manifest.get("raw_evidence", {}).get("frozen"):
            raise RunStateError(
                "This run's raw evidence is already frozen; rerun analysis independently or start a new run"
            )
        validate_resume_compatibility(
            resume_manifest,
            config,
            dataset_sha256=checked.dataset_sha256,
            reference_labels_sha256=checked.reference_labels_sha256,
        )
        update_manifest(
            paths,
            status="partial",
            event="run_resumed",
            event_details={"config": str(config.source)},
        )
    else:
        paths = create_run(
            config,
            dataset_sha256=checked.dataset_sha256,
            reference_labels_sha256=checked.reference_labels_sha256,
            run_id=run_id,
        )
        update_manifest(paths, updates={"final_freeze_audit": freeze_audit})

    logger = setup_run_logger(paths)
    try:
        logger.info("configuration loaded path=%s", config.source)
        logger.info("validation passed phase=%s dataset=%s", config.phase, config.dataset.path)
        update_manifest(paths, status="running", event="inference_started")
        failures: list[str] = []
        interrupted = False
        for index, model in enumerate(config.models):
            if selected is not None and index not in selected:
                logger.info("skipping model not selected for this job model=%s", model.repository)
                continue
            logger.info("starting model worker model=%s revision=%s", model.repository, model.revision)
            try:
                return_code = worker_runner(config, paths, index)
            except KeyboardInterrupt:
                return_code = 130
            if return_code == 0:
                logger.info("model worker succeeded model=%s", model.repository)
                continue
            key = model_key(config, index)
            failures.append(key)
            if return_code == 130:
                interrupted = True
                logger.warning("model worker interrupted model=%s", model.repository)
                break
            logger.error("model worker failed model=%s return_code=%s", model.repository, return_code)
            if not config.failure_policy.continue_on_model_failure:
                break

        manifest = load_manifest(paths)
        model_states = manifest.get("model_runs", {})
        all_completed = all(
            model_states.get(model_key(config, index), {}).get("status") == "completed"
            for index in range(len(config.models))
        )
        if interrupted:
            update_manifest(paths, status="partial", event="run_interrupted")
        elif selected is not None and not failures and not all_completed:
            pending = [model_key(config, i) for i in range(len(config.models))
                       if model_states.get(model_key(config, i), {}).get("status") != "completed"]
            update_manifest(paths, status="partial", event="inference_partial_selection",
                            event_details={"pending_models": pending})
            logger.info("selected models completed; run stays partial until resumed for pending=%s", pending)
        elif failures or not all_completed:
            update_manifest(
                paths,
                status="failed",
                event="inference_failed",
                event_details={"failed_models": failures},
            )
        else:
            logger.info("all configured model inference completed; freezing raw evidence")
            frozen = freeze_raw_evidence(paths)
            update_manifest(
                paths,
                event="inference_verified",
                event_details={"generation_count": frozen["generation_count"]},
            )
            analysis_updates = {"status": "disabled"}
            if config.analysis.enabled:
                logger.info("verified inference complete; primary analysis started")
                outputs = analyze_run(paths)
                analysis_updates = {
                    "status": "completed",
                    "model_summary": str(outputs.model_summary),
                    "device_model_consistency": str(outputs.device_model_consistency),
                    "metadata": str(outputs.metadata),
                }
                update_manifest(
                    paths,
                    event="analysis_completed",
                    event_details=analysis_updates,
                )
                logger.info("primary analysis completed")
            update_manifest(
                paths,
                status="completed",
                event="run_completed",
                updates={"analysis_run": analysis_updates},
            )
            logger.info("study run completed")
        return paths
    except BaseException:
        logger.exception("unexpected study-level exception")
        update_manifest(paths, status="failed", event="unexpected_study_exception")
        raise
    finally:
        close_run_logger(logger)
