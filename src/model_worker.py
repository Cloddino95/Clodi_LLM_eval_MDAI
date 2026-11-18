"""One-model worker process used by the Python study orchestrator.
==> loads one model in one process, runs preflight, writes
only missing responses, and validates the expected response grid."""

from __future__ import annotations

import argparse
import traceback

from .config import StudyConfig, load_config
from .inference import create_engine, run_model_incremental
from .integrity import IntegrityError, validate_model_integrity
from .preflight import dataset_preflight, model_asset_preflight, smoke_generation_preflight
from .results import ResultStore
from .resume import validate_resume_compatibility
from .run_state import (
    RunPaths,
    close_run_logger,
    load_manifest,
    open_run,
    setup_run_logger,
    update_model_run,
)


def model_key(config: StudyConfig, model_index: int) -> str:
    model = config.models[model_index]
    return f"{model.repository}@{model.revision}"


def classify_execution_error(exc: BaseException) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return "recoverable_interruption"
    if isinstance(exc, IntegrityError):
        return "integrity_failure"
    if "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower():
        return "cuda_oom"
    return "fatal_model_execution_failure"


def execute_model(config: StudyConfig, paths: RunPaths, model_index: int) -> None:
    """Preflight, infer/resume, and validate exactly one configured model."""
    logger = setup_run_logger(paths, component=f"model-{model_index}")
    model = config.models[model_index]
    key = model_key(config, model_index)
    try:
        checked = dataset_preflight(config)
        validate_resume_compatibility(
            load_manifest(paths),
            config,
            dataset_sha256=checked.dataset_sha256,
            reference_labels_sha256=checked.reference_labels_sha256,
        )
        update_model_run(paths, key, status="preflight")
        logger.info("model preflight started model=%s revision=%s", model.repository, model.revision)
        assets = model_asset_preflight(
            model,
            checked.devices,
            generation_max_tokens=config.generation.max_tokens,
            phase=config.phase,
        )
        logger.info("model loading started model=%s", model.repository)
        engine = create_engine(model, config.phase)
        logger.info("model loaded model=%s", model.repository)
        smoke = smoke_generation_preflight(engine, model, config)
        logger.info("preflight generation passed model=%s", model.repository)
        update_model_run(
            paths,
            key,
            status="running",
            details={"asset_preflight": assets.__dict__, "smoke_generation": smoke},
        )
        store = ResultStore(paths)
        generated = run_model_incremental(
            engine, model, checked.devices, config.generation, store, logger=logger
        )
        report = validate_model_integrity(
            paths.responses,
            model,
            checked.devices,
            config.generation,
            raise_on_failure=True,
        )
        update_model_run(
            paths,
            key,
            status="completed",
            details={"new_generations": generated, "integrity": report.to_dict()},
        )
        logger.info("model completed model=%s new_generations=%s", model.repository, generated)
    except BaseException as exc:
        category = classify_execution_error(exc)
        status = "partial" if category == "recoverable_interruption" else "failed"
        update_model_run(
            paths,
            key,
            status=status,
            details={
                "error_category": category,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            },
        )
        logger.exception("model execution failed category=%s model=%s", category, model.repository)
        raise
    finally:
        close_run_logger(logger)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-index", required=True, type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if not 0 <= args.model_index < len(config.models):
        parser.error("--model-index is outside the configured model list")
    paths = open_run(config.outputs.root, args.run_id)
    try:
        execute_model(config, paths, args.model_index)
    except KeyboardInterrupt:
        return 130
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
