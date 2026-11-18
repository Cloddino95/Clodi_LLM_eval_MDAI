"""Layered validation performed before expensive full-model inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import ModelConfig, StudyConfig
from .datasets import (
    ContaminationFinding,
    contamination_findings,
    load_inference_dataset,
    load_reference_labels,
    sha256_file,
    validate_phase_compatibility,
)
from .inference import DeviceInput, InferenceEngine, generate_repeat


class PreflightError(RuntimeError):
    """Raised when a prerequisite fails before the full experiment starts."""


@dataclass(frozen=True)
class DatasetPreflight:
    devices: tuple[DeviceInput, ...]
    dataset_sha256: str
    reference_labels_sha256: str | None
    contamination: tuple[ContaminationFinding, ...]


@dataclass(frozen=True)
class ModelAssetPreflight:
    model_path: str
    tokenizer_available: bool
    chat_template_available: bool
    prompt_tokens_max: int | None
    gpu_visible: bool
    gpu_memory_estimate: dict[str, Any]


PREFLIGHT_DEVICE = DeviceInput(
    "PREFLIGHT-SYNTHETIC",
    "A synthetic non-medical calculator used only to verify that text generation works.",
)


def configuration_preflight(config: StudyConfig, *, resume_run_id: str | None = None) -> None:
    validate_phase_compatibility(config.phase, config.dataset.phase)
    if not config.source.is_file():
        raise PreflightError(f"Configuration file is missing: {config.source}")
    if resume_run_id is None and config.outputs.root.exists() and not config.outputs.root.is_dir():
        raise PreflightError(f"Output root is not a directory: {config.outputs.root}")
    if resume_run_id is not None and not resume_run_id.strip():
        raise PreflightError("Resume run ID cannot be blank")
    if config.analysis.enabled:
        if config.analysis.specification is None:
            raise PreflightError("Enabled analysis requires analysis.specification")
        if not config.analysis.specification.is_file():
            raise PreflightError(
                f"Analysis specification does not exist: {config.analysis.specification}"
            )


def dataset_preflight(config: StudyConfig) -> DatasetPreflight:
    try:
        devices = load_inference_dataset(config.dataset.path)
    except Exception as exc:
        raise PreflightError(f"Inference dataset validation failed: {exc}") from exc

    reference_hash = None
    if config.reference_labels.path is not None:
        try:
            labels = load_reference_labels(config.reference_labels.path)
        except Exception as exc:
            raise PreflightError(f"Reference-label validation failed: {exc}") from exc
        device_ids = {device.device_id for device in devices}
        label_ids = {label.device_id for label in labels}
        if device_ids != label_ids:
            missing_labels = sorted(device_ids - label_ids)
            unknown_labels = sorted(label_ids - device_ids)
            raise PreflightError(
                f"Reference-label IDs do not match inference IDs; "
                f"missing={missing_labels}, unknown={unknown_labels}"
            )
        reference_hash = sha256_file(config.reference_labels.path)
    elif config.analysis.enabled or config.phase == "final":
        raise PreflightError("Reference labels are required for analysis and final evaluation")

    findings: list[ContaminationFinding] = []
    if config.phase == "final":
        if not config.dataset.development_paths:
            raise PreflightError("Final configuration must list development datasets for contamination checks")
        for development_path in config.dataset.development_paths:
            try:
                development_devices = load_inference_dataset(development_path)
            except Exception as exc:
                raise PreflightError(f"Development dataset validation failed ({development_path}): {exc}") from exc
            findings.extend(contamination_findings(development_devices, devices))
        if findings:
            rendered = "; ".join(
                f"{item.kind}:{item.development_device_id}->{item.final_device_id}"
                for item in findings
            )
            raise PreflightError(f"Potential development/final contamination found: {rendered}")

    return DatasetPreflight(
        devices=tuple(devices),
        dataset_sha256=sha256_file(config.dataset.path),
        reference_labels_sha256=reference_hash,
        contamination=tuple(findings),
    )


def _resolve_cached_model(model: ModelConfig) -> Path:
    # Models stored as plain folders (config `local_path`) are checked in place;
    # the pinned revision stays recorded in the configuration.
    local_path = getattr(model, "local_path", None)
    if local_path:
        folder = Path(local_path).expanduser()
        if not folder.is_dir():
            raise PreflightError(f"local_path for {model.repository} is not a directory: {folder}")
        if not ((folder / "config.json").is_file() or (folder / "params.json").is_file()):
            raise PreflightError(
                f"local_path for {model.repository} has no config.json or params.json: {folder}"
            )
        has_weights = any(
            path.is_file()
            for pattern in ("*.safetensors", "*.bin")
            for path in folder.glob(pattern)
        )
        if not has_weights:
            raise PreflightError(
                f"local_path for {model.repository} has no weight files (*.safetensors or *.bin): {folder}"
            )
        return folder.resolve()
    local = Path(model.repository).expanduser()
    if local.exists():
        return local.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise PreflightError("huggingface_hub is not installed") from exc
    try:
        resolved = snapshot_download(
            repo_id=model.repository,
            revision=model.revision,
            local_files_only=True,
        )
    except Exception as exc:
        raise PreflightError(
            f"Model {model.repository}@{model.revision} is not completely available in the local cache: {exc}"
        ) from exc
    return Path(resolved).resolve()


def _load_preflight_tokenizer(model: ModelConfig, model_path: Path) -> Any:
    """Load the tokenizer the same way the vLLM engine does (tokenizer_mode "auto").

    vLLM uses a model's own tokenizer format where present (e.g. Mistral's tekken.json)
    and the Hugging Face tokenizer otherwise, so the preflight checks what inference uses.
    """
    try:
        from vllm.tokenizers import get_tokenizer
        return get_tokenizer(str(model_path), revision=model.revision)
    except Exception as exc:
        raise PreflightError(f"Tokenizer preflight failed for {model.repository}: {exc}") from exc


def _prompt_token_count(tokenizer: Any, messages: list[dict[str, str]], model: ModelConfig) -> int:
    """Apply the chat template as inference does and count the prompt tokens."""
    try:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception as exc:
        raise PreflightError(f"Chat template could not be applied for {model.repository}: {exc}") from exc
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if not encoded:
        raise PreflightError(f"Chat template produced no tokens for {model.repository}")
    return len(encoded)


def model_asset_preflight(
    model: ModelConfig,
    devices: tuple[DeviceInput, ...],
    *,
    generation_max_tokens: int,
    phase: str,
) -> ModelAssetPreflight:
    if model.repository.startswith("mock://"):
        if phase != "smoke_test":
            raise PreflightError("Mock model is permitted only for smoke_test")
        return ModelAssetPreflight(model.repository, True, True, 1, True, {"kind": "mock"})

    model_path = _resolve_cached_model(model)
    tokenizer = _load_preflight_tokenizer(model, model_path)

    from .prompt import build_chat_messages
    token_counts = [
        _prompt_token_count(tokenizer, build_chat_messages(device.description), model)
        for device in devices
    ]
    maximum = max(token_counts, default=0)
    if maximum + generation_max_tokens > model.max_model_len:
        raise PreflightError(
            f"Prompt plus max output ({maximum}+{generation_max_tokens}) exceeds "
            f"max_model_len={model.max_model_len}"
        )

    try:
        import torch
    except ImportError as exc:
        raise PreflightError("PyTorch is unavailable; GPU visibility cannot be checked") from exc
    if not torch.cuda.is_available():
        raise PreflightError("No CUDA GPU is visible to PyTorch")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    weight_bytes = sum(
        path.stat().st_size
        for pattern in ("*.safetensors", "*.bin")
        for path in model_path.glob(pattern)
        if path.is_file()
    )
    estimate = {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "model_weight_file_bytes": weight_bytes,
        "plausibly_sufficient_for_weights_only": weight_bytes == 0 or free_bytes >= weight_bytes,
        "warning": "Weights-only comparison is an estimate; runtime/KV-cache allocations may require more memory.",
    }
    return ModelAssetPreflight(str(model_path), True, True, maximum, True, estimate)


def smoke_generation_preflight(
    engine: InferenceEngine,
    model: ModelConfig,
    config: StudyConfig,
) -> dict[str, Any]:
    """Generate only from a built-in synthetic case, never from final observations."""
    records = generate_repeat(
        engine,
        model,
        [PREFLIGHT_DEVICE],
        config.generation,
        0,
    )
    if len(records) != 1 or not str(records[0].get("raw_response", "")).strip():
        raise PreflightError(f"Smoke generation returned no usable text for {model.repository}")
    return {
        "device_id": PREFLIGHT_DEVICE.device_id,
        "nonempty_response": True,
        "parse_ok": bool(records[0].get("parse_ok")),
    }
