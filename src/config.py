"""Strict YAML configuration contract for study execution.
--> rejects incomplete or surprising configuration fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .paths import resolve_config_path


ALLOWED_PHASES = {"smoke_test", "development", "final"}


class ConfigError(ValueError):
    """Raised when a study configuration is incomplete or invalid."""


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    phase: str
    name: str
    development_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ReferenceLabelsConfig:
    path: Path | None


@dataclass(frozen=True)
class ModelConfig:
    repository: str
    revision: str
    quantization: str | None = None
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    local_path: str | None = None
    tensor_parallel_size: int = 1
    max_num_seqs: int | None = None
    gdn_prefill_backend: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    repeats: int
    temperature: float
    seeds: tuple[int, ...]
    max_tokens: int


@dataclass(frozen=True)
class OutputsConfig:
    root: Path


@dataclass(frozen=True)
class AnalysisConfig:
    enabled: bool
    specification: Path | None = None


@dataclass(frozen=True)
class FailurePolicyConfig:
    continue_on_model_failure: bool = False


@dataclass(frozen=True)
class FinalFreezeConfig:
    prompt_sha256: str | None = None
    dataset_sha256: str | None = None
    reference_labels_sha256: str | None = None
    analysis_specification_sha256: str | None = None
    output_parser_sha256: str | None = None
    models_sha256: str | None = None
    generation_sha256: str | None = None
    code_git_commit: str | None = None


@dataclass(frozen=True)
class StudyConfig:
    source: Path
    phase: str
    study_name: str
    run_label: str
    dataset: DatasetConfig
    reference_labels: ReferenceLabelsConfig
    models: tuple[ModelConfig, ...]
    generation: GenerationConfig
    outputs: OutputsConfig
    analysis: AnalysisConfig
    failure_policy: FailurePolicyConfig = field(default_factory=FailurePolicyConfig)
    final_freeze: FinalFreezeConfig = field(default_factory=FinalFreezeConfig)


_TOP_LEVEL_KEYS = {
    "phase", "study", "dataset", "reference_labels", "models", "generation",
    "outputs", "analysis", "failure_policy", "final_freeze",
}


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be a mapping")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"Unknown field(s) in {location}: {', '.join(unknown)}")


def _required(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ConfigError(f"Missing required field: {location}.{key}")
    return mapping[key]


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be a non-empty string")
    return value.strip()


def _optional_path(value: Any, source: Path, location: str) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"{location} must be a path string or null")
    return resolve_config_path(value, source)


def validate_config_dict(raw: Mapping[str, Any], source: Path) -> StudyConfig:
    """Validate parsed YAML and return an immutable, path-resolved config."""
    _reject_unknown(raw, _TOP_LEVEL_KEYS, "configuration")

    phase = _nonempty_string(_required(raw, "phase", "configuration"), "phase")
    if phase not in ALLOWED_PHASES:
        raise ConfigError(f"phase must be one of: {', '.join(sorted(ALLOWED_PHASES))}")

    study = _mapping(_required(raw, "study", "configuration"), "study")
    _reject_unknown(study, {"name", "run_label"}, "study")
    study_name = _nonempty_string(_required(study, "name", "study"), "study.name")
    run_label = _nonempty_string(study.get("run_label", phase), "study.run_label")

    dataset = _mapping(_required(raw, "dataset", "configuration"), "dataset")
    _reject_unknown(dataset, {"path", "phase", "name", "development_paths"}, "dataset")
    dataset_phase = _nonempty_string(_required(dataset, "phase", "dataset"), "dataset.phase")
    if dataset_phase not in ALLOWED_PHASES:
        raise ConfigError("dataset.phase must be smoke_test, development, or final")
    development_values = dataset.get("development_paths", [])
    if not isinstance(development_values, list):
        raise ConfigError("dataset.development_paths must be a list")
    dataset_config = DatasetConfig(
        path=resolve_config_path(_required(dataset, "path", "dataset"), source),
        phase=dataset_phase,
        name=_nonempty_string(_required(dataset, "name", "dataset"), "dataset.name"),
        development_paths=tuple(resolve_config_path(item, source) for item in development_values),
    )
    if phase != dataset_phase:
        raise ConfigError(
            f"Configuration phase {phase!r} cannot consume dataset.phase {dataset_phase!r}"
        )

    labels = _mapping(_required(raw, "reference_labels", "configuration"), "reference_labels")
    _reject_unknown(labels, {"path"}, "reference_labels")
    labels_config = ReferenceLabelsConfig(
        path=_optional_path(labels.get("path"), source, "reference_labels.path")
    )

    model_values = _required(raw, "models", "configuration")
    if not isinstance(model_values, list) or not model_values:
        raise ConfigError("models must be a non-empty list")
    models: list[ModelConfig] = []
    for index, value in enumerate(model_values):
        location = f"models[{index}]"
        model = _mapping(value, location)
        _reject_unknown(
            model,
            {"repository", "revision", "quantization", "max_model_len", "gpu_memory_utilization",
             "local_path", "tensor_parallel_size", "max_num_seqs", "gdn_prefill_backend"},
            location,
        )
        repository = _nonempty_string(_required(model, "repository", location), f"{location}.repository")
        revision = _nonempty_string(_required(model, "revision", location), f"{location}.revision")
        max_model_len = model.get("max_model_len", 8192)
        gpu_util = model.get("gpu_memory_utilization", 0.90)
        if not isinstance(max_model_len, int) or max_model_len <= 0:
            raise ConfigError(f"{location}.max_model_len must be a positive integer")
        if not isinstance(gpu_util, (int, float)) or not 0 < float(gpu_util) <= 1:
            raise ConfigError(f"{location}.gpu_memory_utilization must be in (0, 1]")
        quantization = model.get("quantization")
        if quantization is not None:
            quantization = _nonempty_string(quantization, f"{location}.quantization")
        local_path = model.get("local_path")
        if local_path is not None:
            local_path = _nonempty_string(local_path, f"{location}.local_path")
        tp = model.get("tensor_parallel_size", 1)
        if not isinstance(tp, int) or isinstance(tp, bool) or tp < 1:
            raise ConfigError(f"{location}.tensor_parallel_size must be a positive integer")
        max_seqs = model.get("max_num_seqs")
        if max_seqs is not None and (not isinstance(max_seqs, int) or isinstance(max_seqs, bool) or max_seqs < 1):
            raise ConfigError(f"{location}.max_num_seqs must be a positive integer")
        gdn = model.get("gdn_prefill_backend")
        if gdn is not None and gdn not in ("flashinfer", "triton", "cutedsl"):
            raise ConfigError(f"{location}.gdn_prefill_backend must be flashinfer, triton or cutedsl")
        models.append(ModelConfig(repository, revision, quantization, max_model_len, float(gpu_util),
                                  local_path, tp, max_seqs, gdn))

    generation = _mapping(_required(raw, "generation", "configuration"), "generation")
    _reject_unknown(generation, {"repeats", "temperature", "seeds", "max_tokens"}, "generation")
    repeats = _required(generation, "repeats", "generation")
    temperature = _required(generation, "temperature", "generation")
    seeds = _required(generation, "seeds", "generation")
    max_tokens = _required(generation, "max_tokens", "generation")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ConfigError("generation.repeats must be a positive integer")
    if not isinstance(temperature, (int, float)) or float(temperature) < 0:
        raise ConfigError("generation.temperature must be a non-negative number")
    if not isinstance(seeds, list) or len(seeds) != repeats or any(not isinstance(x, int) for x in seeds):
        raise ConfigError("generation.seeds must contain exactly one integer per repeat")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ConfigError("generation.max_tokens must be a positive integer")
    generation_config = GenerationConfig(repeats, float(temperature), tuple(seeds), max_tokens)

    outputs = _mapping(_required(raw, "outputs", "configuration"), "outputs")
    _reject_unknown(outputs, {"root"}, "outputs")
    outputs_config = OutputsConfig(resolve_config_path(_required(outputs, "root", "outputs"), source))

    analysis = _mapping(_required(raw, "analysis", "configuration"), "analysis")
    _reject_unknown(analysis, {"enabled", "specification"}, "analysis")
    enabled = _required(analysis, "enabled", "analysis")
    if not isinstance(enabled, bool):
        raise ConfigError("analysis.enabled must be true or false")
    analysis_config = AnalysisConfig(
        enabled=enabled,
        specification=_optional_path(analysis.get("specification"), source, "analysis.specification"),
    )

    failure = _mapping(raw.get("failure_policy", {}), "failure_policy")
    _reject_unknown(failure, {"continue_on_model_failure"}, "failure_policy")
    continue_on_failure = failure.get("continue_on_model_failure", False)
    if not isinstance(continue_on_failure, bool):
        raise ConfigError("failure_policy.continue_on_model_failure must be true or false")

    freeze = _mapping(raw.get("final_freeze", {}), "final_freeze")
    freeze_keys = {
        "prompt_sha256", "dataset_sha256", "reference_labels_sha256",
        "analysis_specification_sha256", "output_parser_sha256", "models_sha256",
        "generation_sha256", "code_git_commit",
    }
    _reject_unknown(freeze, freeze_keys, "final_freeze")
    for key, value in freeze.items():
        if value is not None:
            _nonempty_string(value, f"final_freeze.{key}")

    return StudyConfig(
        source=source,
        phase=phase,
        study_name=study_name,
        run_label=run_label,
        dataset=dataset_config,
        reference_labels=labels_config,
        models=tuple(models),
        generation=generation_config,
        outputs=outputs_config,
        analysis=analysis_config,
        failure_policy=FailurePolicyConfig(continue_on_failure),
        final_freeze=FinalFreezeConfig(**freeze),
    )


def load_config(path: str | Path) -> StudyConfig:
    """Load a YAML study configuration with clear dependency errors."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"Configuration file does not exist: {source}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ConfigError("PyYAML is required: install the project requirements") from exc
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {source}: {exc}") from exc
    return validate_config_dict(_mapping(raw, "configuration"), source)
