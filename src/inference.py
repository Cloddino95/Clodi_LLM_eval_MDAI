"""Reusable local inference implementation shared by all entry points.
-> is the only model-running implementation. vLLM turns the chat messages into local GPU generation.
A fake deterministic engine exists only for smoke tests."""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence, TYPE_CHECKING

from .config import GenerationConfig, ModelConfig
from .prompt import SYSTEM_PROMPT, build_chat_messages, parse_response

if TYPE_CHECKING:
    import logging
    from .results import ResultStore


def enforce_offline_environment() -> None:
    """Block Hugging Face/Transformers network access before importing vLLM."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


enforce_offline_environment()


@dataclass(frozen=True)
class DeviceInput:
    device_id: str
    description: str


@dataclass(frozen=True)
class GeneratedText:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None = None


class InferenceEngine(Protocol):
    def generate(
        self,
        messages: Sequence[list[dict[str, str]]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> list[GeneratedText]: ...


class VLLMEngine:
    """Thin adapter around vLLM's offline batched inference API."""

    def __init__(self, model: ModelConfig):
        enforce_offline_environment()
        from vllm import LLM

        self._model = model
        llm_kwargs = dict(
            model=model.local_path or model.repository,
            revision=model.revision,
            quantization=model.quantization,
            max_model_len=model.max_model_len,
            gpu_memory_utilization=model.gpu_memory_utilization,
            tensor_parallel_size=model.tensor_parallel_size,
        )
        if model.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = model.max_num_seqs
        if model.gdn_prefill_backend is not None:
            llm_kwargs["gdn_prefill_backend"] = model.gdn_prefill_backend
        self._llm = LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

    def generate(
        self,
        messages: Sequence[list[dict[str, str]]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> list[GeneratedText]:
        from vllm import SamplingParams

        prompts = [
            self._tokenizer.apply_chat_template(
                item, tokenize=False, add_generation_prompt=True
            )
            for item in messages
        ]
        outputs = self._llm.generate(
            prompts,
            SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed),
        )
        generated: list[GeneratedText] = []
        for output in outputs:
            completion = output.outputs[0]
            generated.append(
                GeneratedText(
                    text=completion.text,
                    prompt_tokens=len(output.prompt_token_ids) if output.prompt_token_ids else None,
                    completion_tokens=len(completion.token_ids) if completion.token_ids else None,
                    finish_reason=getattr(completion, "finish_reason", None),
                )
            )
        return generated


class DeterministicMockEngine:
    """Cheap deterministic engine for software tests; never valid for final runs."""

    def generate(
        self,
        messages: Sequence[list[dict[str, str]]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> list[GeneratedText]:
        payload = json.dumps({
            "classification": "NON_MEDICAL_DEVICE",
            "rationale": "Synthetic smoke-test response.",
            "applicable_regulation": "NOT_APPLICABLE",
            "applicable_provision": "NOT_APPLICABLE",
            "missing_information": [],
        })
        return [GeneratedText(payload, 12, 20, "stop") for _ in messages]


def create_engine(model: ModelConfig, phase: str) -> InferenceEngine:
    if model.repository.startswith("mock://"):
        if phase != "smoke_test":
            raise RuntimeError("Mock inference is permitted only in smoke_test phase")
        return DeterministicMockEngine()
    return VLLMEngine(model)


def generate_repeat(
    engine: InferenceEngine,
    model: ModelConfig,
    devices: Sequence[DeviceInput],
    generation: GenerationConfig,
    repeat_index: int,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, object]]:
    """Generate one deterministic repeat for a batch of device descriptions."""
    seed = generation.seeds[repeat_index]
    messages = [build_chat_messages(device.description, system_prompt) for device in devices]
    started = time.monotonic()
    outputs = engine.generate(
        messages,
        temperature=generation.temperature,
        max_tokens=generation.max_tokens,
        seed=seed,
    )
    elapsed = time.monotonic() - started
    if len(outputs) != len(devices):
        raise RuntimeError(f"Engine returned {len(outputs)} outputs for {len(devices)} prompts")
    per_item_latency = elapsed / max(len(outputs), 1)
    records: list[dict[str, object]] = []
    for device, output in zip(devices, outputs):
        parsed = parse_response(output.text)
        response_truncated = str(output.finish_reason or "").lower() in {"length", "max_tokens"}
        validation_issues = list(parsed.get("validation_issues", []))
        if response_truncated:
            validation_issues.append("response_truncated")
        records.append({
            "model": model.repository,
            "model_revision": model.revision,
            "device_id": device.device_id,
            "run_index": repeat_index,
            "temperature": generation.temperature,
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "latency_seconds": per_item_latency,
            "prompt_tokens": output.prompt_tokens,
            "completion_tokens": output.completion_tokens,
            "finish_reason": output.finish_reason,
            "response_truncated": response_truncated,
            "raw_response": output.text,
            **parsed,
            "validation_issues": validation_issues,
        })
    return records


def run_model(
    engine: InferenceEngine,
    model: ModelConfig,
    devices: Sequence[DeviceInput],
    generation: GenerationConfig,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, object]]:
    """Run all configured repeats using the shared inference implementation."""
    records: list[dict[str, object]] = []
    for repeat_index in range(generation.repeats):
        records.extend(
            generate_repeat(engine, model, devices, generation, repeat_index, system_prompt=system_prompt)
        )
    return records


def run_model_incremental(
    engine: InferenceEngine,
    model: ModelConfig,
    devices: Sequence[DeviceInput],
    generation: GenerationConfig,
    store: "ResultStore",
    *,
    logger: "logging.Logger | None" = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> int:
    """Persist only missing generations, one durable repeat batch at a time."""
    from .results import GenerationKey

    completed_count = 0
    for repeat_index in range(generation.repeats):
        missing_devices = [
            device for device in devices
            if GenerationKey(model.repository, model.revision, device.device_id, repeat_index)
            not in store.completed
        ]
        if not missing_devices:
            if logger:
                logger.info("repeat %s already complete for %s", repeat_index, model.repository)
            continue
        if logger:
            logger.info(
                "generating model=%s repeat=%s missing_devices=%s",
                model.repository,
                repeat_index,
                len(missing_devices),
            )
        records = generate_repeat(
            engine,
            model,
            missing_devices,
            generation,
            repeat_index,
            system_prompt=system_prompt,
        )
        store.append_batch(records)
        completed_count += len(records)
    return completed_count
