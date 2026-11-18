"""Hashing and programmatic environment-provenance collection."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _gpu_information() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "devices": []}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip(), "devices": []}
    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            devices.append({
                "name": parts[0],
                "uuid": parts[1],
                "memory_total_mib": parts[2],
                "driver_version": parts[3],
            })
    return {"available": bool(devices), "error": None, "devices": devices}


def collect_environment(project_root: Path) -> dict[str, Any]:
    """Collect versions at runtime without importing heavyweight GPU libraries."""
    distributions = [
        "vllm", "torch", "transformers", "huggingface-hub", "pandas", "PyYAML"
    ]
    packages = {name: package_version(name) for name in distributions}
    cuda_version = None
    if packages["torch"] is not None:
        try:
            import torch

            cuda_version = torch.version.cuda
        except Exception:  # provenance must not make a run impossible
            cuda_version = None
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_version": cuda_version,
        "gpu": _gpu_information(),
        "git_commit": git_commit(project_root),
    }


def canonical_json_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(serialized)

