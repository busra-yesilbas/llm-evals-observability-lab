"""
Configuration management for LLM Evals Lab.

Loads and merges YAML configs with environment variable overrides.
All path resolution is relative to the project root, not the calling script.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Resolve project root as the directory containing this package's src/
_PACKAGE_DIR = Path(__file__).parent
_PROJECT_ROOT = _PACKAGE_DIR.parents[1]  # src/llm_evals_lab -> src -> project root


def _find_project_root() -> Path:
    """Walk upward from this file to find the project root (contains pyproject.toml)."""
    candidate = _PACKAGE_DIR
    for _ in range(6):
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    return _PROJECT_ROOT


PROJECT_ROOT: Path = _find_project_root()
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict if the file doesn't exist."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class LabConfig:
    """
    Unified configuration object for the lab.

    Provides attribute-style access and convenient path resolution.
    """

    def __init__(self, raw: dict[str, Any], project_root: Path | None = None) -> None:
        self._raw = raw
        self.project_root = project_root or PROJECT_ROOT

    # ── Convenience accessors ────────────────────────────────────────────────

    @property
    def app(self) -> dict[str, Any]:
        return self._raw.get("app", {})

    @property
    def paths(self) -> dict[str, Any]:
        return self._raw.get("paths", {})

    @property
    def retrieval(self) -> dict[str, Any]:
        return self._raw.get("retrieval", {})

    @property
    def chunking(self) -> dict[str, Any]:
        return self._raw.get("chunking", {})

    @property
    def eval(self) -> dict[str, Any]:
        return self._raw.get("eval", {})

    @property
    def generation(self) -> dict[str, Any]:
        return self._raw.get("generation", {})

    @property
    def observability(self) -> dict[str, Any]:
        return self._raw.get("observability", {})

    # ── Path helpers ─────────────────────────────────────────────────────────

    def resolve(self, relative: str) -> Path:
        """Resolve a relative path string against the project root."""
        return self.project_root / relative

    def data_dir(self) -> Path:
        return self.resolve(self.paths.get("data_dir", "data"))

    def raw_dir(self) -> Path:
        return self.resolve(self.paths.get("raw_dir", "data/raw"))

    def processed_dir(self) -> Path:
        return self.resolve(self.paths.get("processed_dir", "data/processed"))

    def eval_dir(self) -> Path:
        return self.resolve(self.paths.get("eval_dir", "data/eval"))

    def results_dir(self) -> Path:
        return self.resolve(self.paths.get("results_dir", "results"))

    def runs_dir(self) -> Path:
        return self.resolve(self.paths.get("runs_dir", "results/runs"))

    def tables_dir(self) -> Path:
        return self.resolve(self.paths.get("tables_dir", "results/tables"))

    def figures_dir(self) -> Path:
        return self.resolve(self.paths.get("figures_dir", "results/figures"))

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def __repr__(self) -> str:
        return f"LabConfig(root={self.project_root})"


def load_config(
    app_config: str | Path | None = None,
    eval_config: str | Path | None = None,
    retrieval_config: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> LabConfig:
    """
    Load and merge all configuration files.

    Parameters
    ----------
    app_config:
        Path to app.yaml. Defaults to configs/app.yaml.
    eval_config:
        Path to eval.yaml. Defaults to configs/eval.yaml.
    retrieval_config:
        Path to retrieval.yaml. Defaults to configs/retrieval.yaml.
    overrides:
        Optional dict of additional overrides applied last.

    Returns
    -------
    LabConfig
        Merged configuration object.
    """
    app_path = Path(app_config) if app_config else CONFIGS_DIR / "app.yaml"
    eval_path = Path(eval_config) if eval_config else CONFIGS_DIR / "eval.yaml"
    retrieval_path = Path(retrieval_config) if retrieval_config else CONFIGS_DIR / "retrieval.yaml"

    raw: dict[str, Any] = {}
    for path in [app_path, eval_path, retrieval_path]:
        raw = _deep_merge(raw, _load_yaml(path))

    # Apply environment variable overrides
    env_overrides = _env_overrides()
    raw = _deep_merge(raw, env_overrides)

    if overrides:
        raw = _deep_merge(raw, overrides)

    return LabConfig(raw, project_root=PROJECT_ROOT)


def _env_overrides() -> dict[str, Any]:
    """Extract supported environment variable overrides into a config dict."""
    overrides: dict[str, Any] = {}

    embedding_backend = os.getenv("EMBEDDING_BACKEND")
    if embedding_backend:
        overrides.setdefault("retrieval", {})["embedding_backend"] = embedding_backend

    generation_backend = os.getenv("GENERATION_BACKEND")
    if generation_backend:
        overrides.setdefault("generation", {})["backend"] = generation_backend

    log_level = os.getenv("LOG_LEVEL")
    if log_level:
        overrides.setdefault("logging", {})["level"] = log_level

    results_dir = os.getenv("RESULTS_DIR")
    if results_dir:
        overrides.setdefault("paths", {})["results_dir"] = results_dir

    return overrides
