"""Configuration loading from YAML file and environment variables."""

import os
from pathlib import Path
from dataclasses import dataclass, field

import yaml


@dataclass
class OCRConfig:
    model: str = "qwen3.7-plus"
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass
class LayoutConfig:
    binarize: bool = True
    sharpen: bool = True
    sharpen_amount: float = 1.5
    deskew: bool = False


@dataclass
class PipelineConfig:
    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    output_dir: str = "out"
    debug: bool = False
    max_workers: int = 2


def _env_override(cfg: PipelineConfig) -> PipelineConfig:
    """Apply environment variable overrides to config."""
    if os.environ.get("GUJI_LLM_BASE_URL"):
        cfg.llm.base_url = os.environ["GUJI_LLM_BASE_URL"]
    if os.environ.get("GUJI_LLM_API_KEY"):
        cfg.llm.api_key = os.environ["GUJI_LLM_API_KEY"]
    if os.environ.get("GUJI_LLM_MODEL"):
        cfg.llm.model = os.environ["GUJI_LLM_MODEL"]
    return cfg


def load_config(config_path: str = None) -> PipelineConfig:
    """Load pipeline config from YAML file, falling back to defaults + env.

    Search order:
    1. Explicit config_path argument
    2. GUJI_CONFIG env var
    3. ./config.yaml (relative to cwd)
    4. Defaults
    """
    cfg = PipelineConfig()

    path = config_path or os.environ.get("GUJI_CONFIG") or "config.yaml"
    yaml_path = Path(path)

    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "ocr" in raw:
            for k, v in raw["ocr"].items():
                if hasattr(cfg.ocr, k):
                    setattr(cfg.ocr, k, v)
        if "llm" in raw:
            for k, v in raw["llm"].items():
                if hasattr(cfg.llm, k):
                    setattr(cfg.llm, k, v)
        if "layout" in raw:
            for k, v in raw["layout"].items():
                if hasattr(cfg.layout, k):
                    setattr(cfg.layout, k, v)
        if "output_dir" in raw:
            cfg.output_dir = raw["output_dir"]
        if "debug" in raw:
            cfg.debug = raw["debug"]
        if "max_workers" in raw:
            cfg.max_workers = raw["max_workers"]

    cfg = _env_override(cfg)
    return cfg
