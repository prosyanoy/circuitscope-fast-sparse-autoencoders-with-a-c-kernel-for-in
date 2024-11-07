"""Run configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, '#' comments; real environment variables win."""
    f = Path(path)
    if not f.is_file():
        return
    for line in f.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw.split("#")[0].strip() == "":
        return default
    return cast(raw.split("#")[0].strip())


@dataclass
class Config:
    out_dir: str = "artifacts"
    seed: int = 0
    seq_len: int = 24
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    model_steps: int = 1200
    model_batch: int = 256
    hook_layer: int = 1  # residual stream after this many blocks
    n_features: int = 512
    k: int = 8
    sae_steps: int = 3000
    sae_batch: int = 2048
    sae_lr: float = 2e-3
    sae_backend: str = "cpp"
    n_sae_seqs: int = 12000  # sequences used to harvest SAE training activations
    n_eval_seqs: int = 3000

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        load_dotenv()
        cfg = cls()
        for f in fields(cls):
            setattr(cfg, f.name, _env("CS_" + f.name.upper(), getattr(cfg, f.name), type(getattr(cfg, f.name))))
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg

    @property
    def out(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
