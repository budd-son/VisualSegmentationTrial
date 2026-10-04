from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class SignatureConfig:
    scale: float = 0.5
    frame_step: int = 5
    h_bins: int = 8
    s_bins: int = 8
    v_bins: int = 8


@dataclass
class GroupingConfig:
    threshold: float = 0.6
    lookback: int = 5
    min_duration: float = 2.0


@dataclass
class PathsConfig:
    raw: str = "data/raw"
    out: str = "data/result_hist"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str | None = None


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    video_exts: list = field(default_factory=lambda: [
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"
    ])
    signature: SignatureConfig = field(default_factory=SignatureConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        print(f"[config] {path} не найден, использую дефолты")
        return Config()

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = Config()
    if "paths" in data:
        cfg.paths = PathsConfig(**data["paths"])
    if "video_exts" in data:
        cfg.video_exts = data["video_exts"]
    if "signature" in data:
        cfg.signature = SignatureConfig(**data["signature"])
    if "grouping" in data:
        cfg.grouping = GroupingConfig(**data["grouping"])
    if "logging" in data:
        cfg.logging = LoggingConfig(**data["logging"])
    return cfg
