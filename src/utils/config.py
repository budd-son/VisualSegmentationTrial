from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class HSVConfig:
    scale: float = 0.5
    frame_step: int = 5
    h_bins: int = 8
    s_bins: int = 8
    v_bins: int = 8


@dataclass
class ClipConfig:
    frame_step: int = 30


@dataclass
class DinoConfig:
    frame_step: int = 30


@dataclass
class SignatureConfig:
    hsv: HSVConfig = field(default_factory=HSVConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    dino: DinoConfig = field(default_factory=DinoConfig)


@dataclass
class ThresholdConfig:
    hsv: float = 0.6
    clip: float = 0.75
    dino: float = 0.75


@dataclass
class GroupingConfig:
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    lookback: int = 5
    min_duration: float = 5.0
    k: float = 1.0


@dataclass
class PathsConfig:
    raw: str = "data/raw"



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


def _build(dc, data):
    """Рекурсивно строит dataclass из словаря."""
    if not isinstance(data, dict):
        return data
    fields = dc.__dataclass_fields__
    kwargs = {}
    for k, v in data.items():
        if k in fields:
            ftype = fields[k].type
            if hasattr(ftype, "__dataclass_fields__"):
                kwargs[k] = _build(ftype, v)
            else:
                kwargs[k] = v
    return dc(**kwargs)


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
        cfg.signature = _build(SignatureConfig, data["signature"])
    if "grouping" in data:
        cfg.grouping = _build(GroupingConfig, data["grouping"])
    if "logging" in data:
        cfg.logging = LoggingConfig(**data["logging"])
    return cfg