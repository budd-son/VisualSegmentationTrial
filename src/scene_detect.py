import csv
import logging
from pathlib import Path
from typing import List, Dict

from scenedetect import detect, ContentDetector

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "shots"


def _cache_path(video_path: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{Path(video_path).stem}_shots.csv"


def _load_cache(path: Path) -> List[Dict[str, float]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [
            {"start_time": float(r["start_time"]), "end_time": float(r["end_time"])}
            for r in reader
        ]


def _save_cache(path: Path, scene_data: List[Dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start_time", "end_time"])
        writer.writeheader()
        writer.writerows(scene_data)


def detect_scenes(
    video_path: str,
    threshold: float = 30.0,
    force: bool = False,
) -> List[Dict[str, float]]:
    cache = _cache_path(video_path)

    if cache.exists() and not force:
        logger.info("load scenes from cache %s", cache)
        return _load_cache(cache)

    logger.info("detect scenes for %s", video_path)
    scenes = detect(video_path, ContentDetector(threshold=threshold))
    scene_data = []
    for i in scenes:
        try:
            start = float(i[0].seconds) if hasattr(i[0], "seconds") else float(i[0])
            end = float(i[1].seconds) if hasattr(i[1], "seconds") else float(i[1])
        except Exception:
            start, end = float(i[0]), float(i[1])
        scene_data.append({"start_time": start, "end_time": end})

    _save_cache(cache, scene_data)
    logger.info("detected %s scenes → %s", len(scene_data), cache)
    return scene_data