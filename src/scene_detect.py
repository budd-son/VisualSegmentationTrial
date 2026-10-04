from scenedetect import detect, ContentDetector
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def detect_scenes(video_path: str, threshold: float = 30.0) -> List[Tuple[float, float]]:
    logger.info("detect scenes for %s", video_path)
    path = video_path
    scene_data = []
    scenes = detect(path, ContentDetector())
    for i in scenes:
        try:
            start = float(i[0].seconds) if hasattr(i[0],"seconds") else float(i[0])
            end = float(i[1].seconds) if hasattr(i[1], "seconds") else float(i[1])

        except Exception as e:
            start, end = float(i[0]), float(i[1])
        scene_data.append({
            "start_time": start,
            "end_time": end
        })
    logger.info("detected %s scenes", len(scene_data))
    return scene_data
