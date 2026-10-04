from src.histogram_method.cut_histogrm import shot_hsv_signature
from src.histogram_method.clip_signature import shot_clip_signature
from src.histogram_method.dino_signature import shot_dino_signature


def shot_signature(video_path, shots, method="hsv", **kwargs):
    if method == "hsv":
        return shot_hsv_signature(video_path, shots, **kwargs)
    elif method == "clip":
        return shot_clip_signature(video_path, shots, **kwargs)
    elif method == "dino":
        return shot_dino_signature(video_path, shots, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")