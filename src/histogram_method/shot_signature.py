import numpy as np
import cv2
import torch
from pathlib import Path

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ МОДЕЛЕЙ ---
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_DINO_MODEL = None
_DINO_PROCESSOR = None


def _get_clip_model():
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None:
        import open_clip
        print("[CLIP] Загрузка модели (один раз)...")
        _CLIP_MODEL, _, _CLIP_PREPROCESS = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        _CLIP_MODEL = _CLIP_MODEL.to("cpu").eval()
    return _CLIP_MODEL, _CLIP_PREPROCESS


def _get_dino_model():
    global _DINO_MODEL, _DINO_PROCESSOR
    if _DINO_MODEL is None:
        from transformers import AutoImageProcessor, AutoModel
        print("[DINO] Загрузка модели (один раз)...")
        _DINO_PROCESSOR = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _DINO_MODEL = AutoModel.from_pretrained("facebook/dinov2-base").to("cpu").eval()
    return _DINO_MODEL, _DINO_PROCESSOR


def _extract_three_frames(video_path, shots):
    """
    Открывает видео ОДИН РАЗ и читает только 3 кадра для каждого шота.
    Возвращает {shot_index: [frame1, frame2, frame3]}.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Не открылось: {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    result = {}
    for idx, shot in enumerate(shots):
        start_f = int(shot["start_time"] * fps)
        end_f = int(shot["end_time"] * fps)

        # Берем 25%, 50%, 75% от шота
        f25 = min(start_f + int((end_f - start_f) * 0.25), total_frames - 1)
        f50 = min(start_f + int((end_f - start_f) * 0.50), total_frames - 1)
        f75 = min(start_f + int((end_f - start_f) * 0.75), total_frames - 1)

        frames = []
        for f in [f25, f50, f75]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                frames.append(None)

        result[idx] = frames

    cap.release()
    return result


def shot_hsv_signature(video_path, shots, scale=0.5, frame_step=5, h_bins=8, s_bins=8, v_bins=8):
    """Извлекает цветовые гистограммы HSV (3 кадра на шот, видео открывается 1 раз)."""
    frames_map = _extract_three_frames(video_path, shots)

    result = []
    for idx, shot in enumerate(shots):
        frames = frames_map.get(idx, [])
        valid_frames = [f for f in frames if f is not None]

        if not valid_frames:
            result.append({**shot, "signature": None})
            continue

        histograms = []
        for frame in valid_frames:
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
            histograms.append(hist)

        result.append({**shot, "signature": histograms})
    return result


def shot_clip_signature(video_path, shots, frame_step=10):
    """Извлекает эмбеддинги CLIP (модель загружается один раз, видео открывается 1 раз)."""
    from PIL import Image

    model, preprocess = _get_clip_model()
    frames_map = _extract_three_frames(video_path, shots)

    result = []
    for idx, shot in enumerate(shots):
        frames = frames_map.get(idx, [])
        valid_frames = [f for f in frames if f is not None]

        if not valid_frames:
            result.append({**shot, "signature": None})
            continue

        embeddings = []
        with torch.no_grad():
            for frame in valid_frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                tensor = preprocess(img).unsqueeze(0)
                emb = model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy().flatten())

        result.append({**shot, "signature": embeddings})
    return result


def shot_dino_signature(video_path, shots, frame_step=10):
    """Извлекает эмбеддинги DINOv2 (модель загружается один раз, видео открывается 1 раз)."""
    from PIL import Image

    model, processor = _get_dino_model()
    frames_map = _extract_three_frames(video_path, shots)

    result = []
    for idx, shot in enumerate(shots):
        frames = frames_map.get(idx, [])
        valid_frames = [f for f in frames if f is not None]

        if not valid_frames:
            result.append({**shot, "signature": None})
            continue

        embeddings = []
        with torch.no_grad():
            for frame in valid_frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                inputs = processor(images=img, return_tensors="pt")
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy().flatten())

        result.append({**shot, "signature": embeddings})
    return result


def shot_signature(video_path, shots, method="hsv", **kwargs):
    if method == "hsv":
        return shot_hsv_signature(video_path, shots, **kwargs)
    elif method == "clip":
        return shot_clip_signature(video_path, shots, **kwargs)
    elif method == "dino":
        return shot_dino_signature(video_path, shots, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")