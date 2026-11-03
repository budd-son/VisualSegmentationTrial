import numpy as np
import cv2
import torch
from pathlib import Path

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ МОДЕЛЕЙ ---
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_DINO_MODEL = None
_DINO_PROCESSOR = None


def _extract_frames(video_path, shot, frame_step=1):
    """Извлекает кадры из шота с заданным шагом."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(shot["start_time"] * fps)
    end_frame = int(shot["end_time"] * fps)

    frames = []
    for f in range(start_frame, end_frame, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


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


def shot_hsv_signature(video_path, shots, scale=0.5, frame_step=5, h_bins=8, s_bins=8, v_bins=8):
    """Извлекает цветовые гистограммы HSV для каждого шота."""
    result = []
    for shot in shots:
        frames = _extract_frames(video_path, shot, frame_step=frame_step)
        if not frames:
            result.append({**shot, "signature": None})
            continue

        n = len(frames)
        idxs = [min(int(n * 0.25), n - 1), min(int(n * 0.5), n - 1), min(int(n * 0.75), n - 1)]

        histograms = []
        for i in idxs:
            frame = frames[i]
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)

        result.append({**shot, "signature": histograms})
    return result


def shot_clip_signature(video_path, shots, frame_step=10):
    """Извлекает эмбеддинги CLIP (модель загружается один раз)."""
    from PIL import Image

    model, preprocess = _get_clip_model()

    result = []
    for shot in shots:
        frames = _extract_frames(video_path, shot, frame_step=frame_step)
        if not frames:
            result.append({**shot, "signature": None})
            continue

        n = len(frames)
        idxs = [min(int(n * 0.25), n - 1), min(int(n * 0.5), n - 1), min(int(n * 0.75), n - 1)]

        embeddings = []
        with torch.no_grad():
            for i in idxs:
                rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                tensor = preprocess(img).unsqueeze(0)
                emb = model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy().flatten())

        result.append({**shot, "signature": embeddings})
    return result


def shot_dino_signature(video_path, shots, frame_step=10):
    """Извлекает эмбеддинги DINOv2 (модель загружается один раз)."""
    from PIL import Image

    model, processor = _get_dino_model()

    result = []
    for shot in shots:
        frames = _extract_frames(video_path, shot, frame_step=frame_step)
        if not frames:
            result.append({**shot, "signature": None})
            continue

        n = len(frames)
        idxs = [min(int(n * 0.25), n - 1), min(int(n * 0.5), n - 1), min(int(n * 0.75), n - 1)]

        embeddings = []
        with torch.no_grad():
            for i in idxs:
                rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                inputs = processor(images=img, return_tensors="pt")
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy().flatten())

        result.append({**shot, "signature": embeddings})
    return result


def shot_signature(video_path, shots, method="hsv", **kwargs):
    """Главная точка входа: выбирает нужный метод извлечения признаков."""
    if method == "hsv":
        return shot_hsv_signature(video_path, shots, **kwargs)
    elif method == "clip":
        return shot_clip_signature(video_path, shots, **kwargs)
    elif method == "dino":
        return shot_dino_signature(video_path, shots, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")