import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
_PROCESSOR = None


def _load_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        name = "facebook/dinov2-base"   # 768-мерный
        _PROCESSOR = AutoImageProcessor.from_pretrained(name)
        _MODEL = AutoModel.from_pretrained(name).to(_DEVICE).eval()
    return _MODEL, _PROCESSOR


@torch.no_grad()
def frame_dino_signature(frame_bgr):
    model, processor = _load_model()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt").to(_DEVICE)
    outputs = model(**inputs)
    # CLS-токен — глобальное представление кадра
    emb = outputs.last_hidden_state[:, 0]
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def shot_dino_signature(video_path, shots, frame_step=30):
    """Возвращает СПИСОК эмбеддингов на шот (не усредняет)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result = []

    for shot in shots:
        start_f = max(0, int(round(shot["start_time"] * fps)))
        end_f = min(total_frames, int(round(shot["end_time"] * fps)))

        L = end_f - start_f
        if L <= 0:
            embs = []
        else:
            # 3 кадра: 25%, 50%, 75%
            positions = [start_f + L // 4, start_f + L // 2, start_f + 3 * L // 4]
            embs = []
            for f in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if not ret:
                    continue
                embs.append(frame_dino_signature(frame))

        result.append({**shot, "signature": embs})   # СПИСОК, не вектор

    cap.release()
    return result