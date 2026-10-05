import cv2
import numpy as np
import torch
import open_clip
from PIL import Image


_DEVICE =  "cpu"
_MODEL = None
_PREPROCESS = None


def _load_model():
    global _MODEL, _PREPROCESS
    if _MODEL is None:
        # ViT-B/32 — быстрый, 512-мерный
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = model.to(_DEVICE).eval()
        _MODEL = model
        _PREPROCESS = preprocess
    return _MODEL, _PREPROCESS


@torch.no_grad()
def frame_clip_signature(frame_bgr):
    """Возвращает L2-нормализованный эмбеддинг CLIP для одного кадра."""
    model, preprocess = _load_model()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = preprocess(pil).unsqueeze(0).to(_DEVICE)
    emb = model.encode_image(x)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def shot_clip_signature(video_path, shots, frame_step=30):
    """Возвращает СПИСОК эмбеддингов на шот (не усредняет)."""
    cap = cv2.VideoCapture(str(video_path))
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
                embs.append(frame_clip_signature(frame))

        result.append({**shot, "signature": embs})   # список, не вектор

    cap.release()
    return result