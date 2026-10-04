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


def shot_clip_signature(video_path, shots,
                        frame_step=15,   # CLIP тяжёлый, берём реже
                        scale=1.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result = []

    for shot in shots:
        start_f = max(0, int(round(shot["start_time"] * fps)))
        end_f = min(total_frames, int(round(shot["end_time"] * fps)))

        embs = []
        for f in range(start_f, end_f, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                continue
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            embs.append(frame_clip_signature(frame))

        if embs:
            signature = np.mean(np.stack(embs, axis=0), axis=0)
            # повторно нормируем после усреднения
            n = np.linalg.norm(signature)
            if n > 1e-9:
                signature = signature / n
        else:
            signature = np.zeros(512, dtype=np.float32)

        result.append({**shot, "signature": signature})

    cap.release()
    return result