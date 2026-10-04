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


def shot_dino_signature(video_path, shots,
                        frame_step=15, scale=1.0):
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
            embs.append(frame_dino_signature(frame))

        if embs:
            signature = np.mean(np.stack(embs, axis=0), axis=0)
            # повторно нормируем после усреднения
            n = np.linalg.norm(signature)
            if n > 1e-9:
                signature = signature / n
        else:
            signature = np.zeros(768, dtype=np.float32)

        result.append({**shot, "signature": signature})

    cap.release()
    return result