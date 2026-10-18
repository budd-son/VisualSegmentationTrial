import csv
import logging
import av
import argparse

from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEECH_DIR = PROJECT_ROOT / "data" / "speech"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
_model = None  # (model_size, WhisperModel)


import numpy as np


def _decode_audio(path, sr=16000):

    chunks = []
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)

    def _collect(out):
        if out is None:
            return
        for f in (out if isinstance(out, list) else [out]):
            chunks.append(f.to_ndarray().reshape(-1))

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            _collect(resampler.resample(frame))
        _collect(resampler.resample(None))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _transcribe(media_path, model_size):
    global _model
    from faster_whisper import WhisperModel
    if _model is None or _model[0] != model_size:
        _model = (model_size, WhisperModel(model_size, device="cpu", compute_type="int8"))
    audio = _decode_audio(media_path)
    segments, _ = _model[1].transcribe(audio, language="ru",
                                       vad_filter=True, beam_size=5)
    return [(s.start, s.end) for s in segments]
def speech_path(video, speech_dir=SPEECH_DIR):
    return Path(speech_dir) / f"{Path(video).stem}_speech.csv"


def _load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(float(r["start_time"]), float(r["end_time"])) for r in csv.DictReader(f)]


def _save(intervals, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_time", "end_time"])
        w.writerows(intervals)


def get_speech(video_path, model_size="small", speech_dir=SPEECH_DIR, force=False):
    path = speech_path(video_path, speech_dir)
    if path.exists() and not force:
        return _load(path)
    logger.info("Whisper (%s): %s", model_size, video_path)
    intervals = _transcribe(video_path, model_size)
    _save(intervals, path)                           # пустой результат тоже кэшируем
    logger.info("speech: %d реплик -> %s", len(intervals), path)
    return intervals


def crosses_speech(t, intervals, margin=0.3):
    return any(s + margin < t < e - margin for s, e in intervals)


def merge_scenes_through_speech(scenes, intervals, margin=0.3):
    out = []
    for sc in scenes:
        if out and crosses_speech(float(sc["start_time"]), intervals, margin):
            prev = out[-1]
            prev["end_time"] = sc["end_time"]
            prev["cuts"] = int(float(prev.get("cuts", 0))) + int(float(sc.get("cuts", 0)))
            if "duration" in prev:
                prev["duration"] = float(prev["end_time"]) - float(prev["start_time"])
        else:
            out.append(dict(sc))
    for i, sc in enumerate(out):
        sc["scene_id"] = i
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(PROJECT_ROOT / "data" / "raw"))
    ap.add_argument("--model", default="small")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    for v in sorted(Path(a.raw).iterdir()):
        if v.suffix.lower() in VIDEO_EXTS:
            print(v.name, len(get_speech(v, a.model, force=a.force)), "реплик")