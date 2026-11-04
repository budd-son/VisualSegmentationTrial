import csv
import torch
import numpy as np
from pathlib import Path
from src.utils.io_util import ensure_dir

DATA_DIR = Path("data")
SPEECH_DIR = DATA_DIR / "speech"

# --- Глобальная модель VAD (загружается один раз) ---
_VAD_MODEL = None


def _get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        print("[VAD] Загрузка Silero VAD...")
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False)
        _VAD_MODEL = model
    return _VAD_MODEL


def _extract_audio(video_path):
    """Извлекает аудио из видео через ffmpeg (во временный wav)."""
    import subprocess
    import tempfile

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp_wav


def get_speech(video_path):
    """
    Возвращает список интервалов речи [(start, end), ...] в секундах.
    Использует Silero VAD (быстро, без распознавания текста).
    """
    stem = Path(video_path).stem
    csv_path = SPEECH_DIR / f"{stem}_speech.csv"

    # Если кэш есть — читаем
    if csv_path.exists():
        intervals = []
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                intervals.append((float(r["start_time"]), float(r["end_time"])))
        return intervals

    # Иначе — извлекаем через VAD
    print(f"[VAD] Анализ речи: {stem}")
    model = _get_vad_model()

    # Извлекаем аудио
    wav_path = _extract_audio(video_path)

    # Читаем wav
    import wave
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    # Прогоняем через VAD
    speech_probs = model(torch.from_numpy(audio), sr).numpy()
    # Если VAD возвращает покадровые вероятности, нужно найти интервалы > 0.5
    # (зависит от версии silero-vad, иногда он возвращает сразу интервалы)

    # Упрощенный вариант: считаем, что интервал речи — это где prob > 0.5
    # Шаг VAD = 30 мс (для 16kHz)
    chunk_size = int(0.03 * sr)
    intervals = []
    in_speech = False
    start = 0.0

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            break
        prob = model(torch.from_numpy(chunk).unsqueeze(0), sr).item()

        t = i / sr
        if prob > 0.5 and not in_speech:
            in_speech = True
            start = t
        elif prob <= 0.5 and in_speech:
            in_speech = False
            intervals.append((start, t))

    if in_speech:
        intervals.append((start, len(audio) / sr))

    # Сохраняем в кэш
    ensure_dir(SPEECH_DIR)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start_time", "end_time"])
        writer.writeheader()
        for s, e in intervals:
            writer.writerow({"start_time": s, "end_time": e})

    # Удаляем временный wav
    Path(wav_path).unlink(missing_ok=True)

    print(f"[VAD] {stem}: {len(intervals)} интервалов речи")
    return intervals


def merge_scenes_through_speech(scenes, speech_intervals, penalty=0.5):
    """
    Штрафует границы, которые пересекают речь.
    Возвращает сцены, где границы, попавшие в речь, сдвинуты или удалены.
    """
    if not speech_intervals:
        return scenes

    filtered = []
    for scene in scenes:
        start = scene["start_time"]
        end = scene["end_time"]

        # Проверяем, пересекает ли граница речь
        crosses = False
        for s, e in speech_intervals:
            if s < start < e:
                crosses = True
                break

        if crosses:
            # Штраф: удаляем границу (сливаем с предыдущей)
            if filtered:
                filtered[-1]["end_time"] = end
                filtered[-1]["cuts"] = filtered[-1].get("cuts", 0) + scene.get("cuts", 0)
            else:
                filtered.append(scene)
        else:
            filtered.append(scene)

    return filtered