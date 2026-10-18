import argparse
import csv
import logging
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.histogram_method.shot_signature import shot_signature
from src.histogram_method.group_by_hist import group_shots_by_hist
from src.scene_detect import detect_scenes
from src.speech.speech_cache import get_speech, merge_scenes_through_speech
from src.utils.config import load_config
from src.utils.io_util import save_scenes, ensure_dir

DATA = PROJECT_ROOT / "data"
SHOTS_DIR = DATA / "shots"


def setup_logging(level="INFO", file_path=None):
    handlers = [logging.StreamHandler()]
    if file_path:
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def _is_valid_signature(sig, method):
    if method == "hsv":
        return float(np.asarray(sig).sum()) > 1e-6
    if not sig:
        return False
    return any(float(np.linalg.norm(e)) > 1e-6 for e in sig)


def get_shots(video_path):
    """Шоты из кэша data/shots/<имя>.csv; иначе PySceneDetect + запись в кэш.
    (Если меняете параметры детектора — удалите соответствующий csv.)"""
    path = SHOTS_DIR / f"{Path(video_path).stem}.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            return [{"start_time": float(r["start_time"]),
                     "end_time": float(r["end_time"])} for r in csv.DictReader(f)]
    shots = detect_scenes(video_path)
    if shots:
        ensure_dir(SHOTS_DIR)
        save_scenes([{**s, "cuts": 1} for s in shots], str(path))
    return shots


def signature_kwargs(cfg, method):
    if method == "hsv":
        h = cfg.signature.hsv
        return {"scale": h.scale, "frame_step": h.frame_step,
                "h_bins": h.h_bins, "s_bins": h.s_bins, "v_bins": h.v_bins}
    if method == "clip":
        return {"frame_step": cfg.signature.clip.frame_step}
    if method == "dino":
        return {"frame_step": cfg.signature.dino.frame_step}
    raise ValueError(f"Unknown method: {method}")


def process_one(video_path, out_dir, cfg, method="hsv", mode="fixed",
                k_val=None, use_speech=False):
    stem = Path(video_path).stem
    csv_path = Path(out_dir) / f"{stem}_data.csv"

    shots = get_shots(video_path)
    if not shots:
        print(f"[skip] {video_path}: шотов не найдено")
        return False

    shots_with_sigs = shot_signature(video_path, shots, method=method,
                                     **signature_kwargs(cfg, method))
    shots_with_sigs = [s for s in shots_with_sigs
                       if _is_valid_signature(s["signature"], method)]
    if not shots_with_sigs:
        print(f"[skip] {video_path}: пустые подписи")
        return False

    if mode == "percentile":
        threshold = None
        k_val = k_val if k_val is not None else 30.0
    else:
        threshold = getattr(cfg.grouping.threshold, method)
        k_val = cfg.grouping.k

    print(f"[debug] method={method}, mode={mode}, threshold={threshold}, k={k_val}")

    scenes = group_shots_by_hist(
        shots_with_sigs,
        threshold=threshold,
        lookback=cfg.grouping.lookback,
        min_duration=cfg.grouping.min_duration,
        method=method,
        k=k_val,
        thr_mode=mode,
    )

    save_scenes(scenes, str(csv_path))
    print(f"[ok] {stem} ({method}): {len(scenes)} сцен → {csv_path.name}")

    # Речевое ограничение: реплика через склейку запрещает границу.
    # Whisper запускается только если нет data/speech/<имя>_speech.csv.
    if use_speech:
        intervals = get_speech(video_path)
        scenes_sp = merge_scenes_through_speech(scenes, intervals)
        sp_dir = Path(out_dir).parent / f"result_{method}_speech"
        ensure_dir(sp_dir)
        save_scenes(scenes_sp, str(sp_dir / f"{stem}_data.csv"))
        print(f"[ok] {stem} ({method}) + speech: {len(scenes_sp)} сцен "
              f"→ {sp_dir.name}/")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--methods", nargs="+", default=["hsv", "clip", "dino"],
                        choices=["hsv", "clip", "dino"],
                        help="какие методы прогнать")
    parser.add_argument("--mode", choices=["fixed", "percentile"], default="fixed",
                        help="Режим порога: fixed (из конфига) или percentile (адаптивный)")
    parser.add_argument("--k", type=float, default=30.0,
                        help="Значение k (перцентиля) для режима percentile")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override threshold для ВСЕХ методов")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--speech", action="store_true",
                        help="Применить речевое ограничение, результат в result_<метод>_speech")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.raw is not None:
        cfg.paths.raw = args.raw
    if args.lookback is not None:
        cfg.grouping.lookback = args.lookback
    if args.min_duration is not None:
        cfg.grouping.min_duration = args.min_duration
    if args.threshold is not None:
        for m in ("hsv", "clip", "dino"):
            setattr(cfg.grouping.threshold, m, args.threshold)

    setup_logging(cfg.logging.level, cfg.logging.file)

    raw_dir = Path(cfg.paths.raw)
    if not raw_dir.exists():
        print(f"Нет папки {raw_dir}")
        return

    exts = {e.lower() for e in cfg.video_exts}
    videos = sorted(p for p in raw_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in exts)
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        print(f"В {raw_dir} нет видеофайлов")
        return

    print(f"Найдено {len(videos)} файлов, методы: {args.methods}, "
          f"режим: {args.mode}, speech: {args.speech}")

    for method in args.methods:
        out_dir = DATA / f"result_{method}"
        sp_dir = DATA / f"result_{method}_speech"
        ensure_dir(out_dir)

        print(f"\n===== МЕТОД: {method.upper()} =====")
        ok, skipped, failed = 0, 0, 0

        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] {video.name}")
            done = (out_dir / f"{video.stem}_data.csv").exists() and (
                not args.speech or (sp_dir / f"{video.stem}_data.csv").exists())
            if args.skip_existing and done:
                print(f"[skip] уже есть: {video.stem}")
                skipped += 1
                continue
            try:
                if process_one(str(video), str(out_dir), cfg, method=method,
                               mode=args.mode, k_val=args.k, use_speech=args.speech):
                    ok += 1
                else:
                    failed += 1
            except Exception:
                print(f"[fail] {video.name}")
                traceback.print_exc()
                failed += 1

        print(f"\n{method}: ok={ok}, skip={skipped}, fail={failed}")


if __name__ == "__main__":
    main()