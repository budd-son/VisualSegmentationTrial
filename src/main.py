import argparse
import logging
import sys
import traceback
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np
from src.histogram_method.shot_signature import shot_signature
from src.utils.config import load_config
from src.utils.io_util import save_scenes, ensure_dir
from src.scene_detect import detect_scenes
from src.histogram_method.group_by_hist import group_shots_by_hist


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
    """Проверка, что подпись шота не пустая.

    HSV — вектор-гистограмма, проверяем сумму.
    CLIP/DINO — список эмбеддингов, проверяем, что хотя бы один не нулевой.
    """
    if method == "hsv":
        return float(np.asarray(sig).sum()) > 1e-6
    # clip / dino
    if not sig:
        return False
    return any(float(np.linalg.norm(e)) > 1e-6 for e in sig)


def process_one(video_path, out_dir, cfg, method="hsv"):
    stem = Path(video_path).stem
    csv_path = Path(out_dir) / f"{stem}_data.csv"

    shots = detect_scenes(video_path)
    if not shots:
        print(f"[skip] {video_path}: шотов не найдено")
        return False

    if method == "hsv":
        sig_kwargs = {
            "scale": cfg.signature.hsv.scale,
            "frame_step": cfg.signature.hsv.frame_step,
            "h_bins": cfg.signature.hsv.h_bins,
            "s_bins": cfg.signature.hsv.s_bins,
            "v_bins": cfg.signature.hsv.v_bins,
        }
    elif method == "clip":
        sig_kwargs = {
            "frame_step": cfg.signature.clip.frame_step,
        }
    elif method == "dino":
        sig_kwargs = {
            "frame_step": cfg.signature.dino.frame_step,
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    shots_with_sigs = shot_signature(video_path, shots, method=method, **sig_kwargs)
    shots_with_sigs = [s for s in shots_with_sigs
                       if _is_valid_signature(s["signature"], method)]
    if not shots_with_sigs:
        print(f"[skip] {video_path}: пустые подписи")
        return False

    if method == "hsv":
        threshold = cfg.grouping.threshold.hsv
    elif method == "clip":
        threshold = cfg.grouping.threshold.clip
    elif method == "dino":
        threshold = cfg.grouping.threshold.dino

    print(f"[debug] method={method}, threshold={threshold}")

    scenes = group_shots_by_hist(
        shots_with_sigs,
        threshold=threshold,
        lookback=cfg.grouping.lookback,
        min_duration=cfg.grouping.min_duration,
        method=method,
        k=cfg.grouping.k,
    )

    save_scenes(scenes, str(csv_path))
    print(f"[ok] {stem} ({method}): {len(scenes)} сцен → {csv_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--methods", nargs="+",
                        default=["hsv", "clip", "dino"],
                        choices=["hsv", "clip", "dino"],
                        help="какие методы прогнать")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override threshold для ВСЕХ методов")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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
    videos = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if args.limit:
        videos = videos[:args.limit]

    if not videos:
        print(f"В {raw_dir} нет видеофайлов")
        return

    print(f"Найдено {len(videos)} файлов, методы: {args.methods}")

    for method in args.methods:
        out_dir = Path(f"data/result_{method}")
        ensure_dir(out_dir)

        print(f"\n===== МЕТОД: {method.upper()} =====")
        ok, skipped, failed = 0, 0, 0

        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] {video.name}")
            csv_path = out_dir / f"{video.stem}_data.csv"
            if args.skip_existing and csv_path.exists():
                print(f"[skip] уже есть: {csv_path.name}")
                skipped += 1
                continue
            try:
                if process_one(str(video), str(out_dir), cfg, method=method):
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