import argparse
import logging
import sys
import traceback
from pathlib import Path

# чтобы можно было запускать из корня проекта
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.io_util import save_scenes, ensure_dir
from src.scene_detect import detect_scenes
from src.histogram_method.cut_histogrm import shot_hsv_signature
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


def process_one(video_path, out_dir, cfg):
    stem = Path(video_path).stem
    csv_path = Path(out_dir) / f"{stem}_data.csv"
    npy_path = Path(out_dir) / f"{stem}_data.npy"

    shots = detect_scenes(video_path)
    if not shots:
        print(f"[skip] {video_path}: шотов не найдено")
        return False

    shots_with_sigs = shot_hsv_signature(
        video_path, shots,
        scale=cfg.signature.scale,
        frame_step=cfg.signature.frame_step,
        h_bins=cfg.signature.h_bins,
        s_bins=cfg.signature.s_bins,
        v_bins=cfg.signature.v_bins,
    )
    shots_with_sigs = [s for s in shots_with_sigs if s["signature"].sum() > 0]
    if not shots_with_sigs:
        print(f"[skip] {video_path}: пустые подписи")
        return False

    scenes = group_shots_by_hist(
        shots_with_sigs,
        threshold=cfg.grouping.threshold,
        lookback=cfg.grouping.lookback,
        min_duration=cfg.grouping.min_duration,
    )
    save_scenes(scenes, str(csv_path))
    print(f"[ok] {stem}: {len(scenes)} сцен → {csv_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw", default=None, help="override paths.raw")
    parser.add_argument("--out", default=None, help="override paths.out")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI перекрывает конфиг
    if args.raw is not None:
        cfg.paths.raw = args.raw
    if args.out is not None:
        cfg.paths.out = args.out
    if args.threshold is not None:
        cfg.grouping.threshold = args.threshold
    if args.lookback is not None:
        cfg.grouping.lookback = args.lookback
    if args.min_duration is not None:
        cfg.grouping.min_duration = args.min_duration

    setup_logging(cfg.logging.level, cfg.logging.file)

    raw_dir = Path(cfg.paths.raw)
    out_dir = Path(cfg.paths.out)
    ensure_dir(out_dir)

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

    print(f"Найдено {len(videos)} файлов")
    ok, skipped, failed = 0, 0, 0

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video.name}")
        csv_path = out_dir / f"{video.stem}_data.csv"
        if args.skip_existing and csv_path.exists():
            print(f"[skip] уже есть: {csv_path.name}")
            skipped += 1
            continue
        try:
            if process_one(str(video), str(out_dir), cfg):
                ok += 1
            else:
                failed += 1
        except Exception:
            print(f"[fail] {video.name}")
            traceback.print_exc()
            failed += 1

    print(f"\nГотово: ok={ok}, skip={skipped}, fail={failed}")
    print(f"Результаты: {out_dir.resolve()}")


if __name__ == "__main__":
    main()