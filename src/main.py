import argparse
import csv
import logging
import sys
import traceback
import os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.histogram_method.shot_signature import shot_signature
from src.histogram_method.group_by_hist import group_shots_by_hist
from src.histogram_method.centering import center_embeddings
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
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=handlers)


def get_shots(video_path):
    path = SHOTS_DIR / f"{Path(video_path).stem}_shots.csv"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return [{"start_time": float(r["start_time"]), "end_time": float(r["end_time"])} for r in csv.DictReader(f)]
    shots = detect_scenes(str(video_path))
    if shots:
        ensure_dir(SHOTS_DIR)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["start_time", "end_time"])
            w.writeheader()
            w.writerows(shots)
    return shots


def process_one(video_path, out_dir, cfg, method="hsv", mode="fixed", k_val=None, use_speech=False):
    stem = Path(video_path).stem
    csv_path = Path(out_dir) / f"{stem}_data.csv"

    shots = get_shots(video_path)
    if not shots: return False

    sig_kwargs = {}
    if method == "hsv":
        h = cfg.signature.hsv
        sig_kwargs = dict(scale=h.scale, frame_step=h.frame_step, h_bins=h.h_bins, s_bins=h.s_bins, v_bins=h.v_bins)
    elif method == "clip":
        sig_kwargs = dict(frame_step=cfg.signature.clip.frame_step)
    elif method == "dino":
        sig_kwargs = dict(frame_step=cfg.signature.dino.frame_step)
    shots_with_sigs = shot_signature(video_path, shots, method=method, **sig_kwargs)

    shots_with_sigs = [s for s in shots_with_sigs if s.get("signature") is not None]

    if method in ("clip", "dino"):
        shots_with_sigs = center_embeddings(shots_with_sigs, method)

    if mode == "percentile":
        threshold, k_val = None, k_val or 30.0
    else:
        threshold, k_val = getattr(cfg.grouping.threshold, method), cfg.grouping.k

    scenes = group_shots_by_hist(
        shots_with_sigs, threshold=threshold, lookback=cfg.grouping.lookback,
        min_duration=cfg.grouping.min_duration, min_shots=2, method=method, k=k_val, thr_mode=mode
    )
    save_scenes(scenes, str(csv_path))

    if use_speech:
        intervals = get_speech(video_path)
        scenes_sp = merge_scenes_through_speech(scenes, intervals)
        sp_dir = Path(out_dir).parent / f"result_{method}_speech"
        ensure_dir(sp_dir)
        save_scenes(scenes_sp, str(sp_dir / f"{stem}_data.csv"))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--methods", nargs="+", default=["hsv", "clip", "dino"])
    parser.add_argument("--mode", choices=["fixed", "percentile"], default="fixed")
    parser.add_argument("--k", type=float, default=30.0)
    parser.add_argument("--speech", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.file)

    raw_dir = Path(cfg.paths.raw)
    if not raw_dir.exists(): return

    exts = {e.lower() for e in cfg.video_exts}
    videos = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if args.limit: videos = videos[:args.limit]

    for method in args.methods:
        out_dir = DATA / f"result_{method}"
        ensure_dir(out_dir)
        print(f"\n===== {method.upper()} =====")
        for i, video in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}] {video.name}")
            try:
                process_one(str(video), str(out_dir), cfg, method, args.mode, args.k, args.speech)
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    main()