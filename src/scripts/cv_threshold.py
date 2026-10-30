import sys
import csv
import bisect
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.scene_detect import detect_scenes
from src.histogram_method.shot_signature import shot_signature
from src.histogram_method.group_by_hist import group_shots_by_hist
from src.matchs.matchs_bound import SceneEvaluator

CFG = load_config(str(PROJECT_ROOT / "config.yaml"))
CFG.paths.raw = str(PROJECT_ROOT / CFG.paths.raw)

LOOKBACK = CFG.grouping.lookback
MIN_DURATION = CFG.grouping.min_duration
MIN_SHOTS = 2

CACHE_DIR = PROJECT_ROOT / "data" / "signatures_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Сетка порогов для перебора в LOO
THRESHOLD_GRIDS = {
    "hsv": np.arange(0.40, 0.85, 0.05),
    "clip": np.arange(0.60, 0.95, 0.05),
    "dino": np.arange(0.40, 0.80, 0.05),
}


def get_signatures_cached(video_path, method):
    cache_file = CACHE_DIR / f"{Path(video_path).stem}_{method}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"  [cache miss] {Path(video_path).name} ({method})...")
    shots = detect_scenes(str(video_path))
    if not shots:
        return None

    if method == "hsv":
        sig_kwargs = {
            "scale": CFG.signature.hsv.scale, "frame_step": CFG.signature.hsv.frame_step,
            "h_bins": CFG.signature.hsv.h_bins, "s_bins": CFG.signature.hsv.s_bins,
            "v_bins": CFG.signature.hsv.v_bins
        }
    elif method == "clip":
        sig_kwargs = {"frame_step": CFG.signature.clip.frame_step}
    else:
        sig_kwargs = {"frame_step": CFG.signature.dino.frame_step}

    shots_with_sigs = shot_signature(str(video_path), shots, method=method, **sig_kwargs)
    valid_sigs = []
    for s in shots_with_sigs:
        if method == "hsv":
            if float(np.asarray(s["signature"]).sum()) > 1e-6:
                valid_sigs.append(s)
        else:
            if s["signature"] and any(float(np.linalg.norm(e)) > 1e-6 for e in s["signature"]):
                valid_sigs.append(s)

    with open(cache_file, "wb") as f:
        pickle.dump(valid_sigs, f)
    return valid_sigs


def load_shots(video_name):
    path = PROJECT_ROOT / "data" / "shots" / f"{video_name}_shots.csv"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.reader(f):
            try:
                out.append(float(r[0]))
            except (ValueError, IndexError):
                pass
    return sorted(set(out))


def dedup(bounds, shots, tol=1):
    if not shots:
        return bounds
    idxs = sorted(set(bisect.bisect_left(shots, b) for b in bounds))
    keep = []
    for i in idxs:
        if not keep or i - keep[-1] > tol:
            keep.append(i)
    return [shots[i] for i in keep if i < len(shots)]


def evaluate_one_video(video_name, sigs, method, threshold):
    # Получаем сцены
    scenes = group_shots_by_hist(
        sigs, threshold=threshold, lookback=LOOKBACK,
        min_duration=MIN_DURATION, min_shots=MIN_SHOTS, method=method
    )
    pred_bounds = [s["end_time"] for s in scenes[:-1]]

    # Применяем дедупликацию (как в ev_all)
    shots = load_shots(video_name)
    pred_bounds = dedup(pred_bounds, shots)

    # Считаем метрики
    ev = SceneEvaluator(tol_shots=0)
    gt_path = str(PROJECT_ROOT / "data" / "ground_truth" / f"{video_name}.csv")
    gt_bounds = ev._read_boundaries(gt_path, "start", "end")

    return ev._match(pred_bounds, gt_bounds, shots)


def main():
    gt_dir = PROJECT_ROOT / "data" / "ground_truth"
    raw_dir = Path(CFG.paths.raw)
    videos = sorted([p.stem for p in gt_dir.glob("*.csv")])

    for method in ["hsv", "clip", "dino"]:
        print(f"\n========== LOO для {method.upper()} ==========")

        sigs_cache = {}
        for v in videos:
            video_path = raw_dir / f"{v}.mp4"
            if not video_path.exists():
                matches = list(raw_dir.glob(f"{v}.*"))
                if matches:
                    video_path = matches[0]
                else:
                    continue
            sigs = get_signatures_cached(video_path, method)
            if sigs:
                sigs_cache[v] = sigs

        loo_tp, loo_fp, loo_fn = 0, 0, 0
        chosen_thresholds = []

        for test_video in videos:
            train_videos = [v for v in videos if v != test_video and v in sigs_cache]
            if not train_videos or test_video not in sigs_cache:
                continue

            best_thr, best_train_f1 = None, -1
            for thr in THRESHOLD_GRIDS[method]:
                train_tp = train_fp = train_fn = 0
                for v in train_videos:
                    tp, fp, fn = evaluate_one_video(v, sigs_cache[v], method, thr)
                    train_tp += tp;
                    train_fp += fp;
                    train_fn += fn

                p = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
                r = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                if f1 > best_train_f1:
                    best_train_f1 = f1
                    best_thr = thr

            chosen_thresholds.append(best_thr)

            # Тест на отложенном видео
            tp, fp, fn = evaluate_one_video(test_video, sigs_cache[test_video], method, best_thr)
            loo_tp += tp;
            loo_fp += fp;
            loo_fn += fn
            print(f"  {test_video:<20} | Лучший thr={best_thr:.2f} | Тест: TP={tp} FP={fp} FN={fn}")

        p = loo_tp / (loo_tp + loo_fp) if (loo_tp + loo_fp) > 0 else 0
        r = loo_tp / (loo_tp + loo_fn) if (loo_tp + loo_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        mean_thr = np.mean(chosen_thresholds)
        print(f"\nИТОГО (LOO, Fixed): P={p:.2f} R={r:.2f} F1={f1:.2f} [{loo_tp}/{loo_fp}/{loo_fn}]")
        print(f"Средний выбранный порог: {mean_thr:.2f}")


if __name__ == "__main__":
    main()