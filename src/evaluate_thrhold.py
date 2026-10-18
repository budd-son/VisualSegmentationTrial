import sys
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.scene_detect import detect_scenes
from src.histogram_method.shot_signature import shot_signature
from src.histogram_method.group_by_hist import group_shots_by_hist
from src.matchs.matchs_bound import SceneEvaluator

CFG = load_config("config.yaml")
LOOKBACK = 5
MIN_DURATION = 5.0

# --- СЕТКИ ПАРАМЕТРОВ ---
# 1. Фиксированные пороги (как было)
FIXED_GRIDS = {
    "hsv": np.arange(0.20, 0.95, 0.05),
    "clip": np.arange(0.20, 0.95, 0.05),
    "dino": np.arange(0.20, 0.95, 0.05),
}

# 2. Перцентили для адаптивного порога (новый метод)
PERCENTILE_GRIDS = {
    "hsv": np.arange(5, 35, 5),
    "clip": np.arange(5, 35, 5),
    "dino": np.arange(5, 35, 5),
}

CACHE_DIR = PROJECT_ROOT / "data" / "signatures_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_signatures_cached(video_path, method):
    cache_file = CACHE_DIR / f"{Path(video_path).stem}_{method}.pkl"

    # 1. Если кэш есть — просто читаем
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # 2. Если кэша нет — считаем
    print(f"  [cache miss] Извлечение сигнатур: {Path(video_path).name} ({method})...")
    shots = detect_scenes(str(video_path))
    if not shots: return None

    if method == "hsv":
        sig_kwargs = {
            "scale": CFG.signature.hsv.scale,
            "frame_step": CFG.signature.hsv.frame_step,
            "h_bins": CFG.signature.hsv.h_bins,
            "s_bins": CFG.signature.hsv.s_bins,
            "v_bins": CFG.signature.hsv.v_bins,
        }
    elif method == "clip":
        sig_kwargs = {"frame_step": CFG.signature.clip.frame_step}
    elif method == "dino":
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


def scenes_to_bounds(scenes):
    return [s["end_time"] for s in scenes[:-1]]


def evaluate_bounds(pred_bounds, gt_path, shots_path):
    ev = SceneEvaluator(tol_shots=1)
    gt_bounds = ev._read_boundaries(str(gt_path), "start", "end")
    shots = ev._load_shots(str(shots_path))
    return ev._match(pred_bounds, gt_bounds, shots)


def run_cv(method, sigs_cache, videos, gt_dir, shots_dir, grid, thr_mode):
    """Запускает Leave-One-Out для заданного режима и сетки."""
    loo_tp, loo_fp, loo_fn = 0, 0, 0

    for test_video in videos:
        train_videos = [v for v in videos if v != test_video and v in sigs_cache]
        if not train_videos or test_video not in sigs_cache:
            continue

        best_val, best_f1 = None, -1
        for val in grid[method]:
            train_tp, train_fp, train_fn = 0, 0, 0
            for v in train_videos:
                if thr_mode == "fixed":
                    scenes = group_shots_by_hist(
                        sigs_cache[v], threshold=float(val),
                        lookback=LOOKBACK, min_duration=MIN_DURATION, method=method
                    )
                else:  # percentile
                    scenes = group_shots_by_hist(
                        sigs_cache[v], threshold=None, k=float(val), thr_mode="percentile",
                        lookback=LOOKBACK, min_duration=MIN_DURATION, method=method
                    )

                tp, fp, fn = evaluate_bounds(
                    scenes_to_bounds(scenes), gt_dir / f"{v}.csv", shots_dir / f"{v}_shots.csv"
                )
                train_tp += tp;
                train_fp += fp;
                train_fn += fn

            p = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
            r = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_val = val

        if thr_mode == "fixed":
            scenes = group_shots_by_hist(
                sigs_cache[test_video], threshold=float(best_val),
                lookback=LOOKBACK, min_duration=MIN_DURATION, method=method
            )
        else:
            scenes = group_shots_by_hist(
                sigs_cache[test_video], threshold=None, k=float(best_val), thr_mode="percentile",
                lookback=LOOKBACK, min_duration=MIN_DURATION, method=method
            )

        tp, fp, fn = evaluate_bounds(
            scenes_to_bounds(scenes), gt_dir / f"{test_video}.csv", shots_dir / f"{test_video}_shots.csv"
        )
        loo_tp += tp;
        loo_fp += fp;
        loo_fn += fn

        val_str = f"{best_val:.2f}%" if thr_mode == "percentile" else f"{best_val:.2f}"
        print(f"  {test_video:<20} | Лучший: {val_str} | Тест: TP={tp} FP={fp} FN={fn}")

    p = loo_tp / (loo_tp + loo_fp) if (loo_tp + loo_fp) > 0 else 0
    r = loo_tp / (loo_tp + loo_fn) if (loo_tp + loo_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1, loo_tp, loo_fp, loo_fn


def main():
    gt_dir = PROJECT_ROOT / "data" / "ground_truth"
    raw_dir = PROJECT_ROOT / CFG.paths.raw
    shots_dir = PROJECT_ROOT / "data" / "shots"
    videos = sorted([p.stem for p in gt_dir.glob("*.csv")])

    for method in ["hsv", "clip", "dino"]:
        print(f"\n========== Метод: {method.upper()} ==========")

        # Кэшируем сигнатуры
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
            if sigs: sigs_cache[v] = sigs

        # 1. Запускаем Fixed Threshold
        print(f"\n--- Режим: Fixed Threshold ---")
        p1, r1, f1_1, tp1, fp1, fn1 = run_cv(
            method, sigs_cache, videos, gt_dir, shots_dir, FIXED_GRIDS, "fixed"
        )
        print(f"ИТОГО (Fixed): P={p1:.2f} R={r1:.2f} F1={f1_1:.2f} [{tp1}/{fp1}/{fn1}]")

        # 2. Запускаем Percentile Threshold
        print(f"\n--- Режим: Percentile ---")
        p2, r2, f1_2, tp2, fp2, fn2 = run_cv(
            method, sigs_cache, videos, gt_dir, shots_dir, PERCENTILE_GRIDS, "percentile"
        )
        print(f"ИТОГО (Percentile): P={p2:.2f} R={r2:.2f} F1={f1_2:.2f} [{tp2}/{fp2}/{fn2}]")


if __name__ == "__main__":
    main()