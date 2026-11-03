import sys
import csv
import bisect
import pickle
import numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.scene_detect import detect_scenes
from src.histogram_method.shot_signature import shot_signature
from src.histogram_method.group_by_hist import group_shots_by_hist
from src.histogram_method.centering import center_embeddings
from src.matchs.matchs_bound import SceneEvaluator

CFG = load_config(str(PROJECT_ROOT / "config.yaml"))
CFG.paths.raw = str(PROJECT_ROOT / CFG.paths.raw)
CACHE_DIR = PROJECT_ROOT / "data" / "signatures_cache"
SHOTS_DIR = PROJECT_ROOT / "data" / "shots"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_GRIDS = {"hsv": np.arange(0.40, 0.85, 0.05), "clip": np.arange(0.60, 0.95, 0.05),
                   "dino": np.arange(0.40, 0.80, 0.05)}


def get_sigs(video_path, method):
    cache = CACHE_DIR / f"{Path(video_path).stem}_{method}.pkl"
    if cache.exists():
        with open(cache, "rb") as f: return pickle.load(f)
    shots = detect_scenes(str(video_path))
    if not shots: return None
    kw = {}
    if method == "hsv":
        h = CFG.signature.hsv
        kw = dict(scale=h.scale, frame_step=h.frame_step, h_bins=h.h_bins, s_bins=h.s_bins, v_bins=h.v_bins)
    elif method == "clip":
        kw = dict(frame_step=CFG.signature.clip.frame_step)
    else:
        kw = dict(frame_step=CFG.signature.dino.frame_step)
    sigs = shot_signature(str(video_path), shots, method=method, **kw)
    sigs = [s for s in sigs if s.get("signature") is not None]
    if method in ("clip", "dino"):
        sigs = center_embeddings(sigs, method)
    with open(cache, "wb") as f:
        pickle.dump(sigs, f)
    return sigs


def load_shots(video_name):
    path = PROJECT_ROOT / "data" / "shots" / f"{video_name}_shots.csv"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # <--- ПРОПУСКАЕМ ЗАГОЛОВОК
        for r in reader:
            try:
                out.append(float(r[0]))
            except (ValueError, IndexError):
                pass
    return sorted(set(out))

def main():
    gt_dir = PROJECT_ROOT / "data" / "ground_truth"
    raw_dir = Path(CFG.paths.raw)
    videos = sorted([p.stem for p in gt_dir.glob("*.csv")])
    ev = SceneEvaluator()

    for method in ["hsv", "clip", "dino"]:
        print(f"\n========== LOO для {method.upper()} ==========")
        sigs_cache = {v: get_sigs(raw_dir / f"{v}.mp4", method) for v in videos if (raw_dir / f"{v}.mp4").exists()}

        loo_tp, loo_fp, loo_fn = 0, 0, 0
        chosen_thr = []

        for test_v in videos:
            train = [v for v in videos if v != test_v and v in sigs_cache]
            if not train or test_v not in sigs_cache: continue

            best_thr, best_f1 = None, -1
            for thr in THRESHOLD_GRIDS[method]:
                tp = fp = fn = 0
                for v in train:
                    scenes = group_shots_by_hist(sigs_cache[v], threshold=thr, lookback=CFG.grouping.lookback,
                                                 min_duration=CFG.grouping.min_duration, min_shots=2, method=method)
                    pred = [s["end_time"] for s in scenes[:-1]]
                    gt = ev._read_boundaries(str(gt_dir / f"{v}.csv"), "start", "end")
                    shots = load_shots(v)
                    t, f_p, f_n = ev._match(pred, gt, shots, tol_shots=0)
                    tp += t;
                    fp += f_p;
                    fn += f_n
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                if f1 > best_f1: best_f1, best_thr = f1, thr

            chosen_thr.append(best_thr)
            scenes = group_shots_by_hist(sigs_cache[test_v], threshold=best_thr, lookback=CFG.grouping.lookback,
                                         min_duration=CFG.grouping.min_duration, min_shots=2, method=method)
            pred = [s["end_time"] for s in scenes[:-1]]
            gt = ev._read_boundaries(str(gt_dir / f"{test_v}.csv"), "start", "end")
            shots = load_shots(test_v)
            t, f_p, f_n = ev._match(pred, gt, shots, tol_shots=0)
            loo_tp += t;
            loo_fp += f_p;
            loo_fn += f_n
            print(f"  {test_v:<15} | thr={best_thr:.2f} | TP={t} FP={f_p} FN={f_n}")

        p = loo_tp / (loo_tp + loo_fp) if (loo_tp + loo_fp) > 0 else 0
        r = loo_tp / (loo_tp + loo_fn) if (loo_tp + loo_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"\nИТОГО (LOO): P={p:.2f} R={r:.2f} F1={f1:.2f} | Средний порог: {np.mean(chosen_thr):.2f}")


if __name__ == "__main__":
    main()