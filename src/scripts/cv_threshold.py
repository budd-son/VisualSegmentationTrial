import sys, pickle
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.histogram_method.shot_signature import shot_signature
from src.histogram_method.group_by_hist import group_shots_by_hist
from src.scene_detect import detect_scenes
from src.matchs.matchs_bound import SceneEvaluator
from src.utils.config import load_config

CACHE = PROJECT_ROOT / "data" / "sig_cache"


def _valid(sig, method):
    if method == "hsv":
        return float(np.asarray(sig).sum()) > 1e-6
    return bool(sig) and any(float(np.linalg.norm(e)) > 1e-6 for e in sig)


def build_cache(cfg, methods=("hsv", "clip", "dino")):
    """Один раз считает сигнатуры и кладёт в data/sig_cache/<method>/<stem>.pkl."""
    raw_dir = Path(cfg.paths.raw)
    exts = {e.lower() for e in cfg.video_exts}

    for method in methods:
        out = CACHE / method
        out.mkdir(parents=True, exist_ok=True)

        for video in sorted(raw_dir.iterdir()):
            if not video.is_file() or video.suffix.lower() not in exts:
                continue
            cache_path = out / f"{video.stem}.pkl"
            if cache_path.exists():
                continue

            shots = detect_scenes(str(video))
            if not shots:
                continue

            if method == "hsv":
                h = cfg.signature.hsv
                kw = dict(scale=h.scale, frame_step=h.frame_step,
                          h_bins=h.h_bins, s_bins=h.s_bins, v_bins=h.v_bins)
            elif method == "clip":
                kw = dict(frame_step=cfg.signature.clip.frame_step)
            else:
                kw = dict(frame_step=cfg.signature.dino.frame_step)

            sigs = shot_signature(str(video), shots, method=method, **kw)
            sigs = [s for s in sigs if _valid(s["signature"], method)]

            with open(cache_path, "wb") as f:
                pickle.dump(sigs, f)
            print(f"[cache] {method}/{video.stem}")


def grouping_predictions(method, k, lookback, min_duration, videos=None):
    """Возвращает {video: [pred_times]} по кэшу сигнатур."""
    out = {}
    for pkl in sorted((CACHE / method).glob("*.pkl")):
        if videos is not None and pkl.stem not in videos:
            continue
        with open(pkl, "rb") as f:
            shots = pickle.load(f)
        scenes = group_shots_by_hist(
            shots, threshold=None,
            lookback=lookback, min_duration=min_duration,
            method=method, k=k,
        )
        # граница между сценами = start_time следующей сцены
        out[pkl.stem] = [s["start_time"] for s in scenes[1:]]
    return out


def score(gt_dir, cuts_dir, ev, preds):
    TP = FP = FN = 0
    for name, pred in preds.items():
        gt_path = Path(gt_dir) / f"{name}.csv"
        if not gt_path.exists():
            continue
        gt_bounds = ev._read_boundaries(str(gt_path))
        cuts = ev._load_cuts(cuts_dir, name)
        tp, fp, fn = ev._match(pred, gt_bounds, cuts)
        TP += tp; FP += fp; FN += fn
    p, r, f1 = ev._prf(TP, FP, FN)
    return (p, r, f1), (TP, FP, FN)


def main():

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    # делаем raw абсолютным — на случай, если в config.yaml относительный путь
    cfg.paths.raw = str(PROJECT_ROOT / cfg.paths.raw)

    gt_dir = str(PROJECT_ROOT / "data" / "ground_truth")
    cuts_dir = str(PROJECT_ROOT / "data" / "shots")


    build_cache(cfg)

    ev = SceneEvaluator(tolerance_shots=1)
    names = sorted(p.stem for p in Path(gt_dir).glob("*.csv"))

    rng = np.random.default_rng(0)
    rng.shuffle(names)
    half = len(names) // 2
    A, B = set(names[:half]), set(names[half:])

    k_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    lookback = cfg.grouping.lookback
    min_dur  = cfg.grouping.min_duration

    for method in ("hsv", "clip", "dino"):
        print(f"\n=== {method.upper()} (current k={cfg.grouping.k}) ===")

        # для справки: F1 с текущим k на всём наборе (оптимистично)
        preds_all = grouping_predictions(method, cfg.grouping.k,
                                         lookback, min_dur)
        (p, r, f1), (TP, FP, FN) = score(gt_dir, cuts_dir, ev, preds_all)
        print(f"  all-data  k={cfg.grouping.k}: F1={f1:.3f} "
              f"P={p:.3f} R={r:.3f} TP={TP} FP={FP} FN={FN}")

        # CV split-half
        folds = []
        for train, test in [(A, B), (B, A)]:
            best_k, best_f1 = None, -1.0
            for k in k_grid:
                preds = grouping_predictions(method, k, lookback, min_dur,
                                             videos=train)
                (_, _, f1), _ = score(gt_dir, cuts_dir, ev, preds)
                if f1 > best_f1:
                    best_f1, best_k = f1, k
            preds_test = grouping_predictions(method, best_k, lookback,
                                              min_dur, videos=test)
            (p, r, f1), (TP, FP, FN) = score(gt_dir, cuts_dir, ev, preds_test)
            folds.append((best_k, best_f1, f1, p, r, TP, FP, FN))

        for k_star, f1_tr, f1_te, p, r, TP, FP, FN in folds:
            print(f"  fold: k*={k_star:.2f} (train F1={f1_tr:.3f}) → "
                  f"test F1={f1_te:.3f} P={p:.3f} R={r:.3f} "
                  f"TP={TP} FP={FP} FN={FN}")
        print(f"  mean test F1 = {np.mean([f[2] for f in folds]):.3f}")


if __name__ == "__main__":

    main()