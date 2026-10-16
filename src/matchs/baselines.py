import numpy as np
from pathlib import Path


def _summ(per_video, ev):
    TP = sum(v["tp"] for v in per_video.values())
    FP = sum(v["fp"] for v in per_video.values())
    FN = sum(v["fn"] for v in per_video.values())
    p, r, f1 = ev._prf(TP, FP, FN)
    f1s = [v["f1"] for v in per_video.values()]
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "n_gt_total": sum(v["gt_bounds"] for v in per_video.values()),
        "precision_micro": p, "recall_micro": r, "f1_micro": f1,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "f1_std":   float(np.std(f1s))  if f1s else 0.0,
        "n_videos": len(per_video),
    }


def baseline_every_cut(gt_dir, cuts_dir, ev):
    """Все склейки = границы сцен."""
    gt_dir = Path(gt_dir)
    per_video = {}
    for gt_path in sorted(gt_dir.glob("*.csv")):
        name = gt_path.stem
        cuts = ev._load_cuts(cuts_dir, name)
        gt_bounds = ev._read_boundaries(str(gt_path))
        pred = list(cuts)
        tp, fp, fn = ev._match(pred, gt_bounds, cuts)
        p, r, f1 = ev._prf(tp, fp, fn)
        per_video[name] = {
            "gt_bounds": len(gt_bounds), "pred_bounds": len(pred),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
        }
    return per_video, _summ(per_video, ev)


def baseline_every_n_seconds(gt_dir, cuts_dir, ev, N=30):
    """Регулярная сетка каждые N секунд от начала видео."""
    gt_dir = Path(gt_dir)
    per_video = {}
    for gt_path in sorted(gt_dir.glob("*.csv")):
        name = gt_path.stem
        cuts = ev._load_cuts(cuts_dir, name)
        if not cuts:
            continue
        duration = cuts[-1]
        gt_bounds = ev._read_boundaries(str(gt_path))
        pred = list(np.arange(N, duration, N))
        tp, fp, fn = ev._match(pred, gt_bounds, cuts)
        p, r, f1 = ev._prf(tp, fp, fn)
        per_video[name] = {
            "gt_bounds": len(gt_bounds), "pred_bounds": len(pred),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
        }
    return per_video, _summ(per_video, ev)