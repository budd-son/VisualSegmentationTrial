import csv
import bisect
from pathlib import Path
import numpy as np


class SceneEvaluator:
    """
    tolerance_shots — допуск в индексах склеек.
      0 — точное совпадение склейки
      1 — соседняя склейка (обычно достаточно)
    """

    def __init__(self, tolerance_shots=1):
        self.tolerance_shots = tolerance_shots
        self._cuts_cache = {}

    # ---------- I/O ----------

    @staticmethod
    def _end_col(csv_path):
        """Определяет имя колонки конца: end_sec (gt) или end_time (pred/shots)."""
        with open(csv_path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        for c in ("end_sec", "end_time"):
            if c in header:
                return c
        raise ValueError(f"{csv_path}: нет end_sec/end_time в {header}")

    def _read_all_ends(self, csv_path):
        """Возвращает список ВСЕХ end-значений (включая последнее)."""
        end_col = self._end_col(csv_path)
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [float(r[end_col]) for r in rows]

    def _read_boundaries(self, csv_path):
        """Границы = все end, кроме последнего (последний — конец видео)."""
        ends = self._read_all_ends(csv_path)
        return ends[:-1] if ends else []

    def _load_cuts(self, cuts_dir, name):
        """Все склейки видео = все end из shots/<name>.csv."""
        if name in self._cuts_cache:
            return self._cuts_cache[name]
        path = Path(cuts_dir) / f"{name}.csv"
        cuts = self._read_all_ends(str(path))
        self._cuts_cache[name] = cuts
        return cuts

    # ---------- матчинг ----------

    @staticmethod
    def _snap_to_cut(t, cuts):
        if not cuts:
            return None
        i = bisect.bisect_left(cuts, t)
        cand = [j for j in (i - 1, i) if 0 <= j < len(cuts)]
        if not cand:
            return None
        return min(cand, key=lambda j: abs(cuts[j] - t))

    def _match(self, pred_times, gt_times, cuts):
        gt_idx = {self._snap_to_cut(t, cuts) for t in gt_times}
        pred_idx = {self._snap_to_cut(t, cuts) for t in pred_times}
        gt_idx.discard(None); pred_idx.discard(None)

        matched_gt, matched_pred = set(), set()
        for g in sorted(gt_idx):
            best_p, best_d = None, float("inf")
            for p in sorted(pred_idx):
                if p in matched_pred:
                    continue
                d = abs(p - g)
                if d <= self.tolerance_shots and d < best_d:
                    best_d, best_p = d, p
            if best_p is not None:
                matched_gt.add(g); matched_pred.add(best_p)

        tp = len(matched_gt)
        fp = len(pred_idx) - len(matched_pred)
        fn = len(gt_idx) - len(matched_gt)
        return tp, fp, fn

    @staticmethod
    def _prf(tp, fp, fn):
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    # ---------- оценка ----------

    def evaluate_file(self, gt_path, pred_path, cuts_dir):
        name = Path(gt_path).stem
        gt_bounds   = self._read_boundaries(gt_path)
        pred_bounds = self._read_boundaries(pred_path)
        cuts        = self._load_cuts(cuts_dir, name)

        tp, fp, fn = self._match(pred_bounds, gt_bounds, cuts)
        p, r, f1 = self._prf(tp, fp, fn)

        return {
            "gt_bounds":   len(gt_bounds),
            "pred_bounds": len(pred_bounds),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
        }

    def evaluate_all(self, gt_dir, pred_dir, cuts_dir):
        gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
        per_video = {}

        for gt_path in sorted(gt_dir.glob("*.csv")):
            name = gt_path.stem
            pred_path = pred_dir / f"{name}_data.csv"
            if not pred_path.exists():
                print(f"[warn] нет prediction для {name}: {pred_path}")
                continue
            per_video[name] = self.evaluate_file(
                str(gt_path), str(pred_path), cuts_dir
            )

        TP = sum(v["tp"] for v in per_video.values())
        FP = sum(v["fp"] for v in per_video.values())
        FN = sum(v["fn"] for v in per_video.values())
        p_micro, r_micro, f1_micro = self._prf(TP, FP, FN)

        f1s = [v["f1"] for v in per_video.values()]
        summary = {
            "n_videos":       len(per_video),
            "n_gt_total":     sum(v["gt_bounds"] for v in per_video.values()),
            "TP": TP, "FP": FP, "FN": FN,
            "precision_micro": p_micro,
            "recall_micro":    r_micro,
            "f1_micro":        f1_micro,
            "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
            "f1_std":   float(np.std(f1s))  if f1s else 0.0,
        }
        return per_video, summary