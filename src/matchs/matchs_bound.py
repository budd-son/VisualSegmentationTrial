import csv
import bisect
from pathlib import Path


class SceneEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def _read_boundaries(csv_path, start_col="start", end_col="end"):
        boundaries = []
        if not Path(csv_path).exists():
            return boundaries
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows: return boundaries
            ek = next((k for k in rows[0] if k.startswith(end_col)), None)
            if not ek: raise KeyError(f"Нет колонки {end_col} в {csv_path}")
            for row in rows[:-1]:
                boundaries.append(float(row[ek]))
        return boundaries

    @staticmethod
    def _load_shots(shots_path):
        shots = []
        if not Path(shots_path).exists():
            return shots
        with open(shots_path, encoding="utf-8") as f:
            for r in csv.reader(f):
                try:
                    shots.append(float(r[0]))
                except (ValueError, IndexError):
                    pass
        return sorted(set(shots))

    @staticmethod
    def _get_shot_idx(time, shots):
        if not shots: return -1
        idx = bisect.bisect_left(shots, time)
        if idx == 0: return 0
        if idx == len(shots): return len(shots) - 1
        # Возвращаем индекс БЛИЖАЙШЕГО шота
        return idx if shots[idx] - time < time - shots[idx - 1] else idx - 1

    def _match(self, pred, gt, shots, tol_shots=0, tol_sec=0.0):
        """Единая функция сопоставления (и по шотам, и по секундам)."""
        tp = 0
        matched_gt = set()

        if shots:
            # Сопоставление по индексам шотов
            pred_idx = [self._get_shot_idx(p, shots) for p in sorted(pred)]
            gt_idx = [self._get_shot_idx(g, shots) for g in sorted(gt)]

            for p_i in pred_idx:
                best_j, best_d = None, 10 ** 9
                for j, g_i in enumerate(gt_idx):
                    if j in matched_gt: continue
                    d = abs(p_i - g_i)
                    if d < best_d:
                        best_d, best_j = d, j
                if best_d <= tol_shots:
                    matched_gt.add(best_j)
                    tp += 1
        else:
            # Фолбэк: сопоставление по секундам
            for p in sorted(pred):
                best_j, best_d = None, 10 ** 9
                for j, g in enumerate(sorted(gt)):
                    if j in matched_gt: continue
                    d = abs(p - g)
                    if d < best_d:
                        best_d, best_j = d, j
                if best_d <= tol_sec:
                    matched_gt.add(best_j)
                    tp += 1

        fp = len(pred) - tp
        fn = len(gt) - tp
        return tp, fp, fn

    def evaluate_all(self, gt_dir, pred_dir, shots_dir, tol_shots=0, tol_sec=0.0):
        gt_dir, pred_dir, shots_dir = Path(gt_dir), Path(pred_dir), Path(shots_dir)
        results = {}

        for gt_path in sorted(gt_dir.glob("*.csv")):
            video = gt_path.stem
            pred_path = pred_dir / f"{video}_data.csv"
            shots_path = shots_dir / f"{video}_shots.csv"

            gt_bounds = self._read_boundaries(str(gt_path), "start", "end")
            shots = self._load_shots(str(shots_path))

            if not pred_path.exists():
                # Если предсказаний нет — все GT пропущены
                tp, fp, fn = 0, 0, len(gt_bounds)
            else:
                pred_bounds = self._read_boundaries(str(pred_path), "start", "end")
                tp, fp, fn = self._match(pred_bounds, gt_bounds, shots, tol_shots, tol_sec)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            results[video] = {
                "precision": p, "recall": r, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn,
                "n_gt": len(gt_bounds), "n_pred": len(pred_bounds) if pred_path.exists() else 0
            }

        return results