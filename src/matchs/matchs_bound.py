import csv
from pathlib import Path



class SceneEvaluator:
    def __init__(self, tolerance=2.0):
        self.tolerance = tolerance

    @staticmethod
    def _read_boundaries(csv_path, start_col, end_col):
        boundaries = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return boundaries

        for row in rows[:-1]:
            boundaries.append(float(row[end_col]))
        return boundaries

    def _match(self, pred, gt):

        pred = sorted(pred)
        gt = sorted(gt)
        matched_gt = set()
        matched_pred = set()

        for i, p in enumerate(pred):
            best_j = None
            best_dist = float("inf")
            for j, g in enumerate(gt):
                if j in matched_gt:
                    continue
                d = abs(p - g)
                if d <= self.tolerance and d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j is not None:
                matched_pred.add(i)
                matched_gt.add(best_j)

        tp = len(matched_pred)
        fp = len(pred) - tp
        fn = len(gt) - tp
        return tp, fp, fn

    def evaluate_file(self, gt_path, pred_path):
        gt_bounds = self._read_boundaries(gt_path, "start_sec", "end_sec")
        pred_bounds = self._read_boundaries(pred_path, "start_time", "end_time")

        tp, fp, fn = self._match(pred_bounds, gt_bounds)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            "gt_bounds": len(gt_bounds),
            "pred_bounds": len(pred_bounds),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate_all(self, gt_dir, pred_dir):
        gt_dir = Path(gt_dir)
        pred_dir = Path(pred_dir)

        results = {}
        for gt_path in sorted(gt_dir.glob("*.csv")):
            name = gt_path.stem
            pred_path = pred_dir / f"{name}_data.csv"

            if not pred_path.exists():
                print(f"[warn] нет prediction для {name}: {pred_path}")
                continue

            results[name] = self.evaluate_file(str(gt_path), str(pred_path))

        return results