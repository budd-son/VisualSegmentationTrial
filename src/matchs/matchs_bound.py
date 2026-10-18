import csv
import bisect
from pathlib import Path


class SceneEvaluator:
    def __init__(self, tol_shots=1):
        # Допуск в шотах, а не в секундах
        self.tol_shots = tol_shots

    @staticmethod
    def _read_boundaries(csv_path, start_col, end_col):
        boundaries = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return boundaries

        end_key = next((k for k in rows[0].keys() if k.startswith(end_col)), None)
        if not end_key:
            raise KeyError(f"Не найдена колонка {end_col} в {csv_path}")

        for row in rows[:-1]:
            boundaries.append(float(row[end_key]))
        return boundaries

    @staticmethod
    def _load_shots(shots_path):
        """Загружает времена склеек (начало каждого шота)."""
        shots = []
        if not Path(shots_path).exists():
            return shots

        with open(shots_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                try:
                    shots.append(float(row[0]))  # Берем start_time
                except ValueError:
                    continue  # Пропускаем заголовок
        return sorted(list(set(shots)))

    @staticmethod
    def _get_shot_idx(time, shots):
        if not shots: return -1
        idx = bisect.bisect_left(shots, time)
        if idx == 0: return 0
        if idx == len(shots): return len(shots) - 1
        # Смотрим, какой шот ближе: слева или справа
        if shots[idx] - time < time - shots[idx - 1]:
            return idx
        return idx - 1

    def _match(self, pred, gt, shots):
        """Сопоставление границ по индексам шотов."""
        pred = sorted(pred)
        gt = sorted(gt)

        # Переводим времена в индексы шотов
        pred_idx = [self._get_shot_idx(p, shots) for p in pred]
        gt_idx = [self._get_shot_idx(g, shots) for g in gt]

        matched_gt = set()
        matched_pred = set()

        for i, p_i in enumerate(pred_idx):
            best_j = None
            best_dist = float("inf")
            for j, g_i in enumerate(gt_idx):
                if j in matched_gt:
                    continue

                # Сравниваем РАЗНИЦУ ИНДЕКСОВ, а не время
                d = abs(p_i - g_i)
                if d <= self.tol_shots and d < best_dist:
                    best_dist = d
                    best_j = j

            if best_j is not None:
                matched_pred.add(i)
                matched_gt.add(best_j)

        tp = len(matched_pred)
        fp = len(pred) - tp
        fn = len(gt) - tp
        return tp, fp, fn

    def evaluate_file(self, gt_path, pred_path, shots_path):
        # На скрине GT: start_sec, end_sec. В предсказаниях: start_time, end_time
        gt_bounds = self._read_boundaries(gt_path, "start", "end")
        pred_bounds = self._read_boundaries(pred_path, "start", "end")

        shots = self._load_shots(shots_path)

        tp, fp, fn = self._match(pred_bounds, gt_bounds, shots)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            "gt_bounds": len(gt_bounds),
            "pred_bounds": len(pred_bounds),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    def evaluate_all(self, gt_dir, pred_dir, shots_dir):
        gt_dir = Path(gt_dir)
        pred_dir = Path(pred_dir)
        shots_dir = Path(shots_dir)

        results = {}
        for gt_path in sorted(gt_dir.glob("*.csv")):
            name = gt_path.stem
            pred_path = pred_dir / f"{name}_data.csv"
            shots_path = shots_dir / f"{name}_shots.csv"

            if not pred_path.exists():
                print(f"[warn] нет prediction для {name}: {pred_path}")
                continue
            if not shots_path.exists():
                print(f"[warn] нет шотов для {name}: {shots_path}")
                continue

            results[name] = self.evaluate_file(
                str(gt_path), str(pred_path), str(shots_path)
            )

        return results