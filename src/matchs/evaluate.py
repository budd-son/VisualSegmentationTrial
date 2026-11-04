import sys
from pathlib import Path

# Поднимаемся до корня проекта, чтобы работали импорты
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matchs.matchs_bound import SceneEvaluator


def get_all_metrics(tolerance=0):
    """Возвращает {метод: {precision, recall, f1}} для всех папок result_*."""
    data_dir = PROJECT_ROOT / "data"
    gt_dir = str(data_dir / "ground_truth")
    shots_dir = str(data_dir / "shots")

    methods = {}
    for p in sorted(data_dir.glob("result_*")):
        if p.is_dir():
            method = p.name.replace("result_", "")
            methods[method] = str(p)

    ev = SceneEvaluator()
    results = {}

    for method, pred_dir in methods.items():
        res = ev.evaluate_all(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            shots_dir=shots_dir,
            tol_shots=tolerance  # <--- ПЕРЕДАЕМ ДОПУСК
        )

        tp = sum(r["tp"] for r in res.values())
        fp = sum(r["fp"] for r in res.values())
        fn = sum(r["fn"] for r in res.values())

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results[method] = {"p": p, "r": r, "f1": f1}

    return results