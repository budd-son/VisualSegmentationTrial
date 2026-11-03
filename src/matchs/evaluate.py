import sys
from pathlib import Path

# Поднимаемся до корня проекта, чтобы работали импорты
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matchs.matchs_bound import SceneEvaluator


def get_all_metrics(tolerance=0):
    """Возвращает словарь {метод: {precision, recall, f1}} для всех папок result_*."""
    data_dir = PROJECT_ROOT / "data"
    gt_dir = str(data_dir / "ground_truth")
    shots_dir = str(data_dir / "shots")

    # Ищем все папки result_*
    methods = {}
    for p in sorted(data_dir.glob("result_*")):
        if p.is_dir():
            method = p.name.replace("result_", "")
            methods[method] = str(p)

    ev = SceneEvaluator(tol_shots=tolerance)
    results = {}

    for method, pred_dir in methods.items():
        # Считаем micro-метрики (по всему корпусу)
        method_results = ev.evaluate_all(gt_dir, pred_dir, shots_dir)

        tp = fp = fn = 0
        for r in method_results.values():
            tp += r.get("tp", 0)
            fp += r.get("fp", 0)
            fn += r.get("fn", 0)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results[method] = {"p": p, "r": r, "f1": f1}

    return results