import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matchs.matchs_bound import SceneEvaluator


def discover_methods(data_dir: Path):
    methods = {}
    for p in sorted(data_dir.glob("result_*")):
        if p.is_dir():
            methods[p.name.replace("result_", "")] = str(p)
    return methods


def run_eval(tolerance, gt_dir, methods, shots_dir):
    print(f"\n{'=' * 60}\n ОЦЕНКА (допуск ±{tolerance} шот)\n{'=' * 60}\n")
    ev = SceneEvaluator()
    all_results = {}

    for method, pred_dir in methods.items():
        all_results[method] = ev.evaluate_all(
            gt_dir=gt_dir, pred_dir=pred_dir, shots_dir=shots_dir, tol_shots=tolerance
        )

    methods_list = list(all_results.keys())
    videos = sorted(set(v for r in all_results.values() for v in r.keys()))
    col_w = 24

    # Заголовок
    header = f"{'video':<24} | " + " | ".join(f"{m:^{col_w}}" for m in methods_list)
    print(header)
    print("-" * len(header))

    for v in videos:
        cells = []
        for m in methods_list:
            r = all_results[m].get(v)
            if not r:
                cells.append(f"{'—':^{col_w}}")
            else:
                cells.append(f"P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}")
        print(f"{v:<24} | " + " | ".join(cells))

    # Сумма (micro)
    print("-" * len(header))
    cells = []
    for m in methods_list:
        tp = sum(r["tp"] for r in all_results[m].values())
        fp = sum(r["fp"] for r in all_results[m].values())
        fn = sum(r["fn"] for r in all_results[m].values())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        cells.append(f"P={p:.2f} R={r:.2f} F1={f1:.2f} [{tp}/{fp}/{fn}]")
    print(f"{'СУММА (micro)':<24} | " + " | ".join(cells))


def main():
    data_dir = PROJECT_ROOT / "data"
    gt_dir = data_dir / "ground_truth"
    shots_dir = data_dir / "shots"  # <--- ИСПОЛЬЗУЕМ ПРАВИЛЬНУЮ ПАПКУ

    methods = discover_methods(data_dir)
    if not methods:
        print("Нет папок result_*")
        return

    print(f"Найдены методы: {list(methods.keys())}")

    run_eval(0, str(gt_dir), methods, str(shots_dir))
    run_eval(1, str(gt_dir), methods, str(shots_dir))


if __name__ == "__main__":
    main()