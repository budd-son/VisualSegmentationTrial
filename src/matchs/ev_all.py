# scripts/eval_all.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matchs.matchs_bound import SceneEvaluator


def discover_methods(data_dir: Path):
    """Находит все папки result_* в data/ → {method: path}."""
    methods = {}
    for p in sorted(data_dir.glob("result_*")):
        if p.is_dir():
            method = p.name.replace("result_", "")
            methods[method] = str(p)
    return methods


def main():
    data_dir = PROJECT_ROOT / "data"
    gt_dir = str(data_dir / "ground_truth")
    tolerance = 2.0

    methods = discover_methods(data_dir)
    if not methods:
        print(f"[skip] нет папок result_* в {data_dir}")
        return

    print(f"Найдены методы: {list(methods.keys())}\n")
    print(f"GT: {gt_dir}\n")

    ev = SceneEvaluator(tolerance=tolerance)
    all_results = {}
    for method, pred_dir in methods.items():
        all_results[method] = ev.evaluate_all(gt_dir=gt_dir, pred_dir=pred_dir)

    videos = set()
    for r in all_results.values():
        videos.update(r.keys())
    videos = sorted(videos)

    methods_list = list(all_results.keys())
    col_w = 24

    header = f"{'video':<24} | " + " | ".join(
        f"{m:^{col_w}}" for m in methods_list
    )
    print(header)
    print("-" * len(header))

    for v in videos:
        cells = []
        for m in methods_list:
            r = all_results[m].get(v)
            if r is None:
                cells.append(f"{'—':^{col_w}}")
            else:
                cells.append(
                    f"P={r['precision']:.2f} R={r['recall']:.2f} "
                    f"F1={r['f1']:.2f}"
                )
        print(f"{v:<24} | " + " | ".join(cells))

    print("-" * len(header))
    cells = []
    for m in methods_list:
        vals = list(all_results[m].values())
        if not vals:
            cells.append(f"{'—':^{col_w}}")
            continue
        p = sum(r["precision"] for r in vals) / len(vals)
        rc = sum(r["recall"] for r in vals) / len(vals)
        f = sum(r["f1"] for r in vals) / len(vals)
        cells.append(f"P={p:.2f} R={rc:.2f} F1={f:.2f}")
    print(f"{'СРЕДНЕЕ':<24} | " + " | ".join(cells))


if __name__ == "__main__":
    main()