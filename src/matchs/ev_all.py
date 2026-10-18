import sys
from pathlib import Path
from baselines import build

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matchs.matchs_bound import SceneEvaluator


def discover_methods(data_dir: Path):
    methods = {}

    for p in sorted(data_dir.glob("result_*")):
        if p.is_dir():
            method = p.name.replace("result_", "")
            methods[method] = str(p)
    return methods

def _counts(r):
    tp = r["tp"]
    fp = r["fp"]
    fn = r["fn"]
    n_gt = r.get("n_gt", tp + fn)
    n_pred = r.get("n_pred", tp + fp)
    return tp, fp, fn, n_gt, n_pred

def main():
    data_dir = PROJECT_ROOT / "data"
    gt_dir = str(data_dir / "ground_truth")
    tolerance = 2.0
    build()
    methods = discover_methods(data_dir)

    if not methods:
        print(f"[skip] нет папок result_* в {data_dir}")
        return

    print(f"Найдены методы: {list(methods.keys())}\n")
    print(f"GT: {gt_dir}\n")

    shots_dir = str(data_dir / "shots")
    ev = SceneEvaluator(tol_shots=1)  # допуск ровно +-1 шот

    all_results = {}
    for method, pred_dir in methods.items():
        all_results[method] = ev.evaluate_all(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            shots_dir=shots_dir
        )

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

    # ---------- п.3: суммарные TP/FP/FN ----------
    print("-" * len(header))
    cells = []
    totals = {}
    for m in methods_list:
        vals = list(all_results[m].values())
        if not vals:
            cells.append(f"{'—':^{col_w}}")
            continue
        tp = fp = fn = n_gt = n_pred = 0
        for r in vals:
            t, f_p, f_n, g, pr = _counts(r)
            tp += t
            fp += f_p
            fn += f_n
            n_gt += g
            n_pred += pr
        p = tp / n_pred if n_pred else 0.0
        rc = tp / n_gt if n_gt else 0.0
        f = 2 * p * rc / (p + rc) if p + rc else 0.0
        totals[m] = (tp, fp, fn, n_gt, n_pred)
        cells.append(
            f"P={p:.2f} R={rc:.2f} F1={f:.2f} "
            f"[{tp}/{fp}/{fn}]"
        )
    print(f"{'СУММА (micro)':<24} | " + " | ".join(cells))
    if totals:
        n_gt = next(iter(totals.values()))[3]
        n_videos = len(videos)
        print(f"\nGT всего: {n_gt} границ на {n_videos} видео "
              f"(~{n_gt / n_videos:.1f} на видео)")
        print("Суммарные TP/FP/FN по методам:")
        for m, (tp, fp, fn, g, pr) in totals.items():
            print(f"  {m:<12} TP={tp:<4} FP={fp:<4} FN={fn:<4} "
                  f"(GT={g}, pred={pr})")


if __name__ == "__main__":
    main()