import sys
import csv
import bisect
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


# --- НОВЫЕ ФУНКЦИИ ДЛЯ ДЕДУПЛИКАЦИИ ---
def load_shots_for_video(video_name, shots_dir):
    path = Path(shots_dir) / f"{video_name}_shots.csv"
    if not path.exists(): return []
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.reader(f):
            try:
                out.append(float(r[0]))
            except (ValueError, IndexError):
                pass
    return sorted(set(out))


def get_shot_idx(time, shots):
    if not shots: return -1
    idx = bisect.bisect_left(shots, time)
    if idx == 0: return 0
    if idx == len(shots): return len(shots) - 1
    return idx if shots[idx] - time < time - shots[idx - 1] else idx - 1


def dedup_boundaries(boundaries, shots, tol=1):
    if not shots: return boundaries
    idxs = sorted(set(get_shot_idx(b, shots) for b in boundaries))
    keep = []
    for i in idxs:
        if not keep or i - keep[-1] > tol:
            keep.append(i)
    return [shots[i] for i in keep if i < len(shots)]


def run_evaluation(tolerance, gt_dir, methods, shots_dir):
    print(f"\n{'=' * 60}")
    print(f" ОЦЕНКА С ДОПУСКОМ: ±{tolerance} шот(а) (с дедупликацией)")
    print(f"{'=' * 60}\n")

    ev = SceneEvaluator(tol_shots=tolerance)
    all_results = {}

    for method, pred_dir in methods.items():
        # Проверяем, является ли метод комбинацией (тогда дедупликация уже сделана)
        is_ensemble = "_or" in method or "_and" in method

        method_results = {}
        for gt_path in sorted(Path(gt_dir).glob("*.csv")):
            video_name = gt_path.stem
            pred_path = Path(pred_dir) / f"{video_name}_data.csv"

            if not pred_path.exists():
                continue

            # 1. Читаем сырые предсказания
            pred_bounds = ev._read_boundaries(str(pred_path), "start", "end")
            gt_bounds = ev._read_boundaries(str(gt_path), "start", "end")
            shots = load_shots_for_video(video_name, shots_dir)

            # 2. Дедупликация для одиночных методов
            if not is_ensemble:
                pred_bounds = dedup_boundaries(pred_bounds, shots, tol=1)

            # 3. Считаем метрики
            tp, fp, fn = ev._match(pred_bounds, gt_bounds, shots)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            method_results[video_name] = {
                "precision": p, "recall": r, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn,
                "n_gt": len(gt_bounds), "n_pred": len(pred_bounds)
            }

        all_results[method] = method_results

    # --- ДАЛЬШЕ ИДЕТ ТВОЙ КОД ВЫВОДА ТАБЛИЦЫ (без изменений) ---
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
    print("\n")


def main():
    data_dir = PROJECT_ROOT / "data"
    gt_dir = str(data_dir / "ground_truth")
    shots_dir = str(data_dir / "old_shots")

    build()
    methods = discover_methods(data_dir)

    if not methods:
        print(f"[skip] нет папок result_* в {data_dir}")
        return

    print(f"Найдены методы: {list(methods.keys())}")
    print(f"GT: {gt_dir}")

    run_evaluation(0, gt_dir, methods, shots_dir)
    run_evaluation(1, gt_dir, methods, shots_dir)


if __name__ == "__main__":
    main()