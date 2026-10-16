import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matchs.matchs_bound import SceneEvaluator
from src.matchs.baselines import baseline_every_cut, baseline_every_n_seconds


def discover_methods(data_dir: Path):
    return {p.name.replace("result_", ""): str(p)
            for p in sorted(data_dir.glob("result_*")) if p.is_dir()}


def fmt_row(label, s):
    return (f"{label:<14} | "
            f"F1μ={s['f1_micro']:.3f} F1m={s['f1_macro']:.3f}±{s['f1_std']:.2f} | "
            f"P={s['precision_micro']:.3f} R={s['recall_micro']:.3f} | "
            f"TP={s['TP']:>4} FP={s['FP']:>4} FN={s['FN']:>4} | "
            f"#GT={s['n_gt_total']:>4}")


def main():
    data_dir = PROJECT_ROOT / "data"
    gt_dir   = str(data_dir / "ground_truth")
    cuts_dir = str(data_dir / "shots")

    if not Path(cuts_dir).exists():
        print(f"[!] Нет {cuts_dir}. Прогоните run_all.py — он сохранит шоты.")
        return

    ev = SceneEvaluator(tolerance_shots=1)

    print("=" * 110); print("БАЗОВЫЕ ЛИНИИ"); print("=" * 110)
    _, s_cut = baseline_every_cut(gt_dir, cuts_dir, ev)
    _, s_n   = baseline_every_n_seconds(gt_dir, cuts_dir, ev, N=30)
    print(fmt_row("every cut",  s_cut))
    print(fmt_row("every 30 s", s_n))

    print(); print("=" * 110); print("МЕТОДЫ"); print("=" * 110)
    results = {}
    for method, pred_dir in discover_methods(data_dir).items():
        per_video, summary = ev.evaluate_all(
            gt_dir=gt_dir, pred_dir=pred_dir, cuts_dir=cuts_dir
        )
        results[method] = (per_video, summary)
        print(fmt_row(method, summary))

    print(); print("=" * 110); print("PER-VIDEO F1"); print("=" * 110)
    videos  = sorted({v for pv, _ in results.values() for v in pv})
    methods = list(results.keys())
    header = f"{'video':<22} | " + " | ".join(f"{m:^10}" for m in methods)
    print(header); print("-" * len(header))
    for v in videos:
        cells = []
        for m in methods:
            r = results[m][0].get(v)
            cells.append(f"{r['f1']:.2f}" if r else "  —  ")
        print(f"{v:<22} | " + " | ".join(f"{c:^10}" for c in cells))


if __name__ == "__main__":
    main()