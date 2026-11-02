import csv
import bisect
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D


class ScenePlotter:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.gt_dir = data_dir / "ground_truth"
        self.shots_dir = data_dir / "shots"

    def _load_shots(self, video_name):
        path = self.shots_dir / f"{video_name}_shots.csv"
        if not path.exists(): return []
        out = []
        with open(path, encoding="utf-8") as f:
            for r in csv.reader(f):
                try:
                    out.append(float(r[0]))
                except (ValueError, IndexError):
                    pass
        return sorted(set(out))

    def _dedup(self, bounds, shots, tol=1):
        if not shots: return bounds
        idxs = sorted(set(bisect.bisect_left(shots, b) for b in bounds))
        keep = []
        for i in idxs:
            if not keep or i - keep[-1] > tol:
                keep.append(i)
        return [shots[i] for i in keep if i < len(shots)]

    def _read_bounds(self, path):
        if not path.exists(): return []
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if not rows: return []
            end_col = next((k for k in rows[0] if k.startswith("end")), None)
            return [float(r[end_col]) for r in rows[:-1]]

    def calculate_metrics(self, pred_dir, is_ensemble=False, tol_shots=0):
        tp = fp = fn = 0
        for gt_path in sorted(self.gt_dir.glob("*.csv")):
            video = gt_path.stem
            pred_path = pred_dir / f"{video}_data.csv"
            if not pred_path.exists(): continue

            gt_bounds = self._read_bounds(gt_path)
            pred_bounds = self._read_bounds(pred_path)
            shots = self._load_shots(video)

            if not is_ensemble:
                pred_bounds = self._dedup(pred_bounds, shots)

            # Перевод в индексы шотов для точного допуска
            gt_idx = [bisect.bisect_left(shots, g) for g in gt_bounds]
            pred_idx = [bisect.bisect_left(shots, p) for p in pred_bounds]

            matched_gt = set()
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
                    fp += 1
            fn += len(gt_idx) - len(matched_gt)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return p, r

    def plot(self, tolerance=0, save_path="pr_curve.png", title="PR-кривая"):
        result_dirs = sorted([d for d in self.data_dir.glob("result_*") if d.is_dir()])
        points = {}

        print(f"Считаем метрики (допуск ±{tolerance}):")
        for d in result_dirs:
            method = d.name.replace("result_", "")
            if not any(d.glob("*_data.csv")): continue

            is_ensemble = "_or" in method or "_and" in method
            p, r = self.calculate_metrics(d, is_ensemble, tol_shots=tolerance)
            if p == 0 and r == 0: continue
            points[method] = {"p": p, "r": r}
            print(f"  {method}: P={p:.2f}, R={r:.2f}")

        if not points:
            print("Нет данных для отображения")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Изолинии F1
        r_vals = np.linspace(0.01, 1.0, 200)
        for f1 in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            p_vals = (f1 * r_vals) / (2 * r_vals - f1)
            mask = (p_vals >= 0.0) & (p_vals <= 1.05)
            ax.plot(r_vals[mask], p_vals[mask], color='lightgray', linestyle='--', alpha=0.7)
            idx = np.where((r_vals[mask] > 0.85) & (p_vals[mask] > 0.05))[0]
            if len(idx) > 0:
                ax.text(0.88, p_vals[mask][idx[-1]], f'F1 = {f1}', color='gray', fontsize=9)

        colors = ['dodgerblue', 'navy', 'purple', 'limegreen', 'coral', 'tomato', 'gold', 'teal']
        markers = ['*', 'p', 'D', 's', 'o', '^', 'h', 'v']

        manual_offsets = {
            "hsv_speech": (-70, -20), "clip_speech": (15, -20), "dino_speech": (-50, 15),
            "hsv": (10, -15), "clip": (10, 15), "dino": (15, 15),
            "allcuts": (10, 10), "grid": (10, -20)
        }

        for i, name in enumerate(sorted(points.keys())):
            data = points[name]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            size = 200 if marker == '*' else 100

            ax.scatter(data['r'], data['p'], color=color, s=size, marker=marker,
                       edgecolor='black', linewidth=0.8, zorder=5)

            offset = manual_offsets.get(name, (8, 8))
            ax.annotate(name, (data['r'], data['p']), xytext=offset, textcoords='offset points',
                        fontsize=10, fontweight='bold' if marker == '*' else 'normal', zorder=6)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.1)
        ax.set_xlabel('Recall (полнота)', fontsize=13)
        ax.set_ylabel('Precision (точность)', fontsize=13)
        ax.set_title(f'{title} (Допуск ±{tolerance} шот)', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)

        legend_elements = [
            Line2D([0], [0], marker='*', color='w', label='Методы', markerfacecolor='gray', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Baselines', markerfacecolor='gray', markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\nГрафик сохранен: {save_path}")
        plt.show()
        plt.close()

