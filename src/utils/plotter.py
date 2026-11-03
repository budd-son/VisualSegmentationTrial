import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matchs.matchs_bound import SceneEvaluator


class ScenePlotter:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.gt_dir = data_dir / "ground_truth"
        self.shots_dir = data_dir / "shots"

    def plot(self, tolerance=0, save_path="plot.png"):
        methods = {}
        for p in sorted(self.data_dir.glob("result_*")):
            if p.is_dir() and any(p.glob("*_data.csv")):
                methods[p.name.replace("result_", "")] = str(p)

        if not methods: return

        ev = SceneEvaluator()
        points = {}
        for name, pred_dir in methods.items():
            res = ev.evaluate_all(str(self.gt_dir), pred_dir, str(self.shots_dir), tol_shots=tolerance)
            tp = sum(r["tp"] for r in res.values())
            fp = sum(r["fp"] for r in res.values())
            fn = sum(r["fn"] for r in res.values())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if p > 0 or r > 0:
                points[name] = (p, r)

        if not points: return

        fig, ax = plt.subplots(figsize=(12, 8))
        r_vals = np.linspace(0.01, 1.0, 200)
        for f1 in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            p_vals = (f1 * r_vals) / (2 * r_vals - f1)
            mask = (p_vals >= 0.0) & (p_vals <= 1.05)
            ax.plot(r_vals[mask], p_vals[mask], color='lightgray', linestyle='--', alpha=0.7)
            idx = np.where((r_vals[mask] > 0.85) & (p_vals[mask] > 0.05))[0]
            if len(idx) > 0:
                ax.text(0.88, p_vals[mask][idx[-1]], f'F1 = {f1}', color='gray', fontsize=9)

        colors = ['dodgerblue', 'navy', 'purple', 'limegreen', 'coral', 'tomato']
        markers = ['*', 'p', 'D', 's', 'o', '^']

        for i, (name, (p, r)) in enumerate(sorted(points.items())):
            size = 200 if markers[i % len(markers)] == '*' else 100
            ax.scatter(r, p, color=colors[i % len(colors)], s=size,
                       marker=markers[i % len(markers)], edgecolor='black', zorder=5)
            ax.annotate(name, (r, p), xytext=(8, 8), textcoords='offset points', fontsize=10)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.1)
        ax.set_xlabel('Recall (полнота)', fontsize=13)
        ax.set_ylabel('Precision (точность)', fontsize=13)
        ax.set_title(f'PR-кривая (Допуск ±{tolerance} шот)', fontsize=15)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()