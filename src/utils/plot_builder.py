import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matchs.evaluate import get_all_metrics


def plot_pr_curve():
    # 1. Получаем свежие цифры для строгой метрики (допуск 0)
    metrics = get_all_metrics(tolerance=0)

    # 2. Автоматически назначаем стили для любых методов
    colors = ['dodgerblue', 'navy', 'purple', 'limegreen', 'coral', 'tomato', 'gold', 'teal']
    markers = ['*', 'p', 'D', 's', 'o', '^', 'h', 'v']

    fig, ax = plt.subplots(figsize=(12, 8))

    # Рисуем изолинии F1
    r_vals = np.linspace(0.01, 1.0, 200)
    f1_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for f1 in f1_levels:
        p_vals = (f1 * r_vals) / (2 * r_vals - f1)
        p_vals[p_vals < 0] = np.nan
        ax.plot(r_vals, p_vals, color='lightgray', linestyle='--', alpha=0.7, zorder=1)

        idx = np.where((r_vals > 0.85) & (p_vals > 0.05))[0]
        if len(idx) > 0:
            ax.text(0.88, p_vals[idx[-1]], f'F1 = {f1}', color='gray', fontsize=9, verticalalignment='bottom')

    # Рисуем точки
    for i, (method_name, data) in enumerate(metrics.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Размер точки: baseline меньше, методы больше
        size = 100 if ("allcuts" in method_name or "grid" in method_name) else 180

        ax.scatter(data['r'], data['p'], color=color, s=size, marker=marker,
                   edgecolor='black', linewidth=0.8, zorder=5)

        # Подписываем метод
        ax.annotate(method_name, (data['r'], data['p']), xytext=(8, 8), textcoords='offset points',
                    fontsize=10, fontweight='bold' if marker == '*' else 'normal',
                    color='black', zorder=6)

    # Легенда
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Наши методы', markerfacecolor='gray', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Baselines', markerfacecolor='gray', markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11, framealpha=0.9)

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel('Recall (полнота) — Допуск 0', fontsize=14)
    ax.set_ylabel('Precision (точность)', fontsize=14)
    ax.set_title('PR-кривая: Сравнение методов сегментации сцен', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig("pr_curve_comparison.png", dpi=300)
    print("График сохранен: pr_curve_comparison.png")
    plt.show()


if __name__ == "__main__":
    plot_pr_curve()