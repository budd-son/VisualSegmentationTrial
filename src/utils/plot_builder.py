import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matchs.evaluate import get_all_metrics


def plot_pr_curve():
    # 1. Получаем свежие цифры для строгой метрики (допуск 0)
    metrics = get_all_metrics(tolerance=0)

    # 2. Настраиваем визуал (какой метод как рисуем)
    # Формат: "имя_метода": (маркер, цвет, размер)
    styles = {
        # Наши методы
        "hsv": ('*', 'dodgerblue', 200),
        "clip": ('*', 'blue', 200),
        "dino": ('*', 'navy', 200),

        # Комбинации
        "hsv_dino_or": ('p', 'purple', 120),
        "hsv_clip_or": ('p', 'mediumorchid', 120),
        "hsv_dino_and": ('p', 'orchid', 120),

        # Речь
        "hsv_speech": ('D', 'limegreen', 100),
        "clip_speech": ('D', 'palegreen', 100),
        "dino_speech": ('D', 'lightgreen', 100),

        # Baseline
        "allcuts": ('o', 'coral', 100),
        "grid": ('s', 'tomato', 100),
    }

    # Красивые имена для легенды и подписей
    nice_names = {
        "hsv": "HSV (Fixed)", "clip": "CLIP (Fixed)", "dino": "DINOv2 (Fixed)",
        "hsv_dino_or": "HSV + DINO (OR)", "hsv_clip_or": "HSV + CLIP (OR)",
        "hsv_dino_and": "HSV + DINO (AND)",
        "hsv_speech": "HSV + Speech", "clip_speech": "CLIP + Speech", "dino_speech": "DINOv2 + Speech",
        "allcuts": "All Cuts", "grid": "Grid 30s"
    }

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
    for method_name, data in metrics.items():
        if method_name not in styles:
            continue  # Пропускаем, если метод не задан в styles (например, перцентиль, если его папки нет)

        marker, color, size = styles[method_name]
        label = nice_names.get(method_name, method_name)

        # Рисуем точку
        ax.scatter(data['r'], data['p'], color=color, s=size, marker=marker,
                   edgecolor='black', linewidth=0.8, zorder=5)

        # Подписываем
        # Для звезд смещаем вправо-вверх, для остальных вправо-вниз (чтобы не перекрывались)
        offset = (8, 8) if marker == '*' else (5, -12)
        ax.annotate(label, (data['r'], data['p']), xytext=offset, textcoords='offset points',
                    fontsize=10, fontweight='bold' if marker == '*' else 'normal',
                    color='black', zorder=6)

    # Добавляем фиктивные элементы в легенду (так как мы рисуем циклом, метки дублируются)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Fixed', markerfacecolor='gray', markersize=15),
        Line2D([0], [0], marker='p', color='w', label='Ensemble (OR/AND)', markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='D', color='w', label='Speech Filter', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Baselines', markerfacecolor='gray', markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11, framealpha=0.9)

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel('Recall (полнота) — Допуск 0', fontsize=14)
    ax.set_ylabel('Precision (точность)', fontsize=14)
    ax.set_title('PR-кривая (строгая метрика): Сравнение методов', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig("pr_curve_comparison.png", dpi=300)
    print("График сохранен: pr_curve_comparison.png (свежие данные, допуск 0)")
    plt.show()


if __name__ == "__main__":
    plot_pr_curve()