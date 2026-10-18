import matplotlib.pyplot as plt
import numpy as np

# --- 1. ФИКСИРОВАННЫЕ МЕТОДЫ (Fixed) ---
our_methods_fixed = {
    "HSV (Fixed)": {"p": 0.62, "r": 0.86, "color": "dodgerblue"},
    "CLIP (Fixed)": {"p": 0.50, "r": 1.00, "color": "blue"},
    "DINOv2 (Fixed)": {"p": 0.61, "r": 0.86, "color": "navy"},
}

# --- 2. ПЕРЦЕНТИЛЬНЫЕ МЕТОДЫ (Percentile) ---
our_methods_pct = {
    "HSV (Pct)": {"p": 0.71, "r": 0.49, "color": "lightskyblue"},
    "CLIP (Pct)": {"p": 0.91, "r": 0.29, "color": "skyblue"},
    "DINOv2 (Pct)": {"p": 0.74, "r": 0.40, "color": "lightblue"},
}

# --- 3. МЕТОДЫ С РЕЧЕВЫМ ФИЛЬТРОМ (Speech) ---
our_methods_speech = {
    "HSV + Speech": {"p": 0.75, "r": 0.43, "color": "limegreen"},
    "DINOv2 + Speech": {"p": 0.71, "r": 0.49, "color": "lightgreen"},
    "CLIP + Speech": {"p": 0.52, "r": 0.49, "color": "palegreen"},
}

# --- 4. БАЗОВЫЕ ЛИНИИ (Baselines) ---
baselines = {
    "All Cuts": {"p": 0.19, "r": 1.00, "marker": "o", "color": "coral"},
    "Grid 30s": {"p": 0.52, "r": 0.43, "marker": "s", "color": "tomato"},
}


def plot_pr_curve():
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

    # 1. Fixed (Звезды)
    for name, data in our_methods_fixed.items():
        ax.scatter(data['r'], data['p'], color=data['color'], s=200, marker='*',
                   edgecolor='black', linewidth=0.8, zorder=5,
                   label='Fixed' if name == "HSV (Fixed)" else "")
        ax.annotate(name, (data['r'], data['p']), xytext=(8, 8), textcoords='offset points',
                    fontsize=11, fontweight='bold', color='black', zorder=6)

    # 2. Percentile (Треугольники)
    for name, data in our_methods_pct.items():
        ax.scatter(data['r'], data['p'], color=data['color'], s=100, marker='^',
                   edgecolor='gray', linewidth=0.8, zorder=4,
                   label='Percentile' if name == "HSV (Pct)" else "")
        ax.annotate(name, (data['r'], data['p']), xytext=(5, -15), textcoords='offset points',
                    fontsize=9, color='dimgray', zorder=6)

    # 3. Speech (Ромбы)
    for name, data in our_methods_speech.items():
        ax.scatter(data['r'], data['p'], color=data['color'], s=100, marker='D',
                   edgecolor='black', linewidth=0.8, zorder=4,
                   label='Fixed + Speech' if name == "HSV + Speech" else "")
        ax.annotate(name, (data['r'], data['p']), xytext=(-10, 12), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='darkgreen', zorder=6)

    # 4. Baselines (Квадраты и круги)
    for name, data in baselines.items():
        ax.scatter(data['r'], data['p'], color=data['color'], s=100, marker=data['marker'],
                   edgecolor='black', linewidth=0.8, zorder=3,
                   label='Baselines' if name == "All Cuts" else "")
        ax.annotate(name, (data['r'], data['p']), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, color='black', zorder=6)

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel('Recall (полнота)', fontsize=14)
    ax.set_ylabel('Precision (точность)', fontsize=14)
    ax.set_title('PR-кривая: Сравнение методов сегментации сцен', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, zorder=0)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("pr_curve_comparison.png", dpi=300)
    print("График сохранен: pr_curve_comparison.png")
    plt.show()


if __name__ == "__main__":
    plot_pr_curve()