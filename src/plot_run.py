import sys
from pathlib import Path

# 1. Добавляем корень проекта в пути
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# 2. Импортируем класс
from src.utils.plotter import ScenePlotter

# 3. Создаем объект и строим графики
plotter = ScenePlotter(ROOT_DIR / "data")

# График для строгой метрики (допуск 0)
plotter.plot(tolerance=0, save_path="plot_tol0.png", title="PR-кривая")

# График для мягкой метрики (допуск 1)
plotter.plot(tolerance=1, save_path="plot_tol1.png", title="PR-кривая")