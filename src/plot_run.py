import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.plotter import ScenePlotter

plotter = ScenePlotter(ROOT_DIR / "data")
results_dir = ROOT_DIR / "results"
results_dir.mkdir(exist_ok=True)

plotter.plot(tolerance=0, save_path=str(results_dir / "plot_tol0.png"))
plotter.plot(tolerance=1, save_path=str(results_dir / "plot_tol1.png"))