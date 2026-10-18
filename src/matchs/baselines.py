import csv
from pathlib import Path

import numpy as np


def load_shots(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # Пропускаем заголовок, если он есть (например, start_time,end_time)
    if rows:
        try:
            float(rows[0][0])
        except ValueError:
            rows = rows[1:]

    return [(float(r[0]), float(r[1])) for r in rows]


def all_cuts(shots, duration):
    return [a for a, _ in shots[1:]]


def grid(shots, duration, step=30.0):
    return [float(t) for t in np.arange(step, duration, step)]


def boundaries_to_scenes(boundaries, duration):
    pts = [0.0] + sorted(boundaries) + [duration]
    return [{"scene_id": i, "start_time": a, "end_time": b, "cuts": 1}
            for i, (a, b) in enumerate(zip(pts, pts[1:])) if b > a]


def write_scenes(path, scenes):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id", "start_time", "end_time", "cuts"])
        w.writeheader()
        w.writerows(scenes)


def build(step=30.0):
    # Автоматически находим корень проекта (папку, где лежит data)
    root = Path(__file__).resolve().parent
    while not (root / "data").is_dir() and root.parent != root:
        root = root.parent

    data = root / "data"
    shots_dir, gt_dir = data / "shots", data / "ground_truth"

    if not shots_dir.is_dir():
        print(f"[build] нет {shots_dir}")
        return 0

    shot_files = sorted(shots_dir.glob("*_shots.csv"))
    if not shot_files:
        print(f"[build] нет *_shots.csv в {shots_dir}")
        return 0

    methods = {"allcuts": all_cuts, "grid": lambda s, d: grid(s, d, step)}
    n = 0
    for path in shot_files:
        # Получаем чистое имя: fightClub_shots.csv -> fightClub
        video = path.stem.replace("_shots", "")
        try:
            shots = load_shots(path)
        except Exception as e:
            print(f"[skip] {video}: ошибка чтения {e}")
            continue

        if len(shots) < 2:
            print(f"[skip] {video}: шотов {len(shots)}")
            continue

        duration = shots[-1][1]
        for name, fn in methods.items():
            out = data / f"result_{name}"
            out.mkdir(parents=True, exist_ok=True)
            scenes = boundaries_to_scenes(fn(shots, duration), duration)
            write_scenes(out / f"{video}_data.csv", scenes)
            n += 1

    return n




