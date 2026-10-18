import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
SHOTS_DIR = DATA / "shots"
GT_DIR = DATA / "ground_truth"


def load_boundaries(csv_path):
    """Читает границы (end_time) из файла предсказаний."""
    if not csv_path.exists():
        return []
    bounds = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        # Берем end_time всех сцен, кроме последней
        for r in rows[:-1]:
            bounds.append(float(r["end_time"]))
    return bounds


def load_shots(video_name):
    """Читает таймкоды всех склеек из кэша."""
    path = SHOTS_DIR / f"{video_name}_shots.csv"
    if not path.exists():
        return []
    shots = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                try:
                    shots.append(float(row[0]))  # start_time шота
                except ValueError:
                    pass
    return sorted(list(set(shots)))


def get_shot_idx(time, shots):
    """Находит индекс ближайшего шота."""
    if not shots: return -1
    import bisect
    idx = bisect.bisect_left(shots, time)
    if idx == 0: return 0
    if idx == len(shots): return len(shots) - 1
    return idx if shots[idx] - time < time - shots[idx - 1] else idx - 1


def combine_two_methods(m1_bounds, m2_bounds, shots, mode="or", tol=1):
    """Комбинирует два списка границ."""
    if not shots: return []

    idx1 = set(get_shot_idx(t, shots) for t in m1_bounds)
    idx2 = set(get_shot_idx(t, shots) for t in m2_bounds)

    combined_idx = set()

    if mode == "and":
        combined_idx = idx1 & idx2
    elif mode == "or":
        # Объединяем, но убираем дубликаты (если разница в 1 шот)
        all_idx = sorted(idx1 | idx2)
        for idx in all_idx:
            if not combined_idx or idx - max(combined_idx) > tol:
                combined_idx.add(idx)

    # Возвращаем время (start_time) этих шотов
    return [shots[i] for i in sorted(combined_idx) if i < len(shots)]


def save_boundaries(bounds, video_name, out_dir):
    """Сохраняет объединенные границы в формате сцен."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_name}_data.csv"

    # Получаем длительность последней сцены из shots
    shots = load_shots(video_name)
    duration = shots[-1] if shots else 0.0
    if not shots:
        return

    # Нам нужна длительность видео. Берем из GT или последнего шота
    # Проще всего взять end_time последнего шота из файла shots
    shots_path = SHOTS_DIR / f"{video_name}_shots.csv"
    with open(shots_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        if rows:
            try:
                duration = float(rows[-1][1])  # end_time последнего шота
            except (ValueError, IndexError):
                pass

    pts = [0.0] + bounds + [duration]
    scenes = []
    for i, (start, end) in enumerate(zip(pts, pts[1:])):
        scenes.append({
            "scene_id": i,
            "start_time": start,
            "end_time": end,
            "cuts": 1  # Условно
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "start_time", "end_time", "cuts"])
        writer.writeheader()
        writer.writerows(scenes)


def main():
    videos = sorted([p.stem for p in GT_DIR.glob("*.csv")])

    # Наши комбинации
    combos = [
        ("hsv_dino", "hsv", "dino", "or"),
        ("hsv_clip", "hsv", "clip", "or"),
        ("hsv_dino", "hsv", "dino", "and"),  # Для сравнения
    ]

    for combo_name, m1, m2, mode in combos:
        # Если это уже сделано, пропускаем
        out_dir = DATA / f"result_{combo_name}_{mode}"
        print(f"\n=== Комбинация: {m1.upper()} + {m2.upper()} ({mode.upper()}) ===")

        for video in videos:
            b1 = load_boundaries(DATA / f"result_{m1}" / f"{video}_data.csv")
            b2 = load_boundaries(DATA / f"result_{m2}" / f"{video}_data.csv")
            shots = load_shots(video)

            combined = combine_two_methods(b1, b2, shots, mode=mode, tol=1)
            save_boundaries(combined, video, out_dir)

        print(f"Сохранено в {out_dir.name}")


if __name__ == "__main__":
    main()