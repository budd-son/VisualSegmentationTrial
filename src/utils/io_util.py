import csv
import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def scenes_to_rows(scenes):
    rows = []
    for i, sc in enumerate(scenes):
        rows.append({
            "scene_id": i,
            "start_time": round(sc["start_time"], 3),
            "end_time": round(sc["end_time"], 3),
            "cuts": sc["cuts"],
        })
    return rows


def save_scenes(scenes, csv_path):
    if not scenes:
        print(f"[save_scenes] пусто, ничего не сохраняю: {csv_path}")
        return

    ensure_dir(os.path.dirname(csv_path) or ".")
    rows = scenes_to_rows(scenes)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)