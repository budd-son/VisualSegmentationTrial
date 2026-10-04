import numpy as np
def hist_similarity(h1, h2):
    return float(np.minimum(h1, h2).sum())
def _make_scene(shots):
    start = shots[0]["start_time"]
    end = shots[-1]["end_time"]
    duration = end - start
    return {
        "start_time": start,
        "end_time": end,
        "duration": duration,
        "cuts": len(shots),
    }

def group_shots_by_hist(shots_with_sigs,
                        threshold=0.6,
                        lookback=5,
                        min_duration=2.0):
    """
    shots_with_sigs: список {start_time, end_time, signature}
    threshold: порог сходства гистограмм
    lookback: сколько последних шотов группы сравнивать
    min_duration: минимальная длительность сцены (в секундах)
    """
    scenes = []
    current = []

    for shot in shots_with_sigs:
        sig = shot["signature"]

        if not current:
            current.append(shot)
            continue

        recent = current[-lookback:]
        sims = [hist_similarity(s["signature"], sig) for s in recent]
        best = max(sims) if sims else 0.0

        if best >= threshold:
            current.append(shot)
            continue

        # новый шот не похож — но группа ещё короткая?
        group_start = current[0]["start_time"]
        group_end = current[-1]["end_time"]
        group_duration = group_end - group_start

        if group_duration < min_duration:
            current.append(shot)
        else:
            scenes.append(_make_scene(current))
            current = [shot]

    if current:
        scenes.append(_make_scene(current))

    return scenes