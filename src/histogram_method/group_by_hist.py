import numpy as np

from src.histogram_method.similarity import similarity


def compute_adaptive_threshold(shots_with_sigs, method, k=1.0):
    """
    Порог = mean(sim между соседними шотами) - k * std(...).
    k=0.5 — мягкий (мало границ),
    k=1.0 — средний,
    k=1.5 — жёсткий (много границ).
    """
    sims = []
    for i in range(len(shots_with_sigs) - 1):
        s = similarity(
            shots_with_sigs[i]["signature"],
            shots_with_sigs[i + 1]["signature"],
            method=method,
        )
        sims.append(s)

    if not sims:
        return 0.0

    sims = np.array(sims, dtype=np.float32)
    mean_sim = float(sims.mean())
    std_sim = float(sims.std())
    thr = mean_sim - k * std_sim

    print(f"[{method}] adaptive threshold: mean={mean_sim:.3f}, "
          f"std={std_sim:.3f}, k={k} → thr={thr:.3f}")

    return thr


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
                        threshold=None,
                        lookback=5,
                        min_duration=2.0,
                        method="hsv",
                        k=1.0):
    """
    threshold: если None → адаптивный порог по соседним шотам.
    k: коэффициент для адаптивного порога.
    """
    if threshold is None:
        threshold = compute_adaptive_threshold(shots_with_sigs, method, k=k)

    scenes = []
    current = []

    for shot in shots_with_sigs:
        sig = shot["signature"]

        if not current:
            current.append(shot)
            continue

        recent = current[-lookback:]
        sims = [similarity(s["signature"], sig, method=method) for s in recent]
        best = max(sims) if sims else 0.0

        if best >= threshold:
            current.append(shot)
            continue

        group_duration = current[-1]["end_time"] - current[0]["start_time"]
        if group_duration < min_duration:
            current.append(shot)
        else:
            scenes.append(_make_scene(current))
            current = [shot]

    if current:
        scenes.append(_make_scene(current))

    return scenes