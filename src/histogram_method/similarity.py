import numpy as np


def hist_similarity(h1, h2):
    return float(np.minimum(h1, h2).sum())


def cosine_similarity(e1, e2):
    n1 = np.linalg.norm(e1)
    n2 = np.linalg.norm(e2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.dot(e1, e2) / (n1 * n2))   # без нормирования


"""def similarity(sig1, sig2, method="hsv"):
    if method == "hsv":
        return hist_similarity(sig1, sig2)
    elif method in ("clip", "dino"):
        return cosine_similarity(sig1, sig2)
    else:
        raise ValueError(f"Unknown method: {method}")"""
def multi_embedding_similarity(embs_a, embs_b):
    """Максимум попарных косинусов между двумя наборами."""
    if not embs_a or not embs_b:
        return 0.0
    best = -1.0
    for a in embs_a:
        na = np.linalg.norm(a)
        if na < 1e-9:
            continue
        for b in embs_b:
            nb = np.linalg.norm(b)
            if nb < 1e-9:
                continue
            c = float(np.dot(a, b) / (na * nb))
            if c > best:
                best = c
    return best


def similarity(sig1, sig2, method="hsv"):
    if method == "hsv":
        return hist_similarity(sig1, sig2)
    elif method in ("clip", "dino"):
        return multi_embedding_similarity(sig1, sig2)
    else:
        raise ValueError(f"Unknown method: {method}")