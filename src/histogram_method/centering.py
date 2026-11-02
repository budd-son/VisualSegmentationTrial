import numpy as np


def center_embeddings(shots_with_sigs, method):
    """
    Вычитает средний эмбеддинг по фильму из каждого вектора и нормирует.
    Работает только для CLIP и DINO.
    """
    if method == "hsv":
        return shots_with_sigs

    # 1. Собираем все эмбеддинги со всех кадров
    all_embs = []
    for shot in shots_with_sigs:
        if shot["signature"]:
            all_embs.extend(shot["signature"])

    if not all_embs:
        return shots_with_sigs

    # 2. Считаем средний вектор
    all_embs = np.array(all_embs, dtype=np.float32)
    mean_emb = np.mean(all_embs, axis=0, keepdims=True)

    # 3. Вычитаем и нормируем
    new_shots = []
    for shot in shots_with_sigs:
        new_shot = dict(shot)
        if shot["signature"]:
            centered = np.array(shot["signature"], dtype=np.float32) - mean_emb
            # Нормировка по L2
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = centered / norms

            # ВАЖНО: Превращаем обратно в список векторов, чтобы similarity.py не падал
            new_shot["signature"] = [e for e in normalized]

        new_shots.append(new_shot)

    return new_shots