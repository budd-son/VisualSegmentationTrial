import cv2
import numpy as np


def frame_hsv_signature(frame_bgr, scale=0.5,
                        h_bins=8, s_bins=8, v_bins=8):
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [h_bins, s_bins, v_bins],
        [0, 180, 0, 256, 0, 256]
    )
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-9)
    return hist.astype(np.float32)


def shot_hsv_signature(video_path, shots,
                       scale=0.5, frame_step=5,
                       h_bins=8, s_bins=8, v_bins=8):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    result = []
    for shot in shots:
        start_f = max(0, int(round(shot["start_time"] * fps)))
        end_f = min(total_frames, int(round(shot["end_time"] * fps)))

        sigs = []
        for f in range(start_f, end_f, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                continue
            sigs.append(frame_hsv_signature(frame,
                                            scale=scale,
                                            h_bins=h_bins,
                                            s_bins=s_bins,
                                            v_bins=v_bins))

        if sigs:
            signature = np.mean(np.stack(sigs, axis=0), axis=0)
        else:
            signature = np.zeros(h_bins * s_bins * v_bins, dtype=np.float32)

        result.append({**shot, "signature": signature})

    cap.release()
    return result