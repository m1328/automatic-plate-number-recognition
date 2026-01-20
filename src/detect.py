import cv2
import numpy as np


def _refine_plate_in_roi(roi_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Szuka tablicy w już wyciętym ROI i zwraca bbox w układzie ROI.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # binarka + krawędzie
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )
    edges = cv2.Canny(th, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0
    h, w = gray.shape[:2]

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < 500:
            continue

        aspect = ww / float(hh + 1e-6)
        if not (2.0 <= aspect <= 8.0):
            continue

        # tablica raczej nie zajmuje całego ROI
        if ww > 0.95 * w or hh > 0.95 * h:
            continue

        score = area * (1.0 / (abs(aspect - 4.5) + 1.0))
        if score > best_score:
            best_score = score
            best = (x, y, x + ww, y + hh)

    return best


def detect_plate_bbox(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = bgr.shape[:2]

    # resize dla szybkości
    target_w = 1280
    scale = 1.0
    if w > target_w:
        scale = target_w / w
        bgr_small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    else:
        bgr_small = bgr

    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 60, 180)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < 800:
            continue

        aspect = ww / float(hh + 1e-6)
        if not (2.0 <= aspect <= 8.0):
            continue

        score = area * (1.0 / (abs(aspect - 4.0) + 1.0))
        if score > best_score:
            best_score = score
            best = (x, y, x + ww, y + hh)

    # fallback: środkowo-dolna część (jak miałaś)
    if best is None:
        hs, ws = bgr_small.shape[:2]
        x1 = int(ws * 0.2)
        x2 = int(ws * 0.8)
        y1 = int(hs * 0.55)
        y2 = int(hs * 0.90)
        best = (x1, y1, x2, y2)

    # refine: szukamy tablicy w ROI
    x1, y1, x2, y2 = best
    roi = bgr_small[y1:y2, x1:x2]
    refined = _refine_plate_in_roi(roi)

    if refined is not None:
        rx1, ry1, rx2, ry2 = refined
        best = (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2)

    # przeskaluj do oryginału
    if scale != 1.0:
        x1, y1, x2, y2 = best
        x1 = int(x1 / scale); y1 = int(y1 / scale)
        x2 = int(x2 / scale); y2 = int(y2 / scale)
        best = (x1, y1, x2, y2)

    # clamp
    x1, y1, x2, y2 = best
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return (x1, y1, x2, y2)


def crop_bbox(bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return bgr[y1:y2, x1:x2].copy()
