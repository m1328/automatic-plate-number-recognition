import re
import cv2
import numpy as np
import pytesseract


def normalize_plate_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def preprocess_for_ocr(plate_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # zwiększ kontrast
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )

    # jeśli obraz jest "za biały" (znaki znikają), spróbuj odwrócić
    white_ratio = (th > 0).mean()
    if white_ratio > 0.85:
        th = cv2.bitwise_not(th)

    # delikatne czyszczenie
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    return th

def read_plate_by_chars(plate_bgr: np.ndarray, whitelist: str) -> str:
    img = preprocess_for_ocr(plate_bgr)  # binarka 0/255

    # upewnij się, że znaki są czarne na białym tle
    if (img > 0).mean() > 0.85:
        img = cv2.bitwise_not(img)

    h, w = img.shape[:2]

    # znajdź kontury potencjalnych znaków
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh

        # filtry szumu (można później dopasować)
        if area < 120:
            continue
        if hh < 0.35 * h:
            continue
        if ww < 0.02 * w:
            continue
        if ww > 0.35 * w:
            continue

        boxes.append((x, y, ww, hh))

    if not boxes:
        return ""

    # sortuj od lewej do prawej
    boxes.sort(key=lambda b: b[0])

    chars = []
    for (x, y, ww, hh) in boxes:
        ch_img = img[y:y + hh, x:x + ww]

        # powiększ znak
        ch_img = cv2.resize(ch_img, (45, 70), interpolation=cv2.INTER_CUBIC)

        cfg = f"--oem 3 --psm 10 -c tessedit_char_whitelist={whitelist}"
        raw = pytesseract.image_to_string(ch_img, config=cfg)
        ch = normalize_plate_text(raw)

        if ch:
            chars.append(ch[0])

    text = "".join(chars)
    return text

def fix_common_ocr_confusions(s: str) -> str:
    # typowe pomyłki dla tablic
    repl = str.maketrans({
        "I": "1",
        "L": "1",
    })
    return s.translate(repl)

def read_plate_text(plate_bgr: np.ndarray, whitelist: str) -> str:
    img = preprocess_for_ocr(plate_bgr)

    # OCR całej tablicy (2 ustawienia)
    configs = [
        f"--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}",
        f"--oem 3 --psm 8 -c tessedit_char_whitelist={whitelist}",
    ]

    best_line = ""
    for cfg in configs:
        raw = pytesseract.image_to_string(img, config=cfg)
        text = normalize_plate_text(raw)
        if len(text) > len(best_line):
            best_line = text

    # OCR znak po znaku
    by_chars = read_plate_by_chars(plate_bgr, whitelist)

    # postprocess pomyłek (łagodnie)
    best_line = fix_common_ocr_confusions(best_line)
    by_chars = fix_common_ocr_confusions(by_chars)

    # wybierz to, co ma sensowną długość (PL zwykle 6–8)
    candidates = [best_line, by_chars]
    candidates.sort(key=lambda s: (abs(len(s) - 7), -len(s)))

    return candidates[0]
