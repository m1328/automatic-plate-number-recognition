import re
import cv2
import numpy as np
import pytesseract
from typing import List, Tuple


def normalize_plate_text(s: str) -> str:
    """Normalizuje tekst tablicy do wielkich liter i cyfr."""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def is_valid_polish_plate(text: str) -> bool:
    """
    Sprawdza, czy tekst ma cechy polskiej tablicy rejestracyjnej.
    """
    if not text or len(text) < 5 or len(text) > 8:
        return False

    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)

    return has_letter and has_digit


def fix_common_ocr_errors(text: str) -> str:
    """
    Koryguje typowe błędy OCR występujące na tablicach rejestracyjnych.
    Korekty stosowane są tylko w kontekście cyfr.
    """
    corrections = {
        'O': '0',
        'I': '1',
        'l': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
    }

    result = []
    for i, char in enumerate(text):
        neighbors_are_digits = False
        if i > 0 and text[i - 1].isdigit():
            neighbors_are_digits = True
        if i < len(text) - 1 and text[i + 1].isdigit():
            neighbors_are_digits = True

        if neighbors_are_digits and char in corrections:
            result.append(corrections[char])
        else:
            result.append(char)

    return ''.join(result)


def preprocess_plate_v2(plate_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Generuje kilka wariantów obrazu tablicy w celu poprawy skuteczności OCR.
    """
    results = []

    scale = 2.5
    h, w = plate_bgr.shape[:2]
    plate_large = cv2.resize(
        plate_bgr, (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(plate_large, cv2.COLOR_BGR2GRAY)

    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh1 = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    results.append(thresh1)

    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(thresh2)

    _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results.append(thresh3)

    return results


def run_tesseract_configs(img: np.ndarray, whitelist: str) -> List[str]:
    """
    Wykonuje pojedyncze wywołanie Tesseract OCR dla danego obrazu.
    """
    results = []

    psm_modes = [7, 8]

    for psm in psm_modes:
        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'
        try:
            text = pytesseract.image_to_string(img, config=config)
            text = normalize_plate_text(text)
            if text:
                results.append(text)
        except:
            continue

    return results


def score_plate_candidate(text: str, expected_length: int = 7) -> float:
    """
    Ocena jakości rozpoznanej tablicy na podstawie heurystyk.
    """
    if not text:
        return 0.0

    score = 0.0

    length_diff = abs(len(text) - expected_length)
    if length_diff == 0:
        score += 0.5
    elif length_diff == 1:
        score += 0.3
    elif length_diff == 2:
        score += 0.1

    if is_valid_polish_plate(text):
        score += 0.3

    if len(text) >= 5:
        first_part = text[:3]
        last_part = text[3:]

        if first_part.isalpha() and any(c.isdigit() for c in last_part):
            score += 0.2

    return min(score, 1.0)


def read_plate_text_improved(plate_bgr: np.ndarray, whitelist: str) -> str:
    """
    Główna funkcja OCR - próbuje wielu wariantów i wybiera najlepszy.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return ""

    # Generuj różne preprocessingi
    preprocessed_images = preprocess_plate_v2(plate_bgr)

    candidates = []

    # Dla każdego preprocessingu próbuj różnych konfiguracji Tesseract
    for img in preprocessed_images:
        texts = run_tesseract_configs(img, whitelist)
        candidates.extend(texts)

    if not candidates:
        return ""

    # Usuń duplikaty
    candidates = list(set(candidates))

    # Popraw błędy OCR
    candidates = [fix_common_ocr_errors(c) for c in candidates]

    # Oceń każdego kandydata
    scored = [(c, score_plate_candidate(c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Zwróć najlepszego
    return scored[0][0] if scored else ""


def read_plate_text_candidates(plate_bgr: np.ndarray, whitelist: str) -> Tuple[str, str]:
    """
    Zwraca dwa najlepsze kandydaty (dla kompatybilności z evaluate.py),
    ale z early-exit (duże przyspieszenie).
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return "", ""

    preprocessed_images = preprocess_plate_v2(plate_bgr)

    best = ""
    second = ""
    best_score = 0.0
    second_score = 0.0

    for img in preprocessed_images[:3]:
        texts = run_tesseract_configs(img, whitelist)

        for t in texts:
            if not t:
                continue

            t = fix_common_ocr_errors(t)
            sc = score_plate_candidate(t)

            if sc > best_score:
                second_score = best_score
                second = best
                best_score = sc
                best = t
            elif sc > second_score and t != best:
                second_score = sc
                second = t

            if best_score >= 0.95 and second_score >= 0.80:
                return best, (second or best)

    if not best:
        return "", ""

    return best, (second or best)


def read_plate_text(plate_bgr: np.ndarray, whitelist: str) -> str:
    return read_plate_text_improved(plate_bgr, whitelist)