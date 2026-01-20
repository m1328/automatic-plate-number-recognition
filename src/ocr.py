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
    Sprawdza czy tekst przypomina polską tablicę.
    Formaty: ABC1234, AB12345, ABC123D itp.
    """
    if not text or len(text) < 5 or len(text) > 8:
        return False

    # Powinny być litery i cyfry
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)

    return has_letter and has_digit


def fix_common_ocr_errors(text: str) -> str:
    """Poprawia typowe błędy OCR dla polskich tablic."""
    # Mapowanie częstych pomyłek
    corrections = {
        'O': '0',  # litera O -> cyfra 0 w kontekście cyfr
        'I': '1',  # litera I -> cyfra 1
        'l': '1',  # mała litera l -> cyfra 1
        'Z': '2',  # czasem Z -> 2
        'S': '5',  # czasem S -> 5
        'B': '8',  # czasem B -> 8
    }

    # Stosuj korekty inteligentnie - tylko w kontekście cyfr
    result = []
    for i, char in enumerate(text):
        # Jeśli znak jest otoczony cyframi, prawdopodobnie też powinien być cyfrą
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
    Zwraca wiele wariantów preprocessingu - zwiększa szansę na sukces.
    """
    results = []

    # Powiększ obraz (ważne dla OCR)
    scale = 3.0
    h, w = plate_bgr.shape[:2]
    plate_large = cv2.resize(plate_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_large, cv2.COLOR_BGR2GRAY)

    # Wariant 1: Bilateral filter + adaptive threshold
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh1 = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    results.append(thresh1)

    # Wariant 2: Otsu thresholding
    blurred2 = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh2 = cv2.threshold(blurred2, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(thresh2)

    # Wariant 3: Inverted Otsu
    results.append(cv2.bitwise_not(thresh2))

    # Wariant 4: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    results.append(morph)

    # Wariant 5: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, thresh5 = cv2.threshold(enhanced, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(thresh5)

    return results


def run_tesseract_configs(img: np.ndarray, whitelist: str) -> List[str]:
    """
    Próbuje różnych konfiguracji Tesseract.
    """
    results = []

    # Różne Page Segmentation Modes
    psm_modes = [7, 8, 6, 13]  # 7=single line, 8=single word, 6=block, 13=raw line

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
    Ocenia jakość kandydata na tablicę (0.0 - 1.0).
    """
    if not text:
        return 0.0

    score = 0.0

    # Długość bliska oczekiwanej
    length_diff = abs(len(text) - expected_length)
    if length_diff == 0:
        score += 0.5
    elif length_diff == 1:
        score += 0.3
    elif length_diff == 2:
        score += 0.1

    # Ma litery i cyfry
    if is_valid_polish_plate(text):
        score += 0.3

    # Format polski: litery na początku, cyfry dalej
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
    Zwraca dwa najlepsze kandydaty (dla kompatybilności z evaluate.py).
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return "", ""

    preprocessed_images = preprocess_plate_v2(plate_bgr)

    all_candidates = []

    for img in preprocessed_images[:3]:  # Pierwsze 3 warianty dla szybkości
        texts = run_tesseract_configs(img, whitelist)
        all_candidates.extend(texts)

    if not all_candidates:
        return "", ""

    # Usuń duplikaty i popraw błędy
    candidates = list(set(all_candidates))
    candidates = [fix_common_ocr_errors(c) for c in candidates]

    # Oceń
    scored = [(c, score_plate_candidate(c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Zwróć top 2
    best = scored[0][0] if len(scored) > 0 else ""
    second = scored[1][0] if len(scored) > 1 else best

    return best, second


# Zachowaj starą funkcję dla kompatybilności
def read_plate_text(plate_bgr: np.ndarray, whitelist: str) -> str:
    return read_plate_text_improved(plate_bgr, whitelist)