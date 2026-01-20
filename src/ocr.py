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
    Szybszy preprocessing: mniej wariantów + lżejsze operacje.
    Cel: zejść z czasem, a accuracy podnieść/utrzymać ostrożnie.
    """
    results = []

    # Powiększ obraz (ważne dla OCR) - ale trochę mniej niż 3.0 (szybciej)
    scale = 2.5
    h, w = plate_bgr.shape[:2]
    plate_large = cv2.resize(
        plate_bgr, (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(plate_large, cv2.COLOR_BGR2GRAY)

    # Wariant 1: Bilateral + adaptive threshold (zwykle najlepszy "bang for buck")
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh1 = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    results.append(thresh1)

    # Wariant 2: Otsu (szybki i często działa)
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(thresh2)

    # Wariant 3: Otsu inverted (czasem tablica ma odwrotne kolory)
    _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results.append(thresh3)

    return results


def run_tesseract_configs(img: np.ndarray, whitelist: str) -> List[str]:
    """
    Szybsze konfiguracje Tesseract:
    - mniej PSM (najczęściej wystarczają 7 i 8)
    """
    results = []

    # mniej trybów = dużo mniej czasu
    psm_modes = [7, 8]  # 7=single line, 8=single word

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

    # tylko 3 warianty (i tak preprocess zwraca 3 po zmianie)
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

            # EARLY EXIT: jeśli mamy już bardzo dobry wynik, kończymy
            # (to najczęściej zbija czas poniżej 60s/100)
            if best_score >= 0.95 and second_score >= 0.80:
                return best, (second or best)

    if not best:
        return "", ""

    return best, (second or best)


# Zachowaj starą funkcję dla kompatybilności
def read_plate_text(plate_bgr: np.ndarray, whitelist: str) -> str:
    return read_plate_text_improved(plate_bgr, whitelist)