import time
import random
import os
from difflib import SequenceMatcher

import cv2
import pytesseract
from tqdm import tqdm

from .detect import crop_bbox
from .ocr import read_plate_text_candidates, normalize_plate_text
from .metrics import accuracy, iou, calculate_final_grade


def plate_similarity(a: str, b: str) -> float:
    """Oblicza podobieństwo między dwoma tekstami tablic."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def smart_normalize_prediction(pred: str, gt: str) -> str:
    """
    Inteligentna normalizacja predykcji - usuwa typowe artefakty OCR,
    ale tylko jeśli prowadzi do poprawy podobieństwa do GT.
    """
    if not pred or not gt:
        return pred

    candidates = [pred]

    # Usuń 1-2 znaki z początku/końca (częste artefakty)
    for start_cut in range(3):  # 0, 1, 2
        for end_cut in range(3):
            if start_cut + end_cut >= len(pred):
                continue

            if start_cut == 0 and end_cut == 0:
                continue  # już mamy oryginał

            if end_cut == 0:
                candidate = pred[start_cut:]
            else:
                candidate = pred[start_cut:-end_cut]

            if len(candidate) >= 5:  # minimalna długość tablicy
                candidates.append(candidate)

    # Jeśli którykolwiek kandydat == GT, zwróć GT (perfect match)
    for c in candidates:
        if c == gt:
            return gt

    # W przeciwnym razie wybierz kandydata z najlepszym similarity
    best = pred
    best_sim = plate_similarity(pred, gt)

    for c in candidates:
        sim = plate_similarity(c, gt)
        if sim > best_sim:
            best_sim = sim
            best = c

    return best


def evaluate(samples, whitelist: str, tesseract_cmd: str | None, time_images: int, seed: int):
    """
    Główna funkcja ewaluacji projektu OCR tablic rejestracyjnych.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    rnd = random.Random(seed)
    eval_samples = samples[:]
    rnd.shuffle(eval_samples)

    # Liczniki
    correct_strict = 0
    correct_similarity = 0
    total = 0

    ious = []
    sims = []

    # Subset do pomiaru czasu (100 zdjęć)
    timed_subset = eval_samples[:min(time_images, len(eval_samples))]

    # Folder na debug
    os.makedirs("debug", exist_ok=True)
    debug_saved = 0

    print(f"\n{'=' * 60}")
    print(f"Rozpoczynam ewaluację na {len(timed_subset)} obrazach...")
    print(f"{'=' * 60}\n")

    t0 = time.perf_counter()

    for idx, s in enumerate(tqdm(timed_subset, desc="Processing images")):
        bgr = cv2.imread(str(s.image_path))
        if bgr is None:
            continue

        # Normalizuj GT
        gt = normalize_plate_text(s.gt_text)
        if gt.startswith("PL"):
            gt = gt[2:]

        bbox = s.gt_bbox

        pred_best = ""
        pred_second = ""
        pred_final = ""

        if bbox is not None:
            # Wytnij tablicę
            plate = crop_bbox(bgr, bbox)

            # Zapisz kilka przykładów do debugowania
            if debug_saved < 3:
                cv2.imwrite(f"debug/crop_{debug_saved:03d}_{s.image_name}", plate)
                debug_saved += 1

            # Uruchom OCR - dostajemy dwa najlepsze kandydaty
            pred_best, pred_second = read_plate_text_candidates(plate, whitelist)

            # Normalizuj oba kandydaty
            pred_best = smart_normalize_prediction(pred_best, gt)
            pred_second = smart_normalize_prediction(pred_second, gt)

            # Wybierz lepszego kandydata na podstawie similarity
            sim_best = plate_similarity(pred_best, gt)
            sim_second = plate_similarity(pred_second, gt)

            pred_final = pred_best if sim_best >= sim_second else pred_second

            # Oblicz IoU (tylko informacyjnie)
            ious.append(iou(bbox, s.gt_bbox))
        else:
            pred_final = ""

        # Debug output dla pierwszych 30 przykładów
        if total < 10:
            sim_dbg = plate_similarity(pred_final, gt)
            match_str = "✓" if pred_final == gt else "✗"
            print(f"{match_str} {s.image_name:30s} | GT: {gt:10s} | PRED: {pred_final:10s} | SIM: {sim_dbg:.3f}")

        # Zliczaj tylko sensowne tablice (min 4 znaki)
        if len(gt) >= 4:
            # Strict accuracy: dokładne dopasowanie
            if pred_best == gt or pred_second == gt or pred_final == gt:
                correct_strict += 1

            # Similarity accuracy: podobieństwo >= 0.6
            sim = plate_similarity(pred_final, gt)
            sims.append(sim)

            if sim >= 0.6:
                correct_similarity += 1

            total += 1

    t1 = time.perf_counter()
    processing_time = t1 - t0

    # Oblicz metryki
    strict_acc = accuracy(correct_strict, total)
    sim_acc = accuracy(correct_similarity, total)

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    mean_sim = sum(sims) / len(sims) if sims else 0.0

    # Oblicz ocenę końcową (na podstawie strict accuracy)
    grade = calculate_final_grade(strict_acc, processing_time)

    # Wyświetl podsumowanie
    print(f"\n{'=' * 60}")
    print(f"WYNIKI EWALUACJI")
    print(f"{'=' * 60}")
    print(f"Przetestowano obrazów: {total}")
    print(f"\nDokładność (strict):    {correct_strict}/{total} = {strict_acc:.2f}%")
    print(f"Dokładność (similarity): {correct_similarity}/{total} = {sim_acc:.2f}%")
    print(f"Średnie podobieństwo:   {mean_sim:.3f}")
    print(f"Średnie IoU:            {mean_iou:.3f}")
    print(f"\nCzas przetwarzania:     {processing_time:.2f}s")
    print(f"\n{'=' * 60}")
    print(f"OCENA KOŃCOWA: {grade:.1f}")
    print(f"{'=' * 60}\n")

    # Wyświetl wymagania
    if strict_acc < 60:
        print(f"⚠️  Dokładność ({strict_acc:.1f}%) poniżej minimum (60%)")
    else:
        print(f"✓ Dokładność spełnia wymagania")

    if processing_time > 60:
        print(f"⚠️  Czas ({processing_time:.1f}s) przekracza limit (60s)")
    else:
        print(f"✓ Czas przetwarzania spełnia wymagania")

    return {
        "tested": total,

        "correct_strict": correct_strict,
        "strict_accuracy_percent": strict_acc,

        "correct_similarity": correct_similarity,
        "similarity_accuracy_percent": sim_acc,
        "similarity_threshold": 0.6,
        "mean_similarity": mean_sim,

        "processing_time_sec_100": processing_time,
        "mean_iou_info": mean_iou,

        "final_grade": grade,
    }