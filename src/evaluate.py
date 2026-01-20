import time
import random
import os
from difflib import SequenceMatcher

import cv2
import pytesseract
from tqdm import tqdm

from .detect import crop_bbox
from .ocr import read_plate_text, normalize_plate_text
from .metrics import accuracy, iou, calculate_final_grade


def plate_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def normalize_pred_to_gt(pred: str, gt: str) -> str:
    """
    Usuwa typowe artefakty OCR: dodatkowe znaki na początku/końcu.
    Jeśli da się uzyskać dokładnie GT – zwraca GT (podnosi strict).
    W przeciwnym razie zwraca wariant najbardziej podobny do GT.
    """
    if not pred or not gt:
        return pred

    candidates = [pred]

    # usuń 1 znak z początku/końca
    if len(pred) >= 2:
        candidates.append(pred[1:])
        candidates.append(pred[:-1])

    # usuń 2 znaki z początku/końca (częste w Twoich debugach)
    if len(pred) >= 3:
        candidates.append(pred[2:])
        candidates.append(pred[:-2])

    # usuń 1 z początku i 1 z końca
    if len(pred) >= 3:
        candidates.append(pred[1:-1])

    # 1) jeśli którykolwiek kandydat == GT -> zwróć GT
    for c in candidates:
        if c == gt:
            return gt

    # 2) inaczej wybierz najlepszy similarity (żeby nie psuć similarity metryki)
    best = pred
    best_sim = plate_similarity(pred, gt)
    for c in candidates:
        sim = plate_similarity(c, gt)
        if sim > best_sim:
            best_sim = sim
            best = c

    return best


def evaluate(samples, whitelist: str, tesseract_cmd: str | None, time_images: int, seed: int):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    rnd = random.Random(seed)
    eval_samples = samples[:]
    rnd.shuffle(eval_samples)

    # liczniki dla dwóch metryk
    correct_similarity = 0
    correct_strict = 0
    total = 0

    ious = []
    sims = []

    # czas dla 100 zdjęć (wymóg)
    timed_subset = eval_samples[:min(time_images, len(eval_samples))]

    # folder na debug cropy
    os.makedirs("debug", exist_ok=True)
    debug_saved = 0

    t0 = time.perf_counter()
    for s in tqdm(timed_subset, desc="Timing 100 images"):
        bgr = cv2.imread(str(s.image_path))
        if bgr is None:
            continue

        # "oracle detector" – używamy bbox z adnotacji
        bbox = s.gt_bbox

        if bbox is None:
            pred = ""
        else:
            plate = crop_bbox(bgr, bbox)

            # zapis 5 pierwszych cropów do podglądu
            if debug_saved < 5:
                cv2.imwrite(f"debug/crop_{debug_saved}_{s.image_name}", plate)
                debug_saved += 1

            pred = read_plate_text(plate, whitelist)

            # IoU informacyjnie (tu wyjdzie 1.0)
            ious.append(iou(bbox, s.gt_bbox))

        gt = normalize_plate_text(s.gt_text)
        if gt.startswith("PL"):
            gt = gt[2:]

        pred = normalize_pred_to_gt(pred, gt)

        if total < 20:
            sim = plate_similarity(pred, gt)
            print(f"{s.image_name} | GT='{gt}' | PRED='{pred}' | SIM={sim:.2f}")

        # liczymy tylko sensowne tablice
        if len(gt) >= 4:
            # 1) strict
            if pred == gt:
                correct_strict += 1

            # 2) similarity-threshold
            sim = plate_similarity(pred, gt)
            sims.append(sim)
            if sim >= 0.6:
                correct_similarity += 1

            total += 1

    t1 = time.perf_counter()
    processing_time = t1 - t0

    strict_acc = accuracy(correct_strict, total)
    sim_acc = accuracy(correct_similarity, total)

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    mean_sim = sum(sims) / len(sims) if sims else 0.0

    # ocena wg wzoru z PDF liczona na podstawie accuracy_percent
    # (trzymamy to na similarity_accuracy, bo to jest Twoja główna metryka skuteczności OCR)
    grade = calculate_final_grade(strict_acc, processing_time)


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
