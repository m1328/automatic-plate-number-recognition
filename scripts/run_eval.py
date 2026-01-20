from src.config import SETTINGS
from src.dataset import load_samples
from src.evaluate import evaluate

def main():
    samples = load_samples(SETTINGS.annotations_path, SETTINGS.photos_dir)
    print(f"Loaded samples: {len(samples)}")

    results = evaluate(
        samples=samples,
        whitelist=SETTINGS.ocr_whitelist,
        tesseract_cmd=SETTINGS.tesseract_cmd,
        time_images=SETTINGS.time_images,
        seed=SETTINGS.seed,
    )

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
