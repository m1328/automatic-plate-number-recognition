from dataclasses import dataclass
from pathlib import Path
@dataclass
class Settings:
    data_dir: Path = Path("data/raw")
    photos_dir: Path = Path("data/raw/photos")
    annotations_path: Path = Path("data/raw/annotations.xml")

    # OCR
    #tesseract_cmd: str | None = os.getenv("TESSERACT_CMD")  # np. C:\Program Files\Tesseract-OCR\tesseract.exe
    tesseract_cmd: str | None = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    tesseract_lang: str = "eng"
    ocr_whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    seed: int = 42
    test_ratio: float = 1.0
    time_images: int = 100

SETTINGS = Settings()
