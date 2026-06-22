"""
prepare_models.py — Download EasyOCR model weights before building the EXE.

Run this ONCE on the Windows build machine before running PyInstaller:

    python prepare_models.py

Downloads:
    craft_mlt_25k.pth   (~90 MB) — CRAFT text detection
    english_g2.pth      (~17 MB) — English CRNN recognition

Output:  easyocr_models/   (sibling of this script, bundled by rtsp_vision.spec)
"""

from pathlib import Path

MODEL_DIR = Path(__file__).parent / "easyocr_models"
MODEL_DIR.mkdir(exist_ok=True)

print(f"Downloading EasyOCR models to: {MODEL_DIR}")
print("This takes a minute on first run; subsequent runs skip already-downloaded files.")

import easyocr
easyocr.Reader(["en"], gpu=False, model_storage_directory=str(MODEL_DIR), verbose=True)

print("\nDone — easyocr_models/ is ready to be bundled.")
print(f"Files: {[f.name for f in MODEL_DIR.iterdir()]}")
