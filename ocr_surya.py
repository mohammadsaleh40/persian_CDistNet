# %%
import cv2
from PIL import Image
import numpy as np

def preprocess_image_for_ocr(image_path, upscale=2, apply_threshold=True):
    """
    پیش‌پردازش تصویر برای OCR:
    - تبدیل به grayscale
    - upscale (بزرگ‌نمایی)
    - threshold اختیاری (برای Tesseract مفید)
    """
    # خواندن تصویر
    img = cv2.imread(image_path)
    
    # تبدیل به grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # upscale با bicubic
    gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    
    # threshold برای Tesseract
    if apply_threshold:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # تبدیل به PIL Image برای Surya
    pil_img = Image.fromarray(gray)
    
    return gray, pil_img  # gray برای Tesseract, pil_img برای Surya

# %%

from PIL import Image




image_path = "dataset/eval_img/0010_زمینی.jpg"

gray, pil_img = preprocess_image_for_ocr(image_path)

import pytesseract
config = "--psm 8 --oem 3"
text = pytesseract.image_to_string(gray, lang="fas", config=config)
print("Tesseract:", text)

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

foundation = FoundationPredictor()
recognition = RecognitionPredictor(foundation)
detection = DetectionPredictor()

results = recognition([pil_img], det_predictor=detection)
if results and results[0].text_lines:
    surya_text = results[0].text_lines[0].text
else:
    surya_text = ""

print("Surya:", surya_text)

