# %%
import warnings
warnings.filterwarnings("ignore")
import sys
import os
from contextlib import redirect_stdout, redirect_stderr
import io
import cv2
from PIL import Image
import numpy as np
import torch

def preprocess_image_for_cdistnet(
    pil_gray_image,
    target_width=128,
    target_height=36
):
    """
    Preprocess مخصوص CDistNet:
    - ورودی: PIL grayscale (threshold شده)
    - خروجی حتماً: 128x36
    مراحل:
    1) resize با حفظ نسبت برای رسیدن به ارتفاع 36
    2) padding سفید افقی اگر عرض < 128
    3) resize افقی اگر عرض > 128
    4) نرمال‌سازی به [-1, 1]
    5) خروجی Tensor (1, 1, 36, 128)
    """
    import numpy as np
    import torch
    from PIL import Image

    # اطمینان از grayscale
    pil_gray_image = pil_gray_image.convert("L")

    w, h = pil_gray_image.size

    # =========================
    # 1) resize با حفظ نسبت (height -> 36)
    # =========================
    if h != target_height:
        scale = target_height / h
        new_w = max(1, int(w * scale))
        pil_gray_image = pil_gray_image.resize(
            (new_w, target_height),
            Image.LANCZOS
        )

    w, h = pil_gray_image.size  # حالا h == 36

    # =========================
    # 2) padding یا resize افقی
    # =========================
    if w < target_width:
        # padding سفید (255)
        pad_total = target_width - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        padded_img = Image.new(
            "L",
            (target_width, target_height),
            color=255  # سفید
        )
        padded_img.paste(pil_gray_image, (pad_left, 0))
        pil_gray_image = padded_img

    elif w > target_width:
        # resize افقی به 128
        pil_gray_image = pil_gray_image.resize(
            (target_width, target_height),
            Image.LANCZOS
        )

    display(pil_gray_image)
    # =========================
    # 3) numpy + normalize
    # =========================
    img_np = np.array(pil_gray_image).astype(np.float32)

    # Normalize to [-1, 1]
    img_np = img_np / 127.5 - 1.0

    # =========================
    # 4) tensor (B, C, H, W)
    # =========================
    img_tensor = torch.from_numpy(img_np)
    img_tensor = img_tensor.unsqueeze(0)  # C
    img_tensor = img_tensor.unsqueeze(0)  # B

    return img_tensor.contiguous()



def preprocess_image_for_ocr(image_path, upscale=2):
    """
    پیش‌پردازش مشترک OCR:
    - grayscale
    - upscale
    - OTSU threshold (مشترک برای همه مدل‌ها)

    خروجی:
    - gray_thresh_np : numpy (برای Tesseract)
    - pil_thresh_gray : PIL L (برای Surya + CDistNet)
    - pil_thresh_rgb  : PIL RGB (برای EasyOCR)
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # upscale
    gray = cv2.resize(
        gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
    )

    # OTSU threshold
    _, gray_thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    pil_thresh_gray = Image.fromarray(gray_thresh, mode="L")
    pil_thresh_rgb  = Image.fromarray(
        cv2.cvtColor(gray_thresh, cv2.COLOR_GRAY2RGB)
    )

    return gray_thresh, pil_thresh_gray, pil_thresh_rgb


# %%

# image_path = "dataset/eval_img/0010_زمینی.jpg"
# image_path = "dataset/a/1.png"
# image_path = "/home/mohammadsaleh/Documents/GitHub/persian_CDistNet/dataset/eval_img/0014_تاریخ.jpg"
# image_path = "dataset/eval_img/0018_پرورش.jpg"
image_path = "dataset/eval_img/0136_تناوبی.jpg"
from matplotlib.pyplot import imshow
import numpy as np
# image_path = "dataset/eval_img/0043_پمپ.jpg"

gray_thresh_np, pil_thresh_gray, pil_thresh_rgb = preprocess_image_for_ocr(
    image_path, upscale=1
)

imshow(np.asarray(pil_thresh_rgb))
# %%
# =========================
# Tesseract
# =========================
import pytesseract

config = "--psm 8 --oem 3"
tess_text = pytesseract.image_to_string(
    gray_thresh_np, lang="fas", config=config
).strip()

print("Tesseract:", tess_text)


# =========================
# Surya OCR
# =========================
# --- قسمت Surya ---
# Redirect stdout and stderr during Surya execution
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    foundation = FoundationPredictor()
    recognition = RecognitionPredictor(foundation)
    detection = DetectionPredictor()

    results = recognition([pil_thresh_gray], det_predictor=detection)

surya_text = (
    results[0].text_lines[0].text
    if results and results[0].text_lines else ""
)

print("Surya:", surya_text)


# =========================
# EasyOCR
# =========================


# Redirect stdout and stderr during EasyOCR execution
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import easyocr
    reader = easyocr.Reader(['fa'], gpu=False)
    easy_results = reader.readtext(
        np.array(pil_thresh_rgb),
        detail=0,
        paragraph=False
    )

easy_text = easy_results[0] if easy_results else ""
print("EasyOCR:", easy_text)


# =================
# CDitNet
# =================

import sys

from mmcv import Config
from cdistnet.model.model import build_CDistNet
from cdistnet.model.translator import Translator
import codecs

def load_vocab(vocab=None, vocab_size=None):
    """
    Load vocab from disk. The first four items in the vocab should be <PAD>, <UNK>, <S>, </S>
    """
    vocab = [' ' if len(line.split()) == 0 else line.split()[0] for line in codecs.open(vocab, 'r', 'utf-8')]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return word2idx, idx2word

def preprocess_image(pil_image, target_width, target_height): # حذف مقادیر پیش‌فرض
    """
    Preprocess the image to the required format.
    """
    from PIL import Image
    try:
        resample_filter = Image.LANCZOS
    except AttributeError:
        resample_filter = Image.ANTIALIAS

    

    # Convert to numpy array
    image_np = np.array(pil_image).astype(np.float32)

    # --- نرمال‌سازی متفاوت ---
    # Normalize to [-1, 1] - این همان نرمال‌سازی است که در batch_test.py دیده شد
    image_np = image_np / 127.5 - 1.0
    # ---
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_np)
    
    # Add channel dimension for grayscale (1, H, W)
    image_tensor = image_tensor.unsqueeze(0)  # ← اضافه کردن بعد کانال
    
    # Add batch dimension (B, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)

    # Ensure the tensor is contiguous
    image_tensor = image_tensor.contiguous()

    return image_tensor

def run_model_on_image(pil_image, model_path, config_path):


    # ...
    # Load configuration
    cfg = Config.fromfile(config_path)

    # Determine the appropriate device
    device = torch.device('cpu')


    # Update device in config
    cfg.test.device = str(device)
    cfg.train.device = str(device)
    cfg.val.device = str(device)

    # Load the model
    model = build_CDistNet(cfg)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move model to the appropriate device
    model = model.to(device)
    model.eval()

    # Load vocabulary
    word2idx, idx2word = load_vocab(cfg.dst_vocab, cfg.dst_vocab_size)

    # Create translator
    translator = Translator(cfg, model=model)

    # Preprocess the image

    # --- تغییر اینجا: استفاده از ابعاد از کانفیگ ---
    image_tensor = preprocess_image_for_cdistnet(
        pil_thresh_gray,
        target_height=36
        ).to(device)
    # ---
    image_tensor = image_tensor.to(device)

    
    # Run the model

    with torch.no_grad():
        batch_hyp, batch_scores = translator.translate_batch(images=image_tensor)

    # Decode the prediction
    predictions = []
    for seqs in batch_hyp:
        for seq in seqs:
            seq = [x for x in seq if x != 3]  # Remove </S> token (index 3)
            pred = [idx2word[x] for x in seq]
            pred_text = ''.join(pred)
            predictions.append(pred_text)

    return predictions[0] if predictions else "No prediction"

model_path = "/home/mohammadsaleh/Documents/GitHub/persian_CDistNet/models/number_word_6_font_persian_cdistnet_128_36/epoch2_best_acc.pth"
config_path = "/home/mohammadsaleh/Documents/GitHub/persian_CDistNet/configs/CDistNet_config5.py"

prediction = run_model_on_image(pil_thresh_gray, model_path, config_path)

print(f"Model prediction: {prediction}")
