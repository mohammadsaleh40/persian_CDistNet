#!/usr/bin/env python3
"""
Script to run the CDistNet model on a single image.
Takes an image path, resizes it to 64x24 if needed, and shows the model output.
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from mmcv import Config

# Add the project root to the path
sys.path.insert(0, '/home/homeai/Documents/GitHub/CDistNet')

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


def preprocess_image(image_path, target_width, target_height): # حذف مقادیر پیش‌فرض
    """
    Preprocess the image to the required format.
    """
    from PIL import Image
    try:
        resample_filter = Image.LANCZOS
    except AttributeError:
        resample_filter = Image.ANTIALIAS

    # Load the image using PIL
    image = Image.open(image_path)

    # Convert to grayscale (با توجه به rgb2gray=True در کانفیگ)
    image = image.convert('L')  # ← تغییر از 'RGB' به 'L'
    
    # Convert to numpy array
    image_np = np.array(image).astype(np.float32)

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

def run_model_on_image(image_path, model_path, config_path):
    # ...
    # Load configuration
    cfg = Config.fromfile(config_path)

    # Determine the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    print(f"Processing image: {image_path}")
    # --- تغییر اینجا: استفاده از ابعاد از کانفیگ ---
    image_tensor = preprocess_image(image_path, target_width=cfg.width, target_height=cfg.height)
    # ---
    image_tensor = image_tensor.to(device)

    
    # Run the model
    print("Running model inference...")
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_model_on_image.py <image_path>")
        sys.exit(1)
    print(sys.argv)
    image_paths = sys.argv[1:]
    
    if not os.path.exists(image_paths[0]):
        print(f"Error: Image file does not exist: {image_paths[0]}")
        sys.exit(1)
    
    # Define paths
    model_path = "/home/homeai/Documents/GitHub/CDistNet/models/number_word_6_font_persian_cdistnet_128_36/epoch2_best_acc.pth"
    config_path = "/home/homeai/Documents/GitHub/CDistNet/configs/CDistNet_config5.py"
    
    print(f"Model path: {image_paths}")
    print(f"Config path: {config_path}")
    
    try:
        # Run the model on the image
        for image_path in image_paths:
            prediction = run_model_on_image(image_path, model_path, config_path)
        
            print(f"\nInput image: {image_path}")
            print(f"Model prediction: {prediction}")
        
    except Exception as e:
        print(f"Error running model on image: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
