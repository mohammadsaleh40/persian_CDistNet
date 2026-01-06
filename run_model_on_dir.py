#!/usr/bin/env python3
"""
Script to evaluate the CDistNet model on a directory of images.
Compares model predictions against a ground truth file (gt.txt).
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from mmcv import Config
import re # برای پردازش gt.txt

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


def preprocess_image(image_path, target_width, target_height):
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
    image = image.convert('L')
    
    # Resize image according to config settings
    # اینجا فقط اگر keep_aspect_ratio=True باشد، منطق پیچیده‌تر می‌شود. برای سادگی، مستقیماً تغییر اندازه می‌دهیم.
    # این همان کاری است که در دیتاست کلاس احتمالاً انجام می‌شود.
    image = image.resize((target_width, target_height), resample_filter)

    # Convert to numpy array
    image_np = np.array(image).astype(np.float32)

    # --- نرمال‌سازی متفاوت ---
    # Normalize to [-1, 1] - این همان نرمال‌سازی است که در batch_test.py دیده شد
    image_np = image_np / 127.5 - 1.0
    # ---
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_np)
    
    # Add channel dimension for grayscale (1, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    
    # Add batch dimension (B, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)

    # Ensure the tensor is contiguous
    image_tensor = image_tensor.contiguous()

    return image_tensor


def run_model_on_image(image_path, model, translator, config, device):
    """
    Run the model on a single preprocessed image tensor.
    """
    # Load vocabulary (from config or pre-loaded)
    # word2idx, idx2word = load_vocab(config.dst_vocab, config.dst_vocab_size) # نیاز نیست دوباره بارگذاری شود
    word2idx = translator.word2idx
    idx2word = translator.idx2word

    # Preprocess the image
    print(f"Processing image: {image_path}")
    image_tensor = preprocess_image(image_path, target_width=config.width, target_height=config.height)
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


def load_ground_truth(gt_file_path):
    """
    Loads ground truth from a file like gt.txt.
    Assumes format: image_name.jpg, "ground_truth_text"
    Returns a dictionary {image_name: gt_text}.
    """
    gt_dict = {}
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # جدا کردن نام فایل و متن gt
            # این الگو فرض می‌کند که فرمت دقیقاً image.jpg, "gt" است
            match = re.match(r'([^,]+),\s*"([^"]*)"', line)
            if match:
                image_name = match.group(1).strip()
                gt_text = match.group(2).strip()
                gt_dict[image_name] = gt_text
            else:
                print(f"Warning: Could not parse line in gt.txt: {line}")
    return gt_dict


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate_directory.py <image_directory>")
        sys.exit(1)

    image_dir = sys.argv[1]
    gt_file_path = os.path.join(image_dir, 'gt.txt')

    if not os.path.isdir(image_dir):
        print(f"Error: Image directory does not exist: {image_dir}")
        sys.exit(1)

    if not os.path.exists(gt_file_path):
        print(f"Error: Ground truth file does not exist: {gt_file_path}")
        sys.exit(1)

    # Define paths
    model_path = "/home/homeai/Documents/GitHub/CDistNet/models/tps_persian_cdistnet_128_36/epoch4_best_acc.pth"
    config_path = "/home/homeai/Documents/GitHub/CDistNet/configs/CDistNet_config3.py"

    print(f"Model path: {model_path}")
    print(f"Config path: {config_path}")
    print(f"Evaluating images in: {image_dir}")
    print(f"Ground truth file: {gt_file_path}\n")

    try:
        # Load configuration
        cfg = Config.fromfile(config_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        cfg.test.device = str(device)
        cfg.train.device = str(device)
        cfg.val.device = str(device)

        # Load the model
        model = build_CDistNet(cfg)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Load vocabulary and create translator
        word2idx, idx2word = load_vocab(cfg.dst_vocab, cfg.dst_vocab_size)
        translator = Translator(cfg, model=model)

        # Load ground truth
        gt_dict = load_ground_truth(gt_file_path)

        # Find image files in the directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
        image_files.sort() # برای ترتیب پیش‌پذیرفته

        if not image_files:
            print(f"No image files found in {image_dir}")
            sys.exit(0)

        correct_predictions = 0
        total_predictions = 0

        print("Starting evaluation...\n")
        for image_name in image_files:
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                print(f"Warning: Image file listed in gt.txt not found: {image_path}")
                continue

            if image_name not in gt_dict:
                print(f"Warning: No ground truth found for image: {image_name}")
                continue

            gt_text = gt_dict[image_name]
            prediction = run_model_on_image(image_path, model, translator, cfg, device)

            total_predictions += 1
            is_correct = (prediction == gt_text)
            if is_correct:
                correct_predictions += 1

            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            print(f"{status}")
            print(f"  Image: {image_name}")
            print(f"  Ground Truth: '{gt_text}'")
            print(f"  Prediction:   '{prediction}'")
            print("-" * 50)

        print("\n" + "="*50)
        print("Evaluation Summary:")
        print(f"Total Images: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            print(f"Accuracy: {accuracy:.2f}%")
        else:
            print("Accuracy: N/A (No valid image-GT pairs found)")
        print("="*50)


    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
