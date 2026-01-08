#!/usr/bin/env python3
"""
OCR Benchmark Script
Compares Surya, Pytesseract, and CDistNet models on images from a directory
"""

import os
import sys
import csv
import torch
import numpy as np
from PIL import Image
from mmengine import Config
import pytesseract
import argparse

# Add the project root to the path
sys.path.insert(0, '/home/homeai/Documents/GitHub/CDistNet')

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
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


def run_cdistnet_model(image_path, model_path, config_path):
    """
    Run the CDistNet model on a single image.
    """
    try:
        # Load configuration
        cfg = Config.fromfile(config_path)

        # Force CPU usage
        device = torch.device('cpu')
        print(f"Using device: {device}")

        # Update device in config
        cfg.test.device = str(device)
        cfg.train.device = str(device)
        cfg.val.device = str(device)

        # Load the model
        model = build_CDistNet(cfg)
        state_dict = torch.load(model_path, map_location=device)

        # Try to load the state dict, handling potential shape mismatches
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict loading failed: {e}")
            print("Attempting non-strict loading...")
            model.load_state_dict(state_dict, strict=False)

        # Move model to the appropriate device
        model = model.to(device)
        model.eval()

        # Load vocabulary
        word2idx, idx2word = load_vocab(cfg.dst_vocab, cfg.dst_vocab_size)

        # Create translator
        translator = Translator(cfg, model=model)

        # Preprocess the image
        print(f"Processing image: {image_path}")
        image_tensor = preprocess_image(image_path, target_width=cfg.width, target_height=cfg.height)
        image_tensor = image_tensor.to(device)

        # Run the model
        print("Running CDistNet model inference...")
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
    except Exception as e:
        print(f"Error running CDistNet model: {e}")
        return f"Error: {str(e)}"


def upscale_image(img, min_width=512, min_height=64):
    """
    Upscale image to minimum dimensions while maintaining aspect ratio
    """
    w, h = img.size

    scale_w = min_width / w
    scale_h = min_height / h
    scale = max(scale_w, scale_h, 1.0)  # only upscale, don't downscale

    new_w = int(w * scale)
    new_h = int(h * scale)

    return img.resize((new_w, new_h), Image.BICUBIC)


def run_surya_ocr(image_path):
    """
    Run Surya OCR on an image
    """
    try:
        # Force CPU usage for Surya by setting environment variables before any model loading
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
        os.environ['FORCE_CPU'] = '1'

        # Also try to set torch to use CPU only
        import torch
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        foundation_predictor = FoundationPredictor()
        recognition_predictor = RecognitionPredictor(foundation_predictor)
        detection_predictor = DetectionPredictor()

        image = Image.open(image_path).convert("RGB")
        image = upscale_image(image)

        page_predictions = recognition_predictor(
            [image],
            det_predictor=detection_predictor
        )

        if page_predictions and page_predictions[0].text_lines:
            # Extract text from the first text line
            text = page_predictions[0].text_lines[0].text
            return text
        else:
            return "No text detected"
    except Exception as e:
        print(f"Error running Surya OCR: {e}")
        return f"Error: {str(e)}"


def run_pytesseract_ocr(image_path):
    """
    Run Pytesseract OCR on an image
    """
    text = pytesseract.image_to_string(image_path, lang='fas+eng')
    return text.strip()


def read_ground_truth(gt_file_path):
    """
    Read ground truth from gt.txt file
    Format: filename, "ground_truth_text"
    """
    ground_truth = {}
    
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split by comma and clean up the parts
                parts = line.split(',', 1)  # Split only on the first comma
                if len(parts) == 2:
                    filename = parts[0].strip().strip('"')
                    gt_text = parts[1].strip().strip('"')
                    ground_truth[filename] = gt_text
    
    return ground_truth


def get_image_files(image_dir):
    """
    Get all image files from the directory
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for filename in os.listdir(image_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            image_files.append(filename)
    
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(description='OCR Benchmark: Compare Surya, Pytesseract, and CDistNet models')
    parser.add_argument('--image_dir', type=str, default='/home/homeai/Documents/GitHub/CDistNet/dataset/eval_img',
                        help='Directory containing images to process')
    parser.add_argument('--output_csv', type=str, default='ocr_benchmark_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--model_path', type=str, 
                        default='/home/homeai/Documents/GitHub/CDistNet/models/tps_persian_cdistnet_128_36/epoch7_best_acc.pth',
                        help='Path to the CDistNet model')
    parser.add_argument('--config_path', type=str, 
                        default='/home/homeai/Documents/GitHub/CDistNet/configs/eval_config.py',
                        help='Path to the configuration file')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.image_dir):
        print(f"Error: Directory does not exist: {args.image_dir}")
        sys.exit(1)
    
    # Validate model and config files
    if not os.path.exists(args.model_path):
        print(f"Error: Model file does not exist: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.config_path):
        print(f"Error: Config file does not exist: {args.config_path}")
        sys.exit(1)
    
    # Read ground truth
    gt_file_path = os.path.join(args.image_dir, 'gt.txt')
    if not os.path.exists(gt_file_path):
        print(f"Warning: Ground truth file not found: {gt_file_path}")
        ground_truth = {}
    else:
        ground_truth = read_ground_truth(gt_file_path)
        print(f"Loaded {len(ground_truth)} ground truth entries")
    
    # Get image files
    image_files = get_image_files(args.image_dir)
    print(f"Found {len(image_files)} image files to process")
    
    # Prepare results list
    results = []
    
    # Process each image
    for i, filename in enumerate(image_files):
        image_path = os.path.join(args.image_dir, filename)
        print(f"Processing {i+1}/{len(image_files)}: {filename}")
        
        # Get ground truth for this image
        gt_text = ground_truth.get(filename, "Ground truth not available")
        
        # Run Surya OCR
        try:
            surya_result = run_surya_ocr(image_path)
            print(f"  Surya result: {surya_result}")
        except Exception as e:
            print(f"  Surya error: {e}")
            surya_result = f"Error: {str(e)}"
        
        # Run Pytesseract OCR
        try:
            pytesseract_result = run_pytesseract_ocr(image_path)
            print(f"  Pytesseract result: {pytesseract_result}")
        except Exception as e:
            print(f"  Pytesseract error: {e}")
            pytesseract_result = f"Error: {str(e)}"
        
        # Run CDistNet OCR
        try:
            cdistnet_result = run_cdistnet_model(image_path, args.model_path, args.config_path)
            print(f"  CDistNet result: {cdistnet_result}")
        except Exception as e:
            print(f"  CDistNet error: {e}")
            cdistnet_result = f"Error: {str(e)}"
        
        # Add to results
        results.append({
            'filename': filename,
            'surya_prediction': surya_result,
            'pytesseract_prediction': pytesseract_result,
            'cdistnet_prediction': cdistnet_result,
            'ground_truth': gt_text
        })
        
        print()  # Empty line for readability
    
    # Write results to CSV
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'surya_prediction', 'pytesseract_prediction', 'cdistnet_prediction', 'ground_truth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {args.output_csv}")
    print(f"Processed {len(results)} images")


if __name__ == "__main__":
    main()