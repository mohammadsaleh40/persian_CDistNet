# combined_eval.py
import argparse
import os
import torch
from PIL import Image
from mmcv import Config
import numpy as np
from cdistnet.model.translator import Translator
from cdistnet.model.model import build_CDistNet
import pytesseract
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CDistNet and Pytesseract on a dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset directory containing images')
    parser.add_argument('--gt_file', type=str, required=True,
                        help='Path to the ground truth file (gt.txt)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the CDistNet model file')
    parser.add_argument('--config', type=str, default='configs/CDistNet_config.py',
                        help='Path to the config file')
    parser.add_argument('--output_file', type=str, default='combined_results.txt',
                        help='Path to save the comparison results')
    return parser.parse_args()

def load_vocab(vocab_path, vocab_size):
    """Load vocabulary from file."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip().split()[0] if line.strip().split() else ' ' for line in f]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def preprocess_image_for_cdistnet(cfg, image_path):
    """Preprocess image for CDistNet model."""
    try:
        if cfg.rgb2gray:
            image = Image.open(image_path).convert('L')  # Grayscale
        else:
            image = Image.open(image_path).convert('RGB')  # RGB
            
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        image = image.resize((cfg.width, cfg.height), Image.LANCZOS)
        image = np.array(image)

        if image.ndim == 3:  # RGB
            image = image.transpose((2, 0, 1))  # CHW
        elif image.ndim == 2:  # Grayscale
            image = np.expand_dims(image, axis=0)  # (1, H, W)

        # Add batch dimension: (1, C, H, W)
        image = np.expand_dims(image, axis=0)

        # Normalize to [-1, 1]
        image = image.astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image)
        image = image.contiguous()
        
        return image
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_with_cdistnet(model, translator, word2idx, idx2word, image_tensor, device):
    """Get prediction from CDistNet model."""
    try:
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            all_hyp, all_scores = translator.translate_batch(image_tensor)
        
        if all_hyp and all_hyp[0]:
            idx_seq = all_hyp[0][0] 
            idx_seq = [x for x in idx_seq if x != 3]  # Remove end token
            predicted_text = ''.join([idx2word.get(idx, '<UNK>') for idx in idx_seq])
            return predicted_text
        else:
            return ""
    except Exception as e:
        print(f"Error predicting with CDistNet: {e}")
        return ""

def predict_with_pytesseract(image_path):
    """Get prediction from Pytesseract."""
    try:
        # Use Persian language if available, otherwise default
        text = pytesseract.image_to_string(Image.open(image_path), lang='fas+eng')
        # Clean the text
        text = text.strip().replace('\n', ' ')
        return text
    except Exception as e:
        print(f"Error predicting with Pytesseract for {image_path}: {e}")
        return ""

def calculate_accuracy(predictions, ground_truths):
    """Calculate accuracy as the percentage of exact matches."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    accuracy = (correct / len(predictions)) * 100 if predictions else 0
    return accuracy, correct, len(predictions)

def load_ground_truth(gt_file_path, dataset_path):
    """Load ground truth from gt.txt file."""
    gt_dict = {}
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Assuming format: relative_path, "label"
                parts = line.split(',', 1)
                if len(parts) == 2:
                    rel_path = parts[0].strip()
                    label = parts[1].strip().strip('"')
                    # Create full path
                    full_path = os.path.join(dataset_path, rel_path)
                    gt_dict[full_path] = label
    return gt_dict

# combined_eval.py (نسخه بهینه‌شده برای batch processing)
# ... (بقیه importها و توابع بدون تغییر) ...

def batch_predict_with_cdistnet(model, translator, word2idx, idx2word, image_tensors, device):
    """Get predictions from CDistNet model for a batch of images."""
    try:
        # Stack tensors into a single batch tensor
        batch_tensor = torch.cat(image_tensors, dim=0) # Shape: (B, C, H, W)
        batch_tensor = batch_tensor.to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(): # استفاده از Mixed Precision
                all_hyp, all_scores = translator.translate_batch(batch_tensor)
        
        predictions = []
        # all_hyp is a list of lists, one for each image in the batch
        for i in range(len(all_hyp)):
            if all_hyp[i]: # اگر پیش‌بینی‌ای برای تصویر i وجود داشته باشد
                idx_seq = all_hyp[i][0] # بهترین پیش‌بینی (اولین در beam)
                idx_seq = [x for x in idx_seq if x != 3] # حذف توکن پایان
                predicted_text = ''.join([idx2word.get(idx, '<UNK>') for idx in idx_seq])
            else:
                predicted_text = ""
            predictions.append(predicted_text)
            
        return predictions
    except Exception as e:
        print(f"Error predicting with CDistNet in batch: {e}")
        # Return empty predictions for the whole batch or handle individually
        return [""] * len(image_tensors) 

def main():
    args = parse_args()
    
    # تنظیم تعداد رشته‌های CPU
    torch.set_num_threads(6)
    
    # Load configuration
    cfg = Config.fromfile(args.config)
    
    # Load ground truth
    print("Loading ground truth...")
    gt_dict = load_ground_truth(args.gt_file, args.dataset_path)
    image_paths = list(gt_dict.keys())
    ground_truths = list(gt_dict.values())
    
    print(f"Found {len(image_paths)} images with ground truth.")
    
    # Initialize CDistNet model
    print("Initializing CDistNet model...")
    model = build_CDistNet(cfg)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Prepare translator and vocabulary
    translator = Translator(cfg, model)
    word2idx, idx2word = load_vocab(cfg.dst_vocab, cfg.dst_vocab_size)
    
    # Prepare for predictions
    cdistnet_predictions = []
    pytesseract_predictions = []
    
    # تنظیم اندازه دسته برای CDistNet
    cdistnet_batch_size = 512 # یا بیشتر، بسته به VRAM شما
    
    print("Processing images...")
    # پردازش تصاویر به صورت دسته‌ای برای CDistNet
    for i in tqdm(range(0, len(image_paths), cdistnet_batch_size)):
        batch_paths = image_paths[i:i+cdistnet_batch_size]
        batch_gt = ground_truths[i:i+cdistnet_batch_size]
        
        # آماده‌سازی دسته برای CDistNet
        batch_tensors = []
        valid_indices = [] # برای ردیابی تصاویری که بارگذاری موفق بوده‌اند
        
        for j, image_path in enumerate(batch_paths):
            img_tensor = preprocess_image_for_cdistnet(cfg, image_path)
            if img_tensor is not None:
                batch_tensors.append(img_tensor)
                valid_indices.append(j)
            else:
                # در صورت خطا، یک placeholder اضافه کنید یا بعداً مدیریت کنید
                pass # در اینجا فقط رد می‌کنیم
        
        # پیش‌بینی دسته‌ای با CDistNet
        if batch_tensors:
            batch_cdistnet_preds = batch_predict_with_cdistnet(model, translator, word2idx, idx2word, batch_tensors, device)
            # تخصیص پیش‌بینی‌ها به لیست اصلی
            pred_idx = 0
            for j in range(len(batch_paths)):
                if j in valid_indices:
                    cdistnet_predictions.append(batch_cdistnet_preds[pred_idx])
                    pred_idx += 1
                else:
                    cdistnet_predictions.append("") # یا یک placeholder برای خطا
        else:
            # اگر هیچ تصویری بارگذاری نشد
            cdistnet_predictions.extend([""] * len(batch_paths))
            
    # پردازش تصاویر به صورت تکی برای Pytesseract (معمولاً موازی‌سازی آن سخت‌تر است)
    print("Processing images with Pytesseract...")
    for image_path in tqdm(image_paths):
        pytesseract_pred = predict_with_pytesseract(image_path)
        pytesseract_predictions.append(pytesseract_pred)
    
    # Calculate accuracies
    print("Calculating accuracies...")
    cdistnet_accuracy, cdistnet_correct, total = calculate_accuracy(cdistnet_predictions, ground_truths)
    pytesseract_accuracy, pytesseract_correct, _ = calculate_accuracy(pytesseract_predictions, ground_truths)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total images evaluated: {total}")
    print(f"CDistNet (Epoch 27) Accuracy: {cdistnet_accuracy:.2f}% ({cdistnet_correct}/{total} correct)")
    print(f"Pytesseract Accuracy: {pytesseract_accuracy:.2f}% ({pytesseract_correct}/{total} correct)")
    print("="*50)
    
    # Save detailed results
    print(f"\nSaving detailed results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("Image Path\tGround Truth\tCDistNet Prediction\tPytesseract Prediction\tCDistNet Correct\tPytesseract Correct\n")
        for i, image_path in enumerate(image_paths):
            gt_text = ground_truths[i]
            cdistnet_pred = cdistnet_predictions[i]
            pytesseract_pred = pytesseract_predictions[i]
            cdistnet_correct_flag = "✓" if cdistnet_pred == gt_text else "✗"
            pytesseract_correct_flag = "✓" if pytesseract_pred == gt_text else "✗"
            
            f.write(f"{image_path}\t{gt_text}\t{cdistnet_pred}\t{pytesseract_pred}\t{cdistnet_correct_flag}\t{pytesseract_correct_flag}\n")
    
    print("Evaluation complete!")

if __name__ == '__main__':
    main()
