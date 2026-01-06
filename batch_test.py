import argparse
import codecs
import os
import torch
from PIL import Image
from mmcv import Config
import numpy as np
from cdistnet.model.translator import Translator
from cdistnet.model.model import build_CDistNet
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Test CDistNet')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Input image directory path')
    parser.add_argument('--output_dir', type=str, default='output_predictions',
                        help='Output directory for prediction images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Input model path')
    parser.add_argument('--config', type=str, default='configs/CDistNet_config.py',
                        help='train config file path')
    args = parser.parse_args()
    return args

def load_vocab(vocab=None, vocab_size=None):
    """
    Load vocab from disk. The fisrt four items in the vocab should be <PAD>, <UNK>, <S>, </S>
    """
    vocab = [' ' if len(line.split()) == 0 else line.split()[0] for line in codecs.open(vocab, 'r', 'utf-8')]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def origin_process_img(cfg, image_path):
    """Process a single image according to the configuration."""
    if cfg.rgb2gray:
        image = Image.open(image_path).convert('L')  # Grayscale -> (H, W)
    else:
        image = Image.open(image_path).convert('RGB')  # RGB -> (H, W, 3)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    image = image.resize((cfg.width, cfg.height), Image.LANCZOS)
    image = np.array(image)  # RGB: (H, W, 3), Grayscale: (H, W)

    if image.ndim == 3:  # RGB
        # Convert (H, W, 3) to (3, H, W)
        image = image.transpose((2, 0, 1))  # CHW
    elif image.ndim == 2:  # Grayscale
        # Convert (H, W) to (1, H, W)
        image = np.expand_dims(image, axis=0)  # (1, H, W)

    # Add batch dimension: (1, C, H, W)
    image = np.expand_dims(image, axis=0)  # (1, C, H, W)

    # Normalize to [-1, 1] - Matching training normalization
    image = image.astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(image)
    # Ensure contiguous memory layout
    image = image.contiguous()

    return image

def batch_test(cfg, args):
    """Process all images in a directory and visualize predictions."""
    # Prepare model
    model = build_CDistNet(cfg)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    device = torch.device(cfg.test.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Prepare translator and vocabulary
    translator = Translator(cfg, model)
    word2idx, idx2word = load_vocab(cfg.dst_vocab, cfg.dst_vocab_size)
    
    # Get list of image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(args.image_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {args.image_dir}")
        return
        
    print(f"Found {len(image_files)} images to process.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each image
    for i, image_file in enumerate(image_files):
        try:
            image_path = os.path.join(args.image_dir, image_file)
            print(f"Processing ({i+1}/{len(image_files)}): {image_file}")
            
            # Process image
            img_tensor = origin_process_img(cfg, image_path)
            img_tensor = img_tensor.to(device)
            
            # Get prediction
            with torch.no_grad():
                all_hyp, all_scores = translator.translate_batch(img_tensor)
            
            # Extract predicted text
            if all_hyp and all_hyp[0]:
                # Get the best prediction (first in the beam)
                idx_seq = all_hyp[0][0] 
                # Remove end token (usually 3) if present
                idx_seq = [x for x in idx_seq if x != 3]
                # Convert indices to characters
                predicted_text = ''.join([idx2word.get(idx, '<UNK>') for idx in idx_seq])
            else:
                predicted_text = ""
                
            # Load original image for visualization
            original_img = Image.open(image_path)
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.imshow(original_img)
            ax.axis('off')
            
            # Add prediction text as title
            plt.title(f"Predicted: {predicted_text}", fontsize=14, pad=20)
            
            # Save the result
            output_path = os.path.join(args.output_dir, f"pred_{image_file}")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
            print(f"  -> Saved prediction to {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    batch_test(cfg, args)

if __name__ == '__main__':
    main()