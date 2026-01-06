import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import re
import sys

# Use xelatex with Persian font support
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{fontspec}
        \setmainfont{XB Zar}  % Persian font
    """,
    "font.family": "serif"
})

def contains_latex(text):
    """Check if text contains LaTeX expressions"""
    if '$' in text:
        return True
    
    latex_patterns = [
        r'\\times', r'\\frac', r'\\sum', r'\\prod', r'\\int', r'\\lim', 
        r'\\log', r'\\sin', r'\\cos', r'\\tan', r'\\sqrt', r'\\alpha', 
        r'\\beta', r'\\gamma', r'\\delta', r'\\theta', r'\\pi', r'\\infty'
    ]
    
    for pattern in latex_patterns:
        if re.search(pattern, text):
            return True
    
    return False

def clean_latex_text(latex_text):
    """Clean and prepare LaTeX text for rendering - keep Persian numbers"""
    latex_text = latex_text.strip()
    
    # Remove surrounding $ if present (we'll wrap it ourselves)
    if latex_text.startswith('$') and latex_text.endswith('$'):
        latex_text = latex_text[1:-1]
    
    # DO NOT convert Persian numbers - keep them for xelatex
    # Wrap the expression in math mode
    return f"${latex_text}$"

def render_latex_to_image(latex_text, original_text, output_path, width=2, height=0.8, dpi=300):
    """Render LaTeX text to an image using matplotlib with xelatex"""
    try:
        # Clean the LaTeX text for rendering (keep Persian numbers)
        cleaned_latex = clean_latex_text(latex_text)
        print(f"Rendering: {cleaned_latex}")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Add the LaTeX text to the axis
        ax.text(0.5, 0.5, cleaned_latex, fontsize=20, ha='center', va='center', 
                transform=ax.transAxes, wrap=True)
        
        # Turn off axis
        ax.axis('off')
        
        # Adjust layout to remove whitespace
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Save the figure
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.05)
        plt.close()
        
        # Resize image to target size (64x24) after rendering
        img = Image.open(output_path)
        img_resized = img.resize((64, 24), Image.Resampling.LANCZOS)
        img_resized.save(output_path)
        
        return True
    except Exception as e:
        print(f"Error rendering LaTeX: {e}")
        return False

def extract_latex_lines(input_file):
    """Extract lines that contain LaTeX from the input file"""
    latex_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and contains_latex(line):
                latex_lines.append((line_num, line))
    
    return latex_lines

def process_latex_file(input_file='littel.txt', output_dir='dataset/images', gt_file='dataset/gt.txt'):
    """Process the input file, render LaTeX lines, and update gt.txt"""
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(gt_file), exist_ok=True)
    
    # Extract LaTeX lines
    latex_lines = extract_latex_lines(input_file)
    
    if not latex_lines:
        print("No LaTeX lines found in the input file.")
        return
    
    print(f"Found {len(latex_lines)} LaTeX lines to process:")
    
    # Open gt.txt in append mode
    with open(gt_file, 'a', encoding='utf-8') as gt_f:
        for idx, (line_num, latex_text) in enumerate(latex_lines):
            print(f"Processing line {line_num}: {latex_text}")
            
            # Generate image filename
            img_filename = f"latex_{idx:03d}.png"
            img_path = os.path.join(output_dir, img_filename)
            
            # Render LaTeX to image (with Persian numbers preserved)
            if render_latex_to_image(latex_text, latex_text, img_path):
                # Write to gt.txt with original Persian text
                relative_path = os.path.join("images", img_filename)
                gt_f.write(f'{relative_path}, "{latex_text}"\n')
                print(f"  -> Saved as {img_path} and added to {gt_file}")
            else:
                print(f"  -> Failed to render LaTeX: {latex_text}")

if __name__ == "__main__":
    # Use command line arguments if provided, otherwise use defaults
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'littel.txt'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'dataset/images'
    gt_file = sys.argv[3] if len(sys.argv) > 3 else 'dataset/gt.txt'
    
    process_latex_file(input_file, output_dir, gt_file)