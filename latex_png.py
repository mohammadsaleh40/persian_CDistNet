import subprocess
import os
from PIL import Image
import pdf2image

def latex_to_image(latex_code, output_path):
    # ایجاد فایل لاتکس موقت با تنظیمات XeLaTeX و فونت فارسی
    tex_content = r'''\documentclass{article}
\usepackage{fontspec}
\usepackage{xepersian}
\settextfont{Yas}
\begin{document}
''' + latex_code + r'\end{document}'

    # ذخیره فایل لاتکس
    with open('temp.tex', 'w', encoding='utf-8') as f:
        f.write(tex_content)
    
    # کامپایل با XeLaTeX
    subprocess.run(['xelatex', '-interaction=nonstopmode', 'temp.tex'])
    
    # تبدیل PDF به تصویر
    images = pdf2image.convert_from_path('temp.pdf')
    images[0].save(output_path, 'PNG')
    
    # پاک کردن فایل‌های موقت
    for ext in ['.tex', '.pdf', '.aux', '.log']:
        if os.path.exists(f'temp{ext}'):
            os.remove(f'temp{ext}')
latex_to_image(r'$3 \times$', "test")