# %%
from PIL import Image

import pytesseract
# %%


# image_path = "dataset/eval_img/0010_زمینی.jpg"
# image_path = "dataset/eval_img/IMG_20241228_145829.jpg"
image_path = "dataset/eval_img/0018_پرورش.jpg"
# %%
# Extract text from image
text = pytesseract.image_to_string(image_path, lang='fas+eng')

print(text)
# %%
