image_path = "dataset/eval_img/0010_زمینی.jpg"
# image_path = "dataset/eval_img/IMG_20241228_145829.jpg"
# image_path = "dataset/eval_img/0018_پرورش.jpg"
# image_path = "dataset/eval_img/0019_زمان.jpg"
# image_path = "dataset/eval_img/0045_پانصد.jpg"
# image_path = "dataset/eval_img/0070_ورقه.jpg"
image_path = "dataset/eval_img/0094_طول.jpg"


import cv2
import pytesseract

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# بزرگ‌نمایی (خیلی مهم)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# threshold
_, th = cv2.threshold(gray, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

config = "--psm 8 --oem 3"
text = pytesseract.image_to_string(th, lang="fas", config=config)
print(text)
