# %%
import os
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor




image_path = "dataset/eval_img/0010_زمینی.jpg"

# %%
def upscale_image(
    img: Image.Image,
    min_width: int = 512,
    min_height: int = 64,
):
    w, h = img.size

    scale_w = min_width / w
    scale_h = min_height / h
    scale = max(scale_w, scale_h, 1.0)  # فقط بزرگ کن، کوچیک نکن

    new_w = int(w * scale)
    new_h = int(h * scale)

    return img.resize((new_w, new_h), Image.BICUBIC)

# %%

foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)
detection_predictor = DetectionPredictor()

# %%
image = Image.open(image_path).convert("RGB")
image = upscale_image(image)

# %%
page_predictions = recognition_predictor(
    [image],
    det_predictor=detection_predictor
)
# %%
print(page_predictions)

# %%
print(page_predictions[0].text_lines[0].text
)
# %%
