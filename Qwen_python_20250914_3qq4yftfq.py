# visualize_lmdb.py
import lmdb
import cv2
import numpy as np
from PIL import Image
import os

def visualize_lmdb_samples(lmdb_path, num_samples=5):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        # دریافت تعداد کل نمونه‌ها
        n_samples = int(txn.get('num-samples'.encode()).decode())
        print(f"Total samples in {lmdb_path}: {n_samples}")

        # نمایش چند نمونه اولیه
        for i in range(1, min(num_samples + 1, n_samples + 1)):
            image_key = f'image-{i:09d}'
            label_key = f'label-{i:09d}'

            image_bin = txn.get(image_key.encode())
            label = txn.get(label_key.encode()).decode()

            if image_bin:
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                # فرض: تصاویر grayscale ذخیره شده‌اند
                img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    print(f"Sample {i}: Label = '{label}', Image shape = {img.shape}")
                    # نمایش تصویر (اختیاری)
                    # cv2.imshow(f'Sample {i}: {label}', img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # یا ذخیره تصویر
                    # cv2.imwrite(f'sample_{i}_{label}.png', img)
                else:
                    print(f"Sample {i}: Failed to decode image.")
            else:
                print(f"Sample {i}: Image not found.")

if __name__ == '__main__':
    # مسیر پوشه LMDB خود را اینجا قرار دهید
    lmdb_train_path = 'dataset/train_lmdb'
    lmdb_val_path = 'dataset/val_lmdb' # اگر دارید

    print("--- Checking Train LMDB ---")
    visualize_lmdb_samples(lmdb_train_path)
    print("\n--- Checking Val LMDB ---")
    visualize_lmdb_samples(lmdb_val_path)