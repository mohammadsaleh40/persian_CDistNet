# create_lmdb_dataset.py
import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def checkImageIsValid(imageBin):
    """بررسی می‌کند که تصویر باینری معتبر است یا نه."""
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    """داده‌های موجود در cache را در دیتابیس LMDB می‌نویسد."""
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def createDataset(outputPath, gtFile, imagePathPrefix="", checkValid=True):
    """
    دیتاست LMDB برای آموزش CDistNet ایجاد می‌کند.

    ARGS:
        outputPath    : مسیر خروجی برای دیتابیس LMDB
        gtFile        : مسیر فایل ground truth (e.g., 'dataset/gt.txt')
        imagePathPrefix: پیشوند مسیر تصاویر (در صورت نیاز). معمولاً پوشه dataset
        checkValid    : اگر True باشد، صحت هر تصویر را بررسی می‌کند
    """
    os.makedirs(outputPath, exist_ok=True)
    # map_size را بر اساس نیاز خود تنظیم کنید. اینجا 10GB فرض شده.
    env = lmdb.open(outputPath, map_size=10995116277760) 
    cache = {}
    cnt = 1 # شروع اندیس‌گذاری از 1

    # خواندن فایل annotation
    if not os.path.exists(gtFile):
        print(f"Error: Ground truth file {gtFile} not found.")
        return

    with open(gtFile, 'r', encoding='utf-8') as f:
        data_list = [line.strip() for line in f if line.strip()]

    nSamples = len(data_list)
    if nSamples == 0:
        print("Error: Ground truth file is empty or all lines are invalid.")
        return
        
    print(f"Total samples to process: {nSamples}")

    for i, line in enumerate(tqdm(data_list, desc='Creating LMDB')):
        if not line:
            continue

        try:
            # فرض فرمت: image_path, "label" یا image_path,label
            # جدا کردن مسیر تصویر و برچسب
            if line.count(',') == 0:
                print(f"Warning: Skipping invalid line (no comma): {line}")
                continue

            # تقسیم خط به دو بخش اول و بقیه
            parts = line.split(',', 1) 
            if len(parts) != 2:
                print(f"Warning: Skipping invalid line (more/less than one comma): {line}")
                continue
                
            imagePathPart = parts[0].strip()
            labelPart = parts[1].strip()
            
            # حذف " از اطراف برچسب اگر وجود داشته باشد
            label = labelPart.strip('"')

            # ساخت مسیر کامل تصویر
            imagePath = os.path.join(imagePathPrefix, imagePathPart) if imagePathPrefix else imagePathPart
            imagePath = os.path.abspath(imagePath) # تبدیل به مسیر مطلق

            if not os.path.exists(imagePath):
                print(f'{imagePath} does not exist')
                continue

            with open(imagePath, 'rb') as f:
                imageBin = f.read()

            if checkValid:
                try:
                    if not checkImageIsValid(imageBin):
                        print(f'{imagePath} is not a valid image')
                        continue
                except Exception as e:
                    print(f'Error checking image validity for {imagePath}: {e}')
                    continue

            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode('utf-8') # ذخیره برچسب به صورت UTF-8

            # هر 1000 نمونه یا در پایان، داده‌ها را در LMDB می‌نویسیم
            if cnt % 1000 == 0 or cnt == nSamples:
                writeCache(env, cache)
                cache = {}
                # print('Written %d / %d' % (cnt, nSamples)) # اختیاری برای لاگ بیشتر

            cnt += 1
            
        except Exception as e:
             print(f"Error processing line {i+1}: {line}. Error: {e}")
             continue # رد کردن خطوط مشکل‌دار

    nSamples = cnt - 1 # تعداد نمونه‌های واقعی پردازش شده
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print(f'Created dataset with {nSamples} samples at {outputPath}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', required=True, help='Path to the folder containing images and gt.txt')
    parser.add_argument('--gtFile', default='gt.txt', help='Name of the ground truth file (default: gt.txt)')
    parser.add_argument('--outputPath', required=True, help='LMDB output path')
    args = parser.parse_args()

    # مسیر فایل annotation
    gt_file_path = os.path.join(args.inputPath, args.gtFile)
    
    # ایجاد دیتاست LMDB
    createDataset(args.outputPath, gt_file_path, imagePathPrefix=args.inputPath, checkValid=True)

    print("LMDB dataset creation completed.")

