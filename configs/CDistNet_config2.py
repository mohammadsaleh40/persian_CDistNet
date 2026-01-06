# configs/CDistNet_config.py

# === تنظیمات مربوط به واژگان (Vocabulary) ===
# مسیر فایل واژگان شما
dst_vocab = 'cdistnet/utils/dict_persian_letters_only.txt' # یا مسیر واقعی فایل در پروژه شما
# تعداد کل کاراکترها در فایل واژگان (32 حرف فارسی + 4 توکن خاص)
dst_vocab_size = 36

# === تنظیمات پیش‌پردازش تصویر ===
# تبدیل تصویر رنگی به خاکستری (طبق کد C++ شما، تصاویر grayscale هستند)
rgb2gray = False
# حفظ نسبت ابعاد (در کد C++ شما اندازه ثابت 128x64 استفاده شده)
keep_aspect_ratio = False
# عرض و ارتفاع تصویر ورودی (طبق کد C++ شما)
width = 64
height = 24 # توجه: در کد C++ ارتفاع 64 است، نه 32
# حداکثر عرض برای تصاویر متغیر (در صورت استفاده)
max_width = 180 # می‌توانید تنظیم کنید
# تبدیل به حروف کوچک (برای فارسی معمولاً لازم نیست)
is_lower = False

# === تنظیمات معماری مدل ===
# تعداد لایه‌های CNN در بک‌بون
cnn_num = 2
# استفاده از LeakyReLU (معمولاً False)
leakyRelu = False
# اندازه واحدهای پنهان در Transformer
hidden_units = 512
# اندازه واحدهای Feed-Forward در Transformer
ff_units = 1024 # معمولاً 2 * hidden_units
# مقیاس‌گذاری جاسازی (Embedding)
scale_embedding = True
# نرخ Dropout برای توجه (Attention)
attention_dropout_rate = 0.0 # می‌توانید افزایش دهید (مثلاً 0.1)
# نرخ Dropout برای باقی‌مانده (Residual)
residual_dropout_rate = 0.1

# === تنظیمات Transformer ===
# تعداد بلاک‌های Encoder
num_encoder_blocks = 3 # یا 4 (طبق پیشنهاد در فایل اصلی)
# تعداد بلاک‌های Decoder
num_decoder_blocks = 3 # یا 4
# تعداد هِد‌های توجه چندگانه (Multi-head Attention)
num_heads = 8

# === تنظیمات جستجوی پرتو (Beam Search) ===
# اندازه پرتو برای تولید متن
beam_size = 10
# تعداد بهترین نتایج برای بازگشت
n_best = 1

# === تنظیمات تقویت داده (Data Augmentation) ===
# فعال‌سازی تقویت داده
data_aug = False

# === تنظیمات TPS (Thin Plate Spline) ===
# تعداد نقاط fiducial برای TPS-STN
num_fiducial = 20

# === تنظیمات آموزش ===
train_method = 'origin' # 'dist' برای آموزش توزیع شده
optim = 'origin' # روش بهینه‌ساز

# === تنظیمات بلوک‌های مدل ===
# بلوک TPS (اگر می‌خواهید استفاده کنید)
tps_block = 'TPS' # یا None
# بلوک استخراج ویژگی (طبق کد C++ شما که شبیه ResNet است)
feature_block = 'Resnet45' # یا 'Resnet31', 'MTB_nrtr'

# === تنظیمات داده و آموزش ===
train = dict(
    grads_clip=5, # برش گرادیان
    optimizer='adam_decay', # بهینه‌ساز
    learning_rate_warmup_steps=40000, # مراحل گرم‌کردن نرخ یادگیری
    label_smoothing=True, # صاف کردن برچسب
    shared_embedding=False, # جاسازی مشترک
    device='cuda', # دستگاه اجرا
    # مسیر فایل LMDB آموزش (باید تنظیم شود) - به صورت لیست
    gt_file=['dataset/train_lmdb'], # <-- تغییر: به صورت لیست
    num_worker=0, # تعداد worker برای بارگذاری داده
    model_dir='models/tps_persian_cdistnet', # مسیر ذخیره مدل
    num_epochs=30, # تعداد epoch های آموزش
    batch_size=276, # <-- اضافه کردن batch_size برای آموزش
    model=None, # <-- اضافه کردن model برای مشخص کردن مدل اولیه (None یعنی از ابتدا)
    # سایر پارامترهای ممکن برای آموزش (اختیاری):
    # current_epoch=29, # اپوک شروع (اگر از مدل قبلی ادامه می‌دهید)
    save_iter=2000, # فاصله ذخیره‌سازی مدل (تکرارها)
    display_iter=100, # فاصله نمایش لاگ (تکرارها)
    tfboard_iter=100, # فاصله نوشتن رویدادها برای Tensorboard (تکرارها)
    eval_iter=2000, # فاصله ارزیابی روی داده‌های اعتبارسنجی (تکرارها)
)

# === تنظیمات اعتبارسنجی ===
val = dict(
    device='cuda',
    # مسیر فایل LMDB اعتبارسنجی (اگر دارید)
    gt_file=['dataset/val_lmdb'], # <-- تنظیم مسیر LMDB اعتبارسنجی (در صورت وجود)
    num_worker=0,
    batch_size=64, # اندازه batch برای اعتبارسنجی
)

# === تنظیمات تست ===
test = dict(
    device='cuda',
    batch_size=32, # اندازه batch برای تست
    num_worker=0,
    model_dir='models/persian_cdistnet', # مسیر بارگذاری مدل آموزش دیده
    script_path='utils/Evaluation_TextRecog/script.py', # مسیر اسکریپت ارزیابی
    python_path='python' # مسیر پایتون برای اسکریپت ارزیابی
)

# === تنظیمات دیگر ===
# استفاده از squential mask برای tgt (در صورت نیاز)
use_squ = True
