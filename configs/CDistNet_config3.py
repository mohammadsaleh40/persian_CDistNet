# configs/CDistNet_config.py

# === تنظیمات مربوط به واژگان (Vocabulary) ===
# مسیر فایل واژگان شما
dst_vocab = 'cdistnet/utils/dict_persian_letters_number.txt' # یا مسیر واقعی فایل در پروژه شما

dst_vocab_size = 81

# === تنظیمات پیش‌پردازش تصویر ===
rgb2gray = True  # چون تصاویر شما grayscale هستند

keep_aspect_ratio = False  # صحیح است - تصاویر را به ابعاد ثابت تغییر می‌دهد
width = 128
height = 36
max_width = 128

is_lower = False

cnn_num = 2

leakyRelu = False

hidden_units = 512

ff_units = 1024 

scale_embedding = True

attention_dropout_rate = 0.05 

residual_dropout_rate = 0.15


num_encoder_blocks = 3 

num_decoder_blocks = 3 

num_heads = 8

beam_size = 5

n_best = 1


data_aug = False


num_fiducial = 20


train_method = 'origin' 
optim = 'origin'

tps_block = 'TPS' 

feature_block = 'Resnet45'  

train = dict(
    grads_clip=5, 
    optimizer='adam_decay', 
    learning_rate_warmup_steps=40000, 
    label_smoothing=True, 
    shared_embedding=False, 
    device='cuda', 
    gt_file=['dataset/train_128_36_lmdb'], 
    num_worker=0,
    model_dir='models/tps_persian_cdistnet_128_36', 
    num_epochs=31, 
    batch_size=160, 
    model=None, 
    
    save_iter=4000,
    display_iter=100,
    tfboard_iter=100, 
    eval_iter=4000, 
)

# === تنظیمات اعتبارسنجی ===
val = dict(
    device='cuda',
    # مسیر فایل LMDB اعتبارسنجی (اگر دارید)
    gt_file=['dataset/little_128_36_lmdb'], # <-- تنظیم مسیر LMDB اعتبارسنجی (در صورت وجود)
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
