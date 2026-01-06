# configs/eval_config.py - Configuration for evaluating the latest Persian CDistNet model

# === تنظیمات مربوط به واژگان (Vocabulary) ===
dst_vocab = "cdistnet/utils/dict_persian_letters_only.txt"
dst_vocab_size = 36

# === تنظیمات پیش‌پردازش تصویر ===
rgb2gray = False
keep_aspect_ratio = False
width = 64
height = 24
max_width = 180
is_lower = False

# === تنظیمات معماری مدل ===
cnn_num = 2
leakyRelu = False
hidden_units = 512
ff_units = 1024
scale_embedding = True
attention_dropout_rate = 0.0
residual_dropout_rate = 0.1

# === تنظیمات Transformer ===
num_encoder_blocks = 3
num_decoder_blocks = 3
num_heads = 8

# === تنظیمات جستجوی پرتو (Beam Search) ===
beam_size = 10
n_best = 1

# === تنظیمات تقویت داده ===
data_aug = False

# === تنظیمات TPS ===
num_fiducial = 20

# === تنظیمات آموزش ===
train_method = "origin"
optim = "origin"

# === تنظیمات بلوک‌های مدل ===
tps_block = "TPS"
feature_block = "Resnet45"

# === تنظیمات داده و آموزش ===
train = dict(
    grads_clip=5,
    optimizer="adam_decay",
    learning_rate_warmup_steps=10000,
    label_smoothing=True,
    shared_embedding=False,
    device="cuda",
    gt_file=["dataset/train_lmdb"],
    num_worker=0,
    model_dir="models/tps_persian_cdistnet",
    num_epochs=20,
    batch_size=256,
    model=None,
    save_iter=2000,
    display_iter=100,
    tfboard_iter=100,
    eval_iter=1000,
)

# === تنظیمات اعتبارسنجی ===
val = dict(
    device="cuda",
    gt_file=["dataset/val_lmdb"],
    num_worker=0,
    batch_size=64,
)

# === تنظیمات تست ===
test = dict(
    test_one=False,
    device="cuda",
    rotate=False,
    best_acc_test=False,  # Don"t test best accuracy models
    eval_all=False,  # Don"t test all models
    s_epoch=-1,  # Not used since eval_all is False
    e_epoch=-1,  # Not used since eval_all is False
    avg_s=-1,
    avg_e=9,
    avg_all=False,
    is_test_gt=False,  # We"ll use our own dataset
    image_dir=None,     # Not needed since we"re using LMDB
    test_list=["/home/homeai/Documents/GitHub/CDistNet/dataset/eval_lmdb"],  # Our new evaluation dataset
    batch_size=32,
    num_worker=0,
    model_dir="/home/homeai/Documents/GitHub/CDistNet/models/tps_persian_cdistnet",  # Directory containing the model
    script_path="utils/Evaluation_TextRecog/script.py",
    python_path="/home/homeai/tf/bin/python3.12"  # Use the specified Python environment
)
