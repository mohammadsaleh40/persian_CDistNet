import argparse
import unicodedata

def create_vocab_from_text(text_file, output_file):
    """
    ایجاد فایل واژگان از یک فایل متنی
    
    Args:
        text_file: مسیر فایل متنی ورودی
        output_file: مسیر فایل خروجی واژگان
    """
    # خواندن فایل متنی
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # استخراج تمام کاراکترهای منحصر به فرد از متن
    all_chars = set(text)
    
    # پالایش کاراکترها - حذف کاراکترهای کنترلی و غیرضروری
    valid_chars = set()
    for char in all_chars:
        # بررسی کاراکترهای معتبر: حروف، اعداد، نمادها، فاصله و نیم‌فاصله
        if char.isprintable() or char.isspace() or ord(char) == 8204:  # 8204 = نیم‌فاصله
            valid_chars.add(char)
    
    # تبدیل به لیست و مرتب‌سازی
    sorted_chars = sorted(valid_chars)
    
    # مطمئن شدن از وجود نیم‌فاصله در واژگان (چون برای فارسی ضروری است)
    if '\u200c' not in sorted_chars:
        sorted_chars.append('\u200c')
        sorted_chars = sorted(sorted_chars)
    
    # حذف کاراکترهای اضافی مثل newline و tab که در واژگان لازم نیستند
    sorted_chars = [char for char in sorted_chars if char not in ['\n', '\r', '\t']]
    
    # تعریف ۴ توکن پیش‌فرض
    default_tokens = ['<blank>', '<unk>', '<s>', '</s>']
    
    # ایجاد فایل واژگان
    with open(output_file, 'w', encoding='utf-8') as f:
        # نوشتن توکن‌های پیش‌فرض
        for token in default_tokens:
            f.write(token + '\n')
        
        # نوشتن سایر کاراکترها
        for char in sorted_chars:
            if char.strip() != '' or char == ' ' or char == '\u200c':  # نگه داشتن فاصله و نیم‌فاصله
                f.write(char + '\n')
    
    print(f"✅ فایل واژگان با موفقیت ایجاد شد: {output_file}")
    print(f"🔢 تعداد کاراکترهای منحصر به فرد (بدون توکن‌های پیش‌فرض): {len(sorted_chars)}")
    print(f"📊 اندازه کل واژگان (با توکن‌های پیش‌فرض): {len(default_tokens) + len(sorted_chars)}")
    print(f"🔍 نمونه‌ای از کاراکترها: {''.join(sorted_chars[:15])}...")
    print(f"✨ نیم‌فاصله {'وجود دارد' if '\u200c' in sorted_chars else 'وجود ندارد'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ایجاد فایل واژگان بر اساس یک فایل متنی')
    parser.add_argument('--text_file', type=str, required=True, help='مسیر فایل متنی ورودی')
    parser.add_argument('--output_file', type=str, default='dict_persian_letters_number.txt', 
                        help='مسیر فایل خروجی واژگان (پیش‌فرض: dict_persian_letters_number.txt)')
    
    args = parser.parse_args()
    
    create_vocab_from_text(args.text_file, args.output_file)