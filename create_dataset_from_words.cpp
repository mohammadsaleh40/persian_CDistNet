#include <ft2build.h>
#include FT_FREETYPE_H
#include <harfbuzz/hb.h>
#include <harfbuzz/hb-ft.h>
#include <string>
#include <vector>
#include <cstdint>
#include <iostream>
#include <random>
#include <filesystem> 
#include <fstream>
#include <sstream>
#include <cmath> // برای توابع ریاضی
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
namespace fs = std::filesystem;

// تابع خواندن لیست کلمات از فایل
std::vector<std::string> read_words_from_file(const std::string& filename) {
    std::vector<std::string> words;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open word list file: " << filename << std::endl;
        return words;
    }

    std::string line;
    while (std::getline(file, line)) {
        // حذف فضاهای خالی از ابتدا و انتها
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (!line.empty()) {
            words.push_back(line);
        }
    }
    file.close();
    std::cout << "Loaded " << words.size() << " words from " << filename << std::endl;
    return words;
}

// تابع اعمال اعوجاج تصادفی ساده به آرایه تصویر
void apply_random_distortion(uint8_t image[24][64], std::mt19937& gen) {
    // تعریف توزیع‌های تصادفی
    std::uniform_real_distribution<> dist_strength(-1.5, 1.5); // قدرت اعوجاج
    std::uniform_real_distribution<> dist_wave_freq(0.1, 0.3); // فرکانس موج
    std::uniform_real_distribution<> dist_offset(0, 2 * M_PI); // افست فاز

    // ایجاد یک تصویر موقت برای نگهداری نتیجه
    uint8_t temp_image[24][64];
    // مقداردهی اولیه با سفید (255)
    std::fill(&temp_image[0][0], &temp_image[0][0] + 24 * 64, 255);

    double strength = dist_strength(gen);
    double frequency = dist_wave_freq(gen);
    double phase_offset = dist_offset(gen);

    // اعمال اعوجاج موج‌دار ساده (در اینجا در جهت افقی/عمودی خفیف)
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 64; ++x) {
            // محاسبه جابجایی بر اساس یک تابع سینوسی
            // اعوجاج عمودی بر اساس موقعیت افقی
            int dy = static_cast<int>(strength * std::sin(frequency * x + phase_offset));
            // اعوجاج افقی بر اساس موقعیت عمودی (اختیاری)
            int dx = static_cast<int>(strength * 0.5 * std::sin(frequency * y + phase_offset * 2));

            // محاسبه مختصات جدید
            int new_x = x + dx;
            int new_y = y + dy;

            // بررسی مرزهای تصویر
            if (new_x >= 0 && new_x < 64 && new_y >= 0 && new_y < 24) {
                // کپی پیکسل از مکان اصلی به مکان جدید
                temp_image[new_y][new_x] = std::min(temp_image[new_y][new_x], image[y][x]);
            }
            // پیکسل‌هایی که خارج از مرز می‌روند به صورت ضمنی سفید باقی می‌مانند
        }
    }

    // کپی کردن نتیجه به آرایه اصلی
    for (int i = 0; i < 24; ++i) {
        for (int j = 0; j < 64; ++j) {
            image[i][j] = temp_image[i][j];
        }
    }
}

// تابع رندر متن با استفاده از FreeType و HarfBuzz
void draw_text(const std::string& text, const std::string& font_path, uint8_t image[24][64]) {
    FT_Library ft;
    if (FT_Init_FreeType(&ft)) {
        std::cerr << "Could not init FreeType library" << std::endl;
        return;
    }
    FT_Face face;
    if (FT_New_Face(ft, font_path.c_str(), 0, &face)) {
        std::cerr << "Could not open font " << font_path << std::endl;
        FT_Done_FreeType(ft);
        return;
    }
    
    // تنظیم اندازه فونت
    FT_Set_Pixel_Sizes(face, 0, 17);

    hb_buffer_t *buf = hb_buffer_create();
    hb_buffer_add_utf8(buf, text.c_str(), -1, 0, -1);
    hb_buffer_set_direction(buf, HB_DIRECTION_RTL); // راست به چپ برای فارسی
    hb_buffer_set_script(buf, HB_SCRIPT_ARABIC);   // اسکریپت عربی/فارسی
    hb_buffer_set_language(buf, hb_language_from_string("fa", -1)); // زبان فارسی
    hb_font_t *hb_font = hb_ft_font_create(face, nullptr);
    hb_shape(hb_font, buf, nullptr, 0);
    unsigned int count;
    hb_glyph_info_t *info = hb_buffer_get_glyph_infos(buf, &count);
    hb_glyph_position_t *pos = hb_buffer_get_glyph_positions(buf, &count);

    // تنظیم موقعیت شروع مناسب برای رندر متن
    int x = 1, y = 15;
    for (unsigned int i = 0; i < count; ++i) {
        if (FT_Load_Glyph(face, info[i].codepoint, FT_LOAD_RENDER)) {
             continue; // در صورت خطا، گلیف را رد کن
        }
        FT_Bitmap bmp = face->glyph->bitmap;
        // موقعیت گلیف با توجه به افست‌ها
        int glyph_x = x + (pos[i].x_offset >> 6);
        int glyph_y = y - face->glyph->bitmap_top;

        // کپی کردن پیکسل‌های گلیف به تصویر اصلی
        for (int row = 0; row < bmp.rows; ++row) {
            for (int col = 0; col < bmp.width; ++col) {
                int px = glyph_x + col;
                int py = glyph_y + row;
                // بررسی مرزهای تصویر (64x24)
                if (px >= 0 && px < 64 && py >= 0 && py < 24) {
                    uint8_t glyph_pixel = bmp.buffer[row * bmp.pitch + col];
                    // اگر گلیف سیاه نیست، پیکسل را به‌روزرسانی کن
                    if (glyph_pixel > 0) {
                         image[py][px] = std::min(image[py][px], static_cast<uint8_t>(255 - glyph_pixel));
                    }
                }
            }
        }
        // به‌روزرسانی موقعیت برای گلیف بعدی
        x += (pos[i].x_advance >> 6);
    }

    // پاک‌سازی منابع HarfBuzz
    hb_buffer_destroy(buf);
    hb_font_destroy(hb_font);
    // پاک‌سازی منابع FreeType
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
}

int main() {
    // لیست فونت‌ها
    std::vector<std::string> font_paths = {
        "Vazirmatn-Regular.ttf", // فونت اصلی
        "B Davat-tamirpc.net.ttf", // فونت دوم
        "BMitra.ttf" // فونت سوم
    };
    
    std::string word_list_path = "big.txt";
    int images_per_folder = 0;

    // خواندن لیست کلمات
    std::vector<std::string> word_list = read_words_from_file(word_list_path);
    if (word_list.empty()) {
        std::cerr << "No words loaded. Exiting." << std::endl;
        return -1;
    }
    images_per_folder = word_list.size();

    // ایجاد پوشه dataset و زیرپوشه images
    fs::create_directory("dataset");
    std::string image_folder = "dataset/images";
    fs::create_directory(image_folder);

    // باز کردن فایل annotation
    std::ofstream annotation_file("dataset/gt.txt");
    if (!annotation_file.is_open()) {
        std::cerr << "Could not create annotation file dataset/gt.txt" << std::endl;
        return -1;
    }

    int total_images = 0;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> word_dist(0, word_list.size() - 1);

    // پردازش هر کلمه با هر فونت
    for (int word_idx = 0; word_idx < word_list.size(); ++word_idx) {
        std::string word = word_list[word_idx];
        
        // رندر کلمه با هر فونت
        for (int font_idx = 0; font_idx < font_paths.size(); ++font_idx) {
            std::string font_path = font_paths[font_idx];
            
            // ایجاد تصویر خالی سفید (8-bit grayscale)
            uint8_t image[24][64];
            std::fill(&image[0][0], &image[0][0] + 24 * 64, 255);

            // رندر کلمه
            draw_text(word, font_path, image);

            // اعمال اعوجاج تصادفی
            apply_random_distortion(image, gen);

            // ایجاد نام فایل منحصر به فرد (word_index_font_index.png)
            std::string filename = "img_" + std::to_string(word_idx) + "_" + std::to_string(font_idx) + ".png";
            std::string full_path = fs::path(image_folder) / filename;

            // ذخیره تصویر PNG با ابعاد 64x24
            int result = stbi_write_png(full_path.c_str(), 64, 24, 1, image, 64);
            if (result == 0) {
                std::cerr << "Failed to write: " << full_path << std::endl;
            } else {
                std::cout << "Created: " << full_path << std::endl;
                // نوشتن مسیر نسبی و برچسب در فایل annotation
                annotation_file << "images/" << filename << ", \"" << word << "\"" << std::endl;
                total_images++;
            }
        }
    }

    annotation_file.close();
    std::cout << "Dataset creation finished. Total images: " << total_images << std::endl;
    std::cout << "Images saved in: " << image_folder << std::endl;
    std::cout << "Annotation file created: dataset/gt.txt" << std::endl;

    return 0;
}