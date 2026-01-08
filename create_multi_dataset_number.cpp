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
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <functional>
#include <memory>
#include <future>
#include <iomanip>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

// تابع جدید: شبیه‌سازی تبدیل TPS با استفاده از نقاط کنترل تصادفی
void apply_tps_like_distortion(uint8_t image[36][128], std::mt19937& gen) {
    // تعریف توزیع‌های تصادفی برای نقاط کنترل
    std::uniform_real_distribution<> control_dist(-2.0, 2.0); // جابجایی حداکثر نقاط کنترل
    std::uniform_real_distribution<> grid_dist(-1.0, 1.0);   // تأثیر نقاط کنترل بر نقاط دیگر

    // تعداد نقاط کنترل (3x3 یا 2x2) - اینجا 2x2 برای سادگی
    const int num_ctrl_x = 2;
    const int num_ctrl_y = 2;
    const int grid_w = 128 / (num_ctrl_x - 1);
    const int grid_h = 36 / (num_ctrl_y - 1);

    double ctrl_x[num_ctrl_y][num_ctrl_x];
    double ctrl_y[num_ctrl_y][num_ctrl_x];

    // تولید جابجایی‌های تصادفی برای نقاط کنترل
    for (int cy = 0; cy < num_ctrl_y; ++cy) {
        for (int cx = 0; cx < num_ctrl_x; ++cx) {
            ctrl_x[cy][cx] = control_dist(gen);
            ctrl_y[cy][cx] = control_dist(gen);
        }
    }

    // ایجاد یک تصویر موقت برای نگهداری نتیجه
    uint8_t temp_image[36][128];
    std::fill(&temp_image[0][0], &temp_image[0][0] + 36 * 128, 255);

    // اعمال تبدیل معکوس (inverse mapping) - برای هر پیکسل خروجی، مکان ورودی را پیدا کن
    for (int y = 0; y < 36; ++y) {
        for (int x = 0; x < 128; ++x) {
            // محاسبه موقعیت نسبی در گرید کنترل
            double rel_x = static_cast<double>(x) / 128.0;
            double rel_y = static_cast<double>(y) / 36.0;

            // محاسبه جابجایی تقریبی با استفاده از درون‌یابی بی‌هنجار (Barycentric)
            // برای سادگی، فقط از چهار گوشه استفاده می‌کنیم
            double dx = 0, dy = 0;
            // وزن‌های ساده برای چهار گوشه گرید
            double w00 = (1 - rel_x) * (1 - rel_y);
            double w01 = (1 - rel_x) * rel_y;
            double w10 = rel_x * (1 - rel_y);
            double w11 = rel_x * rel_y;

            dx = w00 * ctrl_x[0][0] + w01 * ctrl_x[1][0] + w10 * ctrl_x[0][1] + w11 * ctrl_x[1][1];
            dy = w00 * ctrl_y[0][0] + w01 * ctrl_y[1][0] + w10 * ctrl_y[0][1] + w11 * ctrl_y[1][1];

            // محاسبه مختصات منبع
            int src_x = static_cast<int>(x + dx);
            int src_y = static_cast<int>(y + dy);

            // بررسی مرزهای تصویر
            if (src_x >= 0 && src_x < 128 && src_y >= 0 && src_y < 36) {
                temp_image[y][x] = image[src_y][src_x];
            } else {
                temp_image[y][x] = 255; // پیکسل سفید برای نقاط خارج از مرز
            }
        }
    }

    // کپی کردن نتیجه به آرایه اصلی
    for (int i = 0; i < 36; ++i) {
        for (int j = 0; j < 128; ++j) {
            image[i][j] = temp_image[i][j];
        }
    }
}

// تابع اعمال اعوجاج تصادفی ساده به آرایه تصویر
void apply_random_distortion(uint8_t image[36][128], std::mt19937& gen) {
    // تعریف توزیع‌های تصادفی
    std::uniform_real_distribution<> dist_strength(-1.5, 1.5); // قدرت اعوجاج
    std::uniform_real_distribution<> dist_wave_freq(0.1, 0.3); // فرکانس موج
    std::uniform_real_distribution<> dist_offset(0, 2 * M_PI); // افست فاز

    // ایجاد یک تصویر موقت برای نگهداری نتیجه
    uint8_t temp_image[36][128];
    // مقداردهی اولیه با سفید (255)
    std::fill(&temp_image[0][0], &temp_image[0][0] + 36 * 128, 255);

    double strength = dist_strength(gen);
    double frequency = dist_wave_freq(gen);
    double phase_offset = dist_offset(gen);

    // اعمال اعوجاج موج‌دار ساده (در اینجا در جهت افقی/عمودی خفیف)
    for (int y = 0; y < 36; ++y) {
        for (int x = 0; x < 128; ++x) {
            // محاسبه جابجایی بر اساس یک تابع سینوسی
            // اعوجاج عمودی بر اساس موقعیت افقی
            int dy = static_cast<int>(strength * std::sin(frequency * x + phase_offset));
            // اعوجاج افقی بر اساس موقعیت عمودی (اختیاری)
            int dx = static_cast<int>(strength * 0.5 * std::sin(frequency * y + phase_offset * 2));

            // محاسبه مختصات جدید
            int new_x = x + dx;
            int new_y = y + dy;

            // بررسی مرزهای تصویر
            if (new_x >= 0 && new_x < 128 && new_y >= 0 && new_y < 36) {
                // کپی پیکسل از مکان اصلی به مکان جدید
                temp_image[new_y][new_x] = std::min(temp_image[new_y][new_x], image[y][x]);
            }
            // پیکسل‌هایی که خارج از مرز می‌روند به صورت ضمنی سفید باقی می‌مانند
        }
    }

    // کپی کردن نتیجه به آرایه اصلی
    for (int i = 0; i < 36; ++i) {
        for (int j = 0; j < 128; ++j) {
            image[i][j] = temp_image[i][j];
        }
    }
}

std::string to_persian_digits(const std::string& ascii) {
    std::string out;
    for (unsigned char c : ascii) {
        if (c >= '0' && c <= '9') {
            // U+06F0 = Persian Digit Zero
            char32_t pd = 0x06F0 + (c - '0');
            // UTF-8 encode
            out.push_back(static_cast<char>(0xD8));
            out.push_back(static_cast<char>(0xB0 + (c - '0')));
        } else {
            out.push_back(c);
        }
    }
    return out;
}


// تابع رندر متن با استفاده از FreeType و HarfBuzz - کد اصلی که کار می‌کرد
void draw_text(const std::string& text, const std::string& font_path, uint8_t image[36][128], int font_size) {
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
    FT_Set_Pixel_Sizes(face, 0, font_size);

    hb_buffer_t *buf = hb_buffer_create();
    hb_buffer_add_utf8(buf, text.c_str(), -1, 0, -1);
    hb_buffer_guess_segment_properties(buf);
    hb_font_t *hb_font = hb_ft_font_create(face, nullptr);
    hb_shape(hb_font, buf, nullptr, 0);
    unsigned int count;
    hb_glyph_info_t *info = hb_buffer_get_glyph_infos(buf, &count);
    hb_glyph_position_t *pos = hb_buffer_get_glyph_positions(buf, &count);

    // تنظیم موقعیت شروع مناسب برای رندر متن
    int x = 4, y = 20; // این مقادیر ممکن است نیاز به تنظیم داشته باشند
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
                // بررسی مرزهای تصویر (128x36)
                if (px >= 0 && px < 128 && py >= 0 && py < 36) {
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

// تابع جدید: رندر کلمه و سپس تغییر مکان آن بر اساس bounding box
void draw_text_with_random_placement(const std::string& text, const std::string& font_path, uint8_t image[36][128], int font_size, std::mt19937& gen) {
    // ایجاد یک تصویر موقت برای رندر اولیه کلمه
    uint8_t temp_word[36][128];
    std::fill(&temp_word[0][0], &temp_word[0][0] + 36 * 128, 255); // سفید

    // رندر کلمه در تصویر موقت با موقعیت ثابت (همان کد قبلی)
    draw_text(text, font_path, temp_word, font_size);

    // محاسبه bounding box کلمه در تصویر موقت
    int min_x = 128, max_x = -1, min_y = 36, max_y = -1;
    for (int y = 0; y < 36; ++y) {
        for (int x = 0; x < 128; ++x) {
            if (temp_word[y][x] != 255) { // اگر پیکسل غیر سفید بود
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }

    // اگر هیچ پیکسل غیر سفیدی پیدا نشد، خارج شو
    if (min_x > max_x || min_y > max_y) {
        return;
    }

    int word_width = max_x - min_x + 1;
    int word_height = max_y - min_y + 1;

    // تولید مکان جدید تصادفی برای قرار دادن کلمه در تصویر نهایی
    int max_new_x = std::max(0, 128 - word_width);
    int max_new_y = std::max(0, 36 - word_height);
    std::uniform_int_distribution<> new_x_dist(0, max_new_x);
    std::uniform_int_distribution<> new_y_dist(0, max_new_y);

    int new_start_x = new_x_dist(gen);
    int new_start_y = new_y_dist(gen);

    // کپی کردن کلمه از تصویر موقت به تصویر نهایی در مکان جدید
    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            int dest_x = new_start_x + (px - min_x);
            int dest_y = new_start_y + (py - min_y);
            if (dest_x >= 0 && dest_x < 128 && dest_y >= 0 && dest_y < 36) {
                // فقط پیکسل‌های غیر سفید را کپی کن
                if (temp_word[py][px] != 255) {
                    image[dest_y][dest_x] = temp_word[py][px];
                }
            }
        }
    }
}


// تعریف ساختار برای نگهداری کار
struct Task {
    std::string number;
    std::string font_path;
    int number_idx;
    int font_idx;
    int task_id;
};

// تعریف ساختار برای نتیجه
struct Result {
    std::string filename;
    std::string number;
    uint8_t image[36][128];
    bool success;
};

// تابع پردازش یک کار
Result process_task(const Task& task) {
    Result result;
    result.number = task.number;
    result.success = false;

    // ایجاد تصویر خالی سفید (8-bit grayscale)
    std::fill(&result.image[0][0], &result.image[0][0] + 36 * 128, 255);

    // ایجاد موتور تصادفی برای این کار
    std::random_device rd;
    std::mt19937 gen(rd());

    // تولید مقادیر تصادفی برای تنظیمات
    std::uniform_int_distribution<> font_size_dist(21, 26);

    int font_size = font_size_dist(gen);
    std::string persian_number = to_persian_digits(task.number);
    // رندر عدد با تغییر مکان تصادفی - استفاده از تابع جدید
    draw_text_with_random_placement(task.number, task.font_path, result.image, font_size, gen);

    // اعمال تبدیل TPS شبیه‌سازی شده
    apply_tps_like_distortion(result.image, gen);

    // اعمال اعوجاج تصادفی اضافی (اختیاری)
    // apply_random_distortion(result.image, gen);

    // ایجاد نام فایل منحصر به فرد (number_index_font_index.png)
    result.filename = "img_" + std::to_string(task.task_id) + ".png";
    result.success = true;

    return result;
}

int main() {
    // لیست فونت‌ها
    std::vector<std::string> font_paths = {
        "Vazirmatn-Regular.ttf", // فونت اصلی
        "B Davat-tamirpc.net.ttf", // فونت دوم
        "BMitra.ttf", // فونت سوم
        "B Nazanin-tamirpc.net.ttf",
        "B Lotus-tamirpc.net.ttf",
        "B Yas-tamirpc.net.ttf"
    };

    // ایجاد پوشه dataset و زیرپوشه numbers
    fs::create_directory("dataset");
    std::string image_folder = "dataset/numbers";
    fs::create_directory(image_folder);

    // باز کردن فایل annotation
    std::ofstream annotation_file("dataset/numbers/gt.txt");
    if (!annotation_file.is_open()) {
        std::cerr << "Could not create annotation file dataset/numbers/gt.txt" << std::endl;
        return -1;
    }

    // ایجاد لیست کارها برای اعداد 0 تا 99999
    std::vector<Task> tasks;
    int task_id = 0;
    for (int num = 1; num <= 99999; ++num) {
        std::string number = std::to_string(num);
        for (int font_idx = 0; font_idx < font_paths.size(); ++font_idx) {
            Task task;
            task.number = number;
            task.font_path = font_paths[font_idx];
            task.number_idx = num;
            task.font_idx = font_idx;
            task.task_id = task_id++;
            tasks.push_back(task);
        }
    }

    std::cout << "Total tasks to process: " << tasks.size() << std::endl;

    // تعیین تعداد رشته‌ها (8 هسته)
    const int num_threads = 8;
    std::cout << "Using " << num_threads << " threads for processing." << std::endl;

    // متغیرهای همگان‌سازی
    std::atomic<int> completed_tasks(0);
    std::atomic<int> total_images(0);
    std::mutex file_mutex;

    // پردازش کارها با استفاده از چند رشته
    for (size_t i = 0; i < tasks.size(); i += num_threads) {
        // ایجاد بردار کارهای فعلی برای این دور
        std::vector<std::future<Result>> futures;
        int batch_size = std::min(static_cast<size_t>(num_threads), tasks.size() - i);

        // ایجاد رشته‌ها برای پردازش کارهای فعلی
        for (int j = 0; j < batch_size; ++j) {
            futures.push_back(std::async(std::launch::async, process_task, tasks[i + j]));
        }

        // انتظار برای تکمیل همه کارهای این دور
        for (int j = 0; j < batch_size; ++j) {
            Result result = futures[j].get();

            if (result.success) {
                // ایجاد مسیر کامل فایل
                std::string full_path = fs::path(image_folder) / result.filename;

                // ذخیره تصویر PNG با ابعاد 128x36
                int write_result = stbi_write_png(full_path.c_str(), 128, 36, 1, result.image, 128);
                if (write_result == 0) {
                    std::cerr << "Failed to write: " << full_path << std::endl;
                } else {
                    // نوشتن مسیر نسبی و برچسب در فایل annotation
                    std::lock_guard<std::mutex> lock(file_mutex);
                    annotation_file << result.filename << ", \"" << result.number << "\"" << std::endl;
                    total_images++;
                }
            }

            // نمایش پیشرفت
            int completed = completed_tasks.fetch_add(1) + 1;
            if (completed % 1000 == 0 || completed == tasks.size()) {
                double progress = (static_cast<double>(completed) / tasks.size()) * 100;
                std::cout << "\rProgress: " << completed << "/" << tasks.size()
                         << " (" << std::fixed << std::setprecision(2) << progress << "%)" << std::flush;
            }
        }
    }

    annotation_file.close();
    std::cout << "\nDataset creation finished. Total images: " << total_images << std::endl;
    std::cout << "Images saved in: " << image_folder << std::endl;
    std::cout << "Annotation file created: dataset/numbers/gt.txt" << std::endl;

    return 0;
}
