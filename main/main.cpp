#include "lcd.h"
#include "camera.h"
#include "yolo.h"
#include "ofd.h"

#include "esp_log.h"
#include "esp_timer.h"
#include <inttypes.h>
#include <vector>
#include <array>

static const char *TAG = "MAIN_APP";
// YOLO input size
#define YOLO_W 160
#define YOLO_H 160

// YOLO input buffer in DRAM (RGB888)
__attribute__((section(".ext_ram.bss"), aligned(64)))
static uint8_t yolo_buf[YOLO_W * YOLO_H * 3];
// camera resolution (RGB565)
#define CAM_W 800
#define CAM_H 640

__attribute__((section(".ext_ram.bss"), aligned(64)))
uint16_t lcd_buf[LCD_WIDTH * LCD_HEIGHT];

#define OFD_W 160
#define OFD_H 120
__attribute__((section(".ext_ram.bss"), aligned(64)))
static uint8_t ofd_gray[OFD_W * OFD_H];

// Center-crop square from camera, then resize to 320×320 and convert to RGB888
__attribute__((section(".iram1")))
static inline void center_crop_resize_rgb565_to_rgb888(
    const uint16_t *src,  // CAM_W * CAM_H, RGB565
    int src_w,
    int src_h,
    uint8_t *dst,         // YOLO_W * YOLO_H * 3, RGB888
    int dst_w,
    int dst_h)
{
    // square crop size = min(width, height)
    int crop_size = (src_w < src_h) ? src_w : src_h;
    int x0 = (src_w - crop_size) / 2;  // center crop
    int y0 = (src_h - crop_size) / 2;

    for (int y = 0; y < dst_h; ++y) {
        int sy = (y * crop_size) / dst_h;   // nearest neighbor
        int src_row_base = (y0 + sy) * src_w;

        for (int x = 0; x < dst_w; ++x) {
            int sx = (x * crop_size) / dst_w;
            uint16_t p = src[src_row_base + (x0 + sx)];

            // RGB565 → RGB888 (same as your old code but only on 320×320)
            uint8_t r = ((p >> 11) & 0x1F) * 255 / 31;
            uint8_t g = ((p >>  5) & 0x3F) * 255 / 63;
            uint8_t b = (p & 0x1F) * 255 / 31;

            int idx = (y * dst_w + x) * 3;
            dst[idx + 0] = r;
            dst[idx + 1] = g;
            dst[idx + 2] = b;
        }
    }
}

__attribute__((section(".iram1")))
static inline void downsample_rgb565_to_gray(
    const uint16_t *src, int src_w, int src_h,
    uint8_t *dst, int dst_w, int dst_h)
{
    // Nearest-neighbor downsample + luma approximation
    for (int y = 0; y < dst_h; ++y) {
        int sy = (y * src_h) / dst_h;
        const uint16_t *srow = src + sy * src_w;

        for (int x = 0; x < dst_w; ++x) {
            int sx = (x * src_w) / dst_w;
            uint16_t p = srow[sx];

            // RGB565 -> 8-bit R,G,B
            uint8_t r = (uint8_t)(((p >> 11) & 0x1F) * 255 / 31);
            uint8_t g = (uint8_t)(((p >>  5) & 0x3F) * 255 / 63);
            uint8_t b = (uint8_t)(( p        & 0x1F) * 255 / 31);

            // Gray = (0.299R + 0.587G + 0.114B) approx with integers
            // gray ≈ (77R + 150G + 29B) >> 8
            dst[y * dst_w + x] = (uint8_t)((77*r + 150*g + 29*b) >> 8);
        }
    }
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "System boot… Initializing modules…");

    // lcd_select_driver(LCD_TYPE_HX8394);
    // lcd_init();
    // ESP_LOGI(TAG, "LCD initialized.");

    camera_set_sensor(CAMERA_OV5647);
    camera_init();
    ESP_LOGI(TAG, "Camera initialized.");

    // yolo_init();
    // ESP_LOGI(TAG, "YOLO initialized.");

    ofd_init(OFD_W, OFD_H);
    ESP_LOGI(TAG, "OFD initialized.");

    ESP_LOGI(TAG, "Starting main loop…");

    std::vector<std::array<float,5>> boxes;
    // int frame_id = 0;
    // const int YOLO_INTERVAL = 3;   // run YOLO every 3rd frame (tune this)

    while (true)
    {
        int64_t t0, t1;

        camera_frame_t f = camera_get_frame();

        // assume f.data is RGB565, CAM_W x CAM_H

        // 1) Crop/scale to LCD (keep RGB565 – your existing function)

        crop_to_lcd((uint16_t *)f.data, lcd_buf);  // 800x640 → 720x1280


        // 2) Run YOLO 

        // Center-crop square from camera frame and resize to 160×160 RGB888
        // center_crop_resize_rgb565_to_rgb888(
        //         (const uint16_t *)f.data,
        //         CAM_W, CAM_H,
        //         yolo_buf,
        //         YOLO_W, YOLO_H
        //     );

        // int n = yolo_run(yolo_buf, YOLO_W, YOLO_H, boxes);
        // (void)n;


        // 2.5) OFD
        // t0 = esp_timer_get_time();
        downsample_rgb565_to_gray(
            (const uint16_t*)f.data, CAM_W, CAM_H,
            ofd_gray, OFD_W, OFD_H
        );
        // t1 = esp_timer_get_time();
        // ESP_LOGI(TAG, "grayscale: %.2f ms", (t1 - t0)/1000.0);
        
        // t0 = esp_timer_get_time();
        ofd_result_t r = ofd_process_gray(ofd_gray);


        // Example: simple avoidance logic
        // - if divergence is high => obstacle approaching
        // - lr_balance tells which side is "more expanding"
        if (r.valid) {
            // You can tune these thresholds from logs
            const float DIV_DANGER = 0.08f;

            if (r.divergence > DIV_DANGER) {
                // turn away from higher expansion side
                // lr_balance > 0 => right expanding more => turn left
                // lr_balance < 0 => left expanding more  => turn right
                float turn = (r.lr_balance > 0) ? -1.0f : 1.0f;

                ESP_LOGI(TAG, "OFD div=%.4f tau=%.2f lr=%.4f => AVOID turn=%.1f",
                        r.divergence, r.tau, r.lr_balance, turn);

                // TODO: map 'turn' to your servo command
                // servo_set_turn(turn);
            } else {
                ESP_LOGI(TAG, "OFD div=%.4f tau=%.2f (safe)", r.divergence, r.tau);
                // TODO: servo_set_turn(0);
            }
        } else {
            ESP_LOGW(TAG, "OFD not valid (low texture / first frame)");
        }
        // t1 = esp_timer_get_time();
        // ESP_LOGI(TAG, "OFD: %.2f ms", (t1 - t0)/1000.0);


        // 3) Draw to LCD (full frame, RGB565)
        lcd_draw(lcd_buf);

        camera_return_frame(&f);
    }
}
