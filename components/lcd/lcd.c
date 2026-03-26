#include "lcd.h"
#include "esp_log.h"

#include "drivers/lcd_ek79007.h"
#include "drivers/lcd_hx8394.h"

#include <string.h>

static const char *TAG = "LCD";
static esp_lcd_panel_handle_t lcd_panel = NULL;

// Default LCD driver
static lcd_type_t selected_driver = LCD_TYPE_EK79007;

#define TILE_HEIGHT 32

// RGB565 tile buffer (2 bytes per pixel)
static uint16_t tile_buf[LCD_WIDTH * TILE_HEIGHT];

void lcd_select_driver(lcd_type_t type)
{
    selected_driver = type;
}

void lcd_init(void)
{
    ESP_LOGI(TAG, "Initializing LCD, driver=%d", selected_driver);

    switch (selected_driver) {
    case LCD_TYPE_EK79007:
        lcd_panel = lcd_ek79007_adapter_init();
        break;

    case LCD_TYPE_HX8394:
        // NOW TAKES CLOCK, e.g. 48 MHz
        lcd_panel = lcd_hx8394_adapter_init(48);
        break;

    default:
        ESP_LOGE(TAG, "Unknown LCD driver selected!");
        return;
    }

    if (!lcd_panel) {
        ESP_LOGE(TAG, "LCD initialization FAILED!");
        return;
    }

    ESP_LOGI(TAG, "LCD initialized successfully");
}

void lcd_draw(uint16_t *src_psram)
{
    if (!lcd_panel || !src_psram)
        return;

    const int w   = LCD_WIDTH;
    const int h   = LCD_HEIGHT;
    const int bpp = 2; // RGB565

    for (int y = 0; y < h; y += TILE_HEIGHT) {

        int th = TILE_HEIGHT;
        if (y + th > h)
            th = h - y;

        size_t bytes = (size_t) w * th * bpp;

        // Copy from PSRAM → DRAM tile
        memcpy(tile_buf,
               src_psram + (size_t) y * w,
               bytes);

        // DRAW ONLY THIS TILE
        esp_lcd_panel_draw_bitmap(
            lcd_panel,
            0, y,
            w, y + th,
            tile_buf
        );
    }
}

void draw_rect565(
    uint16_t *img,
    int x1, int y1, int x2, int y2,
    uint8_t r, uint8_t g, uint8_t b)
{
    uint16_t color =
        ((r & 0xF8) << 8) |
        ((g & 0xFC) << 3) |
        ((b & 0xF8) >> 3);

    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 >= LCD_WIDTH)  x2 = LCD_WIDTH - 1;
    if (y2 >= LCD_HEIGHT) y2 = LCD_HEIGHT - 1;

    // top & bottom
    for (int x = x1; x <= x2; x++) {
        img[y1 * LCD_WIDTH + x] = color;
        img[y2 * LCD_WIDTH + x] = color;
    }

    // left & right
    for (int y = y1; y <= y2; y++) {
        img[y * LCD_WIDTH + x1] = color;
        img[y * LCD_WIDTH + x2] = color;
    }
}

