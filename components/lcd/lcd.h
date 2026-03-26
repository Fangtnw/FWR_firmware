#pragma once

#include <stdint.h>
#include "esp_lcd_panel_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LCD_WIDTH   720
#define LCD_HEIGHT  1280

typedef enum {
    LCD_TYPE_EK79007 = 0,
    LCD_TYPE_HX8394  = 1,
} lcd_type_t;

void lcd_select_driver(lcd_type_t type);
void lcd_init(void);
void lcd_draw(uint16_t *buffer);
void draw_rect565(uint16_t *img,
                  int x1, int y1, int x2, int y2,
                  uint8_t r, uint8_t g, uint8_t b);

#ifdef __cplusplus
}
#endif
