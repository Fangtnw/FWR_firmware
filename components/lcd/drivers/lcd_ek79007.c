#include "lcd_ek79007.h"
#include "esp_lcd_ek79007.h"

#include "driver/gpio.h"
#include "esp_ldo_regulator.h"
#include "esp_lcd_mipi_dsi.h"
#include "esp_log.h"
#include <stdlib.h>

#define LCD_RST_GPIO        27
#define LCD_BK_LIGHT_GPIO   26
#define LCD_BK_ON_LEVEL     1

static const char *TAG = "LCD_EK79007_ADPT";

esp_lcd_panel_handle_t lcd_ek79007_adapter_init(void)
{
    ESP_LOGI(TAG, "Init EK79007 panel");

    // BACKLIGHT
    gpio_config_t bk_cfg = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = 1ULL << LCD_BK_LIGHT_GPIO,
    };
    gpio_config(&bk_cfg);
    gpio_set_level(LCD_BK_LIGHT_GPIO, LCD_BK_ON_LEVEL);

    // 2.5V LDO for MIPI
    esp_ldo_channel_handle_t ldo = NULL;
    esp_ldo_channel_config_t ldo_cfg = {
        .chan_id = 3,
        .voltage_mv = 2500,
    };
    esp_ldo_acquire_channel(&ldo_cfg, &ldo);

    // MIPI DSI bus
    esp_lcd_dsi_bus_handle_t bus = NULL;
    esp_lcd_dsi_bus_config_t bus_cfg = EK79007_PANEL_BUS_DSI_2CH_CONFIG();
    esp_lcd_new_dsi_bus(&bus_cfg, &bus);

    // DBI IO
    esp_lcd_panel_io_handle_t io = NULL;
    esp_lcd_dbi_io_config_t dbi_cfg = EK79007_PANEL_IO_DBI_CONFIG();
    esp_lcd_new_panel_io_dbi(bus, &dbi_cfg, &io);

    // Install panel
    esp_lcd_dpi_panel_config_t dpi_cfg =
        EK79007_1024_600_PANEL_60HZ_CONFIG(LCD_COLOR_PIXEL_FORMAT_RGB888);

    ek79007_vendor_config_t vendor_cfg = {
        .mipi_config = {
            .dsi_bus = bus,
            .dpi_config = &dpi_cfg,
            .lane_num = 2,
        },
    };

    esp_lcd_panel_handle_t panel = NULL;

    const esp_lcd_panel_dev_config_t panel_cfg = {
        .reset_gpio_num = LCD_RST_GPIO,
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB,
        .bits_per_pixel = 24,
        .vendor_config = &vendor_cfg,
    };

    esp_lcd_new_panel_ek79007(io, &panel_cfg, &panel);
    esp_lcd_panel_reset(panel);
    esp_lcd_panel_init(panel);

    ESP_LOGI(TAG, "EK79007 ready");
    return panel;
}
