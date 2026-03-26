// components/lcd/drivers/lcd_hx8394.c

#include "esp_log.h"
#include "esp_ldo_regulator.h"

#include "esp_lcd_hx8394.h"
#include "esp_lcd_mipi_dsi.h"
#include "esp_lcd_panel_interface.h"
#include "esp_lcd_panel_ops.h"
#include "driver/gpio.h"

static const char *TAG = "LCD_HX8394";

static esp_lcd_panel_handle_t panel_handle = NULL;

/*
 * OPTIONAL: extra init commands (currently unused).
 * You can add custom HX8394 registers here if needed.
 */
// static const hx8394_lcd_init_cmd_t hx8394_extra_init[] = {
//     // { cmd, data_ptr, data_len, delay_ms },
// };

/**
 * @brief Initialize HX8394 panel for 720x1280 RGB888, 30 Hz
 *        - Uses vendor timing macro HX8394_720_1280_PANEL_30HZ_DPI_CONFIG
 *        - Forces use_dma2d = false to avoid DMA2D/GDMA issues
 *        - Keeps RGB888 (24bpp)
 */
esp_lcd_panel_handle_t lcd_hx8394_adapter_init(void)
{
    ESP_LOGI(TAG, "Initializing HX8394 LCD...");
    esp_err_t err;

    /* 1. Power MIPI DSI PHY (2.5V) */
    esp_ldo_channel_handle_t ldo_phy = NULL;
    esp_ldo_channel_config_t ldo_cfg = {
        .chan_id    = 3,
        .voltage_mv = 2500,
    };
    err = esp_ldo_acquire_channel(&ldo_cfg, &ldo_phy);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to enable MIPI DSI PHY LDO: %s", esp_err_to_name(err));
        return NULL;
    }
    ESP_LOGI(TAG, "MIPI DSI PHY power ON");

    /* 2. Create MIPI-DSI bus (2 data lanes) */
    esp_lcd_dsi_bus_handle_t dsi_bus = NULL;
    esp_lcd_dsi_bus_config_t bus_cfg = HX8394_PANEL_BUS_DSI_2CH_CONFIG();
    err = esp_lcd_new_dsi_bus(&bus_cfg, &dsi_bus);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_dsi_bus failed: %s", esp_err_to_name(err));
        return NULL;
    }

    /* 3. Create DBI IO (command channel over DSI) */
    esp_lcd_panel_io_handle_t dbi_io = NULL;
    esp_lcd_dbi_io_config_t dbi_cfg = HX8394_PANEL_IO_DBI_CONFIG();
    err = esp_lcd_new_panel_io_dbi(dsi_bus, &dbi_cfg, &dbi_io);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_io_dbi failed: %s", esp_err_to_name(err));
        return NULL;
    }

    /* 4. DPI timing (720 x 1280, 30 Hz, RGB565) */
    esp_lcd_dpi_panel_config_t dpi_cfg =
        HX8394_720_1280_PANEL_30HZ_DPI_CONFIG(LCD_COLOR_PIXEL_FORMAT_RGB565);

    // IMPORTANT: turn OFF DMA2D to avoid the dma_trans_done_cb crash
    dpi_cfg.flags.use_dma2d = false;
    // Optional: ensure 1 framebuffer
    dpi_cfg.num_fbs = 1;

    ESP_LOGI(TAG, "DPI clock = %d MHz, use_dma2d = %d, num_fbs = %d",
             dpi_cfg.dpi_clock_freq_mhz,
             dpi_cfg.flags.use_dma2d,
             dpi_cfg.num_fbs);

    /* 5. Vendor config */
    hx8394_vendor_config_t vendor_cfg = {
        .init_cmds      = NULL,   // use default init sequence in esp_lcd_hx8394.c
        .init_cmds_size = 0,
        .mipi_config = {
            .dsi_bus    = dsi_bus,
            .dpi_config = &dpi_cfg,
            .lane_num   = 2,
        },
    };

    /* 6. Panel device config */
    esp_lcd_panel_dev_config_t panel_cfg = {
        .reset_gpio_num = -1,                        // no dedicated reset pin
        .rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_RGB, // standard RGB
        .bits_per_pixel = 16,                        // RGB565
        .vendor_config  = &vendor_cfg,
    };

    /* 7. Create HX8394 panel */
    err = esp_lcd_new_panel_hx8394(dbi_io, &panel_cfg, &panel_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_hx8394 failed: %s", esp_err_to_name(err));
        return NULL;
    }

    /* 8. Reset + init */
    err = panel_handle->reset(panel_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "panel reset failed: %s", esp_err_to_name(err));
        return NULL;
    }

    err = panel_handle->init(panel_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "panel init failed: %s", esp_err_to_name(err));
        return NULL;
    }

    ESP_LOGI(TAG, "HX8394 init complete!");
    return panel_handle;
}
