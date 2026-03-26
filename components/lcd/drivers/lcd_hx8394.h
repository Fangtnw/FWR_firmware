#pragma once
#include "esp_lcd_panel_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_lcd_panel_handle_t lcd_hx8394_adapter_init(int dpi_clk_mhz);

#ifdef __cplusplus
}
#endif
