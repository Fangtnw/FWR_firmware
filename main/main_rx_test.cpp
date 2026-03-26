#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/uart.h"
#include "esp_log.h"

static const char *TAG = "SBUS_MAP_TEST";

/* ===== UART config ===== */
#define TEST_UART   UART_NUM_1
#define RX_PIN      20
#define TX_PIN      UART_PIN_NO_CHANGE

#define SBUS_BAUDRATE  100000
#define SBUS_FRAME_LEN 25
#define SBUS_START     0x0F

/* ===== SBUS decode ===== */
static void sbus_decode(const uint8_t *b, uint16_t *ch)
{
    ch[0] = (b[1]       | b[2]  << 8) & 0x07FF;
    ch[1] = (b[2] >> 3  | b[3]  << 5) & 0x07FF;
    ch[2] = (b[3] >> 6  | b[4]  << 2 | b[5] << 10) & 0x07FF;
    ch[3] = (b[5] >> 1  | b[6]  << 7) & 0x07FF;
    ch[4] = (b[6] >> 4  | b[7]  << 4) & 0x07FF;
    ch[5] = (b[7] >> 7  | b[8]  << 1 | b[9] << 9) & 0x07FF;
    ch[6] = (b[9] >> 2  | b[10] << 6) & 0x07FF;
    ch[7] = (b[10] >> 5 | b[11] << 3) & 0x07FF;
}

extern "C" void app_main(void)
{
    uart_config_t cfg = {};
    cfg.baud_rate = SBUS_BAUDRATE;
    cfg.data_bits = UART_DATA_8_BITS;
    cfg.parity    = UART_PARITY_EVEN;
    cfg.stop_bits = UART_STOP_BITS_2;
    cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;

    uart_param_config(TEST_UART, &cfg);
    uart_set_pin(
        TEST_UART,
        TX_PIN,
        RX_PIN,
        UART_PIN_NO_CHANGE,
        UART_PIN_NO_CHANGE
    );

    uart_driver_install(TEST_UART, 1024, 0, 0, NULL, 0);

    /* ✅ IMPORTANT: hardware SBUS inversion (Arduino equivalent of "true") */
    uart_set_line_inverse(TEST_UART, UART_SIGNAL_RXD_INV);

    ESP_LOGI(TAG, "SBUS mapping test on GPIO%d", RX_PIN);

    uint8_t rx[128];
    uint8_t frame[SBUS_FRAME_LEN];
    uint16_t ch[16];
    int idx = 0;
    bool syncing = false;

    while (1) {
        int len = uart_read_bytes(
            TEST_UART,
            rx,
            sizeof(rx),
            pdMS_TO_TICKS(100)
        );

        for (int i = 0; i < len; i++) {
            uint8_t b = rx[i];

            if (!syncing) {
                if (b == SBUS_START) {
                    syncing = true;
                    idx = 0;
                    frame[idx++] = b;
                }
            } else {
                frame[idx++] = b;
                if (idx == SBUS_FRAME_LEN) {
                    syncing = false;
                    sbus_decode(frame, ch);

                    printf(
                        "AIL:%4d ELE:%4d THR:%4d RUD:%4d | "
                        "SW1:%4d SW2:%4d SW3:%4d SW4:%4d\n",
                        ch[0], ch[1], ch[2], ch[3],
                        ch[4], ch[5], ch[6], ch[7]
                    );
                }
            }
        }
    }
}
