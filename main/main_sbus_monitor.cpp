/* main_sbus_monitor.cpp — SBUS live monitor (Futaba T12K, 12ch)
 * Range: 352 (min) 1024 (mid) 1696 (max)  FL=frame_lost  FS=failsafe */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"

#define SBUS_UART      UART_NUM_1
#define SBUS_RX_PIN    20
#define SBUS_BAUDRATE  100000
#define SBUS_FRAME_LEN 25
#define SBUS_START     0x0F
#define SBUS_END       0x00
#define PRINT_MS       50       // print interval (SBUS frame = ~14 ms)

static void sbus_decode(const uint8_t *b, uint16_t *ch, uint8_t *fl, uint8_t *fs)
{
    ch[0]  = ( b[1]        | b[2]  << 8)               & 0x07FF;
    ch[1]  = ( b[2]  >> 3  | b[3]  << 5)               & 0x07FF;
    ch[2]  = ( b[3]  >> 6  | b[4]  << 2  | b[5] << 10) & 0x07FF;
    ch[3]  = ( b[5]  >> 1  | b[6]  << 7)               & 0x07FF;
    ch[4]  = ( b[6]  >> 4  | b[7]  << 4)               & 0x07FF;
    ch[5]  = ( b[7]  >> 7  | b[8]  << 1  | b[9] << 9)  & 0x07FF;
    ch[6]  = ( b[9]  >> 2  | b[10] << 6)               & 0x07FF;
    ch[7]  = ( b[10] >> 5  | b[11] << 3)               & 0x07FF;
    ch[8]  = ( b[11] >> 6  | b[12] << 2  | b[13] << 10)& 0x07FF;
    ch[9]  = ( b[13] >> 1  | b[14] << 7)               & 0x07FF;
    ch[10] = ( b[14] >> 4  | b[15] << 4)               & 0x07FF;
    ch[11] = ( b[15] >> 7  | b[16] << 1  | b[17] << 9) & 0x07FF;
    *fl = (b[23] >> 2) & 1;
    *fs = (b[23] >> 3) & 1;
}

extern "C" void app_main(void)
{
    uart_config_t cfg = {};
    cfg.baud_rate = SBUS_BAUDRATE;
    cfg.data_bits = UART_DATA_8_BITS;
    cfg.parity    = UART_PARITY_EVEN;
    cfg.stop_bits = UART_STOP_BITS_2;
    cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    uart_param_config(SBUS_UART, &cfg);
    uart_set_pin(SBUS_UART, UART_PIN_NO_CHANGE, SBUS_RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(SBUS_UART, 1024, 0, 0, NULL, 0);
    uart_set_line_inverse(SBUS_UART, UART_SIGNAL_RXD_INV);

    ESP_LOGI("SBUS", "monitor GPIO%d  12ch", SBUS_RX_PIN);
    printf("  C1   C2   C3   C4   C5   C6   C7   C8   C9  C10  C11  C12  FL FS\n");

    uint8_t rx[128], frame[SBUS_FRAME_LEN];
    uint16_t ch[12];
    uint8_t fl, fs;
    int idx = 0;
    bool syncing = false;
    int64_t last_ms = 0;

    while (1) {
        int len = uart_read_bytes(SBUS_UART, rx, sizeof(rx), pdMS_TO_TICKS(20));

        for (int i = 0; i < len; i++) {
            if (!syncing) {
                if (rx[i] == SBUS_START) { syncing = true; idx = 0; frame[idx++] = rx[i]; }
            } else {
                frame[idx++] = rx[i];
                if (idx == SBUS_FRAME_LEN) {
                    syncing = false;
                    if (frame[24] != SBUS_END) continue;
                    sbus_decode(frame, ch, &fl, &fs);

                    int64_t now = esp_timer_get_time() / 1000;
                    if (now - last_ms >= PRINT_MS) {
                        printf("%4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d   %d  %d\r\n",
                               ch[0],ch[1],ch[2],ch[3],ch[4],ch[5],
                               ch[6],ch[7],ch[8],ch[9],ch[10],ch[11], fl, fs);
                        last_ms = now;
                    }
                }
            }
        }
    }
}
