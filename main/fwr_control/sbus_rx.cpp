#include "sbus_rx.h"

#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include <stdint.h>

/* ===================== UART config ===================== */

#define SBUS_UART      UART_NUM_1
#define SBUS_RX_PIN    20          // <<< CONFIRMED RX PIN
#define SBUS_TX_PIN    18

#define SBUS_BAUDRATE  100000
#define SBUS_FRAME_LEN 25

/* ===================== SBUS constants ===================== */

#define SBUS_START_BYTE 0x0F
#define SBUS_END_BYTE   0x00

#define PROPO_MIN     352
#define PROPO_NEUTRAL 1024
#define PROPO_MAX     1696
volatile int sw1_raw = PROPO_NEUTRAL;


/* ===================== External control variables ===================== */
/* Defined in fwr_control.cpp */
extern volatile int aileron, elevator, throttle, rudder;
extern volatile int frequency;
extern volatile int servo_adj_l, servo_adj_r;
extern volatile int flag_square_wave;

/* ===================== Utility ===================== */

static inline int clamp(int x, int min, int max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

static inline int map_int(int x, int in_min, int in_max,
                          int out_min, int out_max)
{
    return (x - in_min) * (out_max - out_min) /
           (in_max - in_min) + out_min;
}

/* ===================== SBUS decode ===================== */

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

    sw1_raw = ch[4];
}

/* ===================== SBUS task ===================== */

static void sbus_task(void *arg)
{
    uint8_t rx_buf[128];
    uint8_t frame[SBUS_FRAME_LEN];
    uint16_t ch[16];

    int idx = 0;
    bool syncing = false;

    int64_t last_frame_time = 0;

    while (1) {

        int len = uart_read_bytes(
            SBUS_UART, rx_buf, sizeof(rx_buf), pdMS_TO_TICKS(20)
        );

        if (len > 0) {
            for (int i = 0; i < len; i++) {
                uint8_t b = rx_buf[i];

                if (!syncing) {
                    if (b == SBUS_START_BYTE) {
                        syncing = true;
                        idx = 0;
                        frame[idx++] = b;
                    }
                }
                else {
                    frame[idx++] = b;

                    if (idx == SBUS_FRAME_LEN) {
                        syncing = false;

                        /* Validate end byte */
                        if (frame[24] != SBUS_END_BYTE) {
                            continue;
                        }

                        sbus_decode(frame, ch);
                        last_frame_time = esp_timer_get_time();

                        /* ---- Channel mapping (Arduino-equivalent) ---- */

                        aileron = map_int(
                            clamp(ch[0], PROPO_MIN, PROPO_MAX),
                            PROPO_MIN, PROPO_MAX,
                            -300, 300
                        );

                        elevator = map_int(
                            clamp(ch[1], PROPO_MIN, PROPO_MAX),
                            PROPO_MIN, PROPO_MAX,
                            -400, 400
                        );

                        throttle = map_int(
                            clamp(ch[2], PROPO_MIN + 100, PROPO_MAX),
                            PROPO_MIN + 100, PROPO_MAX,
                            0, 600
                        );

                        rudder = map_int(
                            clamp(ch[3], PROPO_MIN, PROPO_MAX),
                            PROPO_MIN, PROPO_MAX,
                            -300, 300
                        );
                        // flag_square_wave = (ch[4] > PROPO_NEUTRAL);  // SW1 also controls recording; decouple if needed

                        frequency = map_int(
                            clamp(ch[5], PROPO_MIN, PROPO_MAX),
                            PROPO_MIN, PROPO_MAX,
                            1000, 0
                        );

                        servo_adj_l = map_int(
                            clamp(ch[6], PROPO_MIN, PROPO_MAX),
                            PROPO_MIN, PROPO_MAX,
                            -100, 100
                        );

                        servo_adj_r = map_int(
                            clamp(ch[7], PROPO_MIN, PROPO_MAX),
                            PROPO_MIN, PROPO_MAX,
                            -100, 100
                        );
                    }
                }
            }
        }

        /* ---- FAILSAFE: SBUS timeout ---- */
        if (esp_timer_get_time() - last_frame_time > 100000) { // 100 ms
            throttle = 0;
            frequency = 0;
        }
    }
}

/* ===================== Public API ===================== */

void sbus_rx_init(void)
{
    uart_config_t cfg = {};
    cfg.baud_rate = SBUS_BAUDRATE;
    cfg.data_bits = UART_DATA_8_BITS;
    cfg.parity    = UART_PARITY_EVEN;
    cfg.stop_bits = UART_STOP_BITS_2;
    cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;

    uart_param_config(SBUS_UART, &cfg);
    uart_set_pin(
        SBUS_UART,
        SBUS_TX_PIN,
        SBUS_RX_PIN,
        UART_PIN_NO_CHANGE,
        UART_PIN_NO_CHANGE
    );

    uart_driver_install(SBUS_UART, 256, 0, 0, NULL, 0);

    /* ✅ REQUIRED: hardware inversion (same as Arduino sbus_rx(..., true)) */
    uart_set_line_inverse(SBUS_UART, UART_SIGNAL_RXD_INV);
}

void sbus_rx_start(void)
{
    xTaskCreatePinnedToCore(
        sbus_task,
        "sbus_rx",
        4096,
        NULL,
        10,
        NULL,
        1
    );
}
