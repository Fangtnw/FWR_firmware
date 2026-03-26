/**
 * main_fwr_ofd.cpp - FWR + OFD Smooth Obstacle Avoidance
 *
 * - Continuous proportional control
 * - Telemetry (div, lr)
 * - Red log on avoid
 */

#include "fwr_control.h"
#include "sbus_rx.h"
#include "camera.h"
#include "ofd.h"

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <math.h>

static const char *TAG = "FWR_OFD";

/* ---------------- Camera ---------------- */
#define CAM_W  800
#define CAM_H  640
#define OFD_W  160
#define OFD_H  120

/* ------------- OFD tuning --------------- */
#define DIV_DANGER      0.05f
#define KP_AVOID        4000.0f
#define MAX_AVOID_US    600
#define MIN_AVOID_US    0
#define USE_YAW         0
#define USE_ROLL        1

/* ------------- Debug -------------------- */
#define OFD_DEBUG 1
#define TELEMETRY_PERIOD 10

#define ANSI_RED   "\033[31m"
#define ANSI_RESET "\033[0m"

/* ---------------------------------------- */

__attribute__((section(".ext_ram.bss"), aligned(64)))
static uint8_t ofd_gray[OFD_W * OFD_H];

/* Downsample RGB565 -> Grayscale */
static inline void downsample_rgb565_to_gray(
    const uint16_t *src, int src_w, int src_h,
    uint8_t *dst, int dst_w, int dst_h)
{
    for (int y = 0; y < dst_h; ++y) {
        int sy = (y * src_h) / dst_h;
        const uint16_t *srow = src + sy * src_w;
        for (int x = 0; x < dst_w; ++x) {
            int sx = (x * src_w) / dst_w;
            uint16_t p = srow[sx];
            uint8_t r = (uint8_t)(((p >> 11) & 0x1F) * 255 / 31);
            uint8_t g = (uint8_t)(((p >>  5) & 0x3F) * 255 / 63);
            uint8_t b = (uint8_t)((p & 0x1F) * 255 / 31);
            dst[y * dst_w + x] =
                (uint8_t)((77 * r + 150 * g + 29 * b) >> 8);
        }
    }
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "FWR + OFD Smooth Avoidance (DEBUG)");

    fwr_control_init();
    fwr_control_start();
    sbus_rx_init();
    sbus_rx_start();

    camera_set_sensor(CAMERA_OV5647);
    camera_init();
    ESP_LOGI(TAG, "Camera initialized (%dx%d)", CAM_W, CAM_H);

    ofd_init(OFD_W, OFD_H);
    ESP_LOGI(TAG, "OFD initialized (%dx%d), DIV_DANGER=%.2f",
             OFD_W, OFD_H, (double)DIV_DANGER);

    int telem_count = 0;
    int avoid_count = 0;
    float avoid_lp = 0;   // low-pass filtered command

    while (1) {
        camera_frame_t f = camera_get_frame();
        if (!f.data || f.length == 0) {
            camera_return_frame(&f);
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        /* Downsample to grayscale for OFD */
        downsample_rgb565_to_gray(
            (const uint16_t *)f.data, CAM_W, CAM_H,
            ofd_gray, OFD_W, OFD_H
        );

        ofd_result_t r = ofd_process_gray(ofd_gray);

#if OFD_DEBUG
        if ((telem_count++ % TELEMETRY_PERIOD) == 0 && r.valid) {
            ESP_LOGI("OFD_TELEM",
                     "div=%.4f  lr=%.4f  avoid=%.1f",
                     (double)r.divergence,
                     (double)r.lr_balance,
                     (double)avoid_lp);
        }
#endif

        if (r.valid && r.divergence > DIV_DANGER) {

            float avoid_f =
                KP_AVOID * r.divergence;

            /* Clamp */
            if (avoid_f >  MAX_AVOID_US) avoid_f =  MAX_AVOID_US;
            if (avoid_f < -MAX_AVOID_US) avoid_f = -MAX_AVOID_US;

            int avoid_us = (int)avoid_f;

            int ail = USE_ROLL ? avoid_us : 0;
            int rud = USE_YAW  ? avoid_us : 0;

            fwr_set_ofd_avoidance(ail, rud);

            if ((avoid_count++ % 5) == 0) {
                printf(ANSI_RED
                       "[AVOID] div=%.4f lr=%.4f -> cmd=%d\n"
                       ANSI_RESET,
                       (double)r.divergence,
                       (double)r.lr_balance,
                       avoid_us);
                printf("raw avoid_f = %.2f\n", avoid_f);

            }

        } else {
            avoid_lp = 0;
            fwr_set_ofd_avoidance(0, 0);
        }

        camera_return_frame(&f);
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}
