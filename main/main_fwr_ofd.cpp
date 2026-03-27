/**
 * main_fwr_ofd.cpp — FWR Manual Flight + OFD Logging + SD Video Recording
 *
 * - Manual RC flight via SBUS → fwr_control servos
 * - SD video recording (RAW RGB565) triggered by SW1 switch
 * - OFD (optical flow divergence) computed by the recording module and logged
 * - Avoidance is computed and logged; servo output is commented out
 *   → uncomment fwr_set_ofd_avoidance() to enable full obstacle avoidance
 */

#include "video_rec.h"
#include "fwr_control.h"
#include "sbus_rx.h"
#include "camera.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "FWR_OFD";

/* ---- Camera ---- */
#define CAM_W  800
#define CAM_H  640

/* ---- OFD avoidance tuning (logging only for now) ---- */
#define DIV_DANGER    0.05f
#define KP_AVOID      4000.0f
#define MAX_AVOID_US  600
#define USE_ROLL      1
#define USE_YAW       0

/* ---- SW1 switch thresholds ---- */
extern volatile int sw1_raw;
#define SW_LOW_MAX   600
#define SW_HIGH_MIN  1400

/* ======================================================== */

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "FWR Manual + OFD Log + SD Recording");

    fwr_control_init();
    fwr_control_start();
    sbus_rx_init();
    sbus_rx_start();

    video_rec_init();   // SD card, PSRAM ring buffer, OFD engine

    camera_set_sensor(CAMERA_OV5647);
    camera_set_resolution(CAM_W, CAM_H);
    camera_init();

    /* Probe first frame to confirm actual frame size */
    camera_frame_t probe = camera_get_frame();
    ESP_LOGI(TAG, "Camera ready — %d bytes/frame  (%dx%d RGB565 expected %d)",
             probe.length, CAM_W, CAM_H, CAM_W * CAM_H * 2);
    camera_return_frame(&probe);

    int prev_sw = -1;

    while (1) {

        /* ---- SW1: UP = record, DOWN = stop ---- */
        int sw       = sw1_raw;
        int sw_state = (sw >= SW_HIGH_MIN) ? 2 : (sw <= SW_LOW_MAX) ? 0 : 1;

        if (sw_state != prev_sw) {
            ESP_LOGI(TAG, "SW1 %d -> %d", prev_sw, sw_state);
            if (sw_state == 2 && !is_recording()) start_recording();
            if (sw_state == 0 &&  is_recording()) stop_recording();
            prev_sw = sw_state;
        }

        /* ---- Capture and forward to recorder ---- */
        camera_frame_t f = camera_get_frame();
        video_rec_enqueue(&f);   // memcpy to PSRAM ring buffer + camera_return_frame inside

        /* ---- OFD avoidance — logged only, servo output disabled ---- */
        ofd_result_t r = video_rec_last_ofd();
        if (r.valid && r.divergence > DIV_DANGER) {

            float avoid_f = KP_AVOID * r.divergence;
            if (avoid_f > MAX_AVOID_US) avoid_f = MAX_AVOID_US;
            int avoid_us  = (int)avoid_f;
            int direction = (r.lr_balance >= 0.0f) ? -1 : +1;
            int ail = USE_ROLL ? direction * avoid_us : 0;
            int rud = USE_YAW  ? direction * avoid_us : 0;
            (void)rud;

            ESP_LOGW(TAG, "[AVOID] div=%.4f  lr=%.4f  cmd=%d  (output disabled)",
                     (double)r.divergence, (double)r.lr_balance, ail);
            // fwr_set_ofd_avoidance(ail, rud);   /* TODO: uncomment to enable avoidance */

        } else {
            // fwr_set_ofd_avoidance(0, 0);
        }

        /* Yield when idle — skip delay during recording to maximise frame rate */
        if (!is_recording()) vTaskDelay(pdMS_TO_TICKS(5));
    }
}
