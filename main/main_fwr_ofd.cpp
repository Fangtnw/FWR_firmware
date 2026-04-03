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
#include "ofd_config.h"
#include "fwr_control.h"
#include "sbus_rx.h"
#include "camera.h"
#include "esp_log.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"

static const char *TAG = "FWR_OFD";

/* ---- IMU (WITmotion WT901B) ----
 * Dedicated I2C_NUM_1 on GPIO24/25 — fully isolated from camera SCCB on I2C_NUM_0 (GPIO7/8). */
#define IMU_SDA_PIN  GPIO_NUM_25
#define IMU_SCL_PIN  GPIO_NUM_24
#define IMU_I2C_PORT I2C_NUM_1
#define IMU_I2C_FREQ 400000
#define IMU_ADDR     0x50        // WT901B default (try 0x51 if no response)

#define ACCEL_SCALE  (16.0f   / 32768.0f)
#define GYRO_SCALE   (2000.0f / 32768.0f)
#define ANGLE_SCALE  (180.0f  / 32768.0f)

static i2c_master_bus_handle_t s_imu_bus = NULL;
static i2c_master_dev_handle_t s_imu_dev = NULL;

/* Read 2 bytes from a single WT901B register (little-endian int16).
 * Uses individual register reads — same approach as the standalone test
 * (WT901B may not support auto-increment burst reads). */
static bool imu_read_reg(uint8_t reg, int16_t *out)
{
    uint8_t buf[2] = {0};
    esp_err_t ret = i2c_master_transmit_receive(s_imu_dev, &reg, 1, buf, 2, pdMS_TO_TICKS(100));
    if (ret == ESP_OK) {
        *out = (int16_t)((buf[1] << 8) | buf[0]);
    }
    return ret == ESP_OK;
}

/* Read accel (0x34-0x36), gyro (0x37-0x39), angle (0x3D-0x3F) via 9 individual reads. */
static bool imu_read(imu_data_t *d)
{
    int16_t v = 0;
    if (!imu_read_reg(0x34, &v)) return false;
    d->ax = v * ACCEL_SCALE;
    if (!imu_read_reg(0x35, &v)) return false;
    d->ay = v * ACCEL_SCALE;
    if (!imu_read_reg(0x36, &v)) return false;
    d->az = v * ACCEL_SCALE;
    if (!imu_read_reg(0x37, &v)) return false;
    d->gx = v * GYRO_SCALE;
    if (!imu_read_reg(0x38, &v)) return false;
    d->gy = v * GYRO_SCALE;
    if (!imu_read_reg(0x39, &v)) return false;
    d->gz = v * GYRO_SCALE;
    if (!imu_read_reg(0x3D, &v)) return false;
    d->roll = v * ANGLE_SCALE;
    if (!imu_read_reg(0x3E, &v)) return false;
    d->pitch = v * ANGLE_SCALE;
    if (!imu_read_reg(0x3F, &v)) return false;
    d->yaw = v * ANGLE_SCALE;
    return true;
}

static void imu_init(void)
{
    // Dedicated I2C_NUM_1 on GPIO24(SCL)/GPIO25(SDA) — independent from camera SCCB.
    i2c_master_bus_config_t bus_cfg = {};
    bus_cfg.i2c_port          = IMU_I2C_PORT;
    bus_cfg.sda_io_num        = IMU_SDA_PIN;
    bus_cfg.scl_io_num        = IMU_SCL_PIN;
    bus_cfg.clk_source        = I2C_CLK_SRC_DEFAULT;
    bus_cfg.glitch_ignore_cnt = 7;
    bus_cfg.flags.enable_internal_pullup = true;

    if (i2c_new_master_bus(&bus_cfg, &s_imu_bus) != ESP_OK) {
        ESP_LOGE(TAG, "IMU: I2C_NUM_1 bus init failed");
        return;
    }

    // WT901B takes ~3s to boot. Register device once per address and keep it
    // registered while retrying reads — removing/re-adding on every NACK can
    // corrupt the bus state (unlike the standalone test which keeps it registered).
    ESP_LOGI(TAG, "Waiting for IMU to boot (up to 5s)...");
    esp_log_level_set("i2c.master", ESP_LOG_NONE);

    const uint8_t candidates[] = {0x50, 0x51};
    const int RETRY_INTERVAL_MS = 200;
    const int RETRY_MAX         = 5000 / RETRY_INTERVAL_MS;

    for (int a = 0; a < 2; a++) {
        i2c_device_config_t dev_cfg = {};
        dev_cfg.dev_addr_length = I2C_ADDR_BIT_LEN_7;
        dev_cfg.device_address  = candidates[a];
        dev_cfg.scl_speed_hz    = IMU_I2C_FREQ;

        if (i2c_master_bus_add_device(s_imu_bus, &dev_cfg, &s_imu_dev) != ESP_OK) {
            s_imu_dev = NULL;
            continue;
        }

        // Keep device registered; retry only the read until IMU is ready.
        // After each NACK the bus may be stuck in an error state — reset it
        // before the next attempt (mirrors what the I2C scan loop does implicitly).
        for (int i = 0; i < RETRY_MAX; i++) {
            imu_data_t test = {};
            if (imu_read(&test)) {
                esp_log_level_set("i2c.master", ESP_LOG_WARN);
                ESP_LOGI(TAG, "IMU OK at 0x%02X — Accel(g) X:%.3f Y:%.3f Z:%.3f",
                         candidates[a], (double)test.ax, (double)test.ay, (double)test.az);
                return;  // s_imu_dev stays registered for main loop
            }
            vTaskDelay(pdMS_TO_TICKS(RETRY_INTERVAL_MS));
        }

        // This address didn't respond — remove and try next
        i2c_master_bus_rm_device(s_imu_dev);
        s_imu_dev = NULL;
    }

    esp_log_level_set("i2c.master", ESP_LOG_WARN);
    ESP_LOGW(TAG, "IMU not found at 0x50 or 0x51 after 5s — IMU logging disabled");
}

/* ---- Camera ---- */
#define CAM_W  800
#define CAM_H  640

/* ---- OFD avoidance output scaling (thresholds/detection in ofd_task via ofd_config.h) ---- */
#define MAX_AVOID_US  600   // max servo command magnitude (µs)
#define USE_ROLL      1     // apply turn_cmd to aileron channel

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
    imu_init();         // WT901B on I2C_NUM_1 (GPIO25=SDA, GPIO24=SCL)

    camera_set_sensor(CAMERA_OV5647);
    camera_set_resolution(CAM_W, CAM_H);
    camera_init();      // I2C_NUM_0 on GPIO7/GPIO8 for SCCB — independent from IMU

    // IMU occasionally NACKs mid-register-update; we handle it in the loop.
    // Suppress driver-internal error spam — our own logic logs real failures.
    esp_log_level_set("i2c.master", ESP_LOG_NONE);

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

        /* ---- Read IMU and push to recorder ---- */
        if (s_imu_dev) {
            imu_data_t imu = {};
            if (imu_read(&imu)) {
                video_rec_set_imu(imu);
            } else {
                // After a failed read the bus may be stuck — reset it so the
                // next iteration starts clean.  The IMU occasionally NACKs
                // while updating its internal registers; this is transient.
                i2c_master_bus_reset(s_imu_bus);
            }
        }

        /* ---- Capture and forward to recorder ---- */
        camera_frame_t f = camera_get_frame();
        video_rec_enqueue(&f);   // memcpy to PSRAM ring buffer + camera_return_frame inside

        /* ---- OFD avoidance — dual-gate result from ofd_task (logged only) ---- */
        ofd_result_t r = video_rec_last_ofd();
        if (r.looming_detected) {
            int ail = USE_ROLL ? (int)(r.turn_cmd * MAX_AVOID_US) : 0;
            ESP_LOGW(TAG, "[AVOID] lvl=%d tau=%.1fms ema_div=%.4f turn=%.3f az_q=%d cmd=%d (disabled)",
                     r.evasion_level, (double)r.tau_ms,
                     (double)r.ema_div, (double)r.turn_cmd, (int)r.az_quiet, ail);
            // fwr_set_ofd_avoidance(ail, 0);   /* TODO: uncomment to enable avoidance */
        } else {
            // fwr_set_ofd_avoidance(0, 0);
        }

        /* Yield when idle — skip delay during recording to maximise frame rate */
        if (!is_recording()) vTaskDelay(pdMS_TO_TICKS(5));
    }
}
