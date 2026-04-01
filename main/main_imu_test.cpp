/**
 * @file main_imu_test.cpp
 * @brief WITmotion WT901B IMU I2C read test for ESP32-P4 Pico
 *
 * Wiring:
 *   WT901B VCC  →  5V (board 5V pin)
 *   WT901B GND  →  GND
 *   WT901B SDA  →  GPIO7  (SDA labeled pin)
 *   WT901B SCL  →  GPIO8  (SCL labeled pin)
 *
 * ⚠️  IMPORTANT - Logic Level Warning:
 *   WT901B powered at 5V will drive I2C lines at 5V.
 *   ESP32-P4 GPIO is 3.3V tolerant ONLY.
 *   If you see garbage data or the chip gets warm, move VCC to the 3V3 pin instead.
 *
 * I2C port : I2C_NUM_1  (I2C_NUM_0 is reserved for camera SCCB)
 * WT901B default address: 0x50
 */

#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "esp_err.h"

static const char *TAG = "IMU_TEST";

// --- Pin & bus config ---
#define IMU_SDA_PIN  GPIO_NUM_25
#define IMU_SCL_PIN  GPIO_NUM_24
#define IMU_I2C_PORT I2C_NUM_1      // dedicated bus, isolated from camera SCCB on NUM_0
#define IMU_I2C_FREQ 400000
#define IMU_ADDR     0x50           // WT901B default (try 0x51 if scan fails)

// --- WT901B registers (each = 16-bit signed, little-endian) ---
#define REG_ACCEL_X     0x34
#define REG_ACCEL_Y     0x35
#define REG_ACCEL_Z     0x36
#define REG_GYRO_X      0x37
#define REG_GYRO_Y      0x38
#define REG_GYRO_Z      0x39
#define REG_ANGLE_ROLL  0x3D
#define REG_ANGLE_PITCH 0x3E
#define REG_ANGLE_YAW   0x3F

// --- Scale factors ---
#define ACCEL_SCALE  (16.0f  / 32768.0f)   // → g
#define GYRO_SCALE   (2000.0f / 32768.0f)  // → °/s
#define ANGLE_SCALE  (180.0f / 32768.0f)   // → °

static i2c_master_bus_handle_t s_bus = NULL;
static i2c_master_dev_handle_t s_dev = NULL;

static esp_err_t read_reg(uint8_t reg, int16_t *out)
{
    uint8_t buf[2] = {0};
    esp_err_t ret = i2c_master_transmit_receive(s_dev, &reg, 1, buf, 2, pdMS_TO_TICKS(100));
    if (ret == ESP_OK) {
        *out = (int16_t)((buf[1] << 8) | buf[0]);
    }
    return ret;
}

static void i2c_scan(void)
{
    ESP_LOGI(TAG, "--- I2C Scan ---");
    bool found = false;
    for (uint8_t addr = 0x08; addr <= 0x77; addr++) {
        i2c_device_config_t cfg = {};
        cfg.dev_addr_length = I2C_ADDR_BIT_LEN_7;
        cfg.device_address  = addr;
        cfg.scl_speed_hz    = 100000;
        i2c_master_dev_handle_t h = NULL;
        if (i2c_master_bus_add_device(s_bus, &cfg, &h) == ESP_OK) {
            uint8_t dummy, reg = 0x00;
            if (i2c_master_transmit_receive(h, &reg, 1, &dummy, 1, pdMS_TO_TICKS(20)) == ESP_OK) {
                ESP_LOGI(TAG, "  Device at 0x%02X", addr);
                found = true;
            }
            i2c_master_bus_rm_device(h);
        }
    }
    if (!found) ESP_LOGW(TAG, "  No devices found — check wiring!");
    ESP_LOGI(TAG, "----------------");
}

static void imu_task(void *)
{
    // Init bus
    i2c_master_bus_config_t bus_cfg = {};
    bus_cfg.i2c_port          = IMU_I2C_PORT;
    bus_cfg.sda_io_num        = IMU_SDA_PIN;
    bus_cfg.scl_io_num        = IMU_SCL_PIN;
    bus_cfg.clk_source        = I2C_CLK_SRC_DEFAULT;
    bus_cfg.glitch_ignore_cnt = 7;
    bus_cfg.flags.enable_internal_pullup = true;
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &s_bus));
    ESP_LOGI(TAG, "I2C bus ready  SDA=GPIO%d  SCL=GPIO%d", IMU_SDA_PIN, IMU_SCL_PIN);

    i2c_scan();

    // Add WT901B
    i2c_device_config_t dev_cfg = {};
    dev_cfg.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    dev_cfg.device_address  = IMU_ADDR;
    dev_cfg.scl_speed_hz    = IMU_I2C_FREQ;
    ESP_ERROR_CHECK(i2c_master_bus_add_device(s_bus, &dev_cfg, &s_dev));
    ESP_LOGI(TAG, "WT901B at 0x%02X — reading every 100 ms\n", IMU_ADDR);

    while (1) {
        int16_t ax = 0, ay = 0, az = 0, wx = 0, wy = 0, wz = 0, roll = 0, pitch = 0, yaw = 0;
        bool ok = true;
        ok &= read_reg(REG_ACCEL_X,     &ax)    == ESP_OK;
        ok &= read_reg(REG_ACCEL_Y,     &ay)    == ESP_OK;
        ok &= read_reg(REG_ACCEL_Z,     &az)    == ESP_OK;
        ok &= read_reg(REG_GYRO_X,      &wx)    == ESP_OK;
        ok &= read_reg(REG_GYRO_Y,      &wy)    == ESP_OK;
        ok &= read_reg(REG_GYRO_Z,      &wz)    == ESP_OK;
        ok &= read_reg(REG_ANGLE_ROLL,  &roll)  == ESP_OK;
        ok &= read_reg(REG_ANGLE_PITCH, &pitch) == ESP_OK;
        ok &= read_reg(REG_ANGLE_YAW,   &yaw)   == ESP_OK;

        if (!ok) {
            ESP_LOGE(TAG, "Read failed — check wiring or try IMU_ADDR 0x51");
        } else {
            ESP_LOGI(TAG, "Accel (g)    X:%7.3f  Y:%7.3f  Z:%7.3f", ax*ACCEL_SCALE, ay*ACCEL_SCALE, az*ACCEL_SCALE);
            ESP_LOGI(TAG, "Gyro  (d/s)  X:%7.2f  Y:%7.2f  Z:%7.2f", wx*GYRO_SCALE,  wy*GYRO_SCALE,  wz*GYRO_SCALE);
            ESP_LOGI(TAG, "Angle (deg)  Roll:%7.2f  Pitch:%7.2f  Yaw:%7.2f\n", roll*ANGLE_SCALE, pitch*ANGLE_SCALE, yaw*ANGLE_SCALE);
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "=== WT901B IMU Test ===");
    xTaskCreatePinnedToCore(imu_task, "imu_task", 4096, NULL, 5, NULL, 0);
}
