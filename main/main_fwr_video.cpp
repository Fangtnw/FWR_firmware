#include "fwr_control.h"
#include "sbus_rx.h"
#include "camera.h"
#include "lcd.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#include "driver/gpio.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include <stdio.h>
#include <string.h>
#include <cerrno>
#include <sys/stat.h>

static const char *TAG = "MAIN_FWR_VIDEO";

/* ---- shared SBUS variable ---- */
extern volatile int sw1_raw;

#define SW_LOW_MAX      600
#define SW_MID_MIN      800
#define SW_MID_MAX      1200
#define SW_HIGH_MIN     1400

/* ---- RAW video recording structures ---- */
typedef struct {
    uint32_t frame_count;
    uint32_t fps;
    uint32_t width;
    uint32_t height;
    uint32_t frame_size;  // bytes per frame
} raw_video_header_t;

static FILE *video_fp = nullptr;
static bool recording = false;
static int video_id = 0;
static sdmmc_card_t *card = nullptr;
static bool sd_mounted = false;

static uint32_t frame_count = 0;
static int64_t recording_start_time = 0;
static uint32_t first_frame_size = 0;

/* ---- Mount SD card ---- */
static void sdcard_init(void)
{
    const char *SDTAG = "SD";
    const char mount_point[] = "/sdcard";
    
    ESP_LOGI(SDTAG, "Initializing SD card (SDMMC + on-chip LDO)...");
    
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 3,
        .allocation_unit_size = 0,
        .disk_status_check_enable = false,
        .use_one_fat = false
    };

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;  // Faster for video
    host.io_voltage = 3.3f;

    sd_pwr_ctrl_ldo_config_t ldo_cfg = {
        .ldo_chan_id = 4
    };
    
    sd_pwr_ctrl_handle_t pwr_ctrl = NULL;
    esp_err_t ret = sd_pwr_ctrl_new_on_chip_ldo(&ldo_cfg, &pwr_ctrl);
    if (ret != ESP_OK) {
        ESP_LOGE(SDTAG, "Failed to create LDO power ctrl: %s", esp_err_to_name(ret));
        return;
    }
    
    ret = sd_pwr_ctrl_set_io_voltage(pwr_ctrl, 3300);
    if (ret != ESP_OK) {
        ESP_LOGE(SDTAG, "Failed to set SD IO voltage: %s", esp_err_to_name(ret));
        return;
    }
    
    host.pwr_ctrl_handle = pwr_ctrl;
    vTaskDelay(pdMS_TO_TICKS(20));

    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 4;
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
    slot_config.cd = SDMMC_SLOT_NO_CD;
    slot_config.wp = SDMMC_SLOT_NO_WP;
    
    slot_config.clk = GPIO_NUM_43;
    slot_config.cmd = GPIO_NUM_44;
    slot_config.d0  = GPIO_NUM_39;
    slot_config.d1  = GPIO_NUM_40;
    slot_config.d2  = GPIO_NUM_41;
    slot_config.d3  = GPIO_NUM_42;

    card = NULL;
    ESP_LOGI(SDTAG, "Mounting FAT filesystem...");
    
    ret = esp_vfs_fat_sdmmc_mount(
        mount_point,
        &host,
        &slot_config,
        &mount_config,
        &card
    );
    
    if (ret != ESP_OK) {
        ESP_LOGE(SDTAG, "SD mount failed: %s", esp_err_to_name(ret));
        sd_mounted = false;
        return;
    }
    
    ESP_LOGI(SDTAG, "SD mounted successfully");
    sdmmc_card_print_info(stdout, card);
    
    // Test write to verify SD is working
    ESP_LOGI(SDTAG, "Testing SD write access...");
    FILE *test_fp = fopen("/sdcard/test.txt", "w");
    if (test_fp) {
        fprintf(test_fp, "SD card test\n");
        fclose(test_fp);
        
        // Verify we can read it back
        test_fp = fopen("/sdcard/test.txt", "r");
        if (test_fp) {
            char buf[32];
            fgets(buf, sizeof(buf), test_fp);
            fclose(test_fp);
            remove("/sdcard/test.txt");
            ESP_LOGI(SDTAG, "SD read/write test successful");
            sd_mounted = true;
        } else {
            ESP_LOGE(SDTAG, "Test read FAILED (errno=%d)", errno);
            sd_mounted = false;
        }
    } else {
        ESP_LOGE(SDTAG, "Test write FAILED (errno=%d: %s)", errno, strerror(errno));
        sd_mounted = false;
    }
    
    if (sd_mounted) {
        ESP_LOGI(SDTAG, "SD card ready for recording");
    }
}

/* ---- Write RAW RGB565 frame ---- */
static void write_raw_frame(FILE *fp, const uint8_t *data, size_t length)
{
    // For RAW video, all frames must be same size
    if (frame_count == 0) {
        first_frame_size = length;
        ESP_LOGI(TAG, "First frame: %d bytes (800x640 RGB565 = %d bytes)", 
                 length, 800*640*2);
    } else if (length != first_frame_size) {
        ESP_LOGW(TAG, "Frame size mismatch: %d != %d", length, first_frame_size);
        return;  // Skip mismatched frames
    }
    
    size_t written = fwrite(data, 1, length, fp);
    if (written != length) {
        ESP_LOGE(TAG, "Frame write error (%d/%d)", written, length);
    }
    
    frame_count++;
    
    // Log progress every second
    if (frame_count % 30 == 0) {
        int64_t elapsed_us = esp_timer_get_time() - recording_start_time;
        float current_fps = (elapsed_us > 0) ? (frame_count * 1000000.0f / elapsed_us) : 0;
        ESP_LOGI(TAG, "Recording: %d frames, %.1f fps", frame_count, current_fps);
    }
}

/* ---- Start recording ---- */
static void start_recording(void)
{
    if (!sd_mounted) {
        ESP_LOGE(TAG, "SD not mounted, cannot record");
        return;
    }
    
    char fname[64];
    snprintf(fname, sizeof(fname), "/sdcard/V%04d.VID", video_id);
    
    ESP_LOGI(TAG, "Opening file: %s", fname);
    
    video_fp = fopen(fname, "wb");
    if (!video_fp) {
        ESP_LOGE(TAG, "Failed to open %s (errno=%d: %s)", fname, errno, strerror(errno));
        return;
    }
    
    // Write header (20 bytes: 5 x uint32_t)
    // Will update frame_count and fps on close
    raw_video_header_t header = {
        .frame_count = 0,
        .fps = 30,     // nominal
        .width = 800,  // OV5647 resolution
        .height = 640,
        .frame_size = 0  // will be set from first frame
    };
    fwrite(&header, sizeof(header), 1, video_fp);
    
    // Use 8KB buffer for better performance
    setvbuf(video_fp, NULL, _IOFBF, 8192);
    
    recording = true;
    frame_count = 0;
    first_frame_size = 0;
    recording_start_time = esp_timer_get_time();
    
    ESP_LOGI(TAG, "RECORD START: %s", fname);
}

/* ---- Stop recording ---- */
static void stop_recording(void)
{
    if (!recording || !video_fp) {
        return;
    }
    
    // Calculate actual FPS
    int64_t duration_us = esp_timer_get_time() - recording_start_time;
    float actual_fps = (duration_us > 0) ? (frame_count * 1000000.0f / duration_us) : 0;
    
    ESP_LOGI(TAG, "RECORD STOP: %d frames, %.1f fps, frame_size=%d bytes", 
             frame_count, actual_fps, first_frame_size);
    
    // Update header with actual info
    fseek(video_fp, 0, SEEK_SET);
    raw_video_header_t header = {
        .frame_count = frame_count,
        .fps = (uint32_t)(actual_fps + 0.5f),
        .width = 800,
        .height = 640,
        .frame_size = first_frame_size
    };
    fwrite(&header, sizeof(header), 1, video_fp);
    
    fflush(video_fp);
    fclose(video_fp);
    video_fp = nullptr;
    recording = false;
    video_id++;
    
    ESP_LOGI(TAG, "Video saved. Use Python script to convert to MP4.");
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Booting FWR manual flight + RAW video record (RGB565)");

    /* ---- Flight control ---- */
    fwr_control_init();
    fwr_control_start();

    /* ---- SBUS ---- */
    sbus_rx_init();
    sbus_rx_start();

    /* ---- SD ---- */
    sdcard_init();

    /* ---- Camera ---- */
    camera_set_sensor(CAMERA_OV5647);
    camera_init();
    
    ESP_LOGI(TAG, "Camera outputs RGB565 - will record RAW and convert to MP4 later");
    
    // Get first frame to determine actual size
    camera_frame_t test_frame = camera_get_frame();
    if (test_frame.data && test_frame.length > 0) {
        ESP_LOGI(TAG, "Frame size: %d bytes (expected for 800x640 RGB565: %d)", 
                 test_frame.length, 800*640*2);
    }
    camera_return_frame(&test_frame);

    int prev_sw_state = -1;

    while (1) {
        /* ---- classify SW1 ---- */
        int sw = sw1_raw;
        int sw_state;
        
        if (sw >= SW_HIGH_MIN) {
            sw_state = 2;  // UP
        } else if (sw <= SW_LOW_MAX) {
            sw_state = 0;  // DOWN
        } else {
            sw_state = 1;  // MID
        }

        /* ---- state transition ---- */
        if (sw_state != prev_sw_state) {
            ESP_LOGI(TAG, "SW1 state %d -> %d", prev_sw_state, sw_state);

            /* ---- START recording ---- */
            if (sw_state == 2 && !recording) {
                start_recording();
            }

            /* ---- STOP recording ---- */
            if (sw_state == 0 && recording) {
                stop_recording();
            }

            prev_sw_state = sw_state;
        }

        /* ---- camera frame ---- */
        camera_frame_t f = camera_get_frame();
        
        if (recording && video_fp && f.data && f.length > 0) {
            write_raw_frame(video_fp, f.data, f.length);
        }
        
        camera_return_frame(&f);
        
        // Only delay if NOT recording to maximize FPS
        if (!recording) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }
}