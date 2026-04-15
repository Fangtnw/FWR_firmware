#include "video_rec.h"
#include "fwr_control.h"
#include "sbus_rx.h"
#include "camera.h"
#include "example_video_common.h"
#include "linux/videodev2.h"
#include "ofd.h"
#include "ofd_config.h"
#include <math.h>   // fabsf, fminf, fmaxf
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#include "driver/gpio.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "esp_heap_caps.h"
#include <stdio.h>
#include <string.h>
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>
#include "ff.h"   // FatFS for f_getfree() cluster-size detection

#ifndef v4l2_fourcc
#define v4l2_fourcc(a,b,c,d) ((uint32_t)(a) | ((uint32_t)(b)<<8) | ((uint32_t)(c)<<16) | ((uint32_t)(d)<<24))
#endif

static const char *TAG = "VID_REC";

/* ---- Camera & recording resolution ---- */
#define CAM_W  1280
#define CAM_H  960
#define REC_W  CAM_W
#define REC_H  CAM_H

/* ---- OFD sidecar ---- */
#define OFD_W  160
#define OFD_H  120

/* ---- Recording performance tunables ---- */
#define OFD_EVERY_N      1              // Run OFD every Nth written frame
#define FRAME_RING_SLOTS 3              // Metadata slots matching camera MMAP buffers
#define OFD_WORK_SLOTS   3              // Stable grayscale snapshots for background OFD


/* ---- Frame ring buffer ---- */
typedef struct {
    uint8_t *data;           // camera MMAP buffer
    size_t   length;
    int      cam_index;
    int64_t  capture_ts_ms;  // timestamp relative to recording_start_time
} frame_slot_t;

static frame_slot_t  s_slots[FRAME_RING_SLOTS];
static QueueHandle_t s_write_queue = nullptr;  // filled slots ready to write
static QueueHandle_t s_free_queue  = nullptr;  // available (empty) slots
static TaskHandle_t  s_writer_task = nullptr;
static uint32_t      s_writer_frame_count = 0; // owned exclusively by writer task

static void flush_csv_buffer(void);
static ofd_result_t load_last_ofd(void);
static void store_last_ofd(const ofd_result_t &r);

/* ---- OFD task: decoupled from SD write path ---- */
typedef struct {
    uint32_t frame_no;
    imu_data_t imu;
    uint8_t gray[OFD_W * OFD_H];
} ofd_work_item_t;

// Small pool of stable grayscale snapshots owned by the OFD task pipeline.
static ofd_work_item_t s_ofd_work_slots[OFD_WORK_SLOTS];
static QueueHandle_t s_ofd_queue   = nullptr;  // queued ofd_work_item_t*
static QueueHandle_t s_ofd_free_queue = nullptr; // available ofd_work_item_t*
static ofd_result_t  s_last_ofd    = {};       // latest OFD result
static TaskHandle_t  s_ofd_task    = nullptr;
static portMUX_TYPE  s_ofd_lock    = portMUX_INITIALIZER_UNLOCKED;

static imu_data_t    s_last_imu    = {};       // latest IMU sample (set from main loop)

/* ---- SD write buffer (DMA-capable internal SRAM for direct SDMMC DMA) ---- */
static uint8_t     *s_write_buf   = nullptr;
static const size_t WRITE_BUF_SIZE = 64 * 1024;  // 64KB; one SD multi-block transfer; fits in internal SRAM

/* ---- JPEG encoder ---- */
#define JPEG_QUALITY      85
#define JPEG_OUT_BUF_SIZE (1024 * 1024)       // generous for 1280x960 RGB565 JPEG

typedef struct {
    char     magic[8];     // "MJPGSEQ\0"
    uint32_t version;      // file format version
    uint32_t width;
    uint32_t height;
    uint32_t nominal_fps;
    uint32_t quality;
    uint32_t frame_count;
    uint32_t reserved[120];
} mjpg_file_header_t;

static example_encoder_handle_t s_jpeg_enc = nullptr;
static uint8_t                 *s_jpeg_out_buf = nullptr;
static uint32_t                 s_jpeg_out_size = 0;
static size_t                   s_media_bytes_written = 0;

/* ---- SPSC ring buffer for continuous SD streaming ---- */
// Writer task appends encoded data; SD task drains 64KB chunks continuously.
// No idle gaps → SD card SLC cache stays warm → sustained 10 MB/s.
#define RING_BUF_SIZE  (1 * 1024 * 1024)   // 1MB PSRAM ring buffer (8Mbps=1MB/s, SD drains 10x faster)
#define RING_BUF_MASK  (RING_BUF_SIZE - 1) // power-of-2 for fast modulo

static uint8_t *s_ring_buf = nullptr;
static volatile size_t s_ring_wr = 0;  // writer position (writer task only writes)
static volatile size_t s_ring_rd = 0;  // reader position (SD task only writes)

// CSV: simple buffer, flushed by SD task periodically
#define CSV_BUF_SIZE  (128 * 1024)
#define CSV_FLUSH_THRESHOLD (4 * 1024)
static char    *s_csv_buf   = nullptr;
static volatile size_t s_csv_len = 0;

/* SD streaming task */
static TaskHandle_t  s_sd_task  = nullptr;
static volatile bool s_sd_stop  = false;

static FILE    *csv_fp   = nullptr;
static bool     s_raw_dump_pending = false;
static char     s_raw_dump_path[64] = {0};
static char     s_plane_dump_base[64] = {0};


/* Repack I420 planar → O_UYY_E_VYY packed (same resolution, no spatial scaling).
 *
 * I420 layout: [Y: w*h] [U: w/2*h/2] [V: w/2*h/2]
 * O_UYY_E_VYY: row pairs — even row: [U Y0 Y1], odd row: [V Y2 Y3]
 *
 * All accesses are sequential → ~5ms at 1280x720 vs 70ms for the old
 * scattered-read downsample from 1920x1080. */
static void i420_repack_to_yuv420(const uint8_t *src, uint8_t *dst,
                                  int w, int h, int src_stride)
{
    if (src_stride <= 0) {
        src_stride = w;
    }

    const uint8_t *Yp = src;
    const size_t y_plane_size = (size_t)src_stride * h;
    const size_t uv_stride = (size_t)src_stride / 2;
    const uint8_t *Up = src + y_plane_size;
    const uint8_t *Vp = Up  + uv_stride * (h / 2);
    const size_t dst_row  = (size_t)w * 3 / 2;

    for (int y = 0; y < h; y += 2) {
        const uint8_t *Yrow0 = Yp + (size_t)y * src_stride;
        const uint8_t *Yrow1 = Yp + (size_t)(y + 1) * src_stride;
        const uint8_t *Urow  = Up + (size_t)(y / 2) * uv_stride;
        const uint8_t *Vrow  = Vp + (size_t)(y / 2) * uv_stride;
        uint8_t *d0 = dst + (size_t)y * dst_row;
        uint8_t *d1 = dst + (size_t)(y + 1) * dst_row;

        for (int x = 0; x < w; x += 2) {
            int t = x >> 1;
            d0[t*3]   = Urow[t];       // U
            d0[t*3+1] = Yrow0[x];      // Y00
            d0[t*3+2] = Yrow0[x + 1];  // Y01
            d1[t*3]   = Vrow[t];       // V
            d1[t*3+1] = Yrow1[x];      // Y10
            d1[t*3+2] = Yrow1[x + 1];  // Y11
        }
    }
}

static void dump_raw_i420_once(const uint8_t *src, size_t len, int w, int h, int stride)
{
    if (!s_raw_dump_pending || !src || len == 0 || s_raw_dump_path[0] == '\0') {
        return;
    }

    FILE *fp = fopen(s_raw_dump_path, "wb");
    if (!fp) {
        ESP_LOGW(TAG, "Raw I420 dump open failed: %s (errno=%d)", s_raw_dump_path, errno);
        s_raw_dump_pending = false;
        return;
    }

    size_t written = fwrite(src, 1, len, fp);
    fclose(fp);

    if (written != len) {
        ESP_LOGW(TAG, "Raw I420 dump short write: %u/%u bytes to %s",
                 (unsigned)written, (unsigned)len, s_raw_dump_path);
    } else {
        ESP_LOGI(TAG, "Raw camera frame dumped: %s (%ux%u stride=%d len=%u)",
                 s_raw_dump_path, (unsigned)w, (unsigned)h, stride, (unsigned)len);
        ESP_LOGI(TAG, "Inspect with: ffmpeg -f rawvideo -pixel_format yuv420p -video_size %dx%d -i %s frame.png",
                 w, h, s_raw_dump_path);
    }

    s_raw_dump_pending = false;
}

static void dump_raw_rgb565_once(const uint8_t *src, size_t len, int w, int h, int stride)
{
    if (!s_raw_dump_pending || !src || len == 0 || s_raw_dump_path[0] == '\0') {
        return;
    }

    FILE *fp = fopen(s_raw_dump_path, "wb");
    if (!fp) {
        ESP_LOGW(TAG, "Raw RGB565 dump open failed: %s (errno=%d)", s_raw_dump_path, errno);
        s_raw_dump_pending = false;
        return;
    }

    size_t written = fwrite(src, 1, len, fp);
    fclose(fp);

    if (written != len) {
        ESP_LOGW(TAG, "Raw RGB565 dump short write: %u/%u bytes to %s",
                 (unsigned)written, (unsigned)len, s_raw_dump_path);
    } else {
        ESP_LOGI(TAG, "Raw RGB565 frame dumped: %s (%ux%u stride=%d len=%u)",
                 s_raw_dump_path, (unsigned)w, (unsigned)h, stride, (unsigned)len);
        ESP_LOGI(TAG, "Inspect with: ffmpeg -f rawvideo -pixel_format rgb565le -video_size %dx%d -i %s frame.png",
                 w, h, s_raw_dump_path);
    }

    s_raw_dump_pending = false;
}

static void dump_plane_pgm_once(const char *path, const uint8_t *plane,
                                int w, int h, size_t stride, const char *label)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        ESP_LOGW(TAG, "%s dump open failed: %s (errno=%d)", label, path, errno);
        return;
    }

    fprintf(fp, "P5\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; ++y) {
        size_t wrote = fwrite(plane + (size_t)y * stride, 1, (size_t)w, fp);
        if (wrote != (size_t)w) {
            ESP_LOGW(TAG, "%s dump short write on row %d: %u/%u", label, y,
                     (unsigned)wrote, (unsigned)w);
            break;
        }
    }
    fclose(fp);
    ESP_LOGI(TAG, "%s plane dumped: %s (%dx%d stride=%u)", label, path, w, h, (unsigned)stride);
}

static void dump_i420_planes_once(const uint8_t *src, int w, int h, int stride)
{
    if (!src || s_plane_dump_base[0] == '\0') {
        return;
    }

    const size_t y_stride = (size_t)stride;
    const size_t uv_stride = (size_t)stride / 2;
    const uint8_t *y_plane = src;
    const uint8_t *u_plane = y_plane + y_stride * h;
    const uint8_t *v_plane = u_plane + uv_stride * (h / 2);

    char path[80];
    snprintf(path, sizeof(path), "%s_Y.PGM", s_plane_dump_base);
    dump_plane_pgm_once(path, y_plane, w, h, y_stride, "Y");
    snprintf(path, sizeof(path), "%s_U.PGM", s_plane_dump_base);
    dump_plane_pgm_once(path, u_plane, w / 2, h / 2, uv_stride, "U");
    snprintf(path, sizeof(path), "%s_V.PGM", s_plane_dump_base);
    dump_plane_pgm_once(path, v_plane, w / 2, h / 2, uv_stride, "V");
    ESP_LOGI(TAG, "Plane inspect tip: Y should look like a normal grayscale image; U/V should be smooth low-res maps, not dense checkerboard.");
}

/* Extract grayscale from RGB565 for OFD.
 * Green gets the highest weight, which is good enough for flow/debug without a full RGB conversion. */
static void extract_luma_from_rgb565(const uint8_t *rgb565, int src_w, int src_h,
                                     uint8_t *gray, int gray_w, int gray_h)
{
    for (int gy = 0; gy < gray_h; gy++) {
        int sy = gy * src_h / gray_h;
        for (int gx = 0; gx < gray_w; gx++) {
            int sx = gx * src_w / gray_w;
            size_t idx = ((size_t)sy * src_w + sx) * 2;
            uint16_t px = (uint16_t)rgb565[idx] | ((uint16_t)rgb565[idx + 1] << 8);
            uint8_t r = (uint8_t)(((px >> 11) & 0x1F) * 255 / 31);
            uint8_t g = (uint8_t)(((px >> 5)  & 0x3F) * 255 / 63);
            uint8_t b = (uint8_t)((px & 0x1F) * 255 / 31);
            gray[gy * gray_w + gx] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
        }
    }
}

/* ---- shared SBUS variable ---- */
extern volatile int sw1_raw;

#define SW_LOW_MAX      600
#define SW_MID_MIN      800
#define SW_MID_MAX      1200
#define SW_HIGH_MIN     1400

static FILE *video_fp = nullptr;

static bool recording = false;
static int video_id = 0;
static sdmmc_card_t *card = nullptr;
static bool sd_mounted = false;

static uint32_t frame_count = 0;
static int64_t recording_start_time = 0;

static void writer_task(void *arg);  // forward declaration
static void ofd_task(void *arg);     // forward declaration

/* ---- Ring buffer helpers (lock-free SPSC) ---- */
static inline size_t ring_avail(void)  // bytes available to read
{
    return s_ring_wr - s_ring_rd;  // works with unsigned wrap
}
static inline size_t ring_free(void)   // bytes available to write
{
    return RING_BUF_SIZE - ring_avail();
}

/* Append data to ring buffer.  Blocks (spins) if ring is full — backpressure. */
static void ring_write(const uint8_t *data, size_t len)
{
    while (len > 0) {
        size_t free = ring_free();
        if (free == 0) {
            vTaskDelay(1);  // backpressure: SD can't keep up
            continue;
        }
        size_t chunk = (len < free) ? len : free;
        size_t pos = s_ring_wr & RING_BUF_MASK;
        size_t to_end = RING_BUF_SIZE - pos;
        if (chunk <= to_end) {
            memcpy(s_ring_buf + pos, data, chunk);
        } else {
            memcpy(s_ring_buf + pos, data, to_end);
            memcpy(s_ring_buf, data + to_end, chunk - to_end);
        }
        s_ring_wr += chunk;
        data += chunk;
        len  -= chunk;
    }
}

/* SD streaming task — continuously drains ring buffer to SD in 64KB chunks.
 * Runs on Core 0.  Keeps the SD card's write pipeline always fed — no idle
 * gaps that would cause SLC cache drain → NAND write stall. */
static void sd_stream_task(void *arg)
{
    size_t total_written = 0;
    int64_t t_last_log = esp_timer_get_time();

    for (;;) {
        size_t avail = ring_avail();
        if (avail >= WRITE_BUF_SIZE) {
            // Read one WRITE_BUF_SIZE chunk from ring → DMA SRAM → SD
            size_t pos = s_ring_rd & RING_BUF_MASK;
            size_t to_end = RING_BUF_SIZE - pos;
            if (WRITE_BUF_SIZE <= to_end) {
                memcpy(s_write_buf, s_ring_buf + pos, WRITE_BUF_SIZE);
            } else {
                memcpy(s_write_buf, s_ring_buf + pos, to_end);
                memcpy(s_write_buf + to_end, s_ring_buf, WRITE_BUF_SIZE - to_end);
            }
            fwrite(s_write_buf, 1, WRITE_BUF_SIZE, video_fp);
            s_ring_rd += WRITE_BUF_SIZE;
            total_written += WRITE_BUF_SIZE;

            // Log throughput every 5 seconds
            int64_t now = esp_timer_get_time();
            if (now - t_last_log > 5000000) {
                float mb = (float)total_written / (1024.0f * 1024.0f);
                float sec = (float)(now - t_last_log) / 1e6f;
                ESP_LOGI(TAG, "SD stream: %.2f MB in %.1fs = %.1f MB/s",
                         mb, sec, mb / sec);
                total_written = 0;
                t_last_log = now;
            }
        } else if (s_sd_stop && avail == 0) {
            break;  // all data drained, exit
        } else if (avail > 0 && s_sd_stop) {
            // Final partial chunk (< 64KB): flush remainder
            size_t pos = s_ring_rd & RING_BUF_MASK;
            size_t to_end = RING_BUF_SIZE - pos;
            size_t rem = avail;
            if (rem <= to_end) {
                memcpy(s_write_buf, s_ring_buf + pos, rem);
            } else {
                memcpy(s_write_buf, s_ring_buf + pos, to_end);
                memcpy(s_write_buf + to_end, s_ring_buf, rem - to_end);
            }
            fwrite(s_write_buf, 1, rem, video_fp);
            s_ring_rd += rem;
        } else {
            vTaskDelay(1);  // ~1ms poll — keeps latency low without busy-spin
        }
    }

    // Flush CSV
    if (s_csv_len > 0 && csv_fp) {
        fwrite(s_csv_buf, 1, s_csv_len, csv_fp);
        s_csv_len = 0;
    }

    s_sd_task = nullptr;
    vTaskDelete(nullptr);
}

/* ---- Forward declarations for public API (defined after static helpers) ---- */
void start_recording(void);
void stop_recording(void);

/* ---- Mount SD card ---- */
static void sdcard_init(void)
{
    const char *SDTAG = "SD";
    const char mount_point[] = "/sdcard";
    
    ESP_LOGI(SDTAG, "Initializing SD card (SDMMC + on-chip LDO)...");
    
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 512 * 1024,  // 512KB clusters -> 2 FAT updates per 1MB frame
        .disk_status_check_enable = false,
        .use_one_fat = false
    };

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    // High-Speed mode @ 3.3V: ~25 MB/s on 4-bit bus — more than enough for
    // Continuous compressed output to SD. Avoids the UHS-I 1.8V voltage
    // switching that caused LDO failures and SDMMC DMA errors (0x109).
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;

    // On-chip LDO channel 4 for SD IO voltage — keep at 3.3V (no UHS-I switch).
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
        ESP_LOGW(SDTAG, "LDO 3.3V set failed (%s) — continuing anyway", esp_err_to_name(ret));
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
    ESP_LOGI(SDTAG, "SD bus: %d kHz, %d-bit, %s",
             card->max_freq_khz, card->log_bus_width ? (1 << card->log_bus_width) : 1,
             (card->is_sdio) ? "SDIO" :
             (card->is_mmc)  ? "MMC"  :
             (card->max_freq_khz >= 100000) ? "SDR104" :
             (card->max_freq_khz >= 50000)  ? "SDR50/HS" : "Default");
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
    
    if (!sd_mounted) return;

    /* ---- Cluster size detection via FatFS f_getfree ---- */
    {
        FATFS *fs_ptr = nullptr;
        DWORD  fre_clust = 0;
        // Drive "0:" is the first FAT volume mounted by esp_vfs_fat
        if (f_getfree("0:", &fre_clust, &fs_ptr) == FR_OK && fs_ptr) {
            uint32_t cluster_bytes = (uint32_t)fs_ptr->csize * (uint32_t)fs_ptr->ssize;
            uint32_t cluster_kb    = cluster_bytes / 1024;
            uint32_t updates_per_mb = (1024 * 1024 + cluster_bytes - 1) / cluster_bytes;
            ESP_LOGI(SDTAG, "FAT cluster size: %lu KB (%lu FAT updates per 1MB frame)",
                     (unsigned long)cluster_kb, (unsigned long)updates_per_mb);
            if (cluster_bytes < 64 * 1024) {
                ESP_LOGW(SDTAG,
                    "!!! Clusters too small — reformat with 64KB allocation units:");
                ESP_LOGW(SDTAG, "    Windows (admin CMD): format X: /fs:FAT32 /a:64K");
            }
        }
    }

    /* ---- Write speed benchmark: burst (4MB, SLC cache) + sustained (32MB, raw NAND) ---- */
    const size_t BM_BUF  = 64 * 1024;   // 64KB DMA SRAM chunk
    uint8_t *bm_buf = (uint8_t *)heap_caps_malloc(BM_BUF,
                                    MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (bm_buf) {
        memset(bm_buf, 0xAB, BM_BUF);
        FILE *bm_fp = fopen("/sdcard/_bm.tmp", "wb");
        if (bm_fp) {
            setvbuf(bm_fp, NULL, _IONBF, 0);

            // Phase 1: burst speed (4MB — fits in SLC cache)
            const size_t BM_BURST = 4 * 1024 * 1024;
            int64_t t0 = esp_timer_get_time();
            for (size_t w = 0; w < BM_BURST; w += BM_BUF)
                fwrite(bm_buf, 1, BM_BUF, bm_fp);
            fflush(bm_fp);
            int64_t dt_burst = esp_timer_get_time() - t0;
            float burst_mbs = (float)BM_BURST / (dt_burst / 1e6f) / (1024.0f * 1024.0f);

            // Phase 2: sustained speed (additional 28MB after SLC cache exhaustion)
            const size_t BM_SUSTAINED = 28 * 1024 * 1024;
            int64_t t1 = esp_timer_get_time();
            for (size_t w = 0; w < BM_SUSTAINED; w += BM_BUF)
                fwrite(bm_buf, 1, BM_BUF, bm_fp);
            fflush(bm_fp);
            int64_t dt_sust = esp_timer_get_time() - t1;
            float sust_mbs = (float)BM_SUSTAINED / (dt_sust / 1e6f) / (1024.0f * 1024.0f);

            fclose(bm_fp);
            remove("/sdcard/_bm.tmp");

            ESP_LOGI(SDTAG, "SD burst speed:     %.2f MB/s (4MB, SLC cache)", burst_mbs);
            ESP_LOGI(SDTAG, "SD sustained speed: %.2f MB/s (28MB, raw NAND)", sust_mbs);
            ESP_LOGI(SDTAG, "Max sustained fps @100KB/frame: %.0f", sust_mbs * 1024 / 100);
        }
        heap_caps_free(bm_buf);
    }

    ESP_LOGI(SDTAG, "SD card ready for recording");
}

/* ---- Start recording ---- */
void start_recording(void)
{
    if (!sd_mounted) {
        ESP_LOGE(TAG, "SD not mounted, cannot record");
        return;
    }

    char fname[64];
    snprintf(fname, sizeof(fname), "/sdcard/V%04d.MJP", video_id);
    char csv_name[64];
    snprintf(csv_name, sizeof(csv_name), "/sdcard/V%04d.CSV", video_id);
    snprintf(s_raw_dump_path, sizeof(s_raw_dump_path), "/sdcard/R%04d.R16", video_id);
    snprintf(s_plane_dump_base, sizeof(s_plane_dump_base), "/sdcard/P%04d", video_id);

    ESP_LOGI(TAG, "Opening file: %s", fname);

    video_fp = fopen(fname, "wb");
    if (!video_fp) {
        ESP_LOGE(TAG, "Failed to open %s (errno=%d: %s)", fname, errno, strerror(errno));
        return;
    }
    // Unbuffered: every fwrite goes through our 64KB DMA bounce buffer directly.
    // This matches the benchmark path that achieved 10 MB/s.
    setvbuf(video_fp, NULL, _IONBF, 0);

    if (!s_write_buf) {
        ESP_LOGE(TAG, "No bounce buffer — recording will be very slow!");
    }

    mjpg_file_header_t header = {};
    memcpy(header.magic, "MJPGSEQ", 7);
    header.version = 1;
    header.width = REC_W;
    header.height = REC_H;
    header.nominal_fps = 30;
    header.quality = JPEG_QUALITY;
    fwrite(&header, 1, sizeof(header), video_fp);

    /* Open OFD+IMU sidecar CSV */
    csv_fp = fopen(csv_name, "w");
    if (csv_fp) {
        fprintf(csv_fp,
                "frame,timestamp_ms,divergence,lr_balance,tau,vx_mean,vy_mean,"
                "mean_flow_mag,mean_flow_mag_raw,"
                "flow_cnt,div_cnt,valid,"
                "ema_div,ema_lr,ema_flow_mag,tau_ms,looming,evasion_level,turn_cmd,az_quiet,"
                "ax,ay,az,gx,gy,gz,roll,pitch,yaw\n");
        ESP_LOGI(TAG, "OFD+IMU sidecar: %s", csv_name);
    } else {
        ESP_LOGW(TAG, "Could not open CSV sidecar (errno=%d)", errno);
    }

    ofd_reset();

    s_media_bytes_written = sizeof(header);

    // Prepare writer task state
    s_writer_frame_count = 0;
    memset(&s_last_ofd, 0, sizeof(s_last_ofd));
    memset(&s_last_imu, 0, sizeof(s_last_imu));
    frame_count = 0;
    s_ring_wr  = 0;
    s_ring_rd  = 0;
    s_csv_len  = 0;
    s_sd_stop  = false;
    recording_start_time = esp_timer_get_time();
    s_raw_dump_pending = false;

    // Drain any stale items left in queues from a previous recording
    frame_slot_t *stale = nullptr;
    while (xQueueReceive(s_write_queue, &stale, 0) == pdTRUE && stale != nullptr) {
        xQueueSend(s_free_queue, &stale, 0);
    }

    // OFD task: runs on Core 0, receives stable grayscale+IMU snapshots.
    s_ofd_queue = xQueueCreate(OFD_WORK_SLOTS, sizeof(ofd_work_item_t *));
    s_ofd_free_queue = xQueueCreate(OFD_WORK_SLOTS, sizeof(ofd_work_item_t *));
    for (int i = 0; i < OFD_WORK_SLOTS; ++i) {
        ofd_work_item_t *slot = &s_ofd_work_slots[i];
        memset(slot, 0, sizeof(*slot));
        xQueueSend(s_ofd_free_queue, &slot, 0);
    }
    xTaskCreatePinnedToCore(ofd_task, "ofd", 4096,
                            nullptr, 3, &s_ofd_task, 0);

    // SD streaming task: Core 1 (shares with writer task which blocks most of the time).
    // Priority 3 < writer's 5: writer preempts SD when encoding, SD streams during idle.
    // Frees Core 0 for main loop (camera capture + downsample) + OFD.
    xTaskCreatePinnedToCore(sd_stream_task, "sd_stream", 4096,
                            nullptr, 3, &s_sd_task, 1);

    // Writer/encoder task on Core 1 — never blocks on SD, only on encode
    xTaskCreatePinnedToCore(writer_task, "vid_writer", 8192,
                            nullptr, 5, &s_writer_task, 1);

    recording = true;
    ESP_LOGI(TAG, "RECORD START: %s (RGB565 -> JPEG stream, quality=%d)", fname, JPEG_QUALITY);
}

/* ---- Stop recording ---- */
void stop_recording(void)
{
    if (!recording || !video_fp) {
        return;
    }

    recording = false;

    // Stop writer task: NULL sentinel drains all pending frames, then it exits
    if (s_writer_task) {
        frame_slot_t *sentinel = nullptr;
        xQueueSend(s_write_queue, &sentinel, portMAX_DELAY);
        while (s_writer_task != nullptr) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }
    // Stop OFD task: sentinel value 0
    if (s_ofd_task) {
        ofd_work_item_t *sentinel = nullptr;
        xQueueSend(s_ofd_queue, &sentinel, portMAX_DELAY);
        while (s_ofd_task != nullptr) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        vQueueDelete(s_ofd_queue);
        s_ofd_queue = nullptr;
        vQueueDelete(s_ofd_free_queue);
        s_ofd_free_queue = nullptr;
    }

    // Stop SD streaming task: signal stop, it drains remaining ring data then exits
    if (s_sd_task) {
        s_sd_stop = true;
        while (s_sd_task != nullptr) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }

    // Writer task has exited: frame_count and total bytes are now stable
    int64_t duration_us = esp_timer_get_time() - recording_start_time;
    float actual_fps = (duration_us > 0) ? (frame_count * 1000000.0f / duration_us) : 0;
    float avg_kbps   = (duration_us > 0) ?
                       ((float)s_media_bytes_written * 8.0f / (duration_us / 1e6f) / 1000.0f) : 0;

    ESP_LOGI(TAG, "RECORD STOP: %lu frames, %.1f fps, %zu KB JPEG stream, %.0f kbps",
             (unsigned long)frame_count, actual_fps,
             s_media_bytes_written / 1024, avg_kbps);

    if (video_fp) {
        mjpg_file_header_t header = {};
        memcpy(header.magic, "MJPGSEQ", 7);
        header.version = 1;
        header.width = REC_W;
        header.height = REC_H;
        header.nominal_fps = (uint32_t)(actual_fps + 0.5f);
        header.quality = JPEG_QUALITY;
        header.frame_count = frame_count;
        fseek(video_fp, 0, SEEK_SET);
        fwrite(&header, 1, sizeof(header), video_fp);
        fseek(video_fp, 0, SEEK_END);
    }

    fflush(video_fp);
    fclose(video_fp);
    video_fp = nullptr;

    if (csv_fp) {
        fflush(csv_fp);
        fclose(csv_fp);
        csv_fp = nullptr;
    }

    video_id++;
    ESP_LOGI(TAG, "Saved V%04d — convert with: python convert_mjp.py V%04d.MJP",
             video_id - 1, video_id - 1);
}

/* ---- Writer task (Core 1, priority 5) ---- */
// Encode + OFD (~20ms, ring slot held) → free slot → push to SPSC ring buffer.
// SD stream task on Core 0 continuously drains the ring to SD.
static void writer_task(void *arg)
{
    for (;;) {
        frame_slot_t *slot = nullptr;
        xQueueReceive(s_write_queue, &slot, portMAX_DELAY);
        if (!slot) break;  // NULL sentinel: exit

        if (s_writer_frame_count == 0) {
            ESP_LOGI(TAG, "First frame: %d bytes RGB565 → JPEG encode (%dx%d)",
                     (int)slot->length, REC_W, REC_H);
        }

        /* ==== Encode + OFD (ring slot held during JPEG encode) ==== */
        size_t encoded_len = 0;
        int64_t t_enc = 0;

        if (s_jpeg_out_buf && s_jpeg_enc) {
            int64_t t0 = esp_timer_get_time();
            uint32_t jpeg_size = 0;
            esp_err_t err = example_encoder_process(s_jpeg_enc,
                                                    slot->data,
                                                    (uint32_t)slot->length,
                                                    s_jpeg_out_buf,
                                                    s_jpeg_out_size,
                                                    &jpeg_size);
            int64_t t1 = esp_timer_get_time();
            t_enc = (t1 - t0) / 1000;

            if (err == ESP_OK && jpeg_size > 0) {
                encoded_len = jpeg_size;
            } else if (err != ESP_OK) {
                ESP_LOGE(TAG, "JPEG encode error: %s (frame %lu)",
                         esp_err_to_name(err), (unsigned long)s_writer_frame_count);
            }
        }

        s_writer_frame_count++;
        frame_count = s_writer_frame_count;

        if (s_ofd_queue && s_ofd_free_queue &&
            s_writer_frame_count % OFD_EVERY_N == 0) {
            ofd_work_item_t *work = nullptr;
            if (xQueueReceive(s_ofd_free_queue, &work, 0) == pdTRUE && work != nullptr) {
                work->frame_no = s_writer_frame_count;
                work->imu = s_last_imu;
                extract_luma_from_rgb565(slot->data, REC_W, REC_H, work->gray, OFD_W, OFD_H);
                if (xQueueSend(s_ofd_queue, &work, 0) != pdTRUE) {
                    xQueueSend(s_ofd_free_queue, &work, 0);
                }
            }
        }

        int64_t capture_ts_ms = slot->capture_ts_ms;

        camera_frame_t cam_frame = {
            .data = slot->data,
            .length = slot->length,
            .index = slot->cam_index,
        };
        camera_return_frame(&cam_frame);
        slot->data = nullptr;
        slot->length = 0;
        slot->cam_index = -1;

        // ---- FREE metadata slot immediately ----
        xQueueSend(s_free_queue, &slot, portMAX_DELAY);

        /* ==== Push encoded data into SPSC ring buffer ==== */
        if (encoded_len > 0 && s_ring_buf) {
            uint32_t frame_len_le = (uint32_t)encoded_len;
            ring_write((const uint8_t *)&frame_len_le, sizeof(frame_len_le));
            ring_write(s_jpeg_out_buf, encoded_len);
            s_media_bytes_written += sizeof(frame_len_le) + encoded_len;
        }

        // CSV: append to simple buffer (flushed by SD task on stop)
        if (s_csv_buf && csv_fp) {
            size_t clen = s_csv_len;  // snapshot
            if (clen + 300 < CSV_BUF_SIZE) {
                imu_data_t imu = s_last_imu;
                ofd_result_t ofd = load_last_ofd();
                int n = snprintf(s_csv_buf + clen, CSV_BUF_SIZE - clen,
                        "%lu,%lld,%.5f,%.5f,%.5f,%.4f,%.4f,"
                        "%.4f,%.4f,"
                        "%d,%d,%d,"
                        "%.5f,%.5f,%.4f,%.3f,%d,%d,%.4f,%d,"
                        "%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                        (unsigned long)s_writer_frame_count,
                        (long long)capture_ts_ms,
                        (double)ofd.divergence,
                        (double)ofd.lr_balance,
                        (double)ofd.tau,
                        (double)ofd.vx_mean,
                        (double)ofd.vy_mean,
                        (double)ofd.mean_flow_mag,
                        (double)ofd.mean_flow_mag_raw,
                        ofd.flow_cnt,
                        ofd.div_cnt,
                        (int)ofd.valid,
                        (double)ofd.ema_div,
                        (double)ofd.ema_lr,
                        (double)ofd.ema_flow_mag,
                        (double)ofd.tau_ms,
                        (int)ofd.looming_detected,
                        ofd.evasion_level,
                        (double)ofd.turn_cmd,
                        (int)ofd.az_quiet,
                        (double)imu.ax,   (double)imu.ay,   (double)imu.az,
                        (double)imu.gx,   (double)imu.gy,   (double)imu.gz,
                        (double)imu.roll, (double)imu.pitch, (double)imu.yaw);
                if (n > 0) {
                    s_csv_len = clen + (size_t)n;
                    if (s_csv_len >= CSV_FLUSH_THRESHOLD) {
                        flush_csv_buffer();
                    }
                }
            } else {
                flush_csv_buffer();
            }
        }

        // Log periodically
        if (s_writer_frame_count == 1 || s_writer_frame_count % 30 == 0) {
            ESP_LOGI(TAG, "Frame %lu: jpeg=%lldms  encoded=%d bytes  ring=%d/%d",
                     (unsigned long)(s_writer_frame_count - 1),
                     t_enc, (int)encoded_len,
                     (int)ring_avail(), RING_BUF_SIZE);
        }
        if (s_writer_frame_count % 10 == 0) {
            int64_t elapsed_us = esp_timer_get_time() - recording_start_time;
            float fps  = (elapsed_us > 0) ?
                         (s_writer_frame_count * 1000000.0f / elapsed_us) : 0;
            float kbps = (elapsed_us > 0) ?
                         ((float)s_media_bytes_written * 8.0f / (elapsed_us / 1e6f) / 1000.0f) : 0;
            ESP_LOGI(TAG, "Recording: %lu frames, %.1f fps, %zu KB, %.0f kbps",
                     (unsigned long)s_writer_frame_count, fps,
                     s_media_bytes_written / 1024, kbps);
        }
    }

    s_writer_task = nullptr;
    vTaskDelete(nullptr);
}

/* ---- OFD task helpers ---- */

// Finding 3: tau computation with dt clamping.
// Computes τ = div / (Δdiv/dt) — robust to noisy dt (clamped to [DT_MIN, DT_MAX]).
static float compute_tau_ms(float div_now, float div_prev, float dt_ms)
{
    float dt_c = fminf(fmaxf(dt_ms, OFD_DT_MIN_MS), OFD_DT_MAX_MS);
    float dt_s = dt_c / 1000.0f;
    float ddiv = div_now - div_prev;
    if (fabsf(ddiv) < 1e-5f) return OFD_TAU_MAX;
    float tau_s = div_now / (ddiv / dt_s);
    return fminf(fmaxf(tau_s * 1000.0f, 0.0f), OFD_TAU_MAX);
}

// Finding 5: wing-sync gate — returns true when az is near the gravity-only quiet point.
static inline bool is_wing_quiet(float az_g)
{
#if !OFD_USE_AZ_QUIET_GATE
    (void)az_g;
    return true;
#else
    return fabsf(az_g - OFD_AZ_QUIET_CENTER) < OFD_AZ_QUIET_BAND;
#endif
}

/* ---- OFD task (Core 0, priority 3) ---- */
// Receives a frame snapshot from the writer task: grayscale downsample + IMU sample.
// This avoids racing on shared buffers and keeps CSV/OFD rows coherent.
// Filter state is local — automatically resets each recording session
// because this task is re-created at start_recording() / destroyed at stop_recording().
// frame_no==0 = shutdown sentinel.
static void ofd_task(void *arg)
{
    // Filter state
    float   ema_div = 0.0f, ema_lr = 0.0f, ema_flow_mag = 0.0f;
    float   prev_ema_div = 0.0f;
    int64_t prev_ts_us = 0;
    bool    filter_initialized = false;

    for (;;) {
        ofd_work_item_t *work = nullptr;
        xQueueReceive(s_ofd_queue, &work, portMAX_DELAY);
        if (work == nullptr) break;  // shutdown sentinel

        // Wing-sync gate — skip OFD when az indicates active wing stroke
        imu_data_t imu_snap = work->imu;
        bool az_quiet = is_wing_quiet(imu_snap.az);
        {
            ofd_result_t last = load_last_ofd();
            last.az_quiet = az_quiet;
            store_last_ofd(last);
        }
        if (!az_quiet) {
            xQueueSend(s_ofd_free_queue, &work, portMAX_DELAY);
            continue;  // hold previous EMA; do not process this frame
        }

        // Pass gyro rates for derotation
        ofd_result_t r = ofd_process_gray(work->gray,
                                          imu_snap.gx, imu_snap.gy, imu_snap.gz);
        r.az_quiet = true;

        // Hard-reject frames with insufficient tracked points
        if (!r.valid || r.div_cnt < OFD_MIN_DIV_CNT || r.flow_cnt == 0) {
            // Hold EMA values; propagate rejection without poisoning the filter
            r.ema_div      = ema_div;
            r.ema_lr       = ema_lr;
            r.ema_flow_mag = ema_flow_mag;
            {
                ofd_result_t last = load_last_ofd();
                r.tau_ms   = last.tau_ms;
                r.turn_cmd = last.turn_cmd;
            }
            r.looming_detected = false;
            r.evasion_level    = OFD_EVADE_NONE;
            store_last_ofd(r);
            xQueueSend(s_ofd_free_queue, &work, portMAX_DELAY);
            continue;
        }

        // Timestamp for dt used by τ computation (logging only)
        int64_t now_us = esp_timer_get_time();
        float dt_ms = (prev_ts_us > 0)
                    ? (float)((now_us - prev_ts_us) / 1000)
                    : OFD_DT_MIN_MS;
        prev_ts_us = now_us;

        // Bias subtraction + EMA for divergence and lr_balance
        float corrected = r.divergence - OFD_DIV_BIAS;
        if (!filter_initialized) {
            ema_div      = corrected;
            ema_lr       = r.lr_balance;
            ema_flow_mag = r.mean_flow_mag;
            filter_initialized = true;
        } else {
            ema_div      = OFD_EMA_ALPHA * corrected       + (1.0f - OFD_EMA_ALPHA) * ema_div;
            ema_lr       = OFD_EMA_ALPHA * r.lr_balance    + (1.0f - OFD_EMA_ALPHA) * ema_lr;
            ema_flow_mag = OFD_EMA_FLOW_ALPHA * r.mean_flow_mag
                         + (1.0f - OFD_EMA_FLOW_ALPHA) * ema_flow_mag;
        }

        // τ with dt clamping (logged for diagnostics — NOT used for decisions)
        float tau_ms = compute_tau_ms(ema_div, prev_ema_div, dt_ms);
        prev_ema_div = ema_div;

        // ---- Flow-magnitude-primary trigger ----
        // PRIMARY: derotated mean flow magnitude (monotonic with proximity)
        // SECONDARY: divergence confirms looming (expansion pattern)
        bool flow_trigger = (ema_flow_mag > OFD_FLOW_THRESH_BRAKE)
                         || (ema_flow_mag > OFD_FLOW_THRESH_ALERT
                             && fabsf(ema_div) > OFD_DIV_THRESHOLD);
        bool looming = flow_trigger
                    && (r.div_cnt >= OFD_MIN_DIV_CNT)
                    && r.valid;

        int evasion_level = OFD_EVADE_NONE;
        if (looming) {
            if      (ema_flow_mag > OFD_FLOW_THRESH_EVADE)  evasion_level = OFD_EVADE_EVADE;
            else if (ema_flow_mag > OFD_FLOW_THRESH_BRAKE)  evasion_level = OFD_EVADE_BRAKE;
            else                                             evasion_level = OFD_EVADE_ALERT;
        }

        // lr_balance gain on derotated divergence; negative = turn away from expanding side.
        float turn_cmd = -ema_lr * OFD_LR_GAIN;
        turn_cmd = fminf(fmaxf(turn_cmd, -1.0f), 1.0f);

        r.ema_div          = ema_div;
        r.ema_lr           = ema_lr;
        r.ema_flow_mag     = ema_flow_mag;
        r.tau_ms           = tau_ms;
        r.turn_cmd         = turn_cmd;
        r.evasion_level    = evasion_level;
        r.looming_detected = looming;

        store_last_ofd(r);
        if (work->frame_no == 2 || work->frame_no % 60 == 0) {
            ESP_LOGI(TAG, "OFD frame %lu: valid=%d flow=%d div=%d mag=%.3f div=%.4f az_quiet=%d",
                     (unsigned long)work->frame_no, (int)r.valid, r.flow_cnt, r.div_cnt,
                     (double)r.mean_flow_mag, (double)r.divergence, (int)r.az_quiet);
        }
        xQueueSend(s_ofd_free_queue, &work, portMAX_DELAY);
    }

    s_ofd_task = nullptr;
    vTaskDelete(nullptr);
}

/* ================================================================
 * Public API — called from main_fwr_ofd.cpp
 * ================================================================ */

/**
 * video_rec_init — initialise SD card, PSRAM ring buffer, and OFD engine.
 * Must be called once at startup, before camera_init().
 */
void video_rec_init(void)
{
    sdcard_init();

    // Allocate DMA-capable internal SRAM bounce buffer EARLY — before heap
    // fragmentation from ring slots/OFD/JPEG.  SDMMC DMA cannot read PSRAM
    // directly; without this buffer every fwrite falls back to 512B per-sector
    // bounce copies → timeouts (the 0x109 errors seen in earlier sessions).
    s_write_buf = (uint8_t *)heap_caps_malloc(WRITE_BUF_SIZE,
                                MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (s_write_buf) {
        ESP_LOGI(TAG, "DMA bounce buffer: %d KB in internal SRAM", WRITE_BUF_SIZE / 1024);
    } else {
        ESP_LOGW(TAG, "No DMA SRAM for bounce buf — falling back to PSRAM (slower)");
        s_write_buf = (uint8_t *)heap_caps_malloc(WRITE_BUF_SIZE, MALLOC_CAP_SPIRAM);
    }

    // SPSC ring buffer for continuous SD streaming (PSRAM)
    s_ring_buf = (uint8_t *)heap_caps_malloc(RING_BUF_SIZE, MALLOC_CAP_SPIRAM);
    if (s_ring_buf) {
        ESP_LOGI(TAG, "SD ring buffer: %d KB in PSRAM", RING_BUF_SIZE / 1024);
    } else {
        ESP_LOGE(TAG, "Ring buffer alloc failed (%d bytes)", RING_BUF_SIZE);
    }
    s_csv_buf = (char *)heap_caps_malloc(CSV_BUF_SIZE, MALLOC_CAP_SPIRAM);
    if (!s_csv_buf) ESP_LOGE(TAG, "CSV buffer alloc failed");

    s_write_queue = xQueueCreate(FRAME_RING_SLOTS, sizeof(frame_slot_t *));
    s_free_queue  = xQueueCreate(FRAME_RING_SLOTS, sizeof(frame_slot_t *));
    for (int i = 0; i < FRAME_RING_SLOTS; i++) {
        memset(&s_slots[i], 0, sizeof(s_slots[i]));
        frame_slot_t *p = &s_slots[i];
        xQueueSend(s_free_queue, &p, 0);
    }
    ESP_LOGI(TAG, "Frame metadata ring: %d slots (zero-copy camera buffers)", FRAME_RING_SLOTS);

    // Scan SD card for existing V*.MJP files and resume numbering from the next ID.
    if (sd_mounted) {
        char path[64];
        for (int id = 0; id < 10000; id++) {
            snprintf(path, sizeof(path), "/sdcard/V%04d.MJP", id);
            struct stat st;
            if (stat(path, &st) != 0) {
                video_id = id;  // first ID with no file
                break;
            }
        }
        ESP_LOGI(TAG, "Next video ID: %d", video_id);
    }

    ofd_init(OFD_W, OFD_H);
    ESP_LOGI(TAG, "OFD init (%dx%d)", OFD_W, OFD_H);

    // Init JPEG hardware encoder for RGB565 capture.
    {
        example_encoder_config_t enc_cfg = {
            .width = REC_W,
            .height = REC_H,
            .pixel_format = V4L2_PIX_FMT_RGB565,
            .quality = JPEG_QUALITY,
        };
        if (example_encoder_init(&enc_cfg, &s_jpeg_enc) != ESP_OK) {
            ESP_LOGE(TAG, "JPEG encoder init failed");
        } else {
            if (example_encoder_alloc_output_buffer(s_jpeg_enc, &s_jpeg_out_buf, &s_jpeg_out_size) != ESP_OK) {
                ESP_LOGE(TAG, "JPEG output buffer alloc failed");
            } else {
                ESP_LOGI(TAG, "JPEG HW encoder ready: %dx%d quality=%d out_buf=%u KB",
                         REC_W, REC_H, JPEG_QUALITY, s_jpeg_out_size / 1024);
            }
        }
    }
}

/** Returns true while a recording session is active. */
bool is_recording(void) { return recording; }

/** Return the latest OFD result from the background ofd_task. */
ofd_result_t video_rec_last_ofd(void) { return load_last_ofd(); }

/** Store the latest IMU sample for CSV logging. Called from main loop. */
void video_rec_set_imu(imu_data_t d) { s_last_imu = d; }

/**
 * video_rec_enqueue — copy a camera frame into the PSRAM ring buffer
 * and hand it to the writer task.  Always calls camera_return_frame(f).
 */
void video_rec_enqueue(camera_frame_t *f)
{
    if (!recording || !f->data || f->length == 0) {
        camera_return_frame(f);
        return;
    }

    frame_slot_t *slot = nullptr;
    if (xQueueReceive(s_free_queue, &slot, 0) == pdTRUE && slot) {
        int src_stride = camera_get_stride();
        uint32_t pixfmt = camera_get_pixelformat();
        if (pixfmt == v4l2_fourcc('R','G','B','P')) {
            size_t rgb565_row_bytes = f->length / REC_H;
            size_t min_src_len = (size_t)REC_W * REC_H * 2;
            if (rgb565_row_bytes < (size_t)REC_W * 2 || f->length < min_src_len) {
                ESP_LOGE(TAG, "Unexpected RGB565 frame layout: len=%u stride=%d row_bytes=%u expected>=%u",
                         (unsigned)f->length, src_stride, (unsigned)rgb565_row_bytes, (unsigned)min_src_len);
                camera_return_frame(f);
                xQueueSend(s_free_queue, &slot, 0);
                return;
            }
            slot->data          = f->data;
            slot->length        = min_src_len;
            slot->cam_index     = f->index;
            slot->capture_ts_ms = (esp_timer_get_time() - recording_start_time) / 1000;
            if (xQueueSend(s_write_queue, &slot, 0) != pdTRUE) {
                ESP_LOGD(TAG, "write_queue full, frame dropped");
                camera_return_frame(f);
                xQueueSend(s_free_queue, &slot, 0);
            }
            return;
        }
        if (pixfmt != v4l2_fourcc('Y','U','1','2')) {
            ESP_LOGE(TAG, "Unsupported camera pixel format for recorder: %c%c%c%c (0x%08x)",
                     (char)(pixfmt & 0xFF), (char)((pixfmt >> 8) & 0xFF),
                     (char)((pixfmt >> 16) & 0xFF), (char)((pixfmt >> 24) & 0xFF),
                     (unsigned)pixfmt);
            camera_return_frame(f);
            xQueueSend(s_free_queue, &slot, 0);
            return;
        }
        size_t min_src_len = (size_t)src_stride * REC_H + (size_t)src_stride * REC_H / 2;
        if (src_stride < REC_W || f->length < min_src_len) {
            ESP_LOGE(TAG, "Unexpected camera frame layout: len=%u stride=%d expected>=%u",
                     (unsigned)f->length, src_stride, (unsigned)min_src_len);
            camera_return_frame(f);
            xQueueSend(s_free_queue, &slot, 0);
            return;
        }
        if (frame_count == 0) {
            ESP_LOGI(TAG, "Camera frame probe: pixfmt=%c%c%c%c (0x%08x) len=%u stride=%d",
                     (char)(pixfmt & 0xFF), (char)((pixfmt >> 8) & 0xFF),
                     (char)((pixfmt >> 16) & 0xFF), (char)((pixfmt >> 24) & 0xFF),
                     (unsigned)pixfmt, (unsigned)f->length, src_stride);
        }
        // Camera outputs I420 at 1280x720 = same as encoder input.
        // Only format repack needed (I420 planar → O_UYY_E_VYY packed).
        // Sequential memory access → ~5ms vs 70ms for the old 1920→1280 downsample.
        dump_raw_i420_once(f->data, min_src_len, REC_W, REC_H, src_stride);
        if (!s_raw_dump_pending) {
            dump_i420_planes_once(f->data, REC_W, REC_H, src_stride);
            s_plane_dump_base[0] = '\0';
        }
        i420_repack_to_yuv420(f->data, slot->data, REC_W, REC_H, src_stride);
        slot->length        = (size_t)REC_W * REC_H * 3 / 2;
        slot->capture_ts_ms = (esp_timer_get_time() - recording_start_time) / 1000;
        camera_return_frame(f);   // release camera buffer ASAP after downsample

        if (xQueueSend(s_write_queue, &slot, 0) != pdTRUE) {
            ESP_LOGD(TAG, "write_queue full, frame dropped");
            xQueueSend(s_free_queue, &slot, 0);
        }
    } else {
        camera_return_frame(f);
        ESP_LOGD(TAG, "No free slot, frame dropped");
    }
}
static void flush_csv_buffer(void)
{
    if (s_csv_len > 0 && csv_fp) {
        fwrite(s_csv_buf, 1, s_csv_len, csv_fp);
        s_csv_len = 0;
    }
}

static ofd_result_t load_last_ofd(void)
{
    ofd_result_t out;
    taskENTER_CRITICAL(&s_ofd_lock);
    out = s_last_ofd;
    taskEXIT_CRITICAL(&s_ofd_lock);
    return out;
}

static void store_last_ofd(const ofd_result_t &r)
{
    taskENTER_CRITICAL(&s_ofd_lock);
    s_last_ofd = r;
    taskEXIT_CRITICAL(&s_ofd_lock);
}
