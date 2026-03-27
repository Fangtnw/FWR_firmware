#include "video_rec.h"
#include "fwr_control.h"
#include "sbus_rx.h"
#include "camera.h"
#include "ofd.h"
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

static const char *TAG = "VID_REC";

/* ---- Camera resolution --- change these two to test different sizes ---- */
#define CAM_W  800
#define CAM_H  640

/* ---- OFD sidecar ---- */
#define OFD_W  160
#define OFD_H  120

/* ---- Recording performance tunables ---- */
#define OFD_EVERY_N      1              // Run OFD every Nth written frame
#define FRAME_RING_SLOTS 4              // Number of PSRAM frame ring slots
#define FRAME_SLOT_SIZE  (CAM_W * CAM_H * 2)  // bytes per slot (RGB565)

// Pre-allocate this many bytes at recording start by seeking to the end.
// FatFS builds the entire cluster chain upfront (fast - only FAT table writes,
// no data) so recording becomes zero-FAT-update pure sequential writes.
// Tune to your typical recording length. 300MB ~= 300s@1fps or 20s@15fps.
#define PRE_ALLOC_SIZE  (300L * 1024L * 1024L)

/* ---- Frame ring buffer ---- */
typedef struct {
    uint8_t *data;           // PSRAM buffer
    size_t   length;
    int64_t  capture_ts_ms;  // timestamp relative to recording_start_time
} frame_slot_t;

static frame_slot_t  s_slots[FRAME_RING_SLOTS];
static QueueHandle_t s_write_queue = nullptr;  // filled slots ready to write
static QueueHandle_t s_free_queue  = nullptr;  // available (empty) slots
static TaskHandle_t  s_writer_task = nullptr;
static uint32_t      s_writer_frame_count = 0; // owned exclusively by writer task

/* ---- OFD task: decoupled from SD write path ---- */
// A single-slot queue carries one frame pointer to the OFD task.
// The OFD task writes its result into s_last_ofd; the writer task reads it
// (no mutex needed: writer reads only when ofd_queue is empty = OFD idle).
static QueueHandle_t s_ofd_queue   = nullptr;  // capacity 1, frame_slot_t*
static ofd_result_t  s_last_ofd    = {};       // latest OFD result
static TaskHandle_t  s_ofd_task    = nullptr;

/* ---- SD write buffer (DMA-capable internal SRAM for direct SDMMC DMA) ---- */
static uint8_t     *s_write_buf   = nullptr;
static const size_t WRITE_BUF_SIZE = 64 * 1024;  // 64KB; must be in internal SRAM for DMA

static uint8_t *ofd_gray = nullptr;  // allocated in PSRAM at init
static FILE    *csv_fp   = nullptr;

/* Downsample RGB565 -> grayscale (nearest-neighbour) */
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
            dst[y * dst_w + x] = (uint8_t)((77*r + 150*g + 29*b) >> 8);
        }
    }
}

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

static void writer_task(void *arg);  // forward declaration
static void ofd_task(void *arg);     // forward declaration

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

    /* ---- Write speed benchmark (4MB sequential) ---- */
    const size_t BM_BUF  = 64 * 1024;   // 64KB DMA SRAM chunk
    const size_t BM_TOTAL = 4 * 1024 * 1024;
    uint8_t *bm_buf = (uint8_t *)heap_caps_malloc(BM_BUF,
                                    MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (bm_buf) {
        memset(bm_buf, 0xAB, BM_BUF);
        FILE *bm_fp = fopen("/sdcard/_bm.tmp", "wb");
        if (bm_fp) {
            setvbuf(bm_fp, NULL, _IONBF, 0);  // unbuffered -> raw FAT perf
            int64_t t0 = esp_timer_get_time();
            for (size_t written = 0; written < BM_TOTAL; written += BM_BUF)
                fwrite(bm_buf, 1, BM_BUF, bm_fp);
            fflush(bm_fp);
            int64_t dt_us = esp_timer_get_time() - t0;
            fclose(bm_fp);
            remove("/sdcard/_bm.tmp");
            float speed_mbs = (float)BM_TOTAL / (dt_us / 1e6f) / (1024.0f * 1024.0f);
            ESP_LOGI(SDTAG, "SD write speed: %.2f MB/s => max ~%.1f fps @1MB/frame",
                     speed_mbs, speed_mbs);
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
    snprintf(fname, sizeof(fname), "/sdcard/V%04d.VID", video_id);
    char csv_name[64];
    snprintf(csv_name, sizeof(csv_name), "/sdcard/V%04d.CSV", video_id);

    ESP_LOGI(TAG, "Opening file: %s", fname);

    video_fp = fopen(fname, "wb");
    if (!video_fp) {
        ESP_LOGE(TAG, "Failed to open %s (errno=%d: %s)", fname, errno, strerror(errno));
        return;
    }

    // Allocate a DMA-capable internal SRAM bounce buffer used explicitly by
    // the writer task (PSRAM → this buffer → SD via SDMMC DMA).
    // We do NOT use setvbuf with this buffer because newlib bypasses stdio
    // buffering when fwrite size > buffer size, passing the raw PSRAM pointer
    // straight to FatFS/SDMMC. SDMMC can't DMA from PSRAM so it falls back to
    // a per-sector (512B) bounce loop — 2048 SD transactions per 1MB frame.
    // Solution: keep the file unbuffered and manually copy PSRAM→SRAM in chunks.
    bool s_write_buf_dma = false;
    s_write_buf = (uint8_t *)heap_caps_malloc(WRITE_BUF_SIZE,
                                MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (s_write_buf) {
        s_write_buf_dma = true;
    } else {
        ESP_LOGW(TAG, "No DMA SRAM for bounce buf, falling back to PSRAM (slower)");
        s_write_buf = (uint8_t *)heap_caps_malloc(WRITE_BUF_SIZE, MALLOC_CAP_SPIRAM);
    }
    // Unbuffered file: every fwrite goes straight to FatFS with whatever
    // pointer we provide — so we control the pointer (always our bounce buffer).
    setvbuf(video_fp, NULL, _IONBF, 0);
    ESP_LOGI(TAG, "Bounce buffer: %d KB in %s",
             WRITE_BUF_SIZE / 1024,
             s_write_buf ? (s_write_buf_dma ? "DMA SRAM" : "PSRAM") : "none (SLOW)");

    /* Open OFD sidecar CSV */
    csv_fp = fopen(csv_name, "w");
    if (csv_fp) {
        fprintf(csv_fp,
                "frame,timestamp_ms,divergence,lr_balance,tau,vx_mean,vy_mean,"
                "flow_cnt,div_cnt,valid\n");
        ESP_LOGI(TAG, "OFD sidecar: %s", csv_name);
    } else {
        ESP_LOGW(TAG, "Could not open CSV sidecar (errno=%d)", errno);
    }

    ofd_reset();

    // Write header as a full 512-byte sector (sector-aligned padding).
    // CRITICAL: FatFS only issues efficient multi-sector disk_write() calls when
    // fp->fptr % 512 == 0.  A 20-byte header makes all frame writes unaligned,
    // forcing 128 × single-sector CMD24 per 64KB chunk instead of 1 × CMD25 —
    // that is the reason for 1900ms/frame vs the expected 125ms/frame.
    // Padding to 512 bytes keeps every frame write sector-aligned.
    {
        uint8_t header_sector[512] = {};
        raw_video_header_t *h = (raw_video_header_t *)header_sector;
        h->frame_count = 0;
        h->fps         = 30;
        h->width       = CAM_W;
        h->height      = CAM_H;
        h->frame_size  = 0;
        fwrite(header_sector, sizeof(header_sector), 1, video_fp);
    }

    // Pre-allocate recording space by seeking to the end and writing 1 byte.
    // f_lseek (called by fseek in write mode) allocates the full cluster chain
    // up to PRE_ALLOC_SIZE with only FAT table writes — no bulk data is written.
    // After this, sequential frame writes hit no create_chain() calls → no FAT
    // updates → near-maximum SD throughput during recording.
    {
        int64_t pa_t0 = esp_timer_get_time();
        bool ok = (fseek(video_fp, PRE_ALLOC_SIZE - 1, SEEK_SET) == 0) &&
                  (fputc(0, video_fp) != EOF);
        fflush(video_fp);  // flush FAT chain to disk
        fseek(video_fp, 512, SEEK_SET);  // frame data starts at sector 1 (512B)
        float pa_s = (esp_timer_get_time() - pa_t0) / 1e6f;
        if (ok) {
            ESP_LOGI(TAG, "Pre-alloc %ldMB done in %.2fs - recording at max speed",
                     PRE_ALLOC_SIZE / (1024L * 1024L), pa_s);
        } else {
            ESP_LOGW(TAG, "Pre-alloc failed (SD full?), recording without it (%.2fs)", pa_s);
        }
    }

    // Prepare writer task state
    s_writer_frame_count = 0;
    memset(&s_last_ofd, 0, sizeof(s_last_ofd));
    frame_count = 0;
    first_frame_size = 0;
    recording_start_time = esp_timer_get_time();

    // Drain any stale items left in queues from a previous recording
    frame_slot_t *stale = nullptr;
    while (xQueueReceive(s_write_queue, &stale, 0) == pdTRUE && stale != nullptr) {
        xQueueSend(s_free_queue, &stale, 0);
    }

    // OFD task: runs on Core 0, receives a uint32_t "go" signal (frame number).
    // Writer does the fast downsample (PSRAM→gray, ~5ms) then signals OFD to
    // run the block-matching computation without blocking the SD write loop.
    s_ofd_queue = xQueueCreate(1, sizeof(uint32_t));
    xTaskCreatePinnedToCore(ofd_task, "ofd", 4096,
                            nullptr, 3, &s_ofd_task, 0);

    // Writer task on Core 1 — dedicated to SD writes, no CPU competition
    xTaskCreatePinnedToCore(writer_task, "vid_writer", 8192,
                            nullptr, 5, &s_writer_task, 1);

    recording = true;
    ESP_LOGI(TAG, "RECORD START: %s", fname);
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
        uint32_t sentinel = 0;
        xQueueOverwrite(s_ofd_queue, &sentinel);
        while (s_ofd_task != nullptr) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        vQueueDelete(s_ofd_queue);
        s_ofd_queue = nullptr;
    }

    // Writer task has exited: frame_count and first_frame_size are now stable
    int64_t duration_us = esp_timer_get_time() - recording_start_time;
    float actual_fps = (duration_us > 0) ? (frame_count * 1000000.0f / duration_us) : 0;

    ESP_LOGI(TAG, "RECORD STOP: %d frames, %.1f fps, frame_size=%d bytes",
             frame_count, actual_fps, first_frame_size);

    // Patch header with actual frame count and fps (keep full 512B sector intact)
    fseek(video_fp, 0, SEEK_SET);
    {
        uint8_t header_sector[512] = {};
        raw_video_header_t *h = (raw_video_header_t *)header_sector;
        h->frame_count = frame_count;
        h->fps         = (uint32_t)(actual_fps + 0.5f);
        h->width       = CAM_W;
        h->height      = CAM_H;
        h->frame_size  = first_frame_size;
        fwrite(header_sector, sizeof(header_sector), 1, video_fp);
    }

    // Truncate unused pre-allocated space so the file reflects actual content
    // Header occupies the first 512 bytes (1 sector), frames follow immediately
    long actual_size = 512L + (long)frame_count * (long)first_frame_size;
    fflush(video_fp);
    if (ftruncate(fileno(video_fp), actual_size) != 0) {
        ESP_LOGW(TAG, "ftruncate failed (errno=%d), pre-alloc space kept", errno);
    }
    fclose(video_fp);
    video_fp = nullptr;

    if (s_write_buf) {
        heap_caps_free(s_write_buf);
        s_write_buf = nullptr;
    }

    if (csv_fp) {
        fflush(csv_fp);
        fclose(csv_fp);
        csv_fp = nullptr;
        ESP_LOGI(TAG, "OFD CSV sidecar saved.");
    }

    video_id++;
    ESP_LOGI(TAG, "Video saved. Use Python script to convert to MP4 ++++.");
}

/* ---- Writer task (Core 1, priority 5) ---- */
// Receives filled frame slots from s_write_queue, writes to SD, runs OFD,
// writes CSV, then returns the slot to s_free_queue.
// A NULL sentinel pointer signals the task to stop.
static void writer_task(void *arg)
{
    for (;;) {
        frame_slot_t *slot = nullptr;
        xQueueReceive(s_write_queue, &slot, portMAX_DELAY);
        if (!slot) break;  // NULL sentinel: stop recording

        // Validate and record frame size from first frame
        if (s_writer_frame_count == 0) {
            first_frame_size = (uint32_t)slot->length;
            ESP_LOGI(TAG, "First frame: %d bytes (%dx%d RGB565 = %d bytes)",
                     slot->length, CAM_W, CAM_H, CAM_W * CAM_H * 2);
        } else if (slot->length != first_frame_size) {
            ESP_LOGW(TAG, "Frame size mismatch: %d != %d, skipping",
                     slot->length, first_frame_size);
            xQueueSend(s_free_queue, &slot, portMAX_DELAY);
            continue;
        }

        // Write frame to SD card.
        // Frame data is in PSRAM (slot->data). SDMMC DMA cannot read PSRAM
        // directly — it would bounce-copy sector by sector (512B × 2048 = slow).
        // Instead: memcpy PSRAM→DMA SRAM in chunks, then fwrite from DMA SRAM.
        // Each fwrite(DMA_SRAM, chunk) calls FatFS which calls SDMMC DMA once
        // per chunk — 16 large DMA transfers instead of 2048 tiny ones.
        {
            const uint8_t *src = (const uint8_t *)slot->data;
            size_t remaining = slot->length;
            bool frame_ok = (s_write_buf != nullptr);
            while (remaining > 0 && frame_ok) {
                size_t chunk = (remaining < WRITE_BUF_SIZE) ? remaining : WRITE_BUF_SIZE;
                memcpy(s_write_buf, src, chunk);             // PSRAM → DMA SRAM
                frame_ok = (fwrite(s_write_buf, 1, chunk, video_fp) == chunk); // DMA SRAM → SD
                src += chunk;
                remaining -= chunk;
            }
            if (!frame_ok) {
                ESP_LOGE(TAG, "Frame write error");
            }
        }

        s_writer_frame_count++;
        frame_count = s_writer_frame_count;

        // On every Nth frame: downsample into ofd_gray (fast, ~5ms on Core 1),
        // then signal Core 0 OFD task to run block-matching on the gray copy.
        // Downsampling happens here so slot->data is safe to recycle immediately after.
        // xQueueOverwrite: if OFD is still busy with the previous signal, we just
        // skip one OFD computation rather than stalling the SD write path.
        if (ofd_gray && s_ofd_queue &&
            s_writer_frame_count % OFD_EVERY_N == 0) {
            downsample_rgb565_to_gray(
                (const uint16_t *)slot->data, CAM_W, CAM_H,
                ofd_gray, OFD_W, OFD_H);
            uint32_t sig = s_writer_frame_count;
            xQueueOverwrite(s_ofd_queue, &sig);
        }

        if (csv_fp) {
            fprintf(csv_fp, "%lu,%lld,%.5f,%.5f,%.5f,%.4f,%.4f,%d,%d,%d\n",
                    (unsigned long)s_writer_frame_count,
                    (long long)slot->capture_ts_ms,
                    (double)s_last_ofd.divergence,
                    (double)s_last_ofd.lr_balance,
                    (double)s_last_ofd.tau,
                    (double)s_last_ofd.vx_mean,
                    (double)s_last_ofd.vy_mean,
                    s_last_ofd.flow_cnt,
                    s_last_ofd.div_cnt,
                    (int)s_last_ofd.valid);
        }

        // Log fps every 10 written frames
        if (s_writer_frame_count % 10 == 0) {
            int64_t elapsed_us = esp_timer_get_time() - recording_start_time;
            float fps = (elapsed_us > 0) ?
                        (s_writer_frame_count * 1000000.0f / elapsed_us) : 0;
            ESP_LOGI(TAG, "Recording: %lu frames, %.1f fps",
                     (unsigned long)s_writer_frame_count, fps);
        }

        xQueueSend(s_free_queue, &slot, portMAX_DELAY);
    }

    s_writer_task = nullptr;
    vTaskDelete(nullptr);
}

/* ---- OFD task (Core 0, priority 3) ---- */
// Receives a uint32_t frame-number signal from the writer task.
// The writer has already downsampled the frame into ofd_gray before signaling,
// so this task only runs the block-matching computation (the expensive part).
// ofd_gray is safe to read here: writer writes it once every OFD_EVERY_N frames
// (~800ms at 6 fps) and this computation completes in ~20ms — no overlap.
// Signal value 0 = shutdown sentinel.
static void ofd_task(void *arg)
{
    for (;;) {
        uint32_t sig;
        xQueueReceive(s_ofd_queue, &sig, portMAX_DELAY);
        if (!sig) break;  // 0 = shutdown sentinel

        if (ofd_gray) {
            ofd_result_t r = ofd_process_gray(ofd_gray);
            if (r.valid) s_last_ofd = r;
        }
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

    s_write_queue = xQueueCreate(FRAME_RING_SLOTS, sizeof(frame_slot_t *));
    s_free_queue  = xQueueCreate(FRAME_RING_SLOTS, sizeof(frame_slot_t *));
    for (int i = 0; i < FRAME_RING_SLOTS; i++) {
        s_slots[i].data = (uint8_t *)heap_caps_malloc(FRAME_SLOT_SIZE, MALLOC_CAP_SPIRAM);
        if (!s_slots[i].data) {
            ESP_LOGE(TAG, "PSRAM slot %d alloc failed (%d bytes)", i, FRAME_SLOT_SIZE);
        } else {
            frame_slot_t *p = &s_slots[i];
            xQueueSend(s_free_queue, &p, 0);
        }
    }
    ESP_LOGI(TAG, "Frame ring: %d x %d bytes PSRAM", FRAME_RING_SLOTS, FRAME_SLOT_SIZE);

    ofd_gray = (uint8_t *)heap_caps_malloc(OFD_W * OFD_H, MALLOC_CAP_SPIRAM);
    if (!ofd_gray) {
        ESP_LOGE(TAG, "OFD gray buffer alloc failed");
    }
    ofd_init(OFD_W, OFD_H);
    ESP_LOGI(TAG, "OFD init (%dx%d)", OFD_W, OFD_H);
}

/** Returns true while a recording session is active. */
bool is_recording(void) { return recording; }

/** Return the latest OFD result from the background ofd_task. */
ofd_result_t video_rec_last_ofd(void) { return s_last_ofd; }

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
        size_t len = (f->length <= FRAME_SLOT_SIZE) ? f->length : FRAME_SLOT_SIZE;
        memcpy(slot->data, f->data, len);
        slot->length        = len;
        slot->capture_ts_ms = (esp_timer_get_time() - recording_start_time) / 1000;
        camera_return_frame(f);   // release camera buffer ASAP after memcpy

        if (xQueueSend(s_write_queue, &slot, 0) != pdTRUE) {
            ESP_LOGD(TAG, "write_queue full, frame dropped");
            xQueueSend(s_free_queue, &slot, 0);
        }
    } else {
        camera_return_frame(f);
        ESP_LOGD(TAG, "No free slot, frame dropped");
    }
}