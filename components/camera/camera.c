#include "camera.h"
#include "lcd.h"

#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"

#include "example_video_common.h"

#ifndef v4l2_fourcc
#define v4l2_fourcc(a,b,c,d) ((uint32_t)(a) | ((uint32_t)(b)<<8) | ((uint32_t)(c)<<16) | ((uint32_t)(d)<<24))
#endif

#define TAG "CAM_LCD"
#define BUFFER_COUNT 6
#define CAMERA_DEBUG_PROBE_RGB565 1

typedef struct {
    void   *addr;
    size_t  length;
} mmap_buf_t;

static int        s_fd      = -1;
static int        s_type    = V4L2_BUF_TYPE_VIDEO_CAPTURE;
static int        s_stride  = 0;
static int        s_src_w   = 0;
static int        s_src_h   = 0;
static uint32_t   s_pixfmt  = 0;

static mmap_buf_t s_bufs[BUFFER_COUNT] = {0};
static camera_sensor_t s_sensor = CAMERA_OV5647;
static int s_req_w = 0;  // 0 = use sensor default
static int s_req_h = 0;


/* ---------------------------- SENSOR SELECT ---------------------------- */
void camera_set_sensor(camera_sensor_t sensor)
{
    s_sensor = sensor;
}

void camera_set_resolution(int width, int height)
{
    s_req_w = width;
    s_req_h = height;
}


/* ---------------------------- CAMERA INIT ------------------------------ */
void camera_init(void)
{
    ESP_LOGI(TAG, "Initializing camera…");

    ESP_ERROR_CHECK(example_video_init());
    ESP_LOGI(TAG, "esp_video driver ready.");

    s_fd = open(EXAMPLE_CAM_DEV_PATH, O_RDWR);
    if (s_fd < 0) {
        ESP_LOGE(TAG, "Failed to open camera");
        return;
    }

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    /* Enumerate supported frame sizes for the current probe format. */
    {
        struct v4l2_frmsizeenum fse = {0};
        fse.pixel_format = CAMERA_DEBUG_PROBE_RGB565 ? v4l2_fourcc('R','G','B','P')
                                                     : v4l2_fourcc('Y','U','1','2');
        ESP_LOGI(TAG, "Supported %s output sizes:",
                 CAMERA_DEBUG_PROBE_RGB565 ? "RGB565" : "YUV420 (I420)");
        for (fse.index = 0; ioctl(s_fd, VIDIOC_ENUM_FRAMESIZES, &fse) == 0; fse.index++) {
            if (fse.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                ESP_LOGI(TAG, "  [%d] %dx%d", fse.index,
                         fse.discrete.width, fse.discrete.height);
            }
        }
    }

    if (s_req_w > 0 && s_req_h > 0) {
        fmt.fmt.pix.width  = s_req_w;
        fmt.fmt.pix.height = s_req_h;
    } else if (s_sensor == CAMERA_OV5647) {
        fmt.fmt.pix.width  = 800;
        fmt.fmt.pix.height = 640;
    } else {
        fmt.fmt.pix.width  = 1280;
        fmt.fmt.pix.height = 720;
    }

    // YUV420 planar (I420): ISP demosaics RAW10 directly to YUV in hardware.
    // I420 layout: Y plane (w×h) | U plane (w/2 × h/2) | V plane (w/2 × h/2)
    fmt.fmt.pix.pixelformat = CAMERA_DEBUG_PROBE_RGB565 ? v4l2_fourcc('R','G','B','P')
                                                        : v4l2_fourcc('Y','U','1','2');

    if (ioctl(s_fd, VIDIOC_S_FMT, &fmt) != 0) {
        ESP_LOGE(TAG, "Failed to set camera format (%dx%d) — falling back to 800x640",
                 fmt.fmt.pix.width, fmt.fmt.pix.height);
        fmt.fmt.pix.width  = 800;
        fmt.fmt.pix.height = 640;
        if (ioctl(s_fd, VIDIOC_S_FMT, &fmt) != 0) {
            ESP_LOGE(TAG, "Fallback 800x640 also failed — camera unavailable");
            close(s_fd);
            return;
        }
    }

    ioctl(s_fd, VIDIOC_G_FMT, &fmt);

    s_src_w  = fmt.fmt.pix.width;
    s_src_h  = fmt.fmt.pix.height;
    s_stride = fmt.fmt.pix.bytesperline ? fmt.fmt.pix.bytesperline : s_src_w;
    s_pixfmt = fmt.fmt.pix.pixelformat;

    ESP_LOGI(TAG, "Camera active: %dx%d stride=%d pixfmt=%c%c%c%c (0x%08x)",
             s_src_w, s_src_h, s_stride,
             (char)(s_pixfmt & 0xFF), (char)((s_pixfmt >> 8) & 0xFF),
             (char)((s_pixfmt >> 16) & 0xFF), (char)((s_pixfmt >> 24) & 0xFF),
             (unsigned)s_pixfmt);

    struct v4l2_requestbuffers req = {
        .count  = BUFFER_COUNT,
        .type   = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .memory = V4L2_MEMORY_MMAP
    };
    ioctl(s_fd, VIDIOC_REQBUFS, &req);

    for (int i = 0; i < req.count; i++) {
        struct v4l2_buffer buf = {
            .type = req.type,
            .memory = req.memory,
            .index = i
        };
        ioctl(s_fd, VIDIOC_QUERYBUF, &buf);

        s_bufs[i].addr = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                              MAP_SHARED, s_fd, buf.m.offset);
        s_bufs[i].length = buf.length;

        ioctl(s_fd, VIDIOC_QBUF, &buf);
    }

    ioctl(s_fd, VIDIOC_STREAMON, &s_type);
    ESP_LOGI(TAG, "Camera stream started.");
}


/* ---------------------------- GET FRAME ------------------------------ */
camera_frame_t camera_get_frame(void)
{
    camera_frame_t f = {0};

    struct v4l2_buffer buf = {
        .type   = s_type,
        .memory = V4L2_MEMORY_MMAP
    };

    if (ioctl(s_fd, VIDIOC_DQBUF, &buf) != 0)
        return f;

    f.data   = (uint8_t*)s_bufs[buf.index].addr;
    f.length = buf.bytesused;
    f.index  = buf.index;

    return f;
}


/* ---------------------------- RETURN FRAME ------------------------------ */
void camera_return_frame(camera_frame_t *f)
{
    struct v4l2_buffer buf = {
        .type   = s_type,
        .memory = V4L2_MEMORY_MMAP,
        .index  = f->index
    };
    ioctl(s_fd, VIDIOC_QBUF, &buf);
}

int camera_get_stride(void)
{
    return s_stride;
}

uint32_t camera_get_pixelformat(void)
{
    return s_pixfmt;
}


/* --------------------------- RGB565 CROP TO LCD -------------------------- */
void crop_to_lcd(const uint16_t *src, uint16_t *dst)
{
    if (!src || !dst) return;

    const int src_w = 1920;
    const int src_h = 1080;

    const int dst_w = 720;
    const int dst_h = 1280;

    const int bpp = 2;  // RGB565

    // Horizontal crop (800 -> 720)
    int crop_x = (src_w - dst_w) / 2;  // = 40 px

    // Vertical centering pad (1280 LCD height - 640 cam height)
    int pad_top = (dst_h - src_h) / 2;  // = 320 px
    int pad_bottom = dst_h - pad_top - src_h;

    // 1) Fill entire LCD with black
    memset(dst, 0, dst_w * dst_h * bpp);

    // 2) Copy camera image into center of LCD
    for (int y = 0; y < src_h; y++) {

        const uint16_t *src_line = src + (y * src_w) + crop_x;
        uint16_t *dst_line = dst + ((y + pad_top) * dst_w);

        memcpy(dst_line, src_line, dst_w * bpp);
    }
}
