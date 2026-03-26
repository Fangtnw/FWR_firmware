#include "yolo.h"
#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_timer.h"

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

static const char *TAG = "ESPDET";

// ✅ update this symbol to match your .espdl name
extern const uint8_t bas_start[] asm("_binary_bas_espdl_start");

__attribute__((section(".ext_ram.bss")))
static dl::Model *model = nullptr;

// --------------------
// CONFIG (match training/export)
// --------------------
#define INPUT_W 160
#define INPUT_H 160

#define CONF_THRESH 0.50f
#define IOU_THRESH  0.45f
#define MAX_BOXES   50     // keep more before NMS (then trim)
#define MAX_FINAL   10     // final boxes after NMS

// --------------------
// Utils
// --------------------
static inline float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float clampf(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

// Nearest resize RGB888 -> model RGB888
static void resize_to_input(const uint8_t *src, int sw, int sh, uint8_t *dst) {
    const float rx = (float)sw / (float)INPUT_W;
    const float ry = (float)sh / (float)INPUT_H;

    for (int y = 0; y < INPUT_H; y++) {
        int sy = (int)(y * ry);
        for (int x = 0; x < INPUT_W; x++) {
            int sx = (int)(x * rx);
            int si = (sy * sw + sx) * 3;
            int di = (y * INPUT_W + x) * 3;
            dst[di + 0] = src[si + 0];
            dst[di + 1] = src[si + 1];
            dst[di + 2] = src[si + 2];
        }
    }
}

static float iou_xyxy(const std::array<float,5> &A, const std::array<float,5> &B) {
    float x1 = std::max(A[0], B[0]);
    float y1 = std::max(A[1], B[1]);
    float x2 = std::min(A[2], B[2]);
    float y2 = std::min(A[3], B[3]);

    float iw = std::max(0.0f, x2 - x1);
    float ih = std::max(0.0f, y2 - y1);
    float inter = iw * ih;

    float areaA = std::max(0.0f, A[2]-A[0]) * std::max(0.0f, A[3]-A[1]);
    float areaB = std::max(0.0f, B[2]-B[0]) * std::max(0.0f, B[3]-B[1]);
    return inter / (areaA + areaB - inter + 1e-6f);
}

static void nms_inplace(std::vector<std::array<float,5>> &boxes) {
    std::sort(boxes.begin(), boxes.end(),
              [](const auto &a, const auto &b){ return a[4] > b[4]; });

    std::vector<std::array<float,5>> keep;
    keep.reserve(boxes.size());

    for (size_t i = 0; i < boxes.size(); i++) {
        bool ok = true;
        for (size_t j = 0; j < keep.size(); j++) {
            if (iou_xyxy(boxes[i], keep[j]) > IOU_THRESH) {
                ok = false;
                break;
            }
        }
        if (ok) keep.push_back(boxes[i]);
        if (keep.size() >= MAX_FINAL) break;
    }
    boxes.swap(keep);
}

// Decode one scale: box=(1,4,H,W), conf=(1,1,H,W)
// box format assumed: [l,t,r,b] distances from cell center in INPUT pixels
static void decode_scale(
    dl::TensorBase *box_t,
    dl::TensorBase *conf_t,
    int sw, int sh,
    std::vector<std::array<float,5>> &out)
{
    const int H = box_t->shape[2];
    const int W = box_t->shape[3];
    const float stride = (float)INPUT_W / (float)W;

    // ⚠️ IMPORTANT:
    // This assumes tensors are float32.
    // If your model outputs int8/uint8, reading as float will explode and you get "best=1.00 always" or giant numbers.
    float *box  = (float*)box_t->data;
    float *conf = (float*)conf_t->data;

    const int HW = H * W;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;

            float c = sigmoidf_fast(conf[idx]);
            if (c < CONF_THRESH) continue;

            float l = box[0*HW + idx];
            float t = box[1*HW + idx];
            float r = box[2*HW + idx];
            float b = box[3*HW + idx];

            float cx = (x + 0.5f) * stride;
            float cy = (y + 0.5f) * stride;

            // map from INPUT space -> current image space
            float x1 = (cx - l) * (float)sw / (float)INPUT_W;
            float y1 = (cy - t) * (float)sh / (float)INPUT_H;
            float x2 = (cx + r) * (float)sw / (float)INPUT_W;
            float y2 = (cy + b) * (float)sh / (float)INPUT_H;

            x1 = clampf(x1, 0.0f, (float)sw - 1.0f);
            y1 = clampf(y1, 0.0f, (float)sh - 1.0f);
            x2 = clampf(x2, 0.0f, (float)sw - 1.0f);
            y2 = clampf(y2, 0.0f, (float)sh - 1.0f);

            // basic sanity: skip inverted/zero boxes
            if (x2 <= x1 || y2 <= y1) continue;

            out.push_back({x1,y1,x2,y2,c});
        }
    }
}

// --------------------
// init
// --------------------
extern "C" void yolo_init(void)
{
    model = new dl::Model((const char *)bas_start,
                          fbs::MODEL_LOCATION_IN_FLASH_RODATA);

    auto in  = model->get_inputs().begin()->second;
    auto outs = model->get_outputs();

    ESP_LOGI(TAG, "Input  = [%d %d %d %d]",
        in->shape[0], in->shape[1], in->shape[2], in->shape[3]);

    // log outputs count + shapes
    int k = 0;
    for (auto &kv : outs) {
        auto *t = kv.second;
        ESP_LOGI(TAG, "Out[%d] %s = [%d %d %d %d]",
                 k, kv.first.c_str(), t->shape[0], t->shape[1], t->shape[2], t->shape[3]);
        k++;
    }
}

// --------------------
// run
// --------------------
extern "C" int yolo_run(uint8_t *rgb, int sw, int sh,
                        std::vector<std::array<float,5>> &boxes)
{
    boxes.clear();
    if (!model) return 0;

    // 1) preprocess -> model input
    auto tin = model->get_inputs().begin()->second;
    resize_to_input(rgb, sw, sh, (uint8_t*)tin->data);

    // 2) inference timing
    int64_t t0 = esp_timer_get_time();
    model->run();
    int64_t t1 = esp_timer_get_time();
    int infer_ms = (int)((t1 - t0) / 1000);

    // 3) decode all scales (expect 6 outputs: box32, conf32, box16, conf16, box8, conf8)
    auto outs = model->get_outputs();
    if (outs.size() < 2) {
        ESP_LOGW(TAG, "No outputs?");
        return 0;
    }

    // If ordering ever changes, you must map by tensor name.
    auto it = outs.begin();

    std::vector<std::array<float,5>> raw;
    raw.reserve(200);

    // decode pairs in-order
    while (true) {
        if (it == outs.end()) break;
        dl::TensorBase *box_t = it->second; ++it;
        if (it == outs.end()) break;
        dl::TensorBase *conf_t = it->second; ++it;

        // quick shape guard: box channels should be 4, conf channels should be 1
        if (box_t->shape[1] != 4 || conf_t->shape[1] != 1) {
            ESP_LOGW(TAG, "Skip output pair (unexpected C): boxC=%d confC=%d",
                     box_t->shape[1], conf_t->shape[1]);
            continue;
        }

        decode_scale(box_t, conf_t, sw, sh, raw);

        if (raw.size() > MAX_BOXES) {
            // keep top MAX_BOXES by confidence early to reduce NMS work
            std::nth_element(raw.begin(), raw.begin() + MAX_BOXES, raw.end(),
                             [](const auto &a, const auto &b){ return a[4] > b[4]; });
            raw.resize(MAX_BOXES);
        }
    }

    if (raw.empty()) {
        ESP_LOGI(TAG, "Inference %d ms | boxes=0", infer_ms);
        return 0;
    }

    std::sort(raw.begin(), raw.end(),
              [](const auto &a, const auto &b){ return a[4] > b[4]; });

    // NMS -> final
    nms_inplace(raw);
    boxes = raw;

    ESP_LOGI(TAG, "Inference %d ms | boxes=%d | best=%.2f",
             infer_ms, (int)boxes.size(), boxes[0][4]);

    return (int)boxes.size();
}
