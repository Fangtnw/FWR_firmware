#include "ofd.h"
#include "esp_log.h"
#include <string.h>
#include <math.h>
#include <climits>

static const char *TAG = "OFD";

// ---- Tunables (speed/robustness tradeoffs) ----
static constexpr int GRID_STEP = 12;     // spacing between flow points (pixels)
static constexpr int BLK_R     = 4;      // block radius => block size = (2R+1)^2
static constexpr int SRCH_R    = 6;      // search radius in prev frame (pixels)
static constexpr int MIN_TEX   = 25;     // min texture threshold for a point
static constexpr int MAX_SAD   = 4500;   // reject match if SAD too large
static constexpr float EPS_DIV = 1e-6f;

static int gW = 0, gH = 0;

// Previous grayscale frame buffer (owned by OFD)
static uint8_t *gPrev = nullptr;
static bool gHavePrev = false;

// Helper: sum abs diff for two blocks centered at (x,y) in cur and (x+dx,y+dy) in prev
static inline int sad_block(const uint8_t* cur, const uint8_t* prev,
                            int x, int y, int dx, int dy, int w)
{
    int sad = 0;
    const int x2 = x + dx;
    const int y2 = y + dy;

    // block boundaries assumed valid by caller
    for (int j = -BLK_R; j <= BLK_R; ++j) {
        const uint8_t* pc = cur  + (y + j)  * w + (x);
        const uint8_t* pp = prev + (y2 + j) * w + (x2);
        for (int i = -BLK_R; i <= BLK_R; ++i) {
            sad += abs(int(pc[i]) - int(pp[i]));
        }
    }
    return sad;
}

// Helper: simple texture measure (sum of abs gradients in block)
static inline int texture_block(const uint8_t* img, int x, int y, int w)
{
    int tex = 0;
    for (int j = -BLK_R; j <= BLK_R; ++j) {
        const uint8_t* p = img + (y + j) * w + x;
        for (int i = -BLK_R; i <= BLK_R; ++i) {
            int gx = int(p[i+1]) - int(p[i-1]);
            int gy = int(p[i+w]) - int(p[i-w]);
            tex += abs(gx) + abs(gy);
        }
    }
    // normalize a bit
    return tex / ((2*BLK_R+1)*(2*BLK_R+1));
}

void ofd_init(int w, int h)
{
    gW = w;
    gH = h;

    // allocate prev frame once (static-ish)
    // NOTE: for ESP-IDF you can replace malloc with heap_caps_malloc if needed.
    if (gPrev) {
        free(gPrev);
        gPrev = nullptr;
    }
    gPrev = (uint8_t*)malloc(gW * gH);
    if (!gPrev) {
        ESP_LOGE(TAG, "Failed to allocate prev buffer (%d bytes)", gW * gH);
    }
    gHavePrev = false;

    ESP_LOGI(TAG, "OFD init: %dx%d, GRID_STEP=%d BLK_R=%d SRCH_R=%d",
             gW, gH, GRID_STEP, BLK_R, SRCH_R);
}

void ofd_reset(void)
{
    gHavePrev = false;
}

ofd_result_t ofd_process_gray(const uint8_t* cur)
{
    ofd_result_t out{};
    out.valid = false;

    if (!gPrev || !cur || gW <= 0 || gH <= 0) return out;

    // First frame: just store and return invalid
    if (!gHavePrev) {
        memcpy(gPrev, cur, gW * gH);
        gHavePrev = true;
        return out;
    }

    // We estimate flow on a sparse grid and then compute divergence:
    // div = d(u)/dx + d(v)/dy.
    // For speed, we approximate with finite differences on grid neighbors.

    // Store flows for one row of grid to compute dy terms with previous row
    // We keep arrays sized for max number of grid points across width.
    const int margin = BLK_R + SRCH_R + 2;
    const int x0 = margin;
    const int x1 = gW - margin - 1;
    const int y0 = margin;
    const int y1 = gH - margin - 1;

    if (x1 <= x0 || y1 <= y0) {
        memcpy(gPrev, cur, gW * gH);
        return out;
    }

    const int maxCols = (x1 - x0) / GRID_STEP + 1;
    static int16_t u_prev_row[256];
    static int16_t v_prev_row[256];
    static bool    ok_prev_row[256];

    if (maxCols > 256) {
        // very unlikely for your small OFD size, but safe
        ESP_LOGW(TAG, "Too many columns for static buffers: %d", maxCols);
        memcpy(gPrev, cur, gW * gH);
        return out;
    }

    // Clear previous row validity at start
    for (int c = 0; c < maxCols; ++c) ok_prev_row[c] = false;

    float div_sum = 0.0f;
    float vx_sum  = 0.0f;
    float vy_sum  = 0.0f;
    int   div_cnt  = 0;
    int   flow_cnt = 0;

    // Left/right expansion measure for steering
    float div_left_sum = 0.0f;
    float div_right_sum = 0.0f;
    int   left_cnt = 0, right_cnt = 0;

    int rowIdx = 0;
    for (int y = y0; y <= y1; y += GRID_STEP, ++rowIdx) {
        int colIdx = 0;

        // store current row to compute dy with next row
        int16_t u_cur_row[256];
        int16_t v_cur_row[256];
        bool    ok_cur_row[256];

        for (int x = x0; x <= x1; x += GRID_STEP, ++colIdx) {
            ok_cur_row[colIdx] = false;
            u_cur_row[colIdx] = 0;
            v_cur_row[colIdx] = 0;

            // texture gate
            int tex = texture_block(cur, x, y, gW);
            if (tex < MIN_TEX) continue;

            // block matching: search best (dx,dy) in prev for block at (x,y) in cur
            int bestSad = INT_MAX;
            int bestDx = 0, bestDy = 0;

            for (int dy = -SRCH_R; dy <= SRCH_R; ++dy) {
                for (int dx = -SRCH_R; dx <= SRCH_R; ++dx) {
                    int sad = sad_block(cur, gPrev, x, y, dx, dy, gW);
                    if (sad < bestSad) {
                        bestSad = sad;
                        bestDx = dx;
                        bestDy = dy;
                    }
                }
            }

            if (bestSad > MAX_SAD) continue;

            // Flow: from prev to cur, if best match in prev is at (x+dx,y+dy),
            // then motion in image is approx (-dx, -dy) (prev -> cur).
            int16_t u = (int16_t)(-bestDx);
            int16_t v = (int16_t)(-bestDy);

            ok_cur_row[colIdx] = true;
            u_cur_row[colIdx] = u;
            v_cur_row[colIdx] = v;

            vx_sum += (float)u;
            vy_sum += (float)v;
            flow_cnt++;
        }

        // Compute divergence on grid cells where we have neighbors:
        // For each valid point (x,y), if right neighbor exists => du/dx approx (uR-u)/step
        // If bottom neighbor exists (from prev_row buffers) => dv/dy approx (v - vPrevRow)/step
        // We accumulate where we have terms.
        for (int c = 0; c < colIdx; ++c) {
            if (!ok_cur_row[c]) continue;

            float div_here = 0.0f;
            bool have_term = false;

            // du/dx using right neighbor
            if (c + 1 < colIdx && ok_cur_row[c + 1]) {
                div_here += (float)(u_cur_row[c + 1] - u_cur_row[c]) / (float)GRID_STEP;
                have_term = true;
            }

            // dv/dy using previous row (above)
            if (ok_prev_row[c] && rowIdx > 0) {
                div_here += (float)(v_cur_row[c] - v_prev_row[c]) / (float)GRID_STEP;
                have_term = true;
            }

            if (have_term) {
                div_sum += div_here;
                div_cnt++;

                // steering split
                // left/right based on x position; use c index against center
                if (c < (colIdx / 2)) {
                    div_left_sum += div_here;
                    left_cnt++;
                } else {
                    div_right_sum += div_here;
                    right_cnt++;
                }
            }
        }

        // shift current row -> prev row for next iteration
        for (int c = 0; c < colIdx; ++c) {
            u_prev_row[c] = u_cur_row[c];
            v_prev_row[c] = v_cur_row[c];
            ok_prev_row[c] = ok_cur_row[c];
        }
    }

    // Update prev frame
    memcpy(gPrev, cur, gW * gH);

    if (div_cnt < 10) {
        // not enough reliable measurements
        return out;
    }

    const float div_avg = div_sum / (float)div_cnt;

    out.divergence = div_avg;
    out.vx_mean = (flow_cnt > 0) ? (vx_sum / (float)flow_cnt) : 0.0f;
    out.vy_mean = (flow_cnt > 0) ? (vy_sum / (float)flow_cnt) : 0.0f;
    out.flow_cnt = flow_cnt;
    out.div_cnt  = div_cnt;

    // time-to-contact proxy: tau ≈ 1/div (only meaningful if div>0)
    if (div_avg > EPS_DIV) out.tau = 1.0f / div_avg;
    else out.tau = 1e6f;

    float left_avg  = (left_cnt  > 0) ? (div_left_sum  / left_cnt)  : 0.0f;
    float right_avg = (right_cnt > 0) ? (div_right_sum / right_cnt) : 0.0f;
    out.lr_balance = (right_avg - left_avg);

    out.valid = true;
    return out;
}
