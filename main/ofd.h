#pragma once
#include <stdint.h>
#include "ofd_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // Raw optical-flow outputs (set by ofd_process_gray):
    float divergence;       // average divergence (positive => approaching / expansion)
    float tau;              // raw time-to-contact proxy: tau ~ 1/div (diagnostic only)
    float vx_mean;          // mean horizontal flow (raw)
    float vy_mean;          // mean vertical flow (raw, sign flipped for roll=-180° mount)
    float lr_balance;       // (right expansion - left expansion) for turning
    float mean_flow_mag;    // mean derotated flow magnitude (pixels/frame) — PRIMARY signal
    float mean_flow_mag_raw;// mean raw flow magnitude before derotation (diagnostic)
    int   flow_cnt;         // number of valid flow matches (diagnostic)
    int   div_cnt;          // number of valid divergence terms (diagnostic)
    bool  valid;            // false if not enough texture/matches

    // Filter-layer outputs (set by ofd_task, not ofd_process_gray):
    float ema_div;          // bias-corrected, EMA-smoothed divergence
    float ema_lr;           // EMA-smoothed lr_balance
    float ema_flow_mag;     // EMA-smoothed derotated flow magnitude
    float tau_ms;           // time-to-contact in ms (dt-clamped, from EMA signal) — logging only
    float turn_cmd;         // normalized turn command [-1, 1]
    int   evasion_level;    // OFD_EVADE_* constants
    bool  looming_detected; // flow-magnitude-primary trigger output
    bool  az_quiet;         // wing-sync gate: az near gravity-only quiet point?
} ofd_result_t;

void ofd_init(int w, int h);
/** Process grayscale frame with gyro-based derotation.
 *  gx/gy/gz_dps: angular velocity in deg/s from IMU (body frame). */
ofd_result_t ofd_process_gray(const uint8_t* gray,
                              float gx_dps, float gy_dps, float gz_dps);
void ofd_reset(void);

#ifdef __cplusplus
}
#endif
