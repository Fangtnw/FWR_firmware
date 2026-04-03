#pragma once
/**
 * ofd_config.h — OFD tuning constants derived from V0000.CSV baseline analysis
 * (1536 frames, 47s, roll_mean = -169°, az_mean = -0.986g)
 */

// Bias / filtering
#define OFD_DIV_BIAS         (-0.018f)   // measured DC offset from baseline frames
#define OFD_EMA_ALPHA        (0.30f)     // reduces noise floor σ: 0.022 → 0.013
#define OFD_MIN_DIV_CNT      (10)        // reject frames with too few divergence terms

// Dual-gate trigger thresholds
#define OFD_DIV_THRESHOLD    (0.05f)     // |EMA div| gate for looming detection
#define OFD_TAU_EVADE_MS     (30.0f)     // immediate evasion threshold (ms)
#define OFD_TAU_BRAKE_MS     (50.0f)     // dual-gate outer / brake threshold (ms)
#define OFD_TAU_ALERT_MS     (100.0f)    // alert threshold (ms) — for future use
#define OFD_TAU_MAX          (1000.0f)   // sentinel: no obstacle detected

// dt clamping — kills 1106ms frame-interval spikes that corrupt τ
#define OFD_DT_MIN_MS        (8.0f)      // minimum dt clamp (ms)
#define OFD_DT_MAX_MS        (80.0f)     // maximum dt clamp (ms)

// Wing-sync gate (az-based, az mean = -0.986g from V0000.CSV)
#define OFD_AZ_QUIET_CENTER  (-0.986f)   // g, gravity-only reference point
#define OFD_AZ_QUIET_BAND    (0.15f)     // g, ±band around quiet center

// Steering gain
#define OFD_LR_GAIN          (3.0f)      // lr_balance → turn_cmd gain (tune empirically)

// Camera orientation flag
#define OFD_CAMERA_ROLL_180  (1)         // camera mounted at roll ≈ -180°; FoE Y-axis flipped

// Evasion level values (stored as int in ofd_result_t.evasion_level)
#define OFD_EVADE_NONE   0
#define OFD_EVADE_ALERT  1
#define OFD_EVADE_BRAKE  2
#define OFD_EVADE_EVADE  3
