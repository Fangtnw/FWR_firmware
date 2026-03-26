#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float divergence;   // average divergence (positive => approaching / expansion)
    float tau;          // time-to-contact proxy (bigger is safer). tau ~ 1/div
    float vx_mean;      // mean horizontal flow
    float vy_mean;      // mean vertical flow
    float lr_balance;   // (right expansion - left expansion) for turning
    bool  valid;        // false if not enough texture/matches
} ofd_result_t;

void ofd_init(int w, int h);
ofd_result_t ofd_process_gray(const uint8_t* gray); // gray image size w*h
void ofd_reset(void);

#ifdef __cplusplus
}
#endif
