#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void fwr_control_init(void);
void fwr_control_start(void);

/** OFD obstacle avoidance: add offsets to aileron (roll) and rudder (yaw).
 *  Called from main when OFD detects approaching obstacle.
 *  Units: same as SBUS-derived aileron/rudder (microsecond offset, ~±300 = ±30 deg).
 *  Pass 0,0 when no avoidance needed. */
void fwr_set_ofd_avoidance(int aileron_offset_us, int rudder_offset_us);

#ifdef __cplusplus
}
#endif
