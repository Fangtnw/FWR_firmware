#pragma once
/**
 * video_rec.h — Public API for the video recording module (main_fwr_video.cpp)
 *
 * Provides SD card recording with OFD sidecar CSV.
 * Call video_rec_init() once at startup (before camera_init),
 * then start_recording() / stop_recording() to control recording.
 */

#include "camera.h"
#include "ofd.h"

#ifdef __cplusplus
extern "C" {
#endif

/** IMU sample logged alongside OFD data in the sidecar CSV. */
typedef struct {
    float ax, ay, az;       // acceleration (g)
    float gx, gy, gz;       // angular velocity (deg/s)
    float roll, pitch, yaw; // angle (deg)
} imu_data_t;

/** Push the latest IMU reading into the recording module for CSV logging. */
void video_rec_set_imu(imu_data_t d);

/** Init SD card, PSRAM ring buffer, and OFD engine. Call once at startup. */
void video_rec_init(void);

/** Start recording to the next V{id}.VID + V{id}.CSV on the SD card. */
void start_recording(void);

/** Stop recording, flush and close files, patch header. */
void stop_recording(void);

/** Returns true while a recording session is active. */
bool is_recording(void);

/**
 * Copy a camera frame into the PSRAM ring buffer and hand it to the writer task.
 * Always calls camera_return_frame(f) internally (even when not recording).
 */
void video_rec_enqueue(camera_frame_t *f);

/** Return the latest OFD result computed by the recording module's ofd_task. */
ofd_result_t video_rec_last_ofd(void);

#ifdef __cplusplus
}
#endif
