#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CAMERA_OV5647,
    CAMERA_SC2336
} camera_sensor_t;


typedef struct {
    uint8_t *data;   
    size_t   length; // bytes used
    int      index;  // V4L2 buffer index
} camera_frame_t;

void camera_set_sensor(camera_sensor_t sensor);
void camera_set_resolution(int width, int height);  // call before camera_init()
void camera_init(void);
camera_frame_t camera_get_frame(void);
void camera_return_frame(camera_frame_t *frame);
int camera_get_stride(void);
uint32_t camera_get_pixelformat(void);

/**
 * Crop center of camera frame (1280x720 RGB888) to LCD size (1024x600)
 * and store into dst (LCD_WIDTH * LCD_HEIGHT * 3 bytes).
 */
void crop_to_lcd(const uint16_t *src, uint16_t *dst);


#ifdef __cplusplus
}
#endif
