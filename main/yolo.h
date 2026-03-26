#pragma once
#include <stdint.h>
#include <vector>
#include <array>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize YOLO model (load .espdl from flash)
void yolo_init(void);

// Run YOLO inference and output decoded boxes
// boxes: vector of {x1, y1, x2, y2, score}
// return: number of boxes
int yolo_run(uint8_t *rgb_frame,
             int width,
             int height,
             std::vector<std::array<float,5>> &boxes);

#ifdef __cplusplus
}
#endif
