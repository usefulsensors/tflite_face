#ifndef INCLUDE_TFLITE_MAIN_H
#define INCLUDE_TFLITE_MAIN_H

#include <stdbool.h>

typedef struct RectfStruct
{
    float min_x;
    float min_y;
    float max_x;
    float max_y;
} Rectf;

#define NUM_KEYPOINTS_PER_BOX (6)

#define KP_LEFT_EYE (0)
#define KP_RIGHT_EYE (1)
#define KP_NOSE (2)
#define KP_MOUTH (3)
#define KP_LEFT_EAR (4)
#define KP_RIGHT_EAR (5)

typedef struct DetectionStruct
{
    Rectf rect;
    float score;
    float keypoints[NUM_KEYPOINTS_PER_BOX * 2];
} Detection;

void *tflite_main(void *cookie);

bool get_detections(Detection **detections, int *detections_count);

#endif // INCLUDE_TFLITE_MAIN_H