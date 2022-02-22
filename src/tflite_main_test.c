#include "acutest.h"

#include "tflite_main.c"

#include <stdlib.h>

#include "test_image_192.h"

void test_decode_boxes()
{
    cvector_vector_type(Anchor) face_anchors = NULL;
    generate_face_anchors(&face_anchors);

    const int num_boxes = 2304;
    const int num_coords = 16;

    float *coords_data = calloc(1, num_boxes * num_coords * sizeof(float));

    const int test_index = 889;
    float *test_coords = &coords_data[test_index * num_coords];
    test_coords[0] = -0.01f; // Y origin.
    test_coords[1] = -0.02f; // X origin.
    test_coords[2] = 0.1f;   // Height.
    test_coords[3] = 0.2f;   // Width.

    cvector_vector_type(float) boxes;
    decode_face_boxes(coords_data, face_anchors, &boxes);
    cvector_free(face_anchors);
    free(coords_data);

    const float *test_box = &boxes[test_index * num_coords];
    TRACE_FLT(test_box[0]);
    TRACE_FLT(test_box[1]);
    TRACE_FLT(test_box[2]);
    TRACE_FLT(test_box[3]);

    cvector_free(boxes);
}

void test_end_to_end()
{
    const char *model_filename = "models/face_detection_full_range.tflite";
    TfLiteInterpreter *interpreter = init_interpreter(model_filename);

    TfLiteTensor *input_tensor = NULL;
    int input_width;
    int input_height;
    size_t input_byte_count;
    const TfLiteTensor *coords_tensor = NULL;
    float *coords_data = NULL;
    size_t coords_byte_count;
    const TfLiteTensor *score_tensor = NULL;
    float *score_data = NULL;
    size_t score_byte_count;
    init_tensors(interpreter,
                 &input_tensor, &input_width, &input_height, &input_byte_count,
                 &coords_tensor, &coords_data, &coords_byte_count,
                 &score_tensor, &score_data, &score_byte_count);

    cvector_vector_type(Anchor) anchors = NULL;
    generate_face_anchors(&anchors);

    cvector_vector_type(Detection) detections = NULL;
    run_model(interpreter, test_image_192_data, anchors, input_tensor, input_byte_count,
              coords_tensor, coords_data, coords_byte_count,
              score_tensor, score_data, score_byte_count,
              &detections);

    cvector_free(anchors);

    for (int i = 0; i < cvector_size(detections); ++i)
    {
        Detection *detection = &detections[i];
        TRACE_FLT(detection->score);
        TRACE_FLT(detection->rect.min_x);
        TRACE_FLT(detection->rect.min_y);
        TRACE_FLT(detection->rect.max_x);
        TRACE_FLT(detection->rect.max_y);
        for (int k = 0; k < NUM_KEYPOINTS_PER_BOX; ++k)
        {
            TRACE_FLT(detection->keypoints[k * 2]);
            TRACE_FLT(detection->keypoints[(k * 2) + 1]);
        }
    }

    cvector_free(detections);

    free(score_data);
    free(coords_data);

    TfLiteInterpreterDelete(interpreter);
}

TEST_LIST = {
    {"decode_boxes", test_decode_boxes},
    {"end_to_end", test_end_to_end},
    {NULL, NULL},
};