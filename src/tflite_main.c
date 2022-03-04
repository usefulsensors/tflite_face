#include "tflite_main.h"

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <pthread.h>
#include <unistd.h>

#include "tflite/c_api.h"

#include "app_main.h"
#include "capture_main.h"
#include "cvector.h"
#include "string_utils.h"
#include "trace.h"

typedef struct AnchorOptionsStruct
{
    int num_layers;
    float min_scale;
    float max_scale;
    int input_size_height;
    int input_size_width;
    float anchor_offset_x;
    float anchor_offset_y;
    int *strides;
    int strides_count;
    float *aspect_ratios;
    int aspect_ratios_count;
    bool fixed_anchor_size;
    float interpolated_scale_aspect_ratio;
    bool reduce_boxes_in_lowest_layer;
} AnchorOptions;

typedef struct AnchorStruct
{
    float x;
    float y;
    float w;
    float h;
} Anchor;

typedef struct DecodeBoxesOptionsStruct
{
    int num_classes;
    int num_boxes;
    int num_coords;

    int keypoint_coord_offset;
    int num_keypoints;
    int num_values_per_keypoint;
    int box_coord_offset;

    float x_scale;
    float y_scale;
    float w_scale;
    float h_scale;

    bool apply_exponential_on_box_size;

    bool reverse_output_order;

    bool sigmoid_score;
    float score_clipping_thresh;

    bool flip_vertically;
    float min_score_thresh;
} DecodeBoxesOptions;

typedef struct NonMaxSuppressionOptionsStruct
{
    int num_detection_streams;
    int max_num_detections;
    float min_score_threshold;
    float min_suppression_threshold;
    enum OverlapType
    {
        NMS_UNSPECIFIED_OVERLAP_TYPE = 0,
        NMS_JACCARD = 1,
        NMS_MODIFIED_JACCARD = 2,
        NMS_INTERSECTION_OVER_UNION = 3,
    } overlap_type;

    bool return_empty_detections;

    enum NmsAlgorithm
    {
        NMS_DEFAULT = 0,
        NMS_WEIGHTED = 1,
    } algorithm;

    int num_boxes;
    int num_coords;
    int keypoint_coord_offset;
    int num_keypoints;
    int num_values_per_keypoint;
    int box_coord_offset;
} NonMaxSuppressionOptions;

typedef struct IntFloatPairStruct
{
    int first;
    float second;
} IntFloatPair;

static const int box_count = 2304;
static const int coords_count = 16;
static const int num_coords_per_box = 4;

static pthread_mutex_t g_detections_mutex = PTHREAD_MUTEX_INITIALIZER;
static Detection *g_detections = NULL;
static int g_detections_count = -1;

static const char *signal_filename = "/tmp/signals/face_detections.txt";

static const char *tensor_type_name(TfLiteType type)
{
    const char *names[17] = {
        "None",
        "Float32",
        "Int32",
        "UInt8",
        "Int64",
        "String",
        "Bool",
        "Int16",
        "Complex64",
        "Int8",
        "Float16",
        "Float64",
        "Complex128",
        "UInt64",
        "Resource",
        "Variant",
        "UInt32",
    };
    if ((type < 0) || (type > 16))
    {
        return "<Out of range>";
    }
    else
    {
        return names[type];
    }
}

static void print_tensor_info(const TfLiteTensor *tensor)
{
    TfLiteType type = TfLiteTensorType(tensor);
    int32_t num_dims = TfLiteTensorNumDims(tensor);
    int32_t *dims = malloc(sizeof(int32_t) * num_dims);
    for (int i = 0; i < num_dims; ++i)
    {
        dims[i] = TfLiteTensorDim(tensor, i);
    }
    size_t byte_size = TfLiteTensorByteSize(tensor);
    void *data = TfLiteTensorData(tensor);
    const char *name = TfLiteTensorName(tensor);
    TfLiteQuantizationParams quant_params =
        TfLiteTensorQuantizationParams(tensor);

    fprintf(stderr, "-----Tensor------\n");
    fprintf(stderr, "Name: '%s'\n", name);
    fprintf(stderr, "Dims: [");
    for (int i = 0; i < num_dims; ++i)
    {
        fprintf(stderr, "%d, ", dims[i]);
    }
    fprintf(stderr, "]\n");
    fprintf(stderr, "Data: %p\n", data);
    fprintf(stderr, "Byte size: %zu\n", byte_size);
    fprintf(stderr, "Quant params: scale %f, zero point %d\n",
            quant_params.scale, quant_params.zero_point);

    free(dims);
}

static float calculate_scale(float min_scale, float max_scale, int stride_index,
                             int num_strides)
{
    if (num_strides == 1)
    {
        return (min_scale + max_scale) * 0.5f;
    }
    else
    {
        return min_scale +
               (max_scale - min_scale) * 1.0f * stride_index / (num_strides - 1.0f);
    }
}

static void print_anchor(Anchor *anchor)
{
    fprintf(stderr, "x: %.2f, y: %.2f, w: %.2f, h: %.2f\n", anchor->x,
            anchor->y, anchor->w, anchor->h);
}

static void generate_anchors(const AnchorOptions *options,
                             cvector_vector_type(Anchor) * anchors_out)
{
    cvector_vector_type(Anchor) anchors = NULL;
    if (options->strides_count != options->num_layers)
    {
        fprintf(stderr, "Strides count (%d) doesn't equal num_layers (%d).\n",
                options->strides_count, options->num_layers);
        return;
    }

    int layer_id = 0;
    while (layer_id < options->num_layers)
    {
        cvector_vector_type(float) anchor_height = NULL;
        cvector_vector_type(float) anchor_width = NULL;
        cvector_vector_type(float) aspect_ratios = NULL;
        cvector_vector_type(float) scales = NULL;
        int last_same_stride_layer = layer_id;

        while (last_same_stride_layer < options->strides_count &&
               options->strides[last_same_stride_layer] ==
                   options->strides[layer_id])
        {
            const float scale =
                calculate_scale(options->min_scale, options->max_scale,
                                last_same_stride_layer, options->strides_count);
            if (last_same_stride_layer == 0 &&
                options->reduce_boxes_in_lowest_layer)
            {
                // For first layer, it can be specified to use predefined anchors.
                cvector_push_back(aspect_ratios, 1.0f);
                cvector_push_back(aspect_ratios, 2.0f);
                cvector_push_back(aspect_ratios, 0.5f);
                cvector_push_back(scales, 0.1f);
                cvector_push_back(scales, scale);
                cvector_push_back(scales, scale);
            }
            else
            {
                for (int aspect_ratio_id = 0;
                     aspect_ratio_id < options->aspect_ratios_count;
                     ++aspect_ratio_id)
                {
                    cvector_push_back(aspect_ratios,
                                      options->aspect_ratios[aspect_ratio_id]);
                    cvector_push_back(scales, scale);
                }
                if (options->interpolated_scale_aspect_ratio > 0.0f)
                {
                    const float scale_next =
                        last_same_stride_layer == options->strides_count - 1
                            ? 1.0f
                            : calculate_scale(options->min_scale,
                                              options->max_scale,
                                              last_same_stride_layer + 1,
                                              options->strides_count);
                    cvector_push_back(scales, sqrtf(scale * scale_next));
                    cvector_push_back(aspect_ratios,
                                      options->interpolated_scale_aspect_ratio);
                }
            }
            last_same_stride_layer++;
        }

        for (int i = 0; i < cvector_size(aspect_ratios); ++i)
        {
            const float ratio_sqrts = sqrtf(aspect_ratios[i]);
            cvector_push_back(anchor_height, scales[i] / ratio_sqrts);
            cvector_push_back(anchor_width, scales[i] * ratio_sqrts);
        }

        cvector_free(aspect_ratios);
        cvector_free(scales);

        const int stride = options->strides[layer_id];
        const int feature_map_height =
            ceilf(1.0f * options->input_size_height / stride);
        const int feature_map_width =
            ceilf(1.0f * options->input_size_width / stride);

        for (int y = 0; y < feature_map_height; ++y)
        {
            for (int x = 0; x < feature_map_width; ++x)
            {
                for (int anchor_id = 0; anchor_id < cvector_size(anchor_height); ++anchor_id)
                {
                    const float x_center =
                        (x + options->anchor_offset_x) * 1.0f / feature_map_width;
                    const float y_center =
                        (y + options->anchor_offset_y) * 1.0f / feature_map_height;

                    Anchor new_anchor;
                    new_anchor.x = x_center;
                    new_anchor.y = y_center;

                    if (options->fixed_anchor_size)
                    {
                        new_anchor.w = 1.0f;
                        new_anchor.h = 1.0f;
                    }
                    else
                    {
                        new_anchor.w = anchor_width[anchor_id];
                        new_anchor.h = anchor_height[anchor_id];
                    }
                    cvector_push_back(anchors, new_anchor);
                }
            }
        }
        cvector_free(anchor_width);
        cvector_free(anchor_height);
        layer_id = last_same_stride_layer;
    }

    *anchors_out = anchors;
}

static void generate_face_anchors(cvector_vector_type(Anchor) * anchors)
{
    int strides[] = {4};
    const int strides_count = sizeof(strides) / sizeof(strides[0]);
    float aspect_ratios[] = {1.0f};
    const int aspect_ratios_count = sizeof(aspect_ratios) / sizeof(aspect_ratios[0]);
    const AnchorOptions anchor_options = {
        1,                   // num_layers
        0.1484375f,          // min_scale
        0.75f,               // max_scale
        192,                 // input_size_height
        192,                 // input_size_width
        0.5f,                // anchor_offset_x
        0.5f,                // anchor_offset_y
        strides,             // strides
        strides_count,       // strides_count
        aspect_ratios,       // aspect_ratios
        aspect_ratios_count, // aspect_ratios_count
        true,                // fixed_anchor_size
        0.0f,                // interpolated_scale_aspect_ratio
        false,               // reduce_boxes_in_lowest_layer
    };
    generate_anchors(&anchor_options, anchors);
}

static void decode_boxes(
    const DecodeBoxesOptions *options,
    const float *raw_boxes,
    cvector_vector_type(Anchor) anchors,
    cvector_vector_type(float) * boxes_out)
{
    cvector_vector_type(float) boxes = NULL;
    cvector_resize(boxes, options->num_boxes * options->num_coords);
    for (int i = 0; i < options->num_boxes; ++i)
    {
        const int box_offset =
            i * options->num_coords + options->box_coord_offset;

        float y_center;
        float x_center;
        float h;
        float w;
        if (options->reverse_output_order)
        {
            x_center = raw_boxes[box_offset];
            y_center = raw_boxes[box_offset + 1];
            w = raw_boxes[box_offset + 2];
            h = raw_boxes[box_offset + 3];
        }
        else
        {
            y_center = raw_boxes[box_offset];
            x_center = raw_boxes[box_offset + 1];
            h = raw_boxes[box_offset + 2];
            w = raw_boxes[box_offset + 3];
        }

        x_center =
            x_center / options->x_scale * anchors[i].w + anchors[i].x;
        y_center =
            y_center / options->y_scale * anchors[i].h + anchors[i].y;

        if (options->apply_exponential_on_box_size)
        {
            h = expf(h / options->h_scale) * anchors[i].h;
            w = expf(w / options->w_scale) * anchors[i].w;
        }
        else
        {
            h = h / options->h_scale * anchors[i].h;
            w = w / options->w_scale * anchors[i].w;
        }

        const float ymin = y_center - h / 2.0f;
        const float xmin = x_center - w / 2.0f;
        const float ymax = y_center + h / 2.0f;
        const float xmax = x_center + w / 2.0f;

        boxes[i * options->num_coords + 0] = xmin;
        boxes[i * options->num_coords + 1] = ymin;
        boxes[i * options->num_coords + 2] = xmax;
        boxes[i * options->num_coords + 3] = ymax;

        for (int k = 0; k < options->num_keypoints; ++k)
        {
            const int offset = i * options->num_coords +
                               options->keypoint_coord_offset +
                               k * options->num_values_per_keypoint;

            float keypoint_y;
            float keypoint_x;
            if (options->reverse_output_order)
            {
                keypoint_x = raw_boxes[offset];
                keypoint_y = raw_boxes[offset + 1];
            }
            else
            {
                keypoint_y = raw_boxes[offset];
                keypoint_x = raw_boxes[offset + 1];
            }

            boxes[offset] = keypoint_x / options->x_scale * anchors[i].w +
                            anchors[i].x;
            boxes[offset + 1] =
                keypoint_y / options->y_scale * anchors[i].h +
                anchors[i].y;
        }
    }
    *boxes_out = boxes;
}

static void decode_face_boxes(
    const float *raw_boxes,
    cvector_vector_type(Anchor) anchors,
    cvector_vector_type(float) * boxes_out)
{
    const DecodeBoxesOptions options = {
        10,     // num_classes
        2304,   // num_boxes
        16,     // num_coords
        4,      // keypoint_coord_offset
        6,      // num_keypoints
        2,      // num_values_per_keypoint
        0,      // box_coord_offset
        192.0f, // x_scale
        192.0f, // y_scale
        192.0f, // h_scale
        192.0f, // w_scale
        false,  // apply_exponential_on_box_size
        true,   // reverse_output_order
        true,   // sigmoid_score
        100.0f, // score_clipping_thresh
        false,  // flip_vertically
        0.6f,   // min_score_thresh
    };
    decode_boxes(&options, raw_boxes, anchors, boxes_out);
}

static bool sort_by_second(const IntFloatPair *indexed_score_0,
                           const IntFloatPair *indexed_score_1)
{
    return (indexed_score_0->second > indexed_score_1->second);
}

static bool rect_is_empty(const Rectf *rect)
{
    return rect->min_x > rect->max_x || rect->min_y > rect->max_y;
}

static bool rect_intersects(const Rectf *a, const Rectf *b)
{
    return !(rect_is_empty(a) || rect_is_empty(b) || b->max_x < a->min_x || a->max_x < b->min_x ||
             b->max_y < a->min_y || a->max_y < b->min_y);
}

static void rect_set_empty(Rectf *rect)
{
    rect->min_x = __FLT_MAX__;
    rect->min_y = __FLT_MAX__;
    rect->max_x = __FLT_MIN__;
    rect->max_y = __FLT_MIN__;
}

static Rectf *rect_intersect(const Rectf *a, const Rectf *b)
{
    Rectf *result = malloc(sizeof(Rectf));
    result->min_x = fmax(a->min_x, b->min_x);
    result->min_y = fmax(a->min_y, b->min_y);
    result->max_x = fmin(a->max_x, b->max_x);
    result->max_y = fmin(a->max_y, b->max_y);
    if (result->min_x > result->max_x || result->min_y > result->max_y)
    {
        rect_set_empty(result);
    }
    return result;
}

static Rectf *rect_union(const Rectf *a, const Rectf *b)
{
    Rectf *result = malloc(sizeof(Rectf));
    result->min_x = fmin(a->min_x, b->min_x);
    result->min_y = fmin(a->min_y, b->min_y);
    result->max_x = fmax(a->max_x, b->max_x);
    result->max_y = fmax(a->max_y, b->max_y);
    return result;
}

static float rect_width(const Rectf *rect)
{
    return rect->max_x - rect->min_x;
}

static float rect_height(const Rectf *rect)
{
    return rect->max_y - rect->min_y;
}

static float rect_area(const Rectf *rect)
{
    return rect_width(rect) * rect_height(rect);
}

static void rect_from_coords(const float *coords, Rectf *result)
{
    result->min_x = coords[0];
    result->min_y = coords[1];
    result->max_x = coords[2];
    result->max_y = coords[3];
}

static float overlap_similarity(int overlap_type,
                                const Rectf *rect1, const Rectf *rect2)
{
    if (!rect_intersects(rect1, rect2))
        return 0.0f;
    Rectf *intersection = rect_intersect(rect1, rect2);
    const float intersection_area = rect_area(intersection);
    free(intersection);
    float normalization;
    switch (overlap_type)
    {
    case NMS_JACCARD:
    {
        Rectf *union_rect = rect_union(rect1, rect2);
        normalization = rect_area(union_rect);
        free(union_rect);
    }
    break;
    case NMS_MODIFIED_JACCARD:
    {
        normalization = rect_area(rect2);
    }
    break;
    case NMS_INTERSECTION_OVER_UNION:
    {
        normalization = rect_area(rect1) + rect_area(rect2) - intersection_area;
    }
    break;
    default:
    {
        assert(false);
    }
    break;
    }
    return normalization > 0.0f ? intersection_area / normalization : 0.0f;
}

static void unweighted_non_max_suppression(
    const NonMaxSuppressionOptions *options,
    cvector_vector_type(IntFloatPair) indexed_scores,
    const float *boxes,
    int max_num_detections,
    cvector_vector_type(Detection) * detections_out)
{
    cvector_vector_type(Detection) detections = NULL;

    for (int i = 0; i < cvector_size(indexed_scores); ++i)
    {
        IntFloatPair *indexed_score = &indexed_scores[i];
        const float *candidate_coords =
            &boxes[indexed_score->first * options->num_coords];
        Rectf candidate_rect;
        rect_from_coords(candidate_coords + options->box_coord_offset,
                         &candidate_rect);
        const Detection candidate_detection = {
            candidate_rect,
            indexed_score->second,
        };
        if (options->min_score_threshold > 0.0f &&
            candidate_detection.score < options->min_score_threshold)
        {
            break;
        }
        bool suppressed = false;
        for (int j = 0; j < cvector_size(detections); ++j)
        {
            Detection *existing_detection = &detections[j];
            const float similarity = overlap_similarity(
                options->overlap_type,
                &existing_detection->rect,
                &candidate_detection.rect);
            if (similarity > options->min_suppression_threshold)
            {
                suppressed = true;
                break;
            }
            if (!suppressed)
            {
                cvector_push_back(detections, candidate_detection);
            }
            if (cvector_size(detections) >= max_num_detections)
            {
                break;
            }
        }
    }
    *detections_out = detections;
}

static void weighted_non_max_suppression(
    const NonMaxSuppressionOptions *options,
    cvector_vector_type(IntFloatPair) indexed_scores,
    const float *boxes,
    int max_num_detections,
    cvector_vector_type(Detection) * detections_out)
{
    cvector_vector_type(IntFloatPair) remained_indexed_scores = NULL;
    cvector_copy(indexed_scores, remained_indexed_scores);

    cvector_vector_type(Detection) detections = NULL;

    while (cvector_size(remained_indexed_scores) > 0)
    {
        IntFloatPair *indexed_score = &remained_indexed_scores[0];
        const int original_indexed_scores_size =
            cvector_size(remained_indexed_scores);
        const float *candidate_coords =
            &boxes[indexed_score->first * options->num_coords];
        Rectf candidate_rect;
        rect_from_coords(candidate_coords + options->box_coord_offset,
                         &candidate_rect);
        const Detection candidate_detection = {
            candidate_rect,
            indexed_score->second,
        };
        if (options->min_score_threshold > 0.0f &&
            candidate_detection.score < options->min_score_threshold)
        {
            break;
        }
        cvector_vector_type(IntFloatPair) remained = NULL;
        cvector_vector_type(IntFloatPair) candidates = NULL;
        const Rectf candidate_location = candidate_detection.rect;

        for (int i = 0; i < cvector_size(remained_indexed_scores); ++i)
        {
            const IntFloatPair *remained_indexed_score =
                &remained_indexed_scores[i];
            const float *remained_coords =
                &boxes[remained_indexed_score->first * options->num_coords];
            Rectf remained_rect;
            rect_from_coords(remained_coords + options->box_coord_offset,
                             &remained_rect);
            float similarity =
                overlap_similarity(options->overlap_type, &remained_rect,
                                   &candidate_rect);
            if (similarity > options->min_suppression_threshold)
            {
                cvector_push_back(candidates, *remained_indexed_score);
            }
            else
            {
                cvector_push_back(remained, *remained_indexed_score);
            }
        }
        Detection weighted_detection = candidate_detection;
        if (cvector_size(candidates) > 0)
        {
            const int num_keypoints = options->num_keypoints;
            float keypoints[NUM_KEYPOINTS_PER_BOX * 2] = {};
            float w_xmin = 0.0f;
            float w_ymin = 0.0f;
            float w_xmax = 0.0f;
            float w_ymax = 0.0f;
            float total_score = 0.0f;
            for (int j = 0; j < cvector_size(candidates); ++j)
            {
                const IntFloatPair *sub_candidate = &candidates[j];
                const float sub_score = sub_candidate->second;
                total_score += sub_score;
                const float *sub_candidate_coords =
                    &boxes[sub_candidate->first * options->num_coords];
                Rectf bbox;
                rect_from_coords(sub_candidate_coords + options->box_coord_offset,
                                 &bbox);
                w_xmin += bbox.min_x * sub_score;
                w_ymin += bbox.min_y * sub_score;
                w_xmax += bbox.max_x * sub_score;
                w_ymax += bbox.max_y * sub_score;

                const float *sub_keypoint_coords =
                    sub_candidate_coords + options->keypoint_coord_offset;
                for (int k = 0; k < num_keypoints; ++k)
                {
                    keypoints[k * 2] += sub_keypoint_coords[k * 2] * sub_score;
                    keypoints[(k * 2) + 1] +=
                        sub_keypoint_coords[(k * 2) + 1] * sub_score;
                }
            }
            weighted_detection.rect.min_x = w_xmin / total_score;
            weighted_detection.rect.min_y = w_ymin / total_score;
            weighted_detection.rect.max_x = w_xmax / total_score;
            weighted_detection.rect.max_y = w_ymax / total_score;
            for (int k = 0; k < num_keypoints; ++k)
            {
                weighted_detection.keypoints[k * 2] =
                    keypoints[k * 2] / total_score;
                weighted_detection.keypoints[(k * 2) + 1] =
                    keypoints[(k * 2) + 1] / total_score;
            }
        }
        cvector_push_back(detections, weighted_detection);
        if (original_indexed_scores_size == cvector_size(remained))
        {
            break;
        }
        else
        {
            cvector_copy(remained, remained_indexed_scores);
        }
    }
    *detections_out = detections;
}

static int sort_int_float_pair_by_second(const void *raw_a, const void *raw_b)
{
    IntFloatPair *a = (IntFloatPair *)(raw_a);
    IntFloatPair *b = (IntFloatPair *)(raw_b);
    return (b->second - a->second);
}

static void non_max_suppression(const NonMaxSuppressionOptions *options,
                                const float *scores, const float *boxes,
                                cvector_vector_type(Detection) * detections_out)
{
    cvector_vector_type(IntFloatPair) indexed_scores = NULL;
    cvector_resize(indexed_scores, options->num_boxes);
    for (int i = 0; i < options->num_boxes; ++i)
    {
        const IntFloatPair pair = {i, scores[i]};
        indexed_scores[i] = pair;
    }
    qsort(cvector_begin(indexed_scores), options->num_boxes,
          sizeof(IntFloatPair), sort_int_float_pair_by_second);

    const int max_num_detections =
        (options->max_num_detections > -1)
            ? options->max_num_detections
            : options->num_boxes;

    if (options->algorithm == NMS_WEIGHTED)
    {
        weighted_non_max_suppression(
            options, indexed_scores, boxes, max_num_detections, detections_out);
    }
    else
    {
        unweighted_non_max_suppression(
            options, indexed_scores, boxes, max_num_detections,
            detections_out);
    }
    cvector_free(indexed_scores);
}

static void non_max_suppression_faces(const float *scores, const float *boxes,
                                      cvector_vector_type(Detection) * detections_out)
{
    const NonMaxSuppressionOptions options = {
        1,                           // num_detection_streams
        -1,                          // max_num_detections
        0.1f,                        // min_score_threshold
        0.3f,                        // min_suppression_threshold
        NMS_INTERSECTION_OVER_UNION, // overlap_type
        true,                        // return_empty_detections
        NMS_WEIGHTED,                // algorithm
        2304,                        // num_boxes
        16,                          // num_coords
        4,                           // keypoint_coord_offset
        6,                           // num_keypoints
        2,                           // num_values_per_keypoint
        0,                           // box_coord_offset
    };
    non_max_suppression(&options, scores, boxes, detections_out);
}

static float *rescale(uint8_t *input, int input_width, int input_height,
                      int output_width, int output_height)
{
    const int input_channels_per_pixel = 4;
    const int input_bytes_per_pixel =
        input_channels_per_pixel * sizeof(uint8_t);

    const int output_channels_per_pixel = 3;
    const int output_bytes_per_pixel =
        output_channels_per_pixel * sizeof(float);
    const int output_byte_count =
        (output_height * output_width * output_bytes_per_pixel);
    float *output = malloc(output_byte_count);

    const float scale_x = (input_width / (float)(output_width));
    const float scale_y = (input_height / (float)(output_height));

    const float *output_end = output + output_byte_count;

    // Does a simple nearest-neighbor resampling for now.
    for (int output_y = 0; output_y < output_height; ++output_y)
    {
        float *output_row =
            output + (output_y * output_width * output_channels_per_pixel);
        int input_y = (output_y * scale_y);
        if (input_y < 0)
        {
            input_y = 0;
        }
        else if (input_y > (input_height - 1))
        {
            input_y = (input_height - 1);
        }
        uint8_t *input_row =
            input + (input_y * input_width * input_bytes_per_pixel);
        for (int output_x = 0; output_x < output_width; ++output_x)
        {
            int input_x = (output_x * scale_x);
            if (input_x < 0)
            {
                input_x = 0;
            }
            else if (input_x > (input_width - 1))
            {
                input_x = (input_width - 1);
            }
            float *output_pixel =
                output_row + (output_x * output_channels_per_pixel);
            uint8_t *input_pixel =
                input_row + (input_x * input_bytes_per_pixel);
            const uint8_t input_red = input_pixel[0];
            const uint8_t input_green = input_pixel[1];
            const uint8_t input_blue = input_pixel[2];
            const float output_red = (((input_red / 255.0f) * 2.0f) - 1.0f);
            const float output_green = (((input_green / 255.0f) * 2.0f) - 1.0f);
            const float output_blue = (((input_blue / 255.0f) * 2.0f) - 1.0f);
            output_pixel[0] = output_red;
            output_pixel[1] = output_green;
            output_pixel[2] = output_blue;
        }
    }
    return output;
}

static TfLiteInterpreter *init_interpreter(const char *model_filename)
{
    TfLiteModel *model = TfLiteModelCreateFromFile(model_filename);

    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return interpreter;
}

static void init_tensors(
    TfLiteInterpreter *interpreter,
    TfLiteTensor **input_tensor,
    int *input_width, int *input_height, size_t *input_byte_count,
    const TfLiteTensor **coords_tensor,
    float **coords_data, size_t *coords_byte_count,
    const TfLiteTensor **score_tensor,
    float **score_data, size_t *score_byte_count)
{

    TfLiteInterpreterAllocateTensors(interpreter);
    *input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    assert(TfLiteTensorNumDims(*input_tensor) == 4);
    assert(TfLiteTensorDim(*input_tensor, 0) == 1);
    assert(TfLiteTensorDim(*input_tensor, 3) == 3);
    *input_width = TfLiteTensorDim(*input_tensor, 2);
    *input_height = TfLiteTensorDim(*input_tensor, 1);
    *input_byte_count = TfLiteTensorByteSize(*input_tensor);

    const int outputs_count =
        TfLiteInterpreterGetOutputTensorCount(interpreter);
    assert(outputs_count == 2);

    *coords_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    assert(TfLiteTensorNumDims(*coords_tensor) == 3);
    assert(TfLiteTensorDim(*coords_tensor, 0) == 1);
    assert(TfLiteTensorDim(*coords_tensor, 1) == box_count);
    assert(TfLiteTensorDim(*coords_tensor, 2) == coords_count);
    *coords_byte_count = TfLiteTensorByteSize(*coords_tensor);
    *coords_data = malloc(*coords_byte_count);

    *score_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 1);

    assert(TfLiteTensorNumDims(*score_tensor) == 3);
    assert(TfLiteTensorDim(*score_tensor, 0) == 1);
    assert(TfLiteTensorDim(*score_tensor, 1) == box_count);
    assert(TfLiteTensorDim(*score_tensor, 2) == 1);
    *score_byte_count = TfLiteTensorByteSize(*score_tensor);
    *score_data = malloc(*score_byte_count);
}

static void run_model(TfLiteInterpreter *interpreter,
                      const float *input_data,
                      cvector_vector_type(Anchor) anchors,
                      TfLiteTensor *input_tensor, size_t input_byte_count,
                      const TfLiteTensor *coords_tensor, float *coords_data, size_t coords_byte_count,
                      const TfLiteTensor *score_tensor, float *score_data, size_t score_byte_count,
                      cvector_vector_type(Detection) * detections_out)
{
    TfLiteTensorCopyFromBuffer(input_tensor, input_data,
                               input_byte_count);

    TfLiteInterpreterInvoke(interpreter);

    TfLiteTensorCopyToBuffer(coords_tensor, coords_data,
                             coords_byte_count);
    TfLiteTensorCopyToBuffer(score_tensor, score_data,
                             score_byte_count);

    cvector_vector_type(float) boxes;
    decode_face_boxes(coords_data, anchors, &boxes);

    cvector_vector_type(Detection) detections;
    non_max_suppression_faces(score_data, boxes, &detections);

    cvector_free(boxes);

    *detections_out = detections;
}

static bool is_facing(const Detection *detection)
{
    const float left_ear_x = detection->keypoints[KP_LEFT_EAR * 2];
    const float left_eye_x = detection->keypoints[KP_LEFT_EYE * 2];

    const float right_ear_x = detection->keypoints[KP_RIGHT_EAR * 2];
    const float right_eye_x = detection->keypoints[KP_RIGHT_EYE * 2];

    const float left_distance = left_eye_x - left_ear_x;
    const float right_distance = right_ear_x - right_eye_x;

    const float bigger_distance = fmaxf(left_distance, right_distance);
    const float smaller_distance = fminf(left_distance, right_distance);

    const float facing_ratio = smaller_distance / bigger_distance;
    return (facing_ratio > 0.70f);
}

static void output_signal(cvector_vector_type(Detection) detections)
{
    static int facing_count = 0;
    bool are_any_facing = false;

    for (int i = 0; i < cvector_size(detections); ++i)
    {
        Detection *detection = &detections[i];
        if (is_facing(detection))
        {
            are_any_facing = true;
        }
    }
    const int facing_max = 5;
    if (are_any_facing)
    {
        facing_count += 1;
        fprintf(stderr, "+");
        fflush(stderr);
        if (facing_count > facing_max)
        {
            facing_count = facing_max;
        }
    }
    else
    {
        facing_count -= 1;
        if (cvector_size(detections) == 0)
        {
            fprintf(stderr, ".");
        }
        else
        {
            fprintf(stderr, "-");
        }
        fflush(stderr);
        if (facing_count < 0)
        {
            facing_count = 0;
        }
    }

    const int facing_threshold = 2;
    if (facing_count >= facing_threshold)
    {
        FILE *file = fopen(signal_filename, "w");
        const char *contents = "Facing";
        const int contents_length = strlen(contents);
        fwrite(contents, 1, contents_length, file);
        fclose(file);
        fprintf(stderr, "Facing\n");
    }
    else
    {
        remove(signal_filename);
    }
}

void *tflite_main(void *cookie)
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

    while (true)
    {
        int capture_width;
        int capture_height;
        uint8_t *capture_data;
        if (!get_latest_capture(&capture_width, &capture_height, &capture_data))
        {
            // Wait until the camera is ready.
            sleep(1);
            continue;
        }

        float *rescaled_data = rescale(capture_data, capture_width,
                                       capture_height, input_width,
                                       input_height);
        free(capture_data);

        cvector_vector_type(Detection) detections = NULL;
        run_model(interpreter, rescaled_data, anchors, input_tensor, input_byte_count,
                  coords_tensor, coords_data, coords_byte_count,
                  score_tensor, score_data, score_byte_count,
                  &detections);
        free(rescaled_data);

        output_signal(detections);

        pthread_mutex_lock(&g_detections_mutex);
        cvector_free(g_detections);
        g_detections = cvector_begin(detections);
        g_detections_count = cvector_size(detections);
        pthread_mutex_unlock(&g_detections_mutex);

        struct timespec time, remaining;
        time.tv_sec = 0;
        time.tv_nsec = 100000000L;
        nanosleep(&time, &remaining);
    }

    cvector_free(anchors);

    free(score_data);
    free(coords_data);

    TfLiteInterpreterDelete(interpreter);

    return NULL;
}

bool get_detections(Detection **detections, int *detections_count)
{
    pthread_mutex_lock(&g_detections_mutex);
    const bool has_data = (g_detections_count != -1);
    if (has_data)
    {
        const size_t detections_byte_count =
            sizeof(Detection) * g_detections_count;
        *detections = calloc(1, detections_byte_count);
        *detections_count = g_detections_count;
        if (g_detections != NULL)
        {
            memcpy(*detections, g_detections, detections_byte_count);
        }
    }
    else
    {
        *detections = NULL;
    }
    pthread_mutex_unlock(&g_detections_mutex);
    return has_data;
}
