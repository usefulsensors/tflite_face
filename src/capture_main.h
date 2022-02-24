#ifndef INCLUDE_CAPTURE_MAIN_H
#define INCLUDE_CAPTURE_MAIN_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void *capture_main(void *cookie);

    bool get_latest_capture(int *width, int *height, uint8_t **rgba_buffer);

#ifdef __cplusplus
}
#endif

#endif // INCLUDE_CAPTURE_MAIN_H