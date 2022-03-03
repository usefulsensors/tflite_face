#include "capture_main.h"

#include <algorithm>

#include "app_main.h"
#include "core/libcamera_app.h"
#include "core/options.h"
#include "trace.h"

namespace
{
    using namespace std::placeholders;

    static pthread_mutex_t g_frame_mutex = PTHREAD_MUTEX_INITIALIZER;
    uint8_t *g_frame_buffer = NULL;
    const int frame_width = 640;
    const int frame_height = 480;
    const int rgba_bytes_per_pixel = 4;
    const int rgba_bytes_per_row = (frame_width * rgba_bytes_per_pixel);
    const int rgba_byte_count = (frame_height * frame_width * rgba_bytes_per_pixel);

    std::vector<uint8_t> Yuv420ToRgba(const uint8_t *src, StreamInfo &src_info, StreamInfo &dst_info)
    {
        std::vector<uint8_t> output(dst_info.height * dst_info.stride);

        assert(src_info.width >= dst_info.width && src_info.height >= dst_info.height);
        int off_x = ((src_info.width - dst_info.width) / 2) & ~1, off_y = ((src_info.height - dst_info.height) / 2) & ~1;
        int src_Y_size = src_info.height * src_info.stride, src_U_size = (src_info.height / 2) * (src_info.stride / 2);

        // We're going to process 4x2 pixel blocks, as far as alignment allows.
        unsigned int dst_h_aligned = dst_info.height & ~1, dst_w_aligned = dst_info.width & ~3;

        unsigned int y = 0;
        for (; y < dst_h_aligned; y += 2)
        {
            const uint8_t *src_Y0 = src + (y + off_y) * src_info.stride + off_x;
            const uint8_t *src_U = src + src_Y_size + ((y + off_y) / 2) * (src_info.stride / 2) + off_x / 2;
            const uint8_t *src_V = src_U + src_U_size;
            const uint8_t *src_Y1 = src_Y0 + src_info.stride;
            uint8_t *dst0 = &output[y * dst_info.stride];
            uint8_t *dst1 = dst0 + dst_info.stride;

            unsigned int x = 0;
            for (; x < dst_w_aligned; x += 4)
            {
                int Y0 = *(src_Y0++);
                int U0 = *(src_U++);
                int V0 = *(src_V++);
                int Y1 = *(src_Y0++);
                int Y2 = *(src_Y0++);
                int U2 = *(src_U++);
                int V2 = *(src_V++);
                int Y3 = *(src_Y0++);
                int Y4 = *(src_Y1++);
                int Y5 = *(src_Y1++);
                int Y6 = *(src_Y1++);
                int Y7 = *(src_Y1++);

                U0 -= 128;
                V0 -= 128;
                U2 -= 128;
                V2 -= 128;
                int U1 = U0;
                int V1 = V0;
                int U4 = U0;
                int V4 = V0;
                int U5 = U0;
                int V5 = V0;
                int U3 = U2;
                int V3 = V2;
                int U6 = U2;
                int V6 = V2;
                int U7 = U2;
                int V7 = V2;

                int R0 = Y0 + 1.402 * V0;
                int G0 = Y0 - 0.345 * U0 - 0.714 * V0;
                int B0 = Y0 + 1.771 * U0;
                int R1 = Y1 + 1.402 * V1;
                int G1 = Y1 - 0.345 * U1 - 0.714 * V1;
                int B1 = Y1 + 1.771 * U1;
                int R2 = Y2 + 1.402 * V2;
                int G2 = Y2 - 0.345 * U2 - 0.714 * V2;
                int B2 = Y2 + 1.771 * U2;
                int R3 = Y3 + 1.402 * V3;
                int G3 = Y3 - 0.345 * U3 - 0.714 * V3;
                int B3 = Y3 + 1.771 * U3;
                int R4 = Y4 + 1.402 * V4;
                int G4 = Y4 - 0.345 * U4 - 0.714 * V4;
                int B4 = Y4 + 1.771 * U4;
                int R5 = Y5 + 1.402 * V5;
                int G5 = Y5 - 0.345 * U5 - 0.714 * V5;
                int B5 = Y5 + 1.771 * U5;
                int R6 = Y6 + 1.402 * V6;
                int G6 = Y6 - 0.345 * U6 - 0.714 * V6;
                int B6 = Y6 + 1.771 * U6;
                int R7 = Y7 + 1.402 * V7;
                int G7 = Y7 - 0.345 * U7 - 0.714 * V7;
                int B7 = Y7 + 1.771 * U7;

                R0 = std::clamp(R0, 0, 255);
                G0 = std::clamp(G0, 0, 255);
                B0 = std::clamp(B0, 0, 255);
                R1 = std::clamp(R1, 0, 255);
                G1 = std::clamp(G1, 0, 255);
                B1 = std::clamp(B1, 0, 255);
                R2 = std::clamp(R2, 0, 255);
                G2 = std::clamp(G2, 0, 255);
                B2 = std::clamp(B2, 0, 255);
                R3 = std::clamp(R3, 0, 255);
                G3 = std::clamp(G3, 0, 255);
                B3 = std::clamp(B3, 0, 255);
                R4 = std::clamp(R4, 0, 255);
                G4 = std::clamp(G4, 0, 255);
                B4 = std::clamp(B4, 0, 255);
                R5 = std::clamp(R5, 0, 255);
                G5 = std::clamp(G5, 0, 255);
                B5 = std::clamp(B5, 0, 255);
                R6 = std::clamp(R6, 0, 255);
                G6 = std::clamp(G6, 0, 255);
                B6 = std::clamp(B6, 0, 255);
                R7 = std::clamp(R7, 0, 255);
                G7 = std::clamp(G7, 0, 255);
                B7 = std::clamp(B7, 0, 255);

                *(dst0++) = R0;
                *(dst0++) = G0;
                *(dst0++) = B0;
                *(dst0++) = 255;
                *(dst0++) = R1;
                *(dst0++) = G1;
                *(dst0++) = B1;
                *(dst0++) = 255;
                *(dst0++) = R2;
                *(dst0++) = G2;
                *(dst0++) = B2;
                *(dst0++) = 255;
                *(dst0++) = R3;
                *(dst0++) = G3;
                *(dst0++) = B3;
                *(dst0++) = 255;
                *(dst1++) = R4;
                *(dst1++) = G4;
                *(dst1++) = B4;
                *(dst1++) = 255;
                *(dst1++) = R5;
                *(dst1++) = G5;
                *(dst1++) = B5;
                *(dst1++) = 255;
                *(dst1++) = R6;
                *(dst1++) = G6;
                *(dst1++) = B6;
                *(dst1++) = 255;
                *(dst1++) = R7;
                *(dst1++) = G7;
                *(dst1++) = B7;
                *(dst1++) = 255;
            }
            // Straggling pixel columns - we must still do both rows.
            for (; x < dst_info.width; x++)
            {
                int Y0 = *(src_Y0++);
                int U0 = *(src_U);
                int V0 = *(src_V);
                int Y4 = *(src_Y1++);
                src_U += (x & 1);
                src_V += (x & 1);

                U0 -= 128;
                V0 -= 128;
                int U4 = U0;
                int V4 = V0;

                int R0 = Y0 + 1.402 * V0;
                int G0 = Y0 - 0.345 * U0 - 0.714 * V0;
                int B0 = Y0 + 1.771 * U0;
                int R4 = Y4 + 1.402 * V4;
                int G4 = Y4 - 0.345 * U4 - 0.714 * V4;
                int B4 = Y4 + 1.771 * U4;

                R0 = std::clamp(R0, 0, 255);
                G0 = std::clamp(G0, 0, 255);
                B0 = std::clamp(B0, 0, 255);
                R4 = std::clamp(R4, 0, 255);
                G4 = std::clamp(G4, 0, 255);
                B4 = std::clamp(B4, 0, 255);

                *(dst0++) = R0;
                *(dst0++) = G0;
                *(dst0++) = B0;
                *(dst0++) = 255;
                *(dst1++) = R4;
                *(dst1++) = G4;
                *(dst1++) = B4;
                *(dst1++) = 255;
            }
        }
        // Any straggling final row is done with extreme steam power.
        for (; y < dst_info.height; y++)
        {
            const uint8_t *src_Y0 = src + (y + off_y) * src_info.stride + off_x;
            const uint8_t *src_U = src + src_Y_size + ((y + off_y) / 2) * (src_info.stride / 2) + off_x / 2;
            const uint8_t *src_V = src_U + src_U_size;
            uint8_t *dst0 = &output[y * dst_info.stride];

            unsigned int x = 0;
            for (; x < dst_info.width; x++)
            {
                int Y0 = *(src_Y0++);
                int U0 = *(src_U);
                int V0 = *(src_V);
                src_U += (x & 1);
                src_V += (x & 1);

                U0 -= 128;
                V0 -= 128;

                int R0 = Y0 + 1.402 * V0;
                int G0 = Y0 - 0.345 * U0 - 0.714 * V0;
                int B0 = Y0 + 1.771 * U0;

                R0 = std::clamp(R0, 0, 255);
                G0 = std::clamp(G0, 0, 255);
                B0 = std::clamp(B0, 0, 255);

                *(dst0++) = R0;
                *(dst0++) = G0;
                *(dst0++) = B0;
                *(dst0++) = 255;
            }
        }

        return output;
    }

    // The main event loop for the application.

    static void event_loop(LibcameraApp &app)
    {
        Options const *options = app.GetOptions();

        app.OpenCamera();
        app.ConfigureViewfinder();
        app.StartCamera();

        auto start_time = std::chrono::high_resolution_clock::now();

        for (unsigned int count = 0;; count++)
        {
            LibcameraApp::Msg msg = app.Wait();
            if (msg.type == LibcameraApp::MsgType::Quit)
                return;
            else if (msg.type != LibcameraApp::MsgType::RequestComplete)
                throw std::runtime_error("unrecognised message!");

            if (options->verbose)
                std::cerr << "Viewfinder frame " << count << std::endl;

            CompletedRequestPtr &completed_request = std::get<CompletedRequestPtr>(msg.payload);

            libcamera::Stream *stream = app.LoresStream();
            StreamInfo info = app.GetStreamInfo(stream);
            const libcamera::Span<uint8_t> mem = app.Mmap(completed_request->buffers[stream])[0];

            StreamInfo dest_info;
            dest_info.width = frame_width;
            dest_info.height = frame_height;
            dest_info.stride = frame_width * 4;
            std::vector<uint8_t> rgba_image = Yuv420ToRgba(mem.data(), info, dest_info);
            assert(rgba_image.size() == rgba_byte_count);

            uint8_t *rgba_buffer = (uint8_t *)(malloc(rgba_byte_count));
            memcpy(rgba_buffer, rgba_image.data(), rgba_byte_count);

            pthread_mutex_lock(&g_frame_mutex);
            free(g_frame_buffer);
            g_frame_buffer = rgba_buffer;
            pthread_mutex_unlock(&g_frame_mutex);
        }
    }

} // namespace

void *capture_main(void *cookie)
{
    Args *args = (Args *)(cookie);
    int argc = args->argc;
    char **argv = args->argv;

    try
    {
        LibcameraApp app;
        Options *options = app.GetOptions();
        if (options->Parse(argc, argv))
        {
            if (options->verbose)
                options->Print();

            event_loop(app);
        }
    }
    catch (std::exception const &e)
    {
        std::cerr << "ERROR: *** " << e.what() << " ***" << std::endl;
        return nullptr;
    }

    return nullptr;
}

bool get_latest_capture(int *width, int *height, uint8_t **rgba_buffer)
{
    *rgba_buffer = (uint8_t *)(malloc(rgba_byte_count));
    pthread_mutex_lock(&g_frame_mutex);
    const bool has_data = (g_frame_buffer != NULL);
    if (has_data)
    {
        memcpy(*rgba_buffer, g_frame_buffer, rgba_byte_count);
    }
    pthread_mutex_unlock(&g_frame_mutex);
    *width = frame_width;
    *height = frame_height;
    return has_data;
}
