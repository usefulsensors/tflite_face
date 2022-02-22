#include "window_main.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include <X11/X.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>

#include "app_main.h"
#include "capture_main.h"
#include "tflite_main.h"
#include "trace.h"

Display *dpy;
Window root;
GLint att[] = {GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None};
XVisualInfo *vi;
Colormap cmap;
XSetWindowAttributes swa;
Window win;
GLXContext glc;
XWindowAttributes gwa;
XEvent xev;
unsigned int g_texture_id = 0;

#define CHECK_GL_ERRORS()                                                      \
    do                                                                         \
    {                                                                          \
        int err;                                                               \
        while ((err = glGetError()) != 0)                                      \
        {                                                                      \
            fprintf(stderr, "%s:%d OpenGL error %d: %s\n", __FILE__, __LINE__, \
                    err, gluErrorString(err));                                 \
        }                                                                      \
    } while (false)

static void DrawAQuad()
{
    int width;
    int height;
    uint8_t *texture_data = NULL;
    if (!get_latest_capture(&width, &height, &texture_data))
    {
        // Camera capture is not yet ready.
        free(texture_data);
        return;
    }

    Detection *detections = NULL;
    int detections_count = 0;
    if (!get_detections(&detections, &detections_count))
    {
        // Object detection is not yet ready.
        free(detections);
        return;
    }

    CHECK_GL_ERRORS();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, texture_data);
    CHECK_GL_ERRORS();
    free(texture_data);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    CHECK_GL_ERRORS();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1., 1., -1., 1., 1., 20.);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0., 0., 10., 0., 0., 0., 0., 1., 0.);
    CHECK_GL_ERRORS();

    glDisable(GL_TEXTURE_2D);

    for (int i = 0; i < detections_count; ++i)
    {
        const float scale = 2.0f;
        const float offset = -0.5f;
        const Detection *detection = &detections[i];
        const float min_x = (detection->rect.min_x + offset) * scale;
        const float min_y = (detection->rect.min_y + offset) * -scale;
        const float max_x = (detection->rect.max_x + offset) * scale;
        const float max_y = (detection->rect.max_y + offset) * -scale;

        glBegin(GL_LINE_LOOP);
        glColor3f(1.0f, 1.0f, 1.0f);
        glVertex3f(min_x, min_y, 0.1f);
        glVertex3f(min_x, max_y, 0.1f);
        glVertex3f(max_x, max_y, 0.1f);
        glVertex3f(max_x, min_y, 0.1f);
        glEnd();
        CHECK_GL_ERRORS();

        for (int k = 0; k < NUM_KEYPOINTS_PER_BOX; ++k)
        {
            const float keypoint_x = detection->keypoints[k * 2];
            const float keypoint_y = detection->keypoints[(k * 2) + 1];
            const float min_x = (keypoint_x - 0.001f + offset) * scale;
            const float min_y = (keypoint_y - 0.001f + offset) * -scale;
            const float max_x = (keypoint_x + 0.001f + offset) * scale;
            const float max_y = (keypoint_y + 0.001f + offset) * -scale;

            glBegin(GL_LINE_LOOP);
            glColor3f(1.0f, 1.0f, 1.0f);
            glVertex3f(min_x, min_y, 0.1f);
            glVertex3f(min_x, max_y, 0.1f);
            glVertex3f(max_x, max_y, 0.1f);
            glVertex3f(max_x, min_y, 0.1f);
            glEnd();
        }
    }

    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);
    glColor3f(1.0f, 1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(-1.0f, -1.0f, 0.0f);
    glColor3f(1.0f, 1.0f, 1.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(1.0f, -1.0f, 0.0f);
    glColor3f(1.0f, 1.0f, 1.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(1.0f, 1.0f, 0.0f);
    glColor3f(1.0f, 1.0f, 1.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-1.0f, 1.0f, 0.0f);
    glEnd();
    CHECK_GL_ERRORS();
}

void *window_main(void *cookie)
{
    Args *args = (Args *)(cookie);
    int argc = args->argc;
    char **argv = args->argv;

    dpy = XOpenDisplay(NULL);

    if (dpy == NULL)
    {
        printf("\n\tcannot connect to X server\n\n");
        exit(0);
    }

    root = DefaultRootWindow(dpy);

    vi = glXChooseVisual(dpy, 0, att);

    if (vi == NULL)
    {
        printf("\n\tno appropriate visual found\n\n");
        exit(0);
    }
    else
    {
        printf("\n\tvisual %p selected\n", (void *)vi->visualid); /* %p creates hexadecimal output like in glxinfo */
    }

    cmap = XCreateColormap(dpy, root, vi->visual, AllocNone);

    swa.colormap = cmap;
    swa.event_mask = ExposureMask | KeyPressMask;

    win = XCreateWindow(dpy, root, 0, 0, 640, 480, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask, &swa);

    XMapWindow(dpy, win);
    XStoreName(dpy, win, "V4L2 OpenGL Example");

    glc = glXCreateContext(dpy, vi, NULL, GL_TRUE);
    glXMakeCurrent(dpy, win, glc);

    glEnable(GL_DEPTH_TEST);

    glGenTextures(1, &g_texture_id);
    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    while (1)
    {
        XGetWindowAttributes(dpy, win, &gwa);
        glViewport(0, 0, gwa.width, gwa.height);
        DrawAQuad();
        glXSwapBuffers(dpy, win);
    }
}