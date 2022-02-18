#include "app_main.h"

#include <pthread.h>

#include "capture_main.h"
#include "tflite_main.h"
#include "window_main.h"

int app_main(int argc, char **argv)
{
  Args args = {argc, argv};

  pthread_t window_thread;
  pthread_create(&window_thread, NULL, window_main, &args);

  pthread_t capture_thread;
  pthread_create(&capture_thread, NULL, capture_main, &args);

  pthread_t tflite_thread;
  pthread_create(&tflite_thread, NULL, tflite_main, &args);

  pthread_join(window_thread, NULL);
  pthread_join(capture_thread, NULL);
  pthread_join(tflite_thread, NULL);

  return 0;
}