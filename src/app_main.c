#include "app_main.h"

#include <pthread.h>
#include <stdio.h>

#include "capture_main.h"
#include "tflite_main.h"
#include "window_main.h"
#include "yargs.h"

int app_main(int argc, char **argv)
{
  bool show_window = false;
  YargsFlag flags[] = {
      YARGS_BOOL("show_window", "w", &show_window, "Whether to display a debug window."),
  };
  const int flags_length = sizeof(flags) / sizeof(flags[0]);
  bool status = yargs_init(flags, flags_length, NULL, argv, argc);
  if (!status)
  {
    fprintf(stderr, "Problem parsing command line flags.\n");
    yargs_print_usage(flags, flags_length, NULL);
    return 1;
  }

  // Don't pass args to the sub-mains.
  char *dummy_argv[] = {argv[0]};
  Args args = {1, dummy_argv};

  pthread_t window_thread;
  if (show_window)
  {
    pthread_create(&window_thread, NULL, window_main, &args);
  }

  pthread_t capture_thread;
  pthread_create(&capture_thread, NULL, capture_main, &args);

  pthread_t tflite_thread;
  pthread_create(&tflite_thread, NULL, tflite_main, &args);

  if (show_window)
  {
    pthread_join(window_thread, NULL);
  }
  pthread_join(capture_thread, NULL);
  pthread_join(tflite_thread, NULL);

  return 0;
}