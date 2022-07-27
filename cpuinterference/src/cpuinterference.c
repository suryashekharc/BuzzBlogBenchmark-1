// Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
// Systems

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define MAX_THREADS 32768

void *f(void *arg) {
  char *flag_p = (char *) arg;

  // Busy wait.
  while (*flag_p == 0);

  pthread_exit(NULL);
}

void log_msg(char *msg) {
  time_t now;
  struct tm *now_local;
  char now_buf[128];

  time(&now);
  now_local = localtime(&now);
  strftime(now_buf, sizeof(now_buf), "%Y-%m-%d %H:%M:%S", now_local);
  printf("[%s] %s\n", now_buf, msg);
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  char flag;
  pthread_t tid[MAX_THREADS];
  struct timespec start, now;
  float elapsed;

  // Parse command-line arguments.
  if (argc != 6) {
    printf("Usage: cpuinterference [min_threads] [max_threads] [thread_step] [period_in_sec] [duration_in_ms]\n");
    exit(1);
  }
  int min_threads = atoi(argv[1]);
  int max_threads = atoi(argv[2]);
  int thread_step = atoi(argv[3]);
  int period_in_sec = atoi(argv[4]);
  int duration_in_ms = atoi(argv[5]);

  // Log parameters.
  printf("Starting CPU interference program\n");
  printf("    min_threads = %d\n", min_threads);
  printf("    max_threads = %d\n", max_threads);
  printf("    thread_step = %d\n", thread_step);
  printf("    period_in_sec = %d\n", period_in_sec);
  printf("    duration_in_ms = %d\n", duration_in_ms);
  fflush(stdout);

  while (1) {
    for (int n_threads = min_threads; n_threads <= max_threads; n_threads += thread_step) {
      // Set flag to 0.
      flag = 0;
      // Wait period.
      clock_gettime(CLOCK_MONOTONIC, &start);
      elapsed = 0;
      while (1) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        elapsed = (now.tv_sec - start.tv_sec) + 1e-9 * (now.tv_nsec - start.tv_nsec);
        if (elapsed > period_in_sec)
          break;
        sleep(1);
      }
      // Spawn `n_threads` threads executing function `f`.
      log_msg("Spawning threads");
      for (int thread_it = 0; thread_it < n_threads; thread_it++)
        pthread_create(&tid[thread_it], NULL, f, (void *) &flag);
      // Duration period.
      clock_gettime(CLOCK_MONOTONIC, &start);
      elapsed = 0;
      while (1) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        elapsed = (now.tv_sec - start.tv_sec) + 1e-9 * (now.tv_nsec - start.tv_nsec);
        if (elapsed * 1000 > duration_in_ms)
          break;
      }
      // Set flag to 1.
      flag = 1;
      // Join threads.
      log_msg("Joining threads");
      for (int thread_it = 0; thread_it < n_threads; thread_it++)
        pthread_join(tid[thread_it], NULL);
    }
  }

  return 0;
}