# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Define base configuration.
FROM gcc:11.2.0
MAINTAINER ral@gatech.edu
WORKDIR /usr/local/

# Environment variables.
ENV min_threads 1
ENV max_threads 1
ENV thread_step 1
ENV period_in_sec 60
ENV duration_in_ms 500

# Copy source code.
COPY src/cpuinterference.c src

# Compile source code.
RUN gcc -Wall -O0 -o bin/cpuinterference src/cpuinterference.c -lpthread

# Start the program.
CMD ["/bin/bash", "-c", "bin/cpuinterference ${min_threads} ${max_threads} ${thread_step} ${period_in_sec} ${duration_in_ms}"]
