# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Define base configuration.
FROM python:3.9-buster
MAINTAINER ral@gatech.edu
WORKDIR /usr/local/

# Declare environment variables.
ENV hostname null
ENV port null
ENV seed null
ENV n_workers null

# Install ATLoad v1.3.
RUN cd lib \
  && wget https://github.com/rodrigoalveslima/ATLoad/archive/v1.3.tar.gz \
  && tar -xzf v1.3.tar.gz \
  && rm v1.3.tar.gz

# Build directory tree and copy files.
RUN mkdir -p src/loadgen
COPY loadgen.py src/loadgen
COPY requirements.txt src/loadgen

# Install Python dependencies.
RUN pip install --no-cache-dir -r src/loadgen/requirements.txt
RUN pip install --no-cache-dir -r lib/ATLoad-1.3/requirements.txt

# Start the workload generator.
CMD ["/bin/bash", "-c", "PYTHONPATH=/usr/local/lib/ATLoad-1.3/src python3 src/loadgen/loadgen.py --workload_conf etc/loadgen/workload.yml --log /tmp/loadgen.log --hostname $hostname --port $port --seed $seed --n_workers ${n_workers}"]
