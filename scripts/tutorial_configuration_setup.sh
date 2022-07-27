#!/bin/bash

# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Render the specified system and workload configuration templates in
# `controller_node`.


# Process command-line arguments.
set -u
while [[ $# > 1 ]]; do
  case $1 in
    --username )
      username=$2
      ;;
    --controller_node )
      controller_node=$2
      ;;
    --system_template )
      system_template=$2
      ;;
    --workload_template )
      workload_template=$2
      ;;
    * )
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
  shift
done

# Render configuration templates in `controller_node`.
ssh -o StrictHostKeyChecking=no ${username}@${controller_node} "
  # Clone this repository to get the experiment configuration files.
  ssh-keygen -F github.com || ssh-keyscan github.com >> ~/.ssh/known_hosts
  git clone git@github.com:rodrigoalveslima/BuzzBlogBenchmark.git
  mv BuzzBlogBenchmark/controller/conf/tutorial/${workload_template} workload.yml
  mv BuzzBlogBenchmark/controller/conf/tutorial/${system_template} system.yml
  rm -rf BuzzBlogBenchmark

  # Set up configuration files.
  sed -i \"s/{{username}}/${username}/g\" system.yml
"
