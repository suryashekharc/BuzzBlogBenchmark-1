#!/bin/bash

# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Compress experiment results in the specified directory `dir`. If `splitsize`
# is passed, split the tarball.


# Initilize parameters.
splitsize=0

# Process command-line arguments.
set -u
while [[ $# > 1 ]]; do
  case $1 in
    --dir )
      dir=$2
      ;;
    --splitsize )
      splitsize=$2
      ;;
    * )
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
  shift
done

# Compress and split experiment results.
for res in $(ls $dir -1 | grep BuzzBlogBenchmark | grep -v .tar.gz); do
  if [ -f $dir/$res.tar.gz ]; then
    echo "File $dir/$res.tar.gz exists."
  else
    tar -cvzf $dir/$res.tar.gz $dir/$res
    if [ "$splitsize" != "0" ]; then
      split -d -b $splitsize $dir/$res.tar.gz $dir/$res.tar.gz
      rm $dir/$res.tar.gz
    fi
  fi
done
