#!/bin/bash

# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Upload experiment results in `dir` to the Dropbox folder in `path`, using the
# specified API token.
# Copied from: `https://stackoverflow.com/questions/42120767/upload-file-on-linux-cli-to-dropbox-via-bash-sh`


# Process command-line arguments.
set -u
while [[ $# > 1 ]]; do
  case $1 in
    --dir )
      dir=$2
      ;;
    --token )
      token=$2
      ;;
    --path )
      path=$2
      ;;
    * )
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
  shift
done

# Upload experiment results.
cd $dir
for res in $(ls -1 | grep BuzzBlogBenchmark | grep .tar.gz); do
  curl -X POST https://content.dropboxapi.com/2/files/upload \
      --header "Authorization: Bearer $token" \
      --header "Dropbox-API-Arg: {\"path\": \"$path/$res\"}" \
      --header "Content-Type: application/octet-stream" \
      --data-binary @$res
done
