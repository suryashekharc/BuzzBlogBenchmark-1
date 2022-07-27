#!/bin/bash

# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Mount the specified `partition` on `/tmp` in all nodes.


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
    --n_nodes )
      n_nodes=$2
      ;;
    --partition )
      partition=$2
      ;;
    * )
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
  shift
done

# Mount `partition` on `/tmp` in all nodes.
ssh -o StrictHostKeyChecking=no ${username}@${controller_node} "
  for (( node_no=0; node_no<${n_nodes}; node_no++ )); do
    ssh -o StrictHostKeyChecking=no node-\${node_no} \"
      # Make an ext4 filesystem on the specified partition.
      sudo mkfs.ext4 ${partition}
      # Mount the specified partition on /tmp.
      sudo mount ${partition} /tmp
      # Set ownership and permissions of /tmp.
      sudo chown -R ${username}:infosphere-PG0 /tmp
      sudo chmod -R 777 /tmp
    \" &
  done
  wait
"
