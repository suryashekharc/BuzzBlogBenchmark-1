#!/bin/bash

# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Increase the max number of SSH sessions allowed per connection.


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
    * )
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
  shift
done

# Increase the max number of SSH sessions allowed per connection.
ssh -o StrictHostKeyChecking=no ${username}@${controller_node} "
  for (( node_no=0; node_no<${n_nodes}; node_no++ )); do
    ssh -o StrictHostKeyChecking=no node-\${node_no} \"
      sudo sed -i 's/#MaxSessions 10/MaxSessions 50/g' /etc/ssh/sshd_config && sudo service ssh restart
    \" &
  done
  wait
"
