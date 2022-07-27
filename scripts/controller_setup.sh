#!/bin/bash

# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Set up the specified node to run the benchmark controller application.


# Process command-line arguments.
set -u
while [[ $# > 1 ]]; do
  case $1 in
    --username )
      username=$2
      ;;
    --private_ssh_key_path )
      private_ssh_key_path=$2
      ;;
    --controller_node )
      controller_node=$2
      ;;
    * )
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
  shift
done

# Copy the SSH private key to the controller node.
scp -o StrictHostKeyChecking=no ${private_ssh_key_path} ${username}@${controller_node}:.ssh/id_rsa

# Install Docker in the controller node.
ssh -o StrictHostKeyChecking=no ${username}@${controller_node} "
  # Install Docker.
  sudo apt-get update
  sudo apt-get -y install apt-transport-https ca-certificates curl gnupg-agent \
      software-properties-common
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo add-apt-repository \"deb [arch=amd64] https://download.docker.com/linux/ubuntu \$(lsb_release -cs) stable\"
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io
"
