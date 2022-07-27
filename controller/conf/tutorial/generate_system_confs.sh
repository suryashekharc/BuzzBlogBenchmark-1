#!/bin/bash

# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Process command-line arguments.
set -u
while [[ $# > 1 ]]; do
  case $1 in
    --cloudlabprofile )
      cloudlabprofile=$2
      ;;
    --cpu )
      cpu=$2
      ;;
    --gb )
      gb=$2
      ;;
    --monitoringinterval )
      monitoringinterval=$2
      ;;
    --gunicornacceptqueue )
      gunicornacceptqueue=$2
      ;;
    --gunicornworkers )
      gunicornworkers=$2
      ;;
    --gunicornthreads )
      gunicornthreads=$2
      ;;
    --gunicornserviceconnpoolsize )
      gunicornserviceconnpoolsize=$2
      ;;
    --thriftacceptqueue )
      thriftacceptqueue=$2
      ;;
    --thriftthreads )
      thriftthreads=$2
      ;;
    --thriftserviceconnpoolsize )
      thriftserviceconnpoolsize=$2
      ;;
    --thriftpgconnpoolsize )
      thriftpgconnpoolsize=$2
      ;;
    --thriftredisconnpoolsize )
      thriftredisconnpoolsize=$2
      ;;
    --pgmaxconnections )
      pgmaxconnections=$2
      ;;
    --redisacceptqueue )
      redisacceptqueue=$2
      ;;
    --redismaxclients )
      redismaxclients=$2
      ;;
    --redissnapshotinterval )
      redissnapshotinterval=$2
      ;;
    --ninvalidwords )
      ninvalidwords=$2
      ;;
    * )
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
  shift
done

# Set current directory to the directory of this script.
cd "$(dirname "$0")"

# Generate system configuration files.
for it_cpu in $cpu; do
  for it_gb in $gb; do
    for it_monitoringinterval in $monitoringinterval; do
      for it_gunicornacceptqueue in $gunicornacceptqueue; do
        for it_gunicornworkers in $gunicornworkers; do
          for it_gunicornthreads in $gunicornthreads; do
            for it_gunicornserviceconnpoolsize in $gunicornserviceconnpoolsize; do
              for it_thriftacceptqueue in $thriftacceptqueue; do
                for it_thriftthreads in $thriftthreads; do
                  for it_thriftserviceconnpoolsize in $thriftserviceconnpoolsize; do
                    for it_thriftpgconnpoolsize in $thriftpgconnpoolsize; do
                      for it_thriftredisconnpoolsize in $thriftredisconnpoolsize; do
                        for it_pgmaxconnections in $pgmaxconnections; do
                          for it_redisacceptqueue in $redisacceptqueue; do
                            for it_redismaxclients in $redismaxclients; do
                              for it_redissnapshotinterval in $redissnapshotinterval; do
                                for it_ninvalidwords in $ninvalidwords; do
                                  filename="${cloudlabprofile}_"
                                  filename+="${it_cpu}CPU_"
                                  filename+="${it_gb}GB_"
                                  filename+="${it_monitoringinterval}MI_"
                                  filename+="${it_gunicornacceptqueue}GACCEPTQ_"
                                  filename+="${it_gunicornworkers}GWORKERS_"
                                  filename+="${it_gunicornthreads}GTHREADS_"
                                  filename+="${it_gunicornserviceconnpoolsize}GSERVICECP_"
                                  filename+="${it_thriftacceptqueue}TACCEPTQ_"
                                  filename+="${it_thriftthreads}TTHREADS_"
                                  filename+="${it_thriftserviceconnpoolsize}TSERVICECP_"
                                  filename+="${it_thriftpgconnpoolsize}TPGCP_"
                                  filename+="${it_thriftredisconnpoolsize}TREDISCP_"
                                  filename+="${it_pgmaxconnections}PGMAXCONNS_"
                                  filename+="${it_redisacceptqueue}RACCEPTQ_"
                                  filename+="${it_redismaxclients}RMAXCLIENTS_"
                                  filename+="${it_redissnapshotinterval}RSNAPSHOTINT_"
                                  filename+="${it_ninvalidwords}NINVALIDWORDS.yml"
                                  cp ${cloudlabprofile}_${it_cpu}CPU_TEMPLATE.yml $filename
                                  sed -i "s/{{GB}}/${it_gb}g/g" $filename
                                  sed -i "s/{{MONITORINGINTERVAL}}/${it_monitoringinterval}/g" $filename
                                  sed -i "s/{{GUNICORNACCEPTQUEUE}}/${it_gunicornacceptqueue}/g" $filename
                                  sed -i "s/{{GUNICORNWORKERS}}/${it_gunicornworkers}/g" $filename
                                  sed -i "s/{{GUNICORNTHREADS}}/${it_gunicornthreads}/g" $filename
                                  sed -i "s/{{GUNICORNSERVICECONNPOOLSIZE}}/${it_gunicornserviceconnpoolsize}/g" $filename
                                  sed -i "s/{{THRIFTACCEPTQUEUE}}/${it_thriftacceptqueue}/g" $filename
                                  sed -i "s/{{THRIFTTHREADS}}/${it_thriftthreads}/g" $filename
                                  sed -i "s/{{THRIFTSERVICECONNPOOLSIZE}}/${it_thriftserviceconnpoolsize}/g" $filename
                                  sed -i "s/{{THRIFTPGCONNPOOLSIZE}}/${it_thriftpgconnpoolsize}/g" $filename
                                  sed -i "s/{{THRIFTREDISCONNPOOLSIZE}}/${it_thriftredisconnpoolsize}/g" $filename
                                  sed -i "s/{{PGMAXCONNECTIONS}}/${it_pgmaxconnections}/g" $filename
                                  sed -i "s/{{REDISACCEPTQUEUE}}/${it_redisacceptqueue}/g" $filename
                                  sed -i "s/{{REDISMAXCLIENTS}}/${it_redismaxclients}/g" $filename
                                  sed -i "s/{{REDISSNAPSHOTINTERVAL}}/${it_redissnapshotinterval}/g" $filename
                                  sed -i "s/{{NINVALIDWORDS}}/${it_ninvalidwords}/g" $filename
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
