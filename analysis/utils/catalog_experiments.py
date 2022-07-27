# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer Systems

"""Catalog experiment configurations and results.

This script generates a CSV file with experiment configuration and performance
data.
"""

import argparse
import csv
import os
import pandas as pd
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
from analysis.utils.utils import *
from analysis.utils.plot_experiment_graphs import RequestLogAnalysis, CollectlCPULogAnalysis, CollectlDskLogAnalysis, \
    CollectlMemLogAnalysis, QueryLogAnalysis, RedisLogAnalysis, RPCLogAnalysis, ServerRequestLogAnalysis, \
    TCPListenBacklogLogAnalysis, TCPRetransLogAnalysis


def process_config(experiment_dirpath):
  # Get workload attributes.
  with open(os.path.join(experiment_dirpath, "conf", "workload.yml")) as workload_conf_file:
    workload_conf = yaml.load(workload_conf_file, Loader=yaml.Loader)
  attrs = {
      "workload_sessions": workload_conf["sessions"],
      "workload_total_duration": workload_conf["duration"]["total"],
      "workload_ramp_up_duration": workload_conf["duration"]["ramp_up"],
      "workload_ramp_down_duration": workload_conf["duration"]["ramp_down"],
      "workload_think_time": workload_conf["think_time"],
      "workload_think_time_distribution": workload_conf["think_time_distribution"],
      "workload_burstiness_think_time": workload_conf["burstiness"]["think_time"] \
          if "burstiness" in workload_conf else None,
      "workload_burstiness_turn_on_prob": workload_conf["burstiness"]["turn_on_prob"] \
          if "burstiness" in workload_conf else None,
      "workload_burstiness_turn_off_prob": workload_conf["burstiness"]["turn_off_prob"] \
          if "burstiness" in workload_conf else None,
  }
  # Get system attributes.
  with open(os.path.join(experiment_dirpath, "conf", "system.yml")) as system_conf_file:
    system_conf = yaml.load(system_conf_file, Loader=yaml.Loader)
  for node_name in get_node_names(experiment_dirpath):
    node_label = get_node_label(experiment_dirpath, node_name)
    attrs[node_label + "_system_cpu_count"] = get_node_vcpus(experiment_dirpath, node_name)
    attrs[node_label + "_system_cpu_interference"] = has_cpu_interference(experiment_dirpath, node_name)
    for option in ["memory"]:
      if get_container_option(experiment_dirpath, node_name, node_label, option):
        attrs[node_label + "_system_%s" % option] = get_container_option(experiment_dirpath, node_name, node_label,
            option)
    for env in ["backlog", "workers", "threads", "microservice_connection_pool_size", "postgres_connection_pool_size",
        "redis_connection_pool_size", "accept_backlog"]:
      if get_container_env(experiment_dirpath, node_name, node_label, env):
        attrs[node_label + "_system_%s" % env] = get_container_env(experiment_dirpath, node_name, node_label, env)
    for (template, param) in [("redis.conf", "tcp_backlog"), ("redis.conf", "snapshot_interval"),
        ("redis.conf", "maxclients")]:
      if get_template_param(experiment_dirpath, node_name, template, param):
        attrs[node_label + "_system_%s" % param] = get_template_param(experiment_dirpath, node_name, template, param)
    return attrs


def process_loadgen_df(experiment_dirpath):
  request_log_analysis = RequestLogAnalysis(experiment_dirpath)
  return request_log_analysis.calculate_stats()


def process_collectl_cpu_df(experiment_dirpath):
  collectl_cpu_log_analysis = CollectlCPULogAnalysis(experiment_dirpath)
  return collectl_cpu_log_analysis.calculate_stats()


def process_collectl_dsk_df(experiment_dirpath):
  collectl_dsk_log_analysis = CollectlDskLogAnalysis(experiment_dirpath)
  return collectl_dsk_log_analysis.calculate_stats()


def process_collectl_mem_df(experiment_dirpath):
  collectl_mem_log_analysis = CollectlMemLogAnalysis(experiment_dirpath)
  return collectl_mem_log_analysis.calculate_stats()


def process_query_df(experiment_dirpath):
  query_log_analysis = QueryLogAnalysis(experiment_dirpath)
  return query_log_analysis.calculate_stats()


def process_redis_df(experiment_dirpath):
  redis_log_analysis = RedisLogAnalysis(experiment_dirpath)
  return redis_log_analysis.calculate_stats()


def process_rpc_df(experiment_dirpath):
  rpc_log_analysis = RPCLogAnalysis(experiment_dirpath)
  return rpc_log_analysis.calculate_stats()


def process_server_request_logs(experiment_dirpath):
  server_request_log_analysis = ServerRequestLogAnalysis(experiment_dirpath)
  return server_request_log_analysis.calculate_stats()


def process_tcp_listenbacklog_logs(experiment_dirpath):
  tcp_listen_backlog_log_analysis = TCPListenBacklogLogAnalysis(experiment_dirpath)
  return tcp_listen_backlog_log_analysis.calculate_stats()


def process_tcp_retrans_logs(experiment_dirpath):
  tcp_retrans_log_analysis = TCPRetransLogAnalysis(experiment_dirpath)
  return tcp_retrans_log_analysis.calculate_stats()


def main():
  # Parse command-line arguments.
  parser = argparse.ArgumentParser(description="Catalog experiments")
  parser.add_argument("--csv_filename", required=True, action="store", type=str, help="CSV output file")
  parser.add_argument("--hardware", required=True, action="store", type=str, help="Hardware used to run experiments")
  parser.add_argument("--experiment_dirname", required=False, action="store", type=str,
      help="Experiment directory in `data`", default="")
  args = parser.parse_args()
  # List experiment(s) directory.
  experiment_dirpaths = [os.path.join(os.path.abspath(""), "..", "data", dirname)
      for dirname in ([args.experiment_dirname] if args.experiment_dirname else
          os.listdir(os.path.join(os.path.abspath(""), "..", "data")))
      if re.findall("BuzzBlogBenchmark_", dirname) and not re.findall(".tar.gz", dirname)]
  # Load data in the specified CSV file, if it exists.
  try:
    with open(os.path.join(os.path.abspath(""), "..", "data", args.csv_filename)) as csvfile:
      reader = csv.DictReader(csvfile)
      data = [row for row in reader]
  except FileNotFoundError:
    data = []
  # Process experiment(s) results.
  with open(os.path.join(os.path.abspath(""), "..", "data", args.csv_filename), 'w') as csvfile:
    writer = None
    for experiment_dirpath in experiment_dirpaths:
      # Skip experiment already processed.
      for experiment in data:
        if experiment["dirname"] == os.path.basename(experiment_dirpath):
          print("Skipped %s" % os.path.basename(experiment_dirpath))
          # Write experiment stats in the specified CSV file.
          if not writer:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(list(experiment.keys())))
            writer.writeheader()
          writer.writerow(experiment)
          break
      else:
        print("Processing %s" % os.path.basename(experiment_dirpath))
        experiment = {"dirname": os.path.basename(experiment_dirpath), "hardware": args.hardware}
        try:
          print("\tProcessing configuration files")
          experiment.update(process_config(experiment_dirpath))
          print("\tProcessing loadgen dataset")
          experiment.update(process_loadgen_df(experiment_dirpath))
          print("\tProcessing Collectl CPU dataset")
          experiment.update(process_collectl_cpu_df(experiment_dirpath))
          print("\tProcessing Collectl Dsk dataset")
          experiment.update(process_collectl_dsk_df(experiment_dirpath))
          print("\tProcessing Collectl Mem dataset")
          experiment.update(process_collectl_mem_df(experiment_dirpath))
          print("\tProcessing Redis dataset")
          experiment.update(process_redis_df(experiment_dirpath))
          print("\tProcessing RPC dataset")
          experiment.update(process_rpc_df(experiment_dirpath))
          print("\tProcessing query dataset")
          experiment.update(process_query_df(experiment_dirpath))
          print("\tProcessing server request logs")
          experiment.update(process_server_request_logs(experiment_dirpath))
          print("\tProcessing TCP listen backlog logs")
          experiment.update(process_tcp_listenbacklog_logs(experiment_dirpath))
          print("\tProcessing TCP retransmission logs")
          experiment.update(process_tcp_retrans_logs(experiment_dirpath))
        except Exception as e:
          print("\tFailed: %s" % str(e))
          continue
        # Write experiment stats in the specified CSV file.
        if not writer:
          writer = csv.DictWriter(csvfile, fieldnames=sorted(list(experiment.keys())))
          writer.writeheader()
        writer.writerow(experiment)


if __name__ == "__main__":
  main()
