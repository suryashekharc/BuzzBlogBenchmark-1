# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

"""Utility Functions

This module contains utility functions that are used by notebooks and scripts
for data analysis.
"""

import os
import re
import tarfile
import yaml

import pandas as pd


def get_node_names(experiment_dirpath):
  return [dirname for dirname in os.listdir(os.path.join(experiment_dirpath, "logs")) if not dirname.startswith('.')]


def get_node_containers(experiment_dirpath, node_name):
  with open(os.path.join(experiment_dirpath, "conf", "system.yml")) as system_conf_file:
    return list(yaml.load(system_conf_file, Loader=yaml.Loader)[node_name]["containers"].keys())


def get_node_vcpus(experiment_dirpath, node_name):
  with open(os.path.join(experiment_dirpath, "conf", "system.yml")) as system_conf_file:
    for setup_script in yaml.load(system_conf_file, Loader=yaml.Loader)[node_name]["setup"]:
      cpu_no_re_match = re.match(r"for cpu_no in \{(\d+)", setup_script)
      if cpu_no_re_match:
        return int(cpu_no_re_match.groups()[0])
  return None


def has_cpu_interference(experiment_dirpath, node_name):
  for container in get_node_containers(experiment_dirpath, node_name):
    if re.findall("cpu_interference", container):
      return True
  return False


def get_container_option(experiment_dirpath, node_name, container, option):
  with open(os.path.join(experiment_dirpath, "conf", "system.yml")) as system_conf_file:
    system_conf = yaml.load(system_conf_file, Loader=yaml.Loader)
    try:
      return system_conf[node_name]["containers"][container]["options"][option]
    except KeyError:
      return None


def get_container_env(experiment_dirpath, node_name, container, env):
  with open(os.path.join(experiment_dirpath, "conf", "system.yml")) as system_conf_file:
    system_conf = yaml.load(system_conf_file, Loader=yaml.Loader)
    for kv in system_conf[node_name]["containers"][container]["options"]["env"]:
      if kv.split('=')[0] == env:
        return kv.split('=')[1]
  return None


def get_template_param(experiment_dirpath, node_name, template, param):
  with open(os.path.join(experiment_dirpath, "conf", "system.yml")) as system_conf_file:
    system_conf = yaml.load(system_conf_file, Loader=yaml.Loader)
    try:
      return system_conf[node_name]["templates"][template]["params"][param]
    except KeyError:
      return None


def get_node_label(experiment_dirpath, node_name):
  return "/".join(sorted(get_node_containers(experiment_dirpath, node_name)))


def get_rpc_df(experiment_dirpath):
  for node_name in get_node_names(experiment_dirpath):
    for tarball_name in os.listdir(os.path.join(experiment_dirpath, "logs", node_name)):
      tarball_path = os.path.join(experiment_dirpath, "logs", node_name, tarball_name)
      with tarfile.open(tarball_path, "r:gz") as tar:
        for filename in tar.getnames():
          if re.match(".*calls(_[0-9]+)?.csv", filename):
            with tar.extractfile(filename) as csvfile:
              yield (node_name, tarball_name,
                  pd.read_csv(csvfile, parse_dates=["timestamp"]).assign(node_name=node_name))


def get_query_df(experiment_dirpath):
  for node_name in get_node_names(experiment_dirpath):
    for tarball_name in os.listdir(os.path.join(experiment_dirpath, "logs", node_name)):
      tarball_path = os.path.join(experiment_dirpath, "logs", node_name, tarball_name)
      with tarfile.open(tarball_path, "r:gz") as tar:
        for filename in tar.getnames():
          if filename.endswith("queries.csv"):
            with tar.extractfile(filename) as csvfile:
              yield (node_name, tarball_name,
                  pd.read_csv(csvfile, parse_dates=["timestamp"]).assign(node_name=node_name))


def get_redis_df(experiment_dirpath):
  for node_name in get_node_names(experiment_dirpath):
    for tarball_name in os.listdir(os.path.join(experiment_dirpath, "logs", node_name)):
      tarball_path = os.path.join(experiment_dirpath, "logs", node_name, tarball_name)
      with tarfile.open(tarball_path, "r:gz") as tar:
        for filename in tar.getnames():
          if filename.endswith("redis.csv"):
            with tar.extractfile(filename) as csvfile:
              yield (node_name, tarball_name,
                  pd.read_csv(csvfile, parse_dates=["timestamp"]).assign(node_name=node_name))


def get_loadgen_df(experiment_dirpath):
  for node_name in get_node_names(experiment_dirpath):
    for tarball_name in os.listdir(os.path.join(experiment_dirpath, "logs", node_name)):
      tarball_path = os.path.join(experiment_dirpath, "logs", node_name, tarball_name)
      with tarfile.open(tarball_path, "r:gz") as tar:
        for filename in tar.getnames():
          if re.match(".*loadgen([0-9]+)?.csv", filename):
            with tar.extractfile(filename) as csvfile:
              yield (node_name, tarball_name,
                  pd.read_csv(csvfile, parse_dates=["timestamp"]).assign(node_name=node_name))


def get_collectl_cpu_df(experiment_dirpath):
  for node_name in get_node_names(experiment_dirpath):
    for tarball_name in os.listdir(os.path.join(experiment_dirpath, "logs", node_name)):
      tarball_path = os.path.join(experiment_dirpath, "logs", node_name, tarball_name)
      with tarfile.open(tarball_path, "r:gz") as tar:
        for filename in tar.getnames():
          if filename.endswith(".cpu.csv"):
            with tar.extractfile(filename) as csvfile:
              yield (node_name, tarball_name,
                  pd.read_csv(csvfile, parse_dates=["timestamp"]).assign(node_name=node_name))


def get_collectl_mem_df(experiment_dirpath):
  for node_name in get_node_names(experiment_dirpath):
    for tarball_name in os.listdir(os.path.join(experiment_dirpath, "logs", node_name)):
      tarball_path = os.path.join(experiment_dirpath, "logs", node_name, tarball_name)
      with tarfile.open(tarball_path, "r:gz") as tar:
        for filename in tar.getnames():
          if filename.endswith(".numa.csv"):
            with tar.extractfile(filename) as csvfile:
              yield (node_name, tarball_name,
                  pd.read_csv(csvfile, parse_dates=["timestamp"]).assign(node_name=node_name))


def get_collectl_dsk_df(experiment_dirpath):
  for node_name in get_node_names(experiment_dirpath):
    for tarball_name in os.listdir(os.path.join(experiment_dirpath, "logs", node_name)):
      tarball_path = os.path.join(experiment_dirpath, "logs", node_name, tarball_name)
      with tarfile.open(tarball_path, "r:gz") as tar:
        for filename in tar.getnames():
          if filename.endswith(".dsk.csv"):
            with tar.extractfile(filename) as csvfile:
              yield (node_name, tarball_name,
                  pd.read_csv(csvfile, parse_dates=["timestamp"]).assign(node_name=node_name))


def get_tcplistenbl_df(experiment_dirpath):
  tarball_name = "tcplistenbl-bpftrace.tar.gz"
  for node_name in get_node_names(experiment_dirpath):
    if tarball_name in os.listdir(os.path.join(experiment_dirpath, "logs", node_name)):
      tarball_path = os.path.join(experiment_dirpath, "logs", node_name, tarball_name)
      with tarfile.open(tarball_path, "r:gz") as tar:
        if "./log.csv" in tar.getnames():
          with tar.extractfile("./log.csv") as csvfile:
            yield (node_name, tarball_name,
                pd.read_csv(csvfile, parse_dates=["timestamp"]).assign(node_name=node_name))


def get_tcpretrans_df(experiment_dirpath):
  tarball_name = "tcpretrans-bpftrace.tar.gz"
  for node_name in get_node_names(experiment_dirpath):
    if tarball_name in os.listdir(os.path.join(experiment_dirpath, "logs", node_name)):
      tarball_path = os.path.join(experiment_dirpath, "logs", node_name, tarball_name)
      with tarfile.open(tarball_path, "r:gz") as tar:
        if "./log.csv" in tar.getnames():
          with tar.extractfile("./log.csv") as csvfile:
            yield (node_name, tarball_name,
                pd.read_csv(csvfile, parse_dates=["timestamp"]).assign(node_name=node_name))


def get_experiment_start_time(experiment_dirpath):
  requests = pd.concat([df[2] for df in get_loadgen_df(experiment_dirpath)])
  return requests["timestamp"].values.min()


def get_experiment_end_time(experiment_dirpath):
  requests = pd.concat([df[2] for df in get_loadgen_df(experiment_dirpath)])
  return requests["timestamp"].values.max()


def get_experiment_total_duration(experiment_dirpath):
  with open(os.path.join(experiment_dirpath, "conf", "workload.yml")) as workload_conf_file:
    return yaml.load(workload_conf_file, Loader=yaml.Loader)["duration"]["total"]


def get_experiment_ramp_up_duration(experiment_dirpath):
  with open(os.path.join(experiment_dirpath, "conf", "workload.yml")) as workload_conf_file:
    return yaml.load(workload_conf_file, Loader=yaml.Loader)["duration"]["ramp_up"]


def get_experiment_ramp_down_duration(experiment_dirpath):
  with open(os.path.join(experiment_dirpath, "conf", "workload.yml")) as workload_conf_file:
    return yaml.load(workload_conf_file, Loader=yaml.Loader)["duration"]["ramp_down"]


def build_requests_df(experiment_dirpath, exploded_window_in_ms=None):
  # Extract experiment information.
  start_time = get_experiment_start_time(experiment_dirpath)
  # Build data frame.
  requests = pd.concat([df[2] for df in get_loadgen_df(experiment_dirpath)])
  # (Re) Build columns.
  requests["timestamp"] = requests.apply(lambda r: (r["timestamp"] - start_time).total_seconds(), axis=1)
  requests["window_1000"] = requests["timestamp"].round(0).multiply(1000)
  requests["window_10"] = requests["timestamp"].round(2).multiply(1000)
  requests["latency"] = requests["latency"].multiply(1000)
  if exploded_window_in_ms:
    requests["window"] = requests.apply(lambda r: range(int(r["timestamp"] * 1000) // exploded_window_in_ms,
        int(r["timestamp"] * 1000 + r["latency"]) // exploded_window_in_ms + 1), axis=1)
    requests = requests.explode("window")
  # (Re) Create index
  requests.set_index("timestamp", inplace=True)
  requests.sort_index(inplace=True)
  return requests


def build_redis_df(experiment_dirpath, exploded_window_in_ms=None):
  # Extract experiment information.
  start_time = get_experiment_start_time(experiment_dirpath)
  # Build data frame.
  redis = pd.concat([df[2] for df in get_redis_df(experiment_dirpath)])
  # (Re) Build columns.
  redis["timestamp"] = redis.apply(lambda r: (r["timestamp"] - start_time).total_seconds(), axis=1)
  redis["window_10"] = redis["timestamp"].round(2).multiply(1000)
  redis["window_1000"] = redis["timestamp"].round(0).multiply(1000)
  redis["latency"] = redis["latency"].multiply(1000)
  if exploded_window_in_ms:
    redis["window"] = redis.apply(lambda r: range(int(r["timestamp"] * 1000) // exploded_window_in_ms,
        int(r["timestamp"] * 1000 + r["latency"]) // exploded_window_in_ms + 1), axis=1)
    redis = redis.explode("window")
  # (Re) Create index
  redis.set_index("timestamp", inplace=True)
  redis.sort_index(inplace=True)
  return redis


def build_rpc_df(experiment_dirpath, exploded_window_in_ms=None):
  # Extract experiment information.
  start_time = get_experiment_start_time(experiment_dirpath)
  # Build data frame.
  rpc = pd.concat([df[2] for df in get_rpc_df(experiment_dirpath)])
  # (Re) Build columns.
  rpc["timestamp"] = rpc.apply(lambda r: (r["timestamp"] - start_time).total_seconds(), axis=1)
  rpc["window_10"] = rpc["timestamp"].round(2).multiply(1000)
  rpc["window_1000"] = rpc["timestamp"].round(0).multiply(1000)
  rpc["latency"] = rpc["latency"].multiply(1000)
  if exploded_window_in_ms:
    rpc["window"] = rpc.apply(lambda r: range(int(r["timestamp"] * 1000) // exploded_window_in_ms,
        int(r["timestamp"] * 1000 + r["latency"]) // exploded_window_in_ms + 1), axis=1)
    rpc = rpc.explode("window")
  # (Re) Create index
  rpc.set_index("timestamp", inplace=True)
  rpc.sort_index(inplace=True)
  return rpc


def build_query_df(experiment_dirpath, exploded_window_in_ms=None):
  # Extract experiment information.
  start_time = get_experiment_start_time(experiment_dirpath)
  # Build data frame.
  query = pd.concat([df[2] for df in get_query_df(experiment_dirpath)])
  # (Re) Build columns.
  query["timestamp"] = query.apply(lambda r: (r["timestamp"] - start_time).total_seconds(), axis=1)
  query["window_10"] = query["timestamp"].round(2).multiply(1000)
  query["window_1000"] = query["timestamp"].round(0).multiply(1000)
  query["latency"] = query["latency"].multiply(1000)
  if exploded_window_in_ms:
    query["window"] = query.apply(lambda r: range(int(r["timestamp"] * 1000) // exploded_window_in_ms,
        int(r["timestamp"] * 1000 + r["latency"]) // exploded_window_in_ms + 1), axis=1)
    query = query.explode("window")
  # (Re) Create index.
  query.set_index("timestamp", inplace=True)
  query.sort_index(inplace=True)
  return query


def build_collectl_cpu_df(experiment_dirpath):
  # Extract experiment information.
  start_time = get_experiment_start_time(experiment_dirpath)
  # Build data frame.
  cpu = pd.concat([df[2] for df in get_collectl_cpu_df(experiment_dirpath)])
  # (Re) Build columns.
  cpu["timestamp"] = cpu.apply(lambda r: (r["timestamp"] - start_time).total_seconds(), axis=1)
  cpu["window_1000"] = cpu["timestamp"].round(0).multiply(1000)
  # (Re) Create index.
  cpu.set_index("timestamp", inplace=True)
  cpu.sort_index(inplace=True)
  return cpu


def build_collectl_dsk_df(experiment_dirpath):
  # Extract experiment information.
  start_time = get_experiment_start_time(experiment_dirpath)
  # Build data frame.
  dsk = pd.concat([df[2] for df in get_collectl_dsk_df(experiment_dirpath)])
  # (Re) Build columns.
  dsk["timestamp"] = dsk.apply(lambda r: (r["timestamp"] - start_time).total_seconds(), axis=1)
  dsk["window_1000"] = dsk["timestamp"].round(0).multiply(1000)
  # (Re) Create index.
  dsk.set_index("timestamp", inplace=True)
  dsk.sort_index(inplace=True)
  return dsk


def build_collectl_mem_df(experiment_dirpath):
  # Extract experiment information.
  start_time = get_experiment_start_time(experiment_dirpath)
  # Build data frame.
  mem = pd.concat([df[2] for df in get_collectl_mem_df(experiment_dirpath)])
  # (Re) Build columns.
  mem["timestamp"] = mem.apply(lambda r: (r["timestamp"] - start_time).total_seconds(), axis=1)
  mem["window_1000"] = mem["timestamp"].round(0).multiply(1000)
  # (Re) Create index.
  mem.set_index("timestamp", inplace=True)
  mem.sort_index(inplace=True)
  return mem


def build_tcp_listenbl_df(experiment_dirpath):
  # Extract experiment information.
  start_time = get_experiment_start_time(experiment_dirpath) - pd.Timedelta(hours=6)
  # Build data frame.
  bl = pd.concat([df[2] for df in get_tcplistenbl_df(experiment_dirpath)])
  # (Re) Build columns.
  bl["timestamp"] = bl.apply(lambda r: (r["timestamp"] - start_time).total_seconds(), axis=1)
  bl["window_1000"] = bl["timestamp"].round(0).multiply(1000)
  bl["window_10"] = bl["timestamp"].round(2).multiply(1000)
  # (Re) Create index.
  bl.set_index("timestamp", inplace=True)
  bl.sort_index(inplace=True)
  return bl


def build_tcp_retrans_df(experiment_dirpath):
  # Extract experiment information.
  start_time = get_experiment_start_time(experiment_dirpath) - pd.Timedelta(hours=6)
  # Build data frame.
  retrans = pd.concat([df[2] for df in get_tcpretrans_df(experiment_dirpath)])
  # (Re) Build columns.
  retrans["timestamp"] = retrans.apply(lambda r: (r["timestamp"] - start_time).total_seconds(), axis=1)
  retrans["window_1000"] = retrans["timestamp"].round(0).multiply(1000)
  retrans["window_10"] = retrans["timestamp"].round(2).multiply(1000)
  # (Re) Create index.
  retrans.set_index("timestamp", inplace=True)
  retrans.sort_index(inplace=True)
  return retrans
