# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

"""List RPCs, database queries, and Redis commands of VLRT requests.

This script gives details of the execution path of requests with VLRT.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
from analysis.utils.utils import *


def list_vlrt_requests(experiment_dirpath, threshold):
  # Build data frames.
  requests = build_requests_df(experiment_dirpath)
  rpc = build_rpc_df(experiment_dirpath)
  query = build_query_df(experiment_dirpath)
  redis = build_redis_df(experiment_dirpath)
  # Filter VLRT requests.
  requests = requests[(requests["latency"] > threshold)]
  # List RPCs, database queries, and Redis commands of VLRT requests.
  for request in requests.to_dict("records"):
    print("Request ID: %s" % request["request_id"])
    print("  Latency: %s" % request["latency"])
    print("  Type: %s" % request["type"])
    print("  RPCs:")
    for request_rpc in rpc[(rpc["request_id"] == request["request_id"])].to_dict("records"):
      print("    %s - %s" % (request_rpc["function"], request_rpc["latency"]))
    print("  Queries:")
    for request_query in query[(query["request_id"] == request["request_id"])].to_dict("records"):
      print("    %s - %s" % (request_query["dbname"] + ":" + request_query["type"], request_query["latency"]))
    print("  Redis Commands:")
    for redis_command in redis[(redis["request_id"] == request["request_id"])].to_dict("records"):
      print("    %s - %s" % (redis_command["service_name"] + ":" + redis_command["command"], redis_command["latency"]))


def main():
  # Parse command-line arguments.
  parser = argparse.ArgumentParser(description="List RPCs, database queries, and Redis commands of VLRT requests")
  parser.add_argument("--experiment_dirname", required=False, action="store", type=str,
      help="Name of directory containing experiment data in `../data`", default="")
  parser.add_argument("--threshold_in_ms", required=True, action="store", type=str,
      help="Latency threshold for VLRT requests (in ms)")
  args = parser.parse_args()
  # List experiment(s) directory.
  experiment_dirpaths = [os.path.join(os.path.abspath(""), "..", "data", dirname)
      for dirname in ([args.experiment_dirname] if args.experiment_dirname else
          os.listdir(os.path.join(os.path.abspath(""), "..", "data")))
      if re.findall("BuzzBlogBenchmark_", dirname) and not re.findall(".tar.gz", dirname)]
  # Analyze log dataset.
  for experiment_dirpath in experiment_dirpaths:
    print("Processing %s:" % experiment_dirpath)
    list_vlrt_requests(experiment_dirpath, int(args.threshold_in_ms))


if __name__ == "__main__":
  main()
