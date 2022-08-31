# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

"""Plot graphs for experiments.

This script generates image files with system performance graphs.
"""

import argparse
import os
import pprint
import re
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
from analysis.utils.utils import *


class LogAnalysis:
  def __init__(self, experiment_dirpath, output_dirpath):
    self._experiment_dirpath = experiment_dirpath
    self._output_dirpath = output_dirpath
    # Extract experiment information
    self._total_duration = get_experiment_total_duration(experiment_dirpath)
    self._ramp_up_duration = get_experiment_ramp_up_duration(experiment_dirpath)
    self._ramp_down_duration = get_experiment_ramp_down_duration(experiment_dirpath)
    self._node_names = get_node_names(experiment_dirpath)
    self._node_labels = {node_name: get_node_label(experiment_dirpath, node_name) for node_name in self._node_names}

  def plot(self, distribution):
    for attr_name in dir(self):
      if attr_name.startswith("plot_") and (distribution or "distribution" not in attr_name):
        getattr(self, attr_name)()

  def save_fig(func):
    def inner(self, *args, **kwargs):
      fig = func(self, *args, **kwargs)
      if self._output_dirpath and fig:
        fig.savefig(os.path.join(self._output_dirpath, "%s.png" % func.__name__))
    return inner


class RequestLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._requests = build_requests_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_number_of_successful_failed_requests(self):
    # Data frame
    df = self._requests[(self._requests.index >= self._ramp_up_duration) &
        (self._requests.index <= self._total_duration - self._ramp_down_duration)].\
        groupby(["status"]).count()["method"]
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(18, 6))
    ax = fig.gca()
    df.plot(ax=ax, kind="pie", title="Number of Successful/Failed Requests", xlabel="", ylabel="", legend=True)
    return fig

  @LogAnalysis.save_fig
  def plot_http_status_code_of_failed_requests(self):
    # Data frame
    df = self._requests[(self._requests["status"] == "failed") &
        (self._requests.index >= self._ramp_up_duration) &
        (self._requests.index <= self._total_duration - self._ramp_down_duration)].\
        groupby(["status_code"]).count()["method"]
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(18, 6))
    ax = fig.gca()
    df.plot(ax=ax, kind="pie", title="HTTP Status Code of Failed Requests", xlabel="", ylabel="", legend=True)
    return fig

  @LogAnalysis.save_fig
  def plot_number_of_read_write_requests(self):
    # Data frame
    df = self._requests[(self._requests.index >= self._ramp_up_duration) &
        (self._requests.index <= self._total_duration - self._ramp_down_duration)].\
        groupby(["rw"]).count()["method"]
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(18, 6))
    ax = fig.gca()
    df.plot(ax=ax, kind="pie", title="Number of Read/Write Requests", xlabel="", ylabel="", legend=True)
    return fig

  @LogAnalysis.save_fig
  def plot_number_of_requests_of_each_type(self):
    # Data frame
    df = self._requests[(self._requests.index >= self._ramp_up_duration) &
        (self._requests.index <= self._total_duration - self._ramp_down_duration)].\
        groupby(["type", "status"]).count()["method"].unstack().fillna(0)
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(18, 12))
    ax = fig.gca()
    ax.grid(alpha=0.75)
    df.plot(ax=ax, kind="bar", stacked=True, title="Number of Requests of Each Type", xlabel="",
        ylabel="Count (Requests)", color={"failed": "red", "successful": "blue"}, legend=True, grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_latency_distribution_of_requests(self, latency_bin_in_ms=25, text_y=None):
    # Data frame
    df = self._requests[(self._requests["status"] == "successful") &
        (self._requests.index >= self._ramp_up_duration) &
        (self._requests.index <= self._total_duration - self._ramp_down_duration)]
    if df.empty:
      return None
    df["latency_bin"] = df.apply(lambda r: int(r["latency"] // latency_bin_in_ms), axis=1)
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca(xlabel="Latency (millisec)", ylabel="Count (Requests)")
    ax.grid(alpha=0.75)
    ax.set_yscale("log")
    max_latency_in_s = (df["latency"].max() + 2 * latency_bin_in_ms) / 1000
    ax.set_xlim((0, (1000 // latency_bin_in_ms) * max_latency_in_s))
    ax.set_xticks(range(int((1000 // latency_bin_in_ms) * max_latency_in_s) + 1))
    ax.set_xticklabels(range(0, (int((1000 // latency_bin_in_ms) * max_latency_in_s) + 1) * latency_bin_in_ms,
        latency_bin_in_ms))
    if text_y:
      p50 = df["latency"].quantile(0.50)
      ax.axvline(x=p50 / latency_bin_in_ms, ls="dotted", lw=5, color="darkorange")
      ax.text(x=p50 / latency_bin_in_ms, y=text_y, s=" P50", fontsize=22, color="darkorange")
      p999 = df["latency"].quantile(0.999)
      ax.axvline(x=p999 / latency_bin_in_ms, ls="dotted", lw=5, color="darkorange")
      ax.text(x=p999 / latency_bin_in_ms, y=text_y, s=" P99.9", fontsize=22, color="darkorange")
    df["latency_bin"].plot(ax=ax, kind="hist", title="Latency Distribution of Requests",
        bins=range(int((1000 // latency_bin_in_ms) * max_latency_in_s)), grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_requests(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._requests[(self._requests["status"] == "successful") & (self._requests.index >= min_time) &
        (self._requests.index <= max_time)].groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).\
        unstack()
    if df.empty:
      return None
    df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, window), fill_value=0)
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.set_ylim((0, df.values.max()))
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    df.plot(ax=ax, kind="line", title="Instantaneous Latency of Requests", xlabel="Time (millisec)",
        ylabel="Latency (millisec)", grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_request_throughput(self, interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._requests[(self._requests.index >= min_time) & (self._requests.index <= max_time)].\
        groupby(["window_%s" % window, "status"])["window_%s" % window].count().unstack().fillna(0)
    if df.empty:
      return None
    df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, window), fill_value=0)
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    df.plot(ax=ax, kind="line", title="Request Throughput", xlabel="Time (millisec)",
        ylabel="Throughput (Requests/Sec)" if not interval else "Throughput (Requests/10ms)",
        color={"failed": "red", "successful": "blue"}, legend=True, grid=True,
        xticks=range(int(df.index.min()), int(df.index.max()) + 1, 60000))
    return fig

  def calculate_stats(self):
    requests = self._requests[(self._requests.index >= self._ramp_up_duration) &
        (self._requests.index <= self._total_duration - self._ramp_down_duration)]
    throughput = requests.groupby(["window_1000"])["window_1000"].count().\
        reindex(range(int(requests["window_1000"].min()), int(requests["window_1000"].max()) + 1, 1000), fill_value=0)
    return {
        "requests_count_total": requests.shape[0],
        "requests_count_successful": requests[requests["status"] == "successful"]["status"].count(),
        "requests_count_failed": requests[requests["status"] == "failed"]["status"].count(),
        "requests_count_read": requests[requests["rw"] == "read"]["rw"].count(),
        "requests_count_write": requests[requests["rw"] == "write"]["rw"].count(),
        "requests_ratio_successful": requests[requests["status"] == "successful"]["status"].count() / requests.shape[0],
        "requests_ratio_failed": requests[requests["status"] == "failed"]["status"].count() / requests.shape[0],
        "requests_ratio_read": requests[requests["rw"] == "read"]["rw"].count() / requests.shape[0],
        "requests_ratio_write": requests[requests["rw"] == "write"]["rw"].count() / requests.shape[0],
        "requests_latency_p50": requests[requests["status"] == "successful"]["latency"].quantile(0.50),
        "requests_latency_p95": requests[requests["status"] == "successful"]["latency"].quantile(0.95),
        "requests_latency_p99": requests[requests["status"] == "successful"]["latency"].quantile(0.99),
        "requests_latency_p999": requests[requests["status"] == "successful"]["latency"].quantile(0.999),
        "requests_latency_avg": requests[requests["status"] == "successful"]["latency"].mean(),
        "requests_latency_std": requests[requests["status"] == "successful"]["latency"].std(),
        "requests_latency_max": requests[requests["status"] == "successful"]["latency"].max(),
        "requests_throughput_p50": throughput.quantile(0.50),
        "requests_throughput_p95": throughput.quantile(0.95),
        "requests_throughput_p99": throughput.quantile(0.99),
        "requests_throughput_p999": throughput.quantile(0.999),
        "requests_throughput_avg": throughput.mean(),
        "requests_throughput_std": throughput.std(),
        "requests_throughput_max": throughput.max(),
    }


class CollectlCPULogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._cpu = build_collectl_cpu_df(experiment_dirpath)
    self._cpu_cores = {node_name: range(get_node_vcpus(experiment_dirpath, node_name) or 128)
        for node_name in self._node_names}

  @LogAnalysis.save_fig
  def plot_cpu_metric(self, cpu_metric="total", interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = None
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or sorted(self._node_names)):
      # Data frame
      df = self._cpu[(self._cpu["node_name"] == node_name) & (self._cpu["hw_no"].isin(self._cpu_cores[node_name])) &
          (self._cpu.index >= min_time) & (self._cpu.index <= max_time)].\
          groupby(["timestamp" if not window else ("window_%s" % window), "hw_no"])[cpu_metric].mean().unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration if not window else (self._ramp_up_duration * 1000), ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) if not window else
          ((self._total_duration - self._ramp_down_duration) * 1000), ls="--", color="green")
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.set_ylim((0, 100))
      ax.grid(alpha=0.75)
      df.interpolate(method='linear').plot(ax=ax, kind="line",
          title="%s: %s - CPU Utilization" % (node_name, self._node_labels[node_name]),
          xlabel=("Time (%s)" % ("sec" if not window else "millisec")) if not short else "",
          ylabel="%s (%%)" % cpu_metric, grid=True, legend=False, yticks=range(0, 101, 10))
    return fig

  @LogAnalysis.save_fig
  def plot_cpu_metric_comparison(self, cpu_metric="total", interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = None
      (min_time, max_time) = interval
    # Data frame
    cpu = self._cpu[(self._cpu.index >= min_time) & (self._cpu.index <= max_time)]
    cpu["node_label"] = cpu.apply(lambda r: self._node_labels[r["node_name"]], axis=1)
    df = pd.concat([cpu[(cpu["node_name"] == node_name) & (cpu["hw_no"].isin(self._cpu_cores[node_name]))]
        for node_name in self._node_names]).\
        groupby(["timestamp" if not window else ("window_%s" % window), "node_label"])[cpu_metric].mean().unstack()
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration if not window else (self._ramp_up_duration * 1000), ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) if not window else
        ((self._total_duration - self._ramp_down_duration) * 1000), ls="--", color="green")
    ax.set_xlim((df.index.min(), df.index.max()))
    ax.set_ylim((0, 100))
    ax.grid(alpha=0.75)
    df.interpolate(method='linear').plot(ax=ax, kind="line", title="CPU Utilization",
        xlabel="Time (%s)" % ("sec" if not window else "millisec"),
        ylabel="%s (%%)" % cpu_metric, grid=True, yticks=range(0, 101, 10))
    return fig


class CollectlDskLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._dsk = build_collectl_dsk_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_averaged_dsk_metric(self, dsk_metric="writes"):
    fig = plt.figure(figsize=(24, len(self._node_names) * 12))
    for (i, node_name) in enumerate(sorted(self._node_names)):
      # Data frame
      df = self._dsk[(self._dsk["node_name"] == node_name)].\
          groupby(["window_1000", "hw_no"])[dsk_metric].mean().unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.set_xlim((0, df.index.max()))
      ax.grid(alpha=0.75)
      df.plot(ax=ax, kind="line", title="%s: %s - Disk Utilization" % (node_name, self._node_labels[node_name]),
          xlabel="Time (millisec)", ylabel=dsk_metric, grid=True, legend=False)
    return fig

  @LogAnalysis.save_fig
  def plot_dsk_metric(self, dsk_metric="writes", interval=None):
    if not interval:
      return None
    else:
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._node_names) * 12))
    for (i, node_name) in enumerate(sorted(self._node_names)):
      # Data frame
      df = self._dsk[(self._dsk["node_name"] == node_name) & (self._dsk.index >= min_time) &
          (self._dsk.index <= max_time)].groupby(["timestamp"])[dsk_metric].mean()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration, ls="--", color="green")
      ax.axvline(x=self._total_duration - self._ramp_down_duration, ls="--", color="green")
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.grid(alpha=0.75)
      df.plot(ax=ax, kind="line", title="%s: %s - Disk Utilization" % (node_name, self._node_labels[node_name]),
          xlabel="Time (sec)", ylabel=dsk_metric, grid=True, legend=False)
    return fig


class CollectlMemLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._mem = build_collectl_mem_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_averaged_mem_metric(self, mem_metric="free"):
    fig = plt.figure(figsize=(24, len(self._node_names) * 12))
    for (i, node_name) in enumerate(sorted(self._node_names)):
      # Data frame
      df = self._mem[(self._mem["node_name"] == node_name)].\
          groupby(["window_1000", "hw_no"])[mem_metric].mean().unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.set_xlim((0, df.index.max()))
      ax.grid(alpha=0.75)
      df.plot(ax=ax, kind="line", title="%s: %s - Memory Utilization" % (node_name, self._node_labels[node_name]),
          xlabel="Time (millisec)", ylabel=mem_metric, grid=True, legend=False)
    return fig

  @LogAnalysis.save_fig
  def plot_mem_metric(self, mem_metric="free", interval=None):
    if not interval:
      return None
    else:
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._node_names) * 12))
    for (i, node_name) in enumerate(sorted(self._node_names)):
      # Data frame
      df = self._mem[(self._mem["node_name"] == node_name) & (self._mem.index >= min_time) &
          (self._mem.index <= max_time)].groupby(["timestamp"])[mem_metric].mean()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration, ls="--", color="green")
      ax.axvline(x=self._total_duration - self._ramp_down_duration, ls="--", color="green")
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.grid(alpha=0.75)
      df.plot(ax=ax, kind="line", title="%s: %s - Memory Utilization" % (node_name, self._node_labels[node_name]),
          xlabel="Time (sec)", ylabel=mem_metric, grid=True, legend=False)
    return fig


class QueryLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._query = build_query_df(experiment_dirpath, exploded_window_in_ms=None)
    self._dbnames = sorted(self._query["dbname"].unique())

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_queries(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._dbnames) * 12))
    for (i, dbname) in enumerate(self._dbnames):
      # Data frame
      df = self._query[(self._query["dbname"] == dbname) & (self._query.index >= min_time) &
          (self._query.index <= max_time)].groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).\
          unstack()
      if df.empty:
        continue
      df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, window), fill_value=0)
      # Plot
      ax = fig.add_subplot(len(self._dbnames), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.plot(ax=ax, kind="line", title="Instantaneous Latency of Queries - %s" % dbname, xlabel="Time (millisec)",
          ylabel="Latency (millisec)", grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_queries_comparison(self, latency_percentile=0.99, interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._query[(self._query.index >= min_time) & (self._query.index <= max_time)]
    df = df.groupby(["window_%s" % window, "dbname"])["latency"].quantile(latency_percentile).unstack()
    if df.empty:
      return None
    df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, window))
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.set_ylim((0, np.nanmax(df)))
    df.interpolate(method='linear').plot(ax=ax, kind="line",
        title="Instantaneous %s Latency of Queries" % latency_percentile, xlabel="Time (millisec)",
        ylabel="Latency (millisec)", grid=True)
    return fig

  def calculate_stats(self):
    stats = {}
    for dbname in self._dbnames:
      query = self._query[(self._query["dbname"] == dbname) &
          (self._query.index >= self._ramp_up_duration) &
          (self._query.index <= self._total_duration - self._ramp_down_duration)]
      stats.update({
          "db_%s_latency_p50" % dbname: query["latency"].quantile(0.50),
          "db_%s_latency_p95" % dbname: query["latency"].quantile(0.95),
          "db_%s_latency_p99" % dbname: query["latency"].quantile(0.99),
          "db_%s_latency_p999" % dbname: query["latency"].quantile(0.999),
          "db_%s_latency_avg" % dbname: query["latency"].mean(),
          "db_%s_latency_std" % dbname: query["latency"].std(),
          "db_%s_latency_max" % dbname: query["latency"].max(),
      })
    return stats


class RedisLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._redis = build_redis_df(experiment_dirpath, exploded_window_in_ms=None)
    self._services = sorted(self._redis["service_name"].unique())

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_commands(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._services) * 12))
    for (i, service) in enumerate(self._services):
      # Data frame
      df = self._redis[(self._redis["service_name"] == service) & (self._redis.index >= min_time) &
          (self._redis.index <= max_time)].groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).\
          unstack()
      if df.empty:
        continue
      df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, window), fill_value=0)
      # Plot
      ax = fig.add_subplot(len(self._services), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.plot(ax=ax, kind="line", title="Instantaneous Latency of Commands - %s" % service, xlabel="Time (millisec)",
          ylabel="Latency (millisec)", grid=True)
    return fig

  def calculate_stats(self):
    stats = {}
    for service in self._services:
      redis = self._redis[(self._redis["service_name"] == service) &
          (self._redis.index >= self._ramp_up_duration) &
          (self._redis.index <= self._total_duration - self._ramp_down_duration)]
      stats.update({
          "redis_%s_latency_p50" % service: redis["latency"].quantile(0.50),
          "redis_%s_latency_p95" % service: redis["latency"].quantile(0.95),
          "redis_%s_latency_p99" % service: redis["latency"].quantile(0.99),
          "redis_%s_latency_p999" % service: redis["latency"].quantile(0.999),
          "redis_%s_latency_avg" % service: redis["latency"].mean(),
          "redis_%s_latency_std" % service: redis["latency"].std(),
          "redis_%s_latency_max" % service: redis["latency"].max(),
      })
    return stats


class RPCLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._rpc = build_rpc_df(experiment_dirpath, exploded_window_in_ms=None)
    self._function_names = sorted(self._rpc["function"].unique())
    self._service_names = sorted([fn.split(':')[0] for fn in self._function_names])

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_rpcs(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._function_names) * 12))
    for (i, function) in enumerate(self._function_names):
      # Data frame
      df = self._rpc[(self._rpc["function"] == function) & (self._rpc.index >= min_time) &
          (self._rpc.index <= max_time)].groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).\
          unstack()
      if df.empty:
        continue
      df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, window), fill_value=0)
      # Plot
      ax = fig.add_subplot(len(self._function_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.plot(ax=ax, kind="line", title="Instantaneous Latency of RPC - %s" % function, xlabel="Time (millisec)",
          ylabel="Latency (millisec)", grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_services(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None,
      services=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(services or self._service_names) * (12 if not short else 4)))
    for (i, service) in enumerate(services or self._service_names):
      # Data frame
      df = self._rpc[(self._rpc.index >= min_time) & (self._rpc.index <= max_time) &
          (self._rpc["function"].isin([fn for fn in self._function_names if fn.split(':')[0] == service]))].\
          groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(services or self._service_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method='linear').plot(ax=ax, kind="line", title="Instantaneous Latency of Service - %s" % service,
          xlabel="Time (millisec)" if not short else "", ylabel="Latency (millisec)", grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_rpcs_comparison(self, latency_percentile=0.99, interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._rpc[(self._rpc.index >= min_time) & (self._rpc.index <= max_time)]
    df["service"] = df.apply(lambda r: r["function"].split(':')[0], axis=1)
    df = df.groupby(["window_%s" % window, "service"])["latency"].quantile(latency_percentile).unstack()
    if df.empty:
      return None
    df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, window))
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.set_ylim((0, np.nanmax(df)))
    df.interpolate(method='linear').plot(ax=ax, kind="line",
        title="Instantaneous %s Latency of RPC" % latency_percentile, xlabel="Time (millisec)",
        ylabel="Latency (millisec)", grid=True)
    return fig

  def calculate_stats(self):
    stats = {}
    for function in self._function_names:
      rpc = self._rpc[(self._rpc["function"] == function) &
          (self._rpc.index >= self._ramp_up_duration) &
          (self._rpc.index <= self._total_duration - self._ramp_down_duration)]
      if rpc.empty:
        stats.update({
            "rpc_%s_latency_p50" % function: 0,
            "rpc_%s_latency_p95" % function: 0,
            "rpc_%s_latency_p99" % function: 0,
            "rpc_%s_latency_p999" % function: 0,
            "rpc_%s_latency_avg" % function: 0,
            "rpc_%s_latency_std" % function: 0,
            "rpc_%s_latency_max" % function: 0,
        })
      else:
        stats.update({
            "rpc_%s_latency_p50" % function: rpc["latency"].quantile(0.50),
            "rpc_%s_latency_p95" % function: rpc["latency"].quantile(0.95),
            "rpc_%s_latency_p99" % function: rpc["latency"].quantile(0.99),
            "rpc_%s_latency_p999" % function: rpc["latency"].quantile(0.999),
            "rpc_%s_latency_avg" % function: rpc["latency"].mean(),
            "rpc_%s_latency_std" % function: rpc["latency"].std(),
            "rpc_%s_latency_max" % function: rpc["latency"].max(),
        })
    return stats


class ServerRequestLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._requests = build_requests_df(experiment_dirpath, exploded_window_in_ms=1)
    self._apigateway_node_names = sorted(self._requests["node_name"].unique())
    self._rpc = build_rpc_df(experiment_dirpath, exploded_window_in_ms=1)
    self._services = sorted(self._rpc["server"].unique())
    self._query = build_query_df(experiment_dirpath, exploded_window_in_ms=1)
    self._dbnames = sorted(self._query["dbname"].unique())
    self._redis = build_redis_df(experiment_dirpath, exploded_window_in_ms=1)
    self._redis_services = sorted(self._redis["service_name"].unique())

  @LogAnalysis.save_fig
  def plot_number_of_concurrent_server_requests_in_apigateways(self, interval=None):
    if not interval:
      min_time = 0
      max_time = self._total_duration
    else:
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._apigateway_node_names) * 12))
    for (i, node_name) in enumerate(self._apigateway_node_names):
      # Data frame
      df = self._requests[(self._requests["node_name"] == node_name) & (self._requests.index >= min_time) &
          (self._requests.index <= max_time)].groupby(["window"])["window"].count()
      if df.empty:
        continue
      df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1), fill_value=0)
      # Plot
      ax = fig.add_subplot(len(self._apigateway_node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.set_ylim((0, df.values.max()))
      df.plot(ax=ax, kind="line",
          title="%s: %s - Number of Concurrent Requests" % (node_name, self._node_labels[node_name]),
          xlabel="Time (millisec)", ylabel="Count (Requests)", color="black", grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_number_of_concurrent_server_requests_in_microservices(self, interval=None):
    if not interval:
      min_time = 0
      max_time = self._total_duration
    else:
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._services) * 12))
    for (i, service) in enumerate(self._services):
      # Data frame
      df = self._rpc[(self._rpc["server"] == service) & (self._rpc.index >= min_time) & (self._rpc.index <= max_time)].\
          groupby(["window"])["window"].count()
      if df.empty:
        continue
      df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1), fill_value=0)
      # Plot
      ax = fig.add_subplot(len(self._services), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.set_ylim((0, df.values.max()))
      df.plot(ax=ax, kind="line",
          title="%s - Number of Requests Being Processed" % self._node_labels[service.split(':')[0]],
          xlabel="Time (millisec)", ylabel="Count (Requests)", color="black", grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_number_of_concurrent_server_requests_in_databases(self, interval=None):
    if not interval:
      min_time = 0
      max_time = self._total_duration
    else:
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._dbnames) * 12))
    for (i, dbname) in enumerate(self._dbnames):
      # Data frame
      df = self._query[(self._query["dbname"] == dbname) & (self._query.index >= min_time) &
          (self._query.index <= max_time)].groupby(["window"])["window"].count()
      if df.empty:
        continue
      df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1), fill_value=0)
      # Plot
      ax = fig.add_subplot(len(self._dbnames), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.set_ylim((0, df.values.max()))
      df.plot(ax=ax, kind="line", title="%s Database - Number of Requests Being Processed" % dbname,
          xlabel="Time (millisec)", ylabel="Count (Requests)", color="black", grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_number_of_concurrent_server_requests_in_redis(self, interval=None):
    if not interval:
      min_time = 0
      max_time = self._total_duration
    else:
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._redis_services) * 12))
    for (i, service) in enumerate(self._redis_services):
      # Data frame
      df = self._redis[(self._redis["service_name"] == service) & (self._redis.index >= min_time) &
          (self._redis.index <= max_time)].groupby(["window"])["window"].count()
      if df.empty:
        continue
      df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1), fill_value=0)
      # Plot
      ax = fig.add_subplot(len(self._redis_services), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.set_ylim((0, df.values.max()))
      df.plot(ax=ax, kind="line", title="%s Redis - Number of Requests Being Processed" % service,
          xlabel="Time (millisec)", ylabel="Count (Requests)", color="black", grid=True)
    return fig


class RunQueueLengthLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._queue = build_runqlen_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_run_queue_length(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._queue[(self._queue["node_name"] == node_name) & (self._queue.index >= min_time) &
          (self._queue.index <= max_time)].groupby(["window_%s" % window, "cpu"])["qlen"].mean().unstack().fillna(0)
      if df.empty:
        continue
      df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, window), fill_value=0)
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method='linear').plot(ax=ax, kind="line",
          title="%s: %s - CPU Run Queue Length" % (node_name, self._node_labels[node_name]),
          xlabel="Time (millisec)" if not short else "", ylabel="Count (Tasks)", grid=True)
    return fig


class TCPSynBacklogLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._bl = build_tcp_synbl_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_syn_backlog_length(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._bl[(self._bl["node_name"] == node_name) & (self._bl.index >= min_time) &
          (self._bl.index <= max_time)].groupby(["window_%s" % window])["synbl"].max()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method='linear').plot(ax=ax, kind="line",
          title="%s: %s - TCP SYN Backlog Length" % (node_name, self._node_labels[node_name]),
          xlabel="Time (millisec)" if not short else "", ylabel="Count (Requests)", color="black", grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_syn_backlog_length_comparison(self, interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._bl[(self._bl.index >= min_time) & (self._bl.index <= max_time)]
    df["node_label"] = df.apply(lambda r: self._node_labels[r["node_name"]], axis=1)
    df = df.groupby(["window_%s" % window, "node_label"])["synbl"].max().unstack().fillna(0)
    if df.empty:
      return None
    df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, window), fill_value=0)
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.set_ylim((0, np.nanmax(df)))
    df.interpolate(method='linear').plot(ax=ax, kind="line", title="TCP SYN Backlog Length",
        xlabel="Time (millisec)", ylabel="Count (Requests)", grid=True)
    return fig


class TCPAcceptQueueLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._queue = build_tcp_acceptq_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_accept_queue_length(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._queue[(self._queue["node_name"] == node_name) & (self._queue.index >= min_time) &
          (self._queue.index <= max_time)].groupby(["window_%s" % window])["qlen"].max()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method='linear').plot(ax=ax, kind="line",
          title="%s: %s - TCP Accept Queue Length" % (node_name, self._node_labels[node_name]),
          xlabel="Time (millisec)" if not short else "", ylabel="Count (Requests)", color="black", grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_accept_queue_length_comparison(self, interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._queue[(self._queue.index >= min_time) & (self._queue.index <= max_time)]
    df["node_label"] = df.apply(lambda r: self._node_labels[r["node_name"]], axis=1)
    df = df.groupby(["window_%s" % window, "node_label"])["qlen"].max().unstack().fillna(0)
    if df.empty:
      return None
    df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1, window), fill_value=0)
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.set_ylim((0, np.nanmax(df)))
    df.interpolate(method='linear').plot(ax=ax, kind="line", title="TCP Accept Queue Length",
        xlabel="Time (millisec)", ylabel="Count (Requests)", grid=True)
    return fig


class TCPRetransLogAnalysis(LogAnalysis):
  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._retrans = build_tcp_retrans_df(experiment_dirpath)
    self._retrans["addr_port"] = list(zip(self._retrans.addr, self._retrans.port))
    self._addr_port = [(addr, port) for addr, port in self._retrans["addr_port"].unique() if port > 1080 and port < 1100]

  def plot_number_of_tcp_packet_retransmissions(self, interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._addr_port) * 12))
    for (i, addr_port) in enumerate(self._addr_port):
      # Data frame
      df = self._retrans[(self._retrans["addr_port"] == addr_port) & (self._retrans.index >= min_time) &
          (self._retrans.index <= max_time)].groupby(["window_%s" % window])["window_%s" % window].count()
      if df.empty:
        continue
      df = df.reindex(range(int(min_time * 1000), int(max_time * 1000) + 1, window), fill_value=0)
      # Plot
      ax = fig.add_subplot(len(self._addr_port), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.set_ylim((0, df.values.max()))
      df.plot(ax=ax, kind="line",
          title="%s:%s - TCP Packet Retransmissions" % (addr_port[0], addr_port[1]), xlabel="Time (millisec)",
          ylabel="Count (Retransmissions)", color="black", grid=True)
    return fig


def main():
  # Parse command-line arguments.
  parser = argparse.ArgumentParser(description="Plot experiment graphs")
  parser.add_argument("--experiment_dirname", required=False, action="store", type=str,
      help="Name of directory containing experiment data in `../data`", default="")
  parser.add_argument("--distribution", action="store_true", default=False,
      help="Add distribution graphs")
  args = parser.parse_args()
  # List experiment(s) directory.
  experiment_dirpaths = [os.path.join(os.path.abspath(""), "..", "data", dirname)
      for dirname in ([args.experiment_dirname] if args.experiment_dirname else
          os.listdir(os.path.join(os.path.abspath(""), "..", "data")))
      if re.findall("BuzzBlogBenchmark_", dirname) and not re.findall(".tar.gz", dirname)]
  # Retrieve list of experiments whose graphs have already been plotted.
  plotted_dirnames = []
  try:
    os.mkdir(os.path.join(os.path.abspath(""), "..", "graphs"))
  except FileExistsError:
    plotted_dirnames = os.listdir(os.path.join(os.path.abspath(""), "..", "graphs"))
  # Plot experiment graphs.
  for experiment_dirpath in experiment_dirpaths:
    if os.path.basename(experiment_dirpath) in plotted_dirnames:
      continue
    print("Processing %s:" % experiment_dirpath)
    output_dirpath = os.path.join(os.path.abspath(""), "..", "graphs", os.path.basename(experiment_dirpath))
    os.mkdir(output_dirpath)
    for notebook_cls in [RequestLogAnalysis, CollectlCPULogAnalysis, CollectlDskLogAnalysis, CollectlMemLogAnalysis,
        QueryLogAnalysis, RedisLogAnalysis, RPCLogAnalysis, ServerRequestLogAnalysis, RunQueueLengthLogAnalysis,
        TCPSynBacklogLogAnalysis, TCPAcceptQueueLogAnalysis, TCPRetransLogAnalysis]:
      try:
        notebook = notebook_cls(experiment_dirpath, output_dirpath)
        notebook.plot(distribution=args.distribution)
      except Exception as e:
        print("\tFailed: %s" % str(e))
        continue


if __name__ == "__main__":
  main()
