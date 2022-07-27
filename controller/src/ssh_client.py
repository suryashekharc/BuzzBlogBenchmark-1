# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

import datetime

import paramiko
import scp
import threading
import time


class SSHClient:
  """A wrapper around paramiko.SSHClient."""

  def __init__(self, hostname, port, username, key_filename, log_filename, max_sessions=40):
    """Initialize an SSH client."""
    self._hostname = hostname
    self._client = paramiko.SSHClient()
    self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    self._client.connect(hostname=hostname, port=port, username=username,
        key_filename=key_filename)
    self._log_filename = log_filename
    self._semaphore = threading.BoundedSemaphore(value=max_sessions)

  def exec(self, command, retries=10):
    """Execute command remotely.

    Data written to stdout and stderr are appended to files named with the
    log_filename attribute and extensions '.out' and '.err', respectively.
    """
    with self._semaphore:
      print("[{ts}][{hostname}] {command}".format(ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          hostname=self._hostname, command=command))
      while retries + 1 > 0:
        try:
          _, stdout, stderr = self._client.exec_command(command)
          stdout.channel.recv_exit_status()
          break
        except (paramiko.ssh_exception.ChannelException, paramiko.ssh_exception.SSHException) as e:
          retries -= 1
          if retries + 1 > 0:
            time.sleep(12)
          else:
            raise e
      with open(self._log_filename + ".out", "ab+") as stdout_log_file:
        stdout_b = stdout.read()
        stdout_log_file.write(stdout_b)
      with open(self._log_filename + ".err", "ab+") as stderr_log_file:
        stderr_b = stderr.read()
        stderr_log_file.write(stderr_b)
      return (stdout_b, stderr_b)

  def copy(self, remote_path, local_path=""):
    """Copy remote file to the local machine."""
    with self._semaphore:
      with scp.SCPClient(self._client.get_transport(), sanitize=lambda x: x) as scp_client:
        scp_client.get(remote_path, local_path, recursive=True)
