# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

import eventlet
eventlet.monkey_patch(socket=True)

import argparse
import datetime
import gc
import random
import string

import requests
from requests.auth import HTTPBasicAuth

import ATLoad


def _random_string(length):
  letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
  return "".join(random.choice(letters) for _ in range(length))


LOG_PATTERN = "[{ts}] {method} {url} {status_code} - latency={latency}"
N_HASHTAGS = 1024
HASHTAGS = ["#" + _random_string(10) for _ in range(N_HASHTAGS)]


class BuzzBlogSession(ATLoad.Session):
  def __init__(self, hostname, port):
    self._url_prefix = "http://{hostname}:{port}".format(hostname=hostname,
        port=port)
    self._password = _random_string(16)
    self._my_account = None
    self._my_posts = []
    self._my_follows = []
    self._my_likes = []
    self._other_account = None
    self._other_post = None

  def _request(self, method, path, params=None, json=None):
    auth = HTTPBasicAuth(self._my_account["username"], self._password) \
        if self._my_account is not None else None
    params = params or {}
    params.update({"request_id": _random_string(8)})
    start_time = datetime.datetime.now()
    r = None
    with eventlet.Timeout(30, False):
      r = getattr(requests, method)(self._url_prefix + path, auth=auth, params=params, json=json, timeout=30)
    if r is None:
      r = requests.Response()
      r.status_code = 408
    latency = round((datetime.datetime.now() - start_time).total_seconds(), 3)
    query_string = "&".join(["%s=%s" % (k, v) for (k, v) in params.items()]) \
        if params is not None else ""
    self._log(LOG_PATTERN.format(
        ts=start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        method=method.upper(),
        url=self._url_prefix + path +
            ("?" + query_string if query_string else ""),
        status_code=r.status_code,
        latency=format(latency, ".3f")))
    return r

  def create_account(self):
    r = self._request("post", "/account",
        json={
            "username": _random_string(16),
            "password": self._password,
            "first_name": _random_string(16),
            "last_name": _random_string(16)
        })
    assert r.status_code == 200
    self._my_account = r.json()

  def update_account(self):
    r = self._request("put", "/account/%s" % self._my_account["id"],
        json={
            "password": self._password,
            "first_name": _random_string(16),
            "last_name": _random_string(16)
        })
    if r.status_code == 200:
      self._my_account = r.json()

  def create_post(self):
    r = self._request("post", "/post", json={"text": _random_string(128) + " " + random.choice(HASHTAGS)})
    if r.status_code == 200:
      self._my_posts.append(r.json())

  def delete_post(self):
    if self._my_posts:
      post = self._my_posts.pop(random.randrange(len(self._my_posts)))
      self._request("delete", "/post/%s" % post["id"])

  def follow_account(self):
    if self._other_account:
      r = self._request("post", "/follow",
          json={"account_id": self._other_account["id"]})
      if r.status_code == 200:
        self._my_follows.append(r.json())
      self._other_account = None

  def delete_follow(self):
    if self._my_follows:
      follow = self._my_follows.pop(random.randrange(len(self._my_follows)))
      self._request("delete", "/follow/%s" % follow["id"])

  def like_post(self):
    if self._other_post:
      r = self._request("post", "/like",
          json={"post_id": self._other_post["id"]})
      if r.status_code == 200:
        self._my_likes.append(r.json())
      self._other_post = None

  def delete_like(self):
    if self._my_likes:
      like = self._my_likes.pop(random.randrange(len(self._my_likes)))
      self._request("delete", "/like/%s" % like["id"])

  def retrieve_recent_posts(self):
    r = self._request("get", "/post",
        params={"limit": 1, "offset": 0})
    if r.status_code == 200 and r.json():
      self._other_post = random.choice(r.json())
      self._other_account = random.choice(r.json())["author"]

  def retrieve_post(self):
    if self._other_post:
      self._request("get", "/post/%s" % self._other_post["id"])

  def retrieve_post_likes(self):
    if self._other_post:
      self._request("get", "/like", params={"post_id": self._other_post["id"],
          "limit": 1, "offset": 0})

  def retrieve_account(self):
    if self._other_account:
      self._request("get", "/account/%s" % self._other_account["id"])

  def retrieve_account_posts(self):
    if self._other_account:
      r = self._request("get", "/post",
          params={"author_id": self._other_account["id"], "limit": 1,
              "offset": 0})
      if r.status_code == 200 and r.json():
        self._other_post = random.choice(r.json())

  def retrieve_account_followers(self):
    if self._other_account:
      r = self._request("get", "/follow",
          params={"followee_id": self._other_account["id"], "limit": 1,
              "offset": 0})
      if r.status_code == 200 and r.json():
        self._other_account = random.choice(r.json())["follower"]

  def retrieve_account_followees(self):
    if self._other_account:
      r = self._request("get", "/follow",
          params={"follower_id": self._other_account["id"], "limit": 1,
              "offset": 0})
      if r.status_code == 200 and r.json():
        self._other_account = random.choice(r.json())["followee"]

  def retrieve_account_likes(self):
    if self._other_account:
      self._request("get", "/like",
          params={"account_id": self._other_account["id"], "limit": 1,
              "offset": 0})

  def list_trending_hashtags(self):
    r = self._request("get", "/trending",
        params={"limit": 10})


if __name__ == "__main__":
  # Disable garbage collection.
  gc.set_threshold(0)
  # Parse command-line arguments.
  parser = argparse.ArgumentParser(description="Generate a BuzzBlog workload")
  parser.add_argument("--workload_conf", required=True, action="store",
      type=str, help="Path to the workload configuration file")
  parser.add_argument("--log", required=True, action="store", type=str,
      help="Path to the log file")
  parser.add_argument("--hostname", required=True, action="store", type=str,
      help="Load balancer (or API Gateway) hostname")
  parser.add_argument("--port", required=True, action="store", type=str,
      help="Load balancer (or API Gateway) server")
  parser.add_argument("--seed", required=True, action="store", type=str,
      help="Random number generator seed")
  parser.add_argument("--n_workers", required=True, action="store", type=str,
      help="Number of workers")
  args = parser.parse_args()
  # Generate workload.
  workload = ATLoad.Workload(args.workload_conf, args.log, BuzzBlogSession,
      int(args.seed), int(args.n_workers), args.hostname, args.port)
  random.seed(None)
  workload.run()
