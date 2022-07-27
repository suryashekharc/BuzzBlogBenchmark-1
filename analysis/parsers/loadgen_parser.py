# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

import argparse
import re

import pandas as pd

# Constants
COLUMNS = ["timestamp", "method", "url", "request_id", "status_code", "latency", "status", "type", "rw"]
REQUEST_LOG_PATTERN = r"^\[(\d+\-\d+\-\d+ \d+:\d+:\d+.\d+)\] (.+) (.+) (\d+) - latency=(\d+.\d+)$"
URL_PATTERN = r"^http://[\w\.\-]+:\d+/{path}/?\??{qs}$"
REQUEST_TO_TYPE = {
    (URL_PATTERN.format(path="account/\d+", qs=""), "GET"): "retrieve_account",
    (URL_PATTERN.format(path="account", qs=""), "POST"): "create_account",
    (URL_PATTERN.format(path="account/\d+", qs=""), "PUT"): "update_account",
    (URL_PATTERN.format(path="follow", qs="followee_id=\d+"), "GET"): "retrieve_account_followers",
    (URL_PATTERN.format(path="follow", qs="follower_id=\d+"), "GET"): "retrieve_account_followees",
    (URL_PATTERN.format(path="follow", qs=""), "POST"): "follow_account",
    (URL_PATTERN.format(path="follow/\d+", qs=""), "DELETE"): "delete_follow",
    (URL_PATTERN.format(path="like", qs="account_id=\d+"), "GET"): "retrieve_account_likes",
    (URL_PATTERN.format(path="like", qs="post_id=\d+"), "GET"): "retrieve_post_likes",
    (URL_PATTERN.format(path="like", qs=""), "POST"): "like_post",
    (URL_PATTERN.format(path="like/\d+", qs=""), "DELETE"): "delete_like",
    (URL_PATTERN.format(path="post", qs=""), "GET"): "retrieve_recent_posts",
    (URL_PATTERN.format(path="post", qs="author_id=\d+"), "GET"): "retrieve_account_posts",
    (URL_PATTERN.format(path="post/\d+", qs=""), "GET"): "retrieve_post",
    (URL_PATTERN.format(path="post", qs=""), "POST"): "create_post",
    (URL_PATTERN.format(path="post/\d+", qs=""), "DELETE"): "delete_post",
    (URL_PATTERN.format(path="trending", qs=""), "GET"): "list_trending_hashtags"
}


class LoadgenParser:
    @classmethod
    def df(cls, logfile):
        data = [cls.extract_values_from_log(log) for log in logfile]
        return pd.DataFrame(data=[d for d in data if d], columns=COLUMNS)

    @staticmethod
    def extract_values_from_log(log):
        match = re.match(REQUEST_LOG_PATTERN, log)
        if not match:
            return None
        timestamp, method, url, status_code, latency = match.groups()
        request_id = re.findall(r"request_id=([a-zA-Z0-9]+)&?", url)[0]
        url = re.sub("limit=\d+&?", "", url)
        url = re.sub("offset=\d+&?", "", url)
        url = re.sub("request_id=[a-zA-Z0-9]+&?", "", url)
        url = re.sub("&$", "", url)
        url = re.sub("\?$", "", url)
        status = "successful" if int(status_code) == 200 else "failed"
        type = [t for ((p, m), t) in REQUEST_TO_TYPE.items() if m == method and re.match(p, url)][0]
        rw = "read" if method.upper() == "GET" else "write"
        return (timestamp, method, url, request_id, status_code, latency, status, type, rw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV file")
    parser.add_argument("--log_filepath", required=True, action="store",
            type=str, help="Path to log file (input)")
    parser.add_argument("--csv_filepath", required=True, action="store",
            type=str, help="Path to CSV file (output)")
    args = parser.parse_args()
    with open(args.log_filepath) as logfile:
        LoadgenParser.df(logfile).to_csv(args.csv_filepath, index=False)
