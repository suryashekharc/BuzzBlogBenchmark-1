# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

import argparse
import re

import pandas as pd

# Constants
COLUMNS = ["timestamp", "request_id", "dbname", "type", "latency"]
QUERY_LOG_PATTERN = r"^\[([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+)\] pid=([^\s]+) tid=([^\s]+) request_id=([^\s]+) latency=([^\s]+) query=\"([^\"]+)\"$"


class QueryParser:
    @classmethod
    def df(cls, logfile):
        data = [cls.extract_values_from_log(log) for log in logfile]
        return pd.DataFrame(data=[d for d in data if d], columns=COLUMNS)

    @staticmethod
    def extract_values_from_log(log):
        match = re.match(QUERY_LOG_PATTERN, log)
        if not match:
            return None
        timestamp, _, _, request_id, latency, query_str = match.groups()
        query_type = query_str.strip().split()[0].upper()
        if query_type == "SELECT":
            dbname = re.findall(r"[Ff][Rr][Oo][Mm]\s+(\w+)", query_str)[0]
        elif query_type == "INSERT":
            dbname = re.findall(r"[Ii][Nn][Tt][Oo]\s+(\w+)", query_str)[0]
        elif query_type == "UPDATE":
            dbname = re.findall(r"[Uu][Pp][Dd][Aa][Tt][Ee]\s+(\w+)", query_str)[0]
        elif query_type == "DELETE":
            dbname = re.findall(r"[Ff][Rr][Oo][Mm]\s+(\w+)", query_str)[0]
        return (timestamp, request_id, dbname, query_type, latency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV file")
    parser.add_argument("--log_filepath", required=True, action="store",
            type=str, help="Path to log file (input)")
    parser.add_argument("--csv_filepath", required=True, action="store",
            type=str, help="Path to CSV file (output)")
    args = parser.parse_args()
    with open(args.log_filepath) as logfile:
        QueryParser.df(logfile).to_csv(args.csv_filepath, index=False)
