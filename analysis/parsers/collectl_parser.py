# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

import argparse
import datetime
import gzip
import re

import pandas as pd

# Constants
HW_METRICS = {
    "cpu": ["user", "nice", "system", "wait", "irq", "soft", "steal", "idle", "total", "guest", "guest_n", "intrpt"],
    "mem": ["used", "free", "slab", "mapped", "anon", "anonh", "inactive", "hits"],
    "dsk": ["name", "reads", "rmerge", "rkbytes", "waitr", "writes", "wmerge", "wkbytes", "waitw", "request", "quelen",
            "wait", "svctim", "util"]
}
COLUMNS = {
    "cpu": ["timestamp", "hw_no"] + HW_METRICS["cpu"],
    "mem": ["timestamp", "hw_no"] + HW_METRICS["mem"],
    "dsk": ["timestamp", "hw_no"] + HW_METRICS["dsk"]
}
FILE_EXTENSION_TO_HW_TYPE = {
    "cpu": "cpu",
    "numa": "mem",
    "dsk": "dsk"
}


class CollectlParser:
    @classmethod
    def df(cls, logfile, hw_type):
        for log in logfile:
            if log[0] == '#':
                timezone = re.findall(r"TZ: ([-+]\d{4})", log)
                if timezone:
                    offset = datetime.timedelta(hours=(1 if timezone[0][0] == '+' else -1) * int(timezone[0][1:3]))
            else:
                break
        df = pd.DataFrame([entry for log in logfile for entry in cls.extract_values_from_log(log, hw_type, offset)],
                columns=COLUMNS[hw_type])
        for hw_no in df.hw_no.unique():
            if (hw_type == "cpu" and df[(df["hw_no"] == hw_no)]["total"].unique().size == 1) or \
                    (hw_type == "mem" and df[(df["hw_no"] == hw_no)]["used"].unique().size == 1) or \
                    (hw_type == "dsk" and df[(df["hw_no"] == hw_no)]["reads"].unique().size == 1):
                df = df[(df["hw_no"] != hw_no)]
        return df

    @staticmethod
    def extract_values_from_log(log, hw_type, offset):
        values = log.split()
        timestamp = datetime.datetime.strptime(" ".join(values[:2]), "%Y%m%d %H:%M:%S.%f") - offset
        for hw_no in range((len(values) - 2) // len(HW_METRICS[hw_type])):
            yield (timestamp, hw_no,
                    *values[2 + hw_no * len(HW_METRICS[hw_type]):2 + (hw_no + 1) * len(HW_METRICS[hw_type])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV file")
    parser.add_argument("--log_filepath", required=True, action="store",
            type=str, help="Path to log file (input)")
    parser.add_argument("--csv_filepath", required=True, action="store",
            type=str, help="Path to CSV file (output)")
    args = parser.parse_args()
    with gzip.open(args.log_filepath, "rt") as logfile:
        CollectlParser.df(logfile,
                FILE_EXTENSION_TO_HW_TYPE[args.log_filepath.split('.')[-2]]).to_csv(args.csv_filepath, index=False)
