import pandas as pd
import time
from influxdb_client import InfluxDBClient
import urllib3
import os

urllib3.disable_warnings()


class InfluxDb:
    def __init__(self):
        self.client = InfluxDBClient(
            url=os.environ['INFLUX_URL'],
            org=os.environ['INFLUX_ORG'],
            token=os.environ['INFLUX_TOKEN'],
            verify_ssl=False
        )
        # self.client = InfluxDBClient.from_config_file("config.ini")
        self.space_keys = [
            'active_core_count, bytesDownloaded, bytesUploaded, chunkSize, concurrency, parallelism, pipelining, destination_rtt, source_rtt, read_throughput, write_throughput, ']
        self.query_api = self.client.query_api()

    def query_space(self, job_uuid, bucket_name="OdsTransferNodes", transfer_node_name="jgoldverg@gmail.com-mac",
                    time_window='-2m', keys_to_expect=None, retry_count=10) -> pd.DataFrame:
        if keys_to_expect is None:
            keys_to_expect = ['bytesDownloaded', 'bytesUploaded', 'chunkSize', 'concurrency', 'destination_latency',
                              'destination_rtt', 'jobSize', 'parallelism', 'pipelining', 'read_throughput',
                              'source_latency', 'source_rtt', 'write_throughput', 'jobId']
        q = '''from(bucket: "{}")
  |> range(start: {})
  |> filter(fn: (r) => r["_measurement"] == "transfer_data")
  |> filter(fn: (r) => r["APP_NAME"] == "{}")
  |> filter(fn: (r) => r["jobUuid"] == "{}")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  '''.format(bucket_name, time_window, transfer_node_name, job_uuid)
        done = False

        df = pd.DataFrame()
        while not done:
            df = self.query_api.query_data_frame(q)
            if isinstance(df, list):
                df = pd.concat(df, axis=0, ignore_index=True)
                break
            if df.empty:
                time.sleep(1)
                continue
            else:
                break
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        return df

    def query_job_data(self, bucket_name="OdsTransferNodes", transfer_node_name="jgoldverg@gmail.com-mac",
                    time_window='-2m', keys_to_expect=None, retry_count=10) -> pd.DataFrame:
        if keys_to_expect is None:
            keys_to_expect = ['bytesDownloaded', 'bytesUploaded', 'chunkSize', 'concurrency', 'destination_latency',
                              'destination_rtt', 'jobSize', 'parallelism', 'pipelining', 'read_throughput',
                              'source_latency', 'source_rtt', 'write_throughput', 'jobId']
        q = '''from(bucket: "{}")
        |> range(start: {})
        |> filter(fn: (r) => r["_measurement"] == "transfer_data")
        |> filter(fn: (r) => r["APP_NAME"] == "{}")
        |> filter(fn: (r) => r["jobId"] >= 0)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''.format(bucket_name, time_window, transfer_node_name)
        done = False

        df = pd.DataFrame()
        while not done:
            df = self.query_api.query_data_frame(q)
            if isinstance(df, list):
                df = pd.concat(df, axis=0, ignore_index=True)
                break
            if df.empty:
                time.sleep(1)
                continue
            else:
                break
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        return df

    def close_client(self):
        self.client.close()

    def __reduce__(self):
        state = self.__dict__.copy()
        state.pop('client', None)
        return (self.__class__, ())