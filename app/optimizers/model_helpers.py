def reward_func(influx_row, reward_type):
    if reward_type == "thrpt":
        return influx_row['read_throughput']
