import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
import wandb

pd.options.mode.chained_assignment = None

@dataclass(frozen=True)
class Dataset:
    path: Path
    channel_names:dict
    first_ts:pd.Timestamp

set_no_1 = Dataset(
    path=Path("data/1st_test/1st_test"),
    channel_names={
        0: "B1x",
        1: "B1y",
        2: "B2x",
        3: "B2y",
        4: "B3x",
        5: "B3y",
        6: "B4x",
        7: "B4y"
    },
    first_ts=pd.to_datetime("2003-10-22 12:06:24")
)

set_no_2 = Dataset(
    path=Path("data/2nd_test/2nd_test"),
    channel_names={
        0: "B1x",
        1: "B2x",
        2: "B3x",
        3: "B4x",
    },
    first_ts=pd.to_datetime("2004-02-12 10:32:39")
)


set_no_3 = Dataset(
    path=Path("data/3rd_test/4th_test/txt"),
    channel_names={
        0: "B1x",
        1: "B2x",
        2: "B3x",
        3: "B4x",
    },
    first_ts=pd.to_datetime("2004-03-04 09:27:46")
)

def read_dataset(dataset: Dataset):
    all_dfs = []
    for file_counter, f in enumerate(tqdm(sorted(dataset.path.iterdir()))):
        df = pd.read_csv(f, sep='\t', header=None, dtype=np.float32).rename(
            columns = dataset.channel_names
        )[["B1x", "B2x", "B3x", "B4x"]]
        ts = pd.to_datetime(f.name, format="%Y.%m.%d.%H.%M.%S")
        measurement_delta = (ts - dataset.first_ts).total_seconds()
        step_s = 1 / 20000  # 20 kHz sampling
        df["time"] = measurement_delta + np.arange(len(df)) * step_s
        df["measurement_id"] = file_counter
        df["measurement_id"] = df["measurement_id"].astype(np.uint32)
        if len(df) != 20480:
            raise RuntimeError(f"Unexpected file length {len(df)} in {f}")
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)

if __name__ == '__main__':
    wandb.login()
    run = wandb.init(
        project = "gctdemo_1",
        job_type="create_dataset",
        config={}
    )

    table = wandb.Table(
        columns=["B1x", "B2x", "B3x", "B4x", "time", "measurement_id"]
    )

    raw_data = read_dataset(set_no_1)
    print(raw_data)
    print(raw_data[["B1x", "B2x", "B3x", "B4x"]].describe())


    num_train_measurements = (
        raw_data.groupby("measurement_id")["time"].first().gt(3600 * 24 * 25).idxmax()
    )
    print("First measurement ID after 25 days:", num_train_measurements)

    sensor_cols = ["B1x", "B2x", "B3x", "B4x"]

    train_data = raw_data.query("measurement_id < @num_train_measurements")

    # Find a single set of data input normalization parameters
    training_mean = train_data[sensor_cols].values.flatten().mean()
    training_std = train_data[sensor_cols].values.flatten().std()

    train_data.loc[:, sensor_cols] = (
                train_data[sensor_cols] - training_mean
) / training_std




