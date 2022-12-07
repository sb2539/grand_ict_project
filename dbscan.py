import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
import wandb
from sklearn.cluster import DBSCAN

pd.options.mode.chained_assignment = None

@dataclass(frozen=True)
class Dataset:
    path: Path
    channel_names:dict

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
    }
)

set_no_2 = Dataset(
    path=Path("data/2nd_test/2nd_test"),
    channel_names={
        0: "B1x",
        1: "B2x",
        2: "B3x",
        3: "B4x",
    }
)


set_no_3 = Dataset(
    path=Path("data/3rd_test/4th_test/txt"),
    channel_names={
        0: "B1x",
        1: "B2x",
        2: "B3x",
        3: "B4x",
    },
)

def read_dataset(dataset: Dataset):
    all_dfs = []
    for file_counter, f in enumerate(tqdm(sorted(dataset.path.iterdir()))):
        df = pd.read_csv(f, sep='\t', header=None, dtype=np.float32).rename(
            columns = dataset.channel_names
        )[["B1x", "B2x", "B3x", "B4x"]]
        if len(df) != 20480:
            raise RuntimeError(f"Unexpected file length {len(df)} in {f}")
        all_dfs.append(df[["B1x", "B2x", "B3x", "B4x"]].mean())

    return all_dfs




if __name__ == '__main__':
    wandb.login()
    w_config = wandb.config

    sweep_config = {
        'name': 'dbscan_wandb_demo',
        'method': 'random',
        'parameters': {
            'eps': {
                'values': [0.1, 0.2,0.05,0.08]
            },
            'min_sample': {
                'values': [5,9,10]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='dbscan_demo_outlier', entity='sb2539')
    # raw_data = read_dataset(set_no_1)
    # df = pd.DataFrame(raw_data)
    # df.to_csv("no_1.csv",index=False)
    df=pd.read_csv("no_1.csv")

    def run_sweep(config=None):
        wandb.init(config=config, entity='sb2539')
        w_config = wandb.config
        dbscan = DBSCAN(eps=w_config.eps, min_samples=w_config.min_sample)
        labels = dbscan.fit_predict(df)
        print(labels)
        wandb.sklearn.plot_clusterer(dbscan, df, labels)
        wandb.sklearn.plot_silhoutte(dbscan, df, labels)
        print(pd.Series(labels).value_counts())
        wandb.log({"data_table": wandb.Table(data=df, columns=["B1x", "B2x", "B3x", "B4x"])})
        # plt.figure(figsize=(12, 12))
        #
        # unique_labels = set(labels)
        # colors = ['#586fab', '#f55354']
        #
        # fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(20, 10))
        # x = np.arange(2156)
        #
        # for (sensor, ax) in zip(["B1x", "B2x", "B3x", "B4x"], axes.ravel()):
        #     for color, label in zip(colors, unique_labels):
        #         sample_mask = [True if l == label else False for l in labels]
        #         ax.plot(x[sample_mask], df[sensor][sample_mask], 'o', color=color, label="snapshot")
        #         ax.set_ylabel(f"DBSCAN {sensor}")
        #         ax.set_xlabel("Sample [0-2155]")
        # plt.tight_layout()
        # #wandb.Image(plt)

    wandb.agent(sweep_id, run_sweep, count = 5)





