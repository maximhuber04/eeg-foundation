import os
import shutil
import gc
import json
from tqdm import tqdm
import pandas as pd
import yaml
import sys
import mne
import random
from socket import gethostname

sys.path.append("/home/maxihuber/eeg-foundation")

from src.data.mae_rope_datamodule import TrainDataModule
from src.utils.preloading.utils import (
    filter_index_simple,
    load_from_path,
    prepare_info_to_load,
    create_raw,
    load_edf_to_dataframe,
)

main_config_file = "/home/maxihuber/eeg-foundation/configs/experiment/rope.yaml"
with open(main_config_file, "r") as file:
    config = yaml.safe_load(file)
    data_config = config["data"]

source_indices = [
    "/itet-stor/maxihuber/deepeye_storage/index_files/half2_tueg_index2.json",
]
store_path = "/dev/shm/mae"
start_id = 4

self = TrainDataModule(
    source_indices=source_indices,
    path_prefix=data_config["path_prefix"],
    min_duration=1,
    max_duration=data_config["max_duration"],
    split_duration=data_config["split_duration"],
    discard_sr=data_config["discard_sr"],
    discard_datasets=data_config["discard_datasets"],
    channel_name_map_path=data_config["channel_name_map_path"],
)


def load_to_memory(self, store_path, start_id):

    print(f"Collecting data from {gethostname()}", file=sys.stderr)

    index = filter_index_simple(
        index_paths=self.source_indices,
        path_prefix=self.path_prefix,
        min_duration=self.min_duration,
        max_duration=self.max_duration,
        discard_sr=self.discard_sr,
        discard_datasets=self.discard_datasets,
    )

    # Shuffle the index (list) randomly, to load balance across processes in expectation
    random.seed(42)
    random.shuffle(index)

    print(f"[load_to_memory] # Files {len(index)}", file=sys.stderr)

    trial_info_index = prepare_info_to_load(
        index_chunk=index,
        min_duration=self.min_duration,
        max_duration=self.max_duration,
        split_duration=self.split_duration,
    )

    print(f"[load_to_memory] # Trials {len(trial_info_index)}", file=sys.stderr)

    slurm_rank = int(os.getenv("SLURM_PROCID", "0"))
    slurm_size = int(os.getenv("SLURM_NPROCS", "1"))

    print(
        f"[load_to_memory] Process={slurm_rank}, World Size={slurm_size}",
        file=sys.stderr,
    )

    indices = list(range(len(trial_info_index)))
    samples_per_process = len(indices) // slurm_size
    start_index = slurm_rank * samples_per_process
    end_index = (
        start_index + samples_per_process
        if slurm_rank < slurm_size - 1
        else len(indices)
    )
    indices = indices[start_index:end_index]
    trial_info_index = {
        trial_idx: trial_info
        for trial_idx, trial_info in trial_info_index.items()
        if trial_idx in indices
    }

    print(
        f"[load_to_memory] # Trials on process={slurm_rank}: {len(trial_info_index)}",
        file=sys.stderr,
    )

    slurm_rank += start_id

    index_path = f"{store_path}/index_{slurm_rank}.json"
    subdir_name = None
    file_index = {}

    print(f"[load_to_memory] Storing data to {store_path}", file=sys.stderr)
    print(f"[load_to_memory] Index path: {index_path}", file=sys.stderr)

    for i, (trial_idx, trial_info) in enumerate(
        tqdm(
            trial_info_index.items(),
            desc=f"Moving data to node [rank={slurm_rank-start_id}]",
            position=0,
            leave=True,
        )
    ):
        if i % 10_000 == 0:
            subdir_name = f"{slurm_rank}_{i // 10_000}"
            os.makedirs(f"{store_path}/{subdir_name}", exist_ok=True)
        trial_info["new_path"] = f"{store_path}/{subdir_name}/trial_{trial_idx}"

        if trial_info["origin_dur"] > self.split_duration:
            # df = load_edf_to_dataframe(trial_info["origin_path"])
            # df = df[trial_info["channels"]]
            df = load_from_path(
                path=trial_info["origin_path"],
                channels=trial_info["channels"],
                sr=trial_info["sr"],
            )
            start_sample = int(trial_info["start"] * trial_info["sr"])
            end_sample = start_sample + int(trial_info["dur"] * trial_info["sr"])
            df = df.iloc[start_sample:end_sample, :]
            raw = create_raw(
                data=df, ch_names1=trial_info["channels"], sr=trial_info["sr"]
            )
            trial_info["new_path"] += ".edf"
            mne.export.export_raw(
                trial_info["new_path"], raw, fmt="edf", overwrite=True
            )
            raw.close()
            del raw, df
        else:
            trial_info["new_path"] += (
                ".edf" if trial_info["origin_path"].endswith("edf") else ".pkl"
            )
            shutil.copyfile(trial_info["origin_path"], trial_info["new_path"])

        file_index[trial_idx] = trial_info

        if i % 100 == 0:
            with open(index_path, "w") as f:
                json.dump(file_index, f, indent=4)
            gc.collect()

    # Save the final index
    with open(index_path, "w") as f:
        json.dump(file_index, f, indent=4)

    gc.collect()

    print(
        f"[load_to_memory] Finished loading data to memory on process={slurm_rank-start_id}.",
        file=sys.stderr,
    )


load_to_memory(self, store_path, start_id)
