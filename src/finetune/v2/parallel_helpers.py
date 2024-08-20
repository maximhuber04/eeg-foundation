print("Hello world")

# Add custom path
import sys

sys.path.append("/home/maxihuber/eeg-foundation/")

# Standard library imports
import os
import gc
import json

# Third-party library imports
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
import multiprocessing as mp


# MNE imports
import mne
from mne.preprocessing import Xdawn

mne.set_log_level("warning")

# Custom imports
from src.utils.preloading.utils import load_edf_to_dataframe

# PyTorch imports
import torch
import lightning.pytorch as L

# Seed everything
L.seed_everything(42)

print("Bye world")


def get_generic_channel_name(channel_name):
    channel_name = channel_name.lower()
    # Remove "eeg " prefix if present
    if channel_name.startswith("eeg "):
        channel_name = channel_name[4:]
    # Simplify names with a dash and check if it ends with "-"
    if "-" in channel_name:
        if channel_name.endswith("-"):
            return "None"
        return channel_name.split("-")[0]
    return channel_name


def get_full_paths(input_files, prefix_filepath, filename_to_nodepath):
    adjusted_files = []
    for file in input_files:
        file_ = os.path.basename(file)
        if file_ in filename_to_nodepath:
            adjusted_files.append(filename_to_nodepath[file_])
        else:
            file = (
                prefix_filepath + file
                if "/itet-stor" not in file
                else file.replace("/itet-stor/kard", "/itet-stor/maxihuber")
            )
            adjusted_files.append(file)
    return adjusted_files


def load_sample(
    sample,
    load_mode,
    task_channels,
    filename_to_nodepath,
    task_name,
    time_col,
    prefix_filepath,
):
    try:
        input_files = get_full_paths(
            sample["input"], prefix_filepath, filename_to_nodepath
        )

        if load_mode == 2:
            file = input_files[0]
            df = load_edf_to_dataframe(file)
        else:
            dataframes = [pd.read_pickle(file) for file in input_files]
            df = pd.concat(dataframes, axis=0)

        start = int(sample["start"])
        length = int(sample["length"]) if "length" in sample else int(sample["end"])
        df = (
            df.loc[start : start + length, :]
            if load_mode == 1
            else df.iloc[start:length, :]
        )
        assert len(df) > 0, f"Empty dataframe for sample: {sample}"

        if load_mode != 1:
            output = sample.get("output", sample.get("label"))
        else:
            output = (
                list(sample["output"].values())
                if task_name == "EyeNetPosition"
                else list(sample["output"].values())[0]
            )

        sr = int(1 / (df[time_col].iloc[1] - df[time_col].iloc[0]))
        dur = len(df) / sr

        df.columns = [get_generic_channel_name(col) for col in df.columns]

        valid_channels = set(df.columns) & set(task_channels)
        sorted_valid_channels = sorted(
            valid_channels, key=lambda x: list(task_channels).index(x)
        )
        df = df[sorted_valid_channels].astype(float)
        data = torch.tensor(df.to_numpy(), dtype=torch.float32).T
        del df  # Free memory

        return (data, output, sr, dur, sorted_valid_channels)

    except Exception as e:
        print(f"Failed to process sample: {sample}. Error: {e}", file=sys.stderr)
        return None


# Helper function to resample a single signal
def resample_signal(idx, signal, sr, target_sfreq):
    try:
        signal_numpy = signal.numpy().astype(np.float64)
        signal_resampled = mne.filter.resample(
            signal_numpy,
            up=target_sfreq / sr,
            npad="auto",
            window="boxcar",
            n_jobs=1,
        )
        resampled_signal = torch.tensor(signal_resampled, dtype=torch.float32)
        # Free memory
        del signal_numpy, signal_resampled
        if idx % 10 == 0:
            gc.collect()
        return idx, resampled_signal
    except Exception as e:
        print(f"Failed to resample signal at index {idx}. Error: {e}", file=sys.stderr)
        return idx, None


def pad_or_truncate_signal(idx, signal, common_length):
    try:
        signal_numpy = signal.numpy().astype(
            np.float32
        )  # Ensure we're working with numpy arrays
        signal_length = signal_numpy.shape[1]
        if signal_length < common_length:
            pad_width = common_length - signal_length
            signal_padded = np.pad(
                signal_numpy, ((0, 0), (0, pad_width)), mode="constant"
            )
        else:
            signal_padded = signal_numpy[:, :common_length]
        # Free memory
        del signal_numpy
        if idx % 10 == 0:
            gc.collect()
        return idx, torch.tensor(signal_padded, dtype=torch.float32)
    except Exception as e:
        print(
            f"Failed to pad/truncate signal at index {idx}. Error: {e}", file=sys.stderr
        )
        return idx, None
