print("Hello world")

# Add custom path
import sys

sys.path.append("/home/maxihuber/eeg-foundation/")

# Standard library imports
import os
import gc
import glob
import json
import pickle
from datetime import datetime
from collections import Counter, defaultdict
from functools import partial

# Third-party library imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import lightning.pytorch as L
import xgboost as xgb
import torchaudio
from natsort import natsorted

# Sklearn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

# MNE imports
import mne

mne.set_log_level("warning")

# Custom imports
from src.data.transforms import crop_spg, normalize_spg
from src.models.mae_rope_encoder import EncoderViTRoPE
from src.utils.preloading.utils import load_edf_to_dataframe
from timm.models.vision_transformer import Mlp
from torch.nn import TransformerEncoderLayer
from src.finetune.v2.parallel_helpers import load_sample

# Seed everything
L.seed_everything(42)

print("Bye world")

########################################################################################################################
# TUAB and Epilepsy

yc_class = {
    "class_name": "YC",
    "time_col": "Time in Seconds",
    "prefix_filepath": "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf",
    "load_mode": 2,
}

tuab = {
    "task_name": "TUAB",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/tuab_light.json",
    "out_dim": 2,
}

epilepsy = {
    "task_name": "Epilepsy",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/epilepsy_light.json",
    "out_dim": 2,
}

yc_tasks = [tuab, epilepsy]

########################################################################################################################
# Clinical JSONs

cli_class = {
    "class_name": "Clinical",
    "time_col": "Time in Seconds",
    "prefix_filepath": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_prepared/",
    "load_mode": 0,
}

age = {
    "task_name": "Age",
    "task_type": "Regression",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/age2_light.json",
    "out_dim": 1,
}

depression = {
    "task_name": "Depression",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/cli_depression_light.json",
    "out_dim": 2,
}

parkinsons = {
    "task_name": "Parkinsons",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/parkinsons2_light.json",
    "out_dim": 2,
}

schizophrenia = {
    "task_name": "Schizophrenia",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/cli_schizophrenia_light.json",
    "out_dim": 2,
}

sex = {
    "task_name": "Sex",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/sex2_light.json",
    "out_dim": 2,
}

cli_tasks = [age, depression, parkinsons, schizophrenia, sex]


########################################################################################################################
# Motor-Imagery JSONs

mi_class = {
    "class_name": "Motor Imagery",
    "time_col": "time in seconds",
    "prefix_filepath": "/itet-stor/maxihuber/deepeye_storage/foundation_prepared/",
    "load_mode": 0,
}

eye_open_closed = {
    "task_name": "EyeOpenClosed",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/eye_open_closed_light.json",
    "out_dim": 2,
    "outputs": set(["eye open", "eye closed"]),
    "short_mode": False,
}

eye_vh = {
    "task_name": "EyeVH",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/eye_vh_light.json",
    "out_dim": 2,
    "outputs": set(["vertical", "horizontal"]),
    "short_mode": False,
}

flexion_extension_imaginary = {
    "task_name": "FlexionExtensionImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/flexion_extension_imaginary_light.json",
    "out_dim": 2,
    "outputs": set(
        [
            "hand movement imagined elbow flexion",
            "hand movement imagined elbow extension",
        ]
    ),
    "short_mode": False,
}

flexion_extension_real = {
    "task_name": "FlexionExtensionReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/flexion_extension_real_light.json",
    "out_dim": 2,
    "outputs": set(["hand movement elbow extension", "hand movement elbow flexion"]),
    "short_mode": False,
}

grasp_imaginary = {
    "task_name": "GraspImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/grasp_imaginary_light.json",
    "out_dim": 2,
    "outputs": set(["imagined palmar grasp", "imagined lateral grasp"]),
    "short_mode": False,
}

grasp_real = {
    "task_name": "GraspReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/grasp_real_light.json",
    "out_dim": 2,
    "outputs": set(["movement palmar grasp", "movement lateral grasp"]),
    "short_mode": False,
}

lr_imaginary = {
    "task_name": "LRImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/lr_imaginary_light.json",
    "out_dim": 2,
    "outputs": set(["left hand imagined movement", "right hand imagined movement"]),
    "short_mode": True,
}

lr_real = {
    "task_name": "LRReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/lr_real_light.json",
    "out_dim": 2,
    "outputs": set(["right hand movement", "left hand movement"]),
    "short_mode": True,
}

mi_task_body_parts_imagined = {
    "task_name": "BodyPartsImagined",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/mi_task_imagined_body_parts_light.json",
    "out_dim": 5,
    "outputs": set(
        [
            "rest",
            "right hand imagined movement",
            "foot imagined movement",
            "left hand imagined movement",
            "tongue imagined movement",
        ]
    ),
    "short_mode": True,
}

mi_task_body_parts_real = {
    "task_name": "BodyPartsReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/mi_task_body_parts_light.json",
    "out_dim": 4,
    "outputs": set(
        ["rest", "right hand movement", "foot movement", "left hand movement"]
    ),
    "short_mode": True,
}

pronation_supination_imaginary = {
    "task_name": "PronationSupinationImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/pronation_supination_imaginary_light.json",
    "out_dim": 2,
    "outputs": set(["imagined supination", "imagined pronation"]),
    "short_mode": False,
}

pronation_supination_real = {
    "task_name": "PronationSupinationReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/pronation_supination_real_light.json",
    "out_dim": 2,
    "outputs": set(["movement supination", "movement pronation"]),
    "short_mode": False,
}

mi_tasks = [
    eye_open_closed,
    eye_vh,
    flexion_extension_imaginary,
    flexion_extension_real,
    grasp_imaginary,
    grasp_real,
    lr_imaginary,
    lr_real,
    mi_task_body_parts_imagined,
    mi_task_body_parts_real,
    pronation_supination_imaginary,
    pronation_supination_real,
]

########################################################################################################################
# ERP JSONs

erp_class = {
    "class_name": "Error-Related Potential",
    "time_col": "time in seconds",
    "prefix_filepath": "/itet-stor/maxihuber/deepeye_storage/foundation_prepared/",
    "load_mode": 0,
}

erp = {
    "task_name": "ERP",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/new_erp_light.json",
    "out_dim": 2,
    "outputs": set(
        [
            "with event-related potential",
            "without event-related potential",
        ]
    ),
}

errp = {
    "task_name": "ERRP",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/net_scratch/finetune_files/errp_all_light.json",
    "out_dim": 2,
    "outputs": set(
        [
            "without error-related potential",
            "with error-related potential",
        ]
    ),
}

erp_tasks = [erp, errp]

########################################################################################################################
# EyeNet JSONs

eye_class = {
    "class_name": "EyeNet",
    "time_col": "time",
    "prefix_filepath": "/itet-stor/maxihuber/deepeye_storage/foundation_prepared/",
    "load_mode": 1,
}

eye_dir_amp = {
    "task_name": "EyeNetDirectionAmp",
    "task_type": "Regression",
    "json_path": [
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Amp_train.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Amp_val.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Amp_test.json",
    ],
    "out_dim": 1,
}

eye_dir_ang = {
    "task_name": "EyeNetDirectionAng",
    "task_type": "Regression",
    "json_path": [
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Ang_train.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Ang_val.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Ang_test.json",
    ],
    "out_dim": 1,
}

eye_lr = {
    "task_name": "EyeNetLR",
    "task_type": "Classification",
    "json_path": [
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_LR_train.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_LR_val.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_LR_test.json",
    ],
    "out_dim": 2,
}

eye_position = {
    "task_name": "EyeNetPosition",
    "task_type": "Regression",
    "json_path": [
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Position_train.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Position_val.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Position_test.json",
    ],
    "out_dim": 2,
}

eye_tasks = [eye_dir_amp, eye_dir_ang, eye_lr, eye_position]

classes = {
    "YC": [yc_class, yc_tasks],
    "Clinical": [cli_class, cli_tasks],
    "MI": [mi_class, mi_tasks],
    "ERP": [erp_class, erp_tasks],
    "EyeNet": [eye_class, eye_tasks],
}

########################################################################################################################
# Select the class and task

# used_class = yc_class
used_class = cli_class
# used_class = mi_class
# used_class = erp_class
# used_class = eye_class
#
# used_task = tuab
# used_task = epilepsy
# used_task = age
used_task = depression
# used_task = parkinsons
# used_task = schizophrenia
# used_task = sex
#
# used_task = eye_open_closed
# used_task = eye_vh
# used_task = flexion_extension_imaginary
# used_task = flexion_extension_real
# used_task = grasp_real
# used_task = lr_imaginary
# used_task = lr_real
# used_task = mi_task_body_parts_real
# used_task = mi_task_body_parts_imagined
# used_task = pronation_supination_real
# used_task = pronation_supination_imaginary
#
# used_task = erp
# used_task = errp
#
# used_task = eye_dir_amp
# used_task = eye_dir_ang
# used_task = eye_lr
# used_task = eye_position

class_name = used_class["class_name"]
time_col = used_class["time_col"]
prefix_filepath = used_class["prefix_filepath"]
load_mode = used_class["load_mode"]
task_name = used_task["task_name"]
task_type = used_task["task_type"]
json_path = used_task["json_path"]
out_dim = used_task["out_dim"]
short_mode = used_task["short_mode"] if "short_mode" in used_task else False

truncate = False
num_keep = 100

with open(
    f"/itet-stor/maxihuber/net_scratch/finetune_files/channels/{class_name.replace(' ', '_')}_{task_name}_cleaned.json",
    "r",
) as f:
    task_channels = set(natsorted(list(json.load(f))))
print(f"Task channels: {task_channels}")


def load_index0(data_index_path):
    with open(data_index_path, "r") as f:
        train_test_dict = json.load(f)
    train_samples = train_test_dict["train"]
    test_samples = train_test_dict["test"]
    return train_samples, test_samples


def load_index1(data_index_paths):
    all_samples = []
    for data_index_path in data_index_paths:
        with open(data_index_path, "r") as f:
            subset_dict = json.load(f)
        all_samples.append(list(subset_dict.values())[0])
    return all_samples[0], all_samples[1], all_samples[2]


def truncate0(train_index, test_index, num_keep, truncate=False):
    train_index = (
        train_index[:num_keep] + train_index[-num_keep:] if truncate else train_index
    )
    test_index = (
        test_index[:num_keep] + test_index[-num_keep:] if truncate else test_index
    )
    return train_index, test_index


def truncate1(train_index, val_index, test_index, num_keep, truncate=False):
    train_index = (
        train_index[:num_keep] + train_index[-num_keep:] if truncate else train_index
    )
    val_index = val_index[:num_keep] + val_index[-num_keep:] if truncate else val_index
    test_index = (
        test_index[:num_keep] + test_index[-num_keep:] if truncate else test_index
    )
    return train_index, val_index, test_index


def get_node_index(index_patterns):
    index_paths = []
    for pattern in index_patterns:  # regex the index_patterns
        index_paths.extend(glob.glob(pattern))
    num_trials = 0
    trial_info_index = {}
    for index_path in index_paths:
        with open(index_path, "r") as f:
            new_trial_info_index = json.load(f)
            for trial_info in new_trial_info_index.values():
                trial_info_index[num_trials] = trial_info
                num_trials += 1
    print(f"[get_node_index] # Trials = {num_trials}", file=sys.stderr)
    return trial_info_index


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


def parallel_load_file_data(
    data_index,
    task_channels,
    filename_to_nodepath,
    load_mode,
    task_name,
    time_col,
    prefix_filepath,
    pool,
):
    num_samples = 0
    data = {}
    outputs = {}
    srs = {}
    durs = {}
    channels = {}
    all_channels = set(task_channels)

    load_func = partial(
        load_sample,
        load_mode=load_mode,
        task_channels=task_channels,
        filename_to_nodepath=filename_to_nodepath,
        task_name=task_name,
        time_col=time_col,
        prefix_filepath=prefix_filepath,
    )

    results = list(
        tqdm(
            pool.imap(load_func, data_index),
            total=len(data_index),
            desc="Loading data",
            position=0,
            leave=True,
        )
    )
    gc.collect()

    for result in results:
        if result is None:
            continue

        (
            data[num_samples],
            outputs[num_samples],
            srs[num_samples],
            durs[num_samples],
            sorted_valid_channels,
        ) = result

        all_channels &= set(sorted_valid_channels)
        channels[num_samples] = sorted_valid_channels
        num_samples += 1

    return data, outputs, srs, durs, channels, all_channels


print(f"Preparing local paths...")
index_patterns = ["/dev/shm/mae/index_*.json", "/scratch/mae/index_*.json"]
node_index = get_node_index(index_patterns=index_patterns)
filename_to_nodepath = {
    os.path.basename(ie["origin_path"]): ie["new_path"]
    for trial_idx, ie in node_index.items()
}
filename_to_nodepath = {}
print(f"Prepared local paths. {len(filename_to_nodepath)} files found on node.")

# Parallel processing pool
pool = mp.Pool(mp.cpu_count())


# Use the new parallel function in your main code
if load_mode != 1:
    train_index, test_index = load_index0(json_path)
    train_index, test_index = truncate0(train_index, test_index, num_keep, truncate)

    print("=" * 10 + "Load train data" + "=" * 100)
    (
        train_data,
        train_outputs,
        train_sr,
        train_dur,
        train_channels,
        train_all_channels,
    ) = parallel_load_file_data(
        train_index,
        task_channels,
        filename_to_nodepath,
        load_mode,
        task_name,
        time_col,
        prefix_filepath,
        pool,
    )

    print("=" * 10 + "Load test data" + "=" * 101)
    (
        test_data,
        test_outputs,
        test_sr,
        test_dur,
        test_channels,
        test_all_channels,
    ) = parallel_load_file_data(
        test_index,
        task_channels,
        filename_to_nodepath,
        load_mode,
        task_name,
        time_col,
        prefix_filepath,
        pool,
    )

    common_channels = train_all_channels & test_all_channels
    assert len(common_channels) > 0, "No common channel found across samples!"
    print(f"Common Channels: {common_channels}")

########################################################################################################################


class FinetuneDataset(Dataset):
    def __init__(
        self,
        data,
        outputs,
        srs,
        durs,
        channels,
        task_type,
        label_encoder=None,
    ):
        self.data = data
        self.outputs = outputs
        self.srs = srs
        self.durs = durs
        self.channels = channels
        self.task_type = task_type
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signals = self.data[idx]
        output = self.outputs[idx]
        sr = self.srs[idx]
        dur = self.durs[idx]
        channels = self.channels[idx]

        if self.task_type == "Classification" and self.label_encoder is not None:
            output = self.label_encoder.transform([output])[0]
            output_tensor = torch.tensor(output, dtype=torch.long)
        else:
            if task_name == "EyeNetPosition":
                output_tensor = torch.tensor(output, dtype=torch.float32)
            else:
                output_tensor = torch.tensor([output], dtype=torch.float32)

        return {
            "signals": signals,
            "output": output_tensor,
            "sr": sr,
            "dur": dur,
            "channels": channels,
        }


########################################################################################################################


def get_nr_y_patches(win_size, sr, self_patch_size):
    return int((sr / 2 * win_size + 1) / self_patch_size)


def get_nr_x_patches(win_size, dur, self_win_shift_factor, self_patch_size):
    win_shift = win_size * self_win_shift_factor
    x_datapoints_per_second = 1 / win_shift
    x_datapoints = dur * x_datapoints_per_second + 1
    return int(x_datapoints // self_patch_size)


# DataLoaders
self_win_shifts = [2, 4, 8]
self_patch_size = 16
self_win_shift_factor = 0.25
self_max_win_shift = self_win_shifts[-1]
self_max_y_datapoints = 4_000
self_max_nr_patches = 6_000

srs = list(train_sr.values()) + list(test_sr.values())
durs = list(train_dur.values()) + list(test_dur.values())
valid_win_shifts = [
    win_shift
    for win_shift in self_win_shifts
    for sr, dur in zip(srs, durs)
    if get_nr_y_patches(win_shift, sr, self_patch_size) >= 1
    and get_nr_x_patches(win_shift, dur, self_win_shift_factor, self_patch_size) >= 1
]
assert len(valid_win_shifts) > 0, "No valid win_shifts found!"

# Sort the valid win_shifts to ensure order
valid_win_shifts.sort()

# Select the middle element, if even number of elements, select the larger one
middle_index = (len(valid_win_shifts) - 1) // 2
win_size = valid_win_shifts[middle_index]

print(f"Win size: {win_size}")


def self_get_generic_channel_name(channel_name):
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


def self_encode_mean(mean, win_size, self_max_y_datapoints, self_max_win_shift):
    y_datapoints = mean.shape[0]
    encoded_mean = torch.zeros(self_max_y_datapoints)
    step_size = int(self_max_win_shift // win_size)
    end_idx = step_size * y_datapoints
    indices = torch.arange(0, end_idx, step_size)
    encoded_mean[indices] = mean.squeeze_().float()
    encoded_mean.unsqueeze_(1)
    return encoded_mean


def get_max_dur(
    n_chns, win_size, sr, self_win_shift_factor, self_patch_size, self_max_nr_patches
):
    single_channel_max_dur = int(
        ((self_patch_size**2) * self_max_nr_patches - sr * win_size / 2 - 1)
        / (sr / self_win_shift_factor / 2 + 1 / self_win_shift_factor / win_size)
    )
    max_dur = int(single_channel_max_dur / n_chns)
    return max_dur


def sample_collate_fn(
    batch,
    self_channel_name_map,
    win_size,
    self_patch_size,
    self_win_shift_factor,
    self_max_nr_patches,
):

    signals, output, sr, dur, channels = (
        batch[0]["signals"],
        batch[0]["output"],
        batch[0]["sr"],
        batch[0]["dur"],
        batch[0]["channels"],
    )

    # truncate signals along time axis to stay below cuda memory limit
    max_dur = get_max_dur(
        len(channels),
        win_size,
        sr,
        self_win_shift_factor,
        self_patch_size,
        self_max_nr_patches,
    )
    if dur > max_dur:
        dur = max_dur
        signals = signals[:, : int(sr * dur)]

    fft = torchaudio.transforms.Spectrogram(
        n_fft=int(sr * win_size),
        win_length=int(sr * win_size),
        hop_length=int(sr * win_size * self_win_shift_factor),
        normalized=True,
    )

    spg_list = []
    chn_list = []
    mean_list = []
    std_list = []

    for signal, channel in zip(signals, channels):

        # Channel information
        channel_name = self_get_generic_channel_name(channel)
        channel = (
            self_channel_name_map[channel_name]
            if channel_name in self_channel_name_map
            else self_channel_name_map["None"]
        )

        # Spectrogram Computation & Cropping
        spg = fft(signal)
        spg = spg**2
        spg = crop_spg(spg, self_patch_size)

        H_new, W_new = spg.shape[0], spg.shape[1]
        h_new, w_new = H_new // self_patch_size, W_new // self_patch_size

        # Prepare channel information (per-patch)
        channel = torch.full((h_new, w_new), channel, dtype=torch.float16)

        spg, mean, std = normalize_spg(spg)
        mean = self_encode_mean(
            mean, win_size, self_max_y_datapoints, self_max_win_shift
        )
        std = self_encode_mean(std, win_size, self_max_y_datapoints, self_max_win_shift)

        spg_list.append(spg)
        chn_list.append(channel)
        mean_list.append(mean)
        std_list.append(std)

    batch = torch.stack(spg_list)
    channels_encoded = torch.stack(chn_list)
    means = torch.stack(mean_list)
    stds = torch.stack(std_list)

    batch.unsqueeze_(1)
    channels_encoded = channels_encoded.flatten(1)
    means = means.transpose(1, 2)
    stds = stds.transpose(1, 2)

    full_batch = {
        "batch": batch,
        "channels": channels_encoded,
        "means": means,
        "stds": stds,
        "win_size": win_size,
        "channels_raw": channels,
    }

    # == Finished iterating over all possible window shifts

    return full_batch, output


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    # Add other types as necessary (e.g., sets)
    return obj


########################################################################################################################

# Mapping of channel names to IDs
channel_name_map_path = (
    "/home/maxihuber/eeg-foundation/src/data/components/channels_to_id3.json"
)
with open(channel_name_map_path, "r") as f:
    self_channel_name_map = json.load(f)

# Label encoder
all_outputs = list(set(list(train_outputs.values()) + list(test_outputs.values())))
label_encoder = LabelEncoder()
label_encoder.fit(all_outputs)

train_dataset = FinetuneDataset(
    train_data,
    train_outputs,
    train_sr,
    train_dur,
    train_channels,
    task_type=task_type,
    label_encoder=label_encoder,
)
test_dataset = FinetuneDataset(
    test_data,
    test_outputs,
    test_sr,
    test_dur,
    test_channels,
    task_type=task_type,
    label_encoder=label_encoder,
)

# Create a partially applied function
partial_collate_fn = partial(
    sample_collate_fn,
    self_channel_name_map=self_channel_name_map,
    win_size=win_size,
    self_patch_size=self_patch_size,
    self_win_shift_factor=self_win_shift_factor,
    self_max_nr_patches=self_max_nr_patches,
)

train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=partial_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=partial_collate_fn)

print(f"Train: {len(train_loader)} | Test: {len(test_loader)}")


########################################################################################################################


class FineTuningModel(L.LightningModule):
    def __init__(
        self,
        encoder,
        frozen_encoder,
        out_dim,
        task_name,
        task_type,
        learning_rate,
        mask_ratio,
    ):
        super(FineTuningModel, self).__init__()

        self.task_name = task_name
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio

        # Pretrained network
        self.encoder = encoder
        if frozen_encoder:
            self.freeze_encoder()

        self.head = nn.Linear(encoder.encoder_embed_dim, out_dim)
        self.criterion = nn.BCELoss()

    def forward(self, x):

        spgs = x["batch"]
        channels = x["channels"]
        means = x["means"]
        stds = x["stds"]
        win_size = x["win_size"]

        x_emb, _, _, _, _ = self.encoder(
            x=spgs,
            means=means,
            stds=stds,
            channels=channels,
            win_size=win_size,
            mask_ratio=self.mask_ratio,
        )

        del spgs, channels, means, stds
        torch.cuda.empty_cache()

        # Get CLS tokens
        x_emb = x_emb[:, 0, :]

        # Could be done better
        x_emb_allcls = x_emb
        x_emb = torch.mean(x_emb, dim=0)

        return x_emb, x_emb_allcls, x["channels_raw"]

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


########################################################################################################################

# Load the checkpoint
chkpt_path = (
    "/itet-stor/maxihuber/net_scratch/checkpoints/1004189/manual-epoch-end.ckpt"
)
checkpoint = torch.load(chkpt_path, map_location=torch.device("cpu"))
state_dict = checkpoint["state_dict"]
state_dict = {
    k.replace("net.encoder.", ""): v
    for k, v in state_dict.items()
    if "net.encoder." in k
}

# Initialize the encoder and load the state dict
encoder = EncoderViTRoPE(
    channel_names_path=channel_name_map_path,
    mask_ratio=0.0,
    encoder_embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    encoder_mlp_ratio=4,
    encoder_qkv_bias=True,
    encoder_drop_rate=0.1,
    encoder_attn_drop_rate=0.1,
    encoder_drop_path_rate=0.1,
    encoder_init_scale=1e-4,
    encoder_rope_theta=100.0,
)
encoder.load_state_dict(state_dict)

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the fine-tuning model
fine_tuning_model = FineTuningModel(
    encoder=encoder,
    frozen_encoder=True,
    out_dim=out_dim,
    task_name=task_name,
    task_type=task_type,
    learning_rate=0.01,
    mask_ratio=0,
).to(device)

########################################################################################################################


# Step 6: Define the function to extract embeddings
def extract_embeddings(loader):
    embeddings = []
    embeddings_allcls = []
    labels = []
    embeddings_channels = []  # List to store channel names for each batch
    for i, (full_batch, label) in tqdm(
        enumerate(loader), desc="Extracting encoder embeddings", position=0, leave=True
    ):
        full_batch = move_to_device(full_batch, device)  # Move inputs to the GPU
        with torch.no_grad():  # No need to compute gradients for inference
            x_emb, x_emb_allcls, x_channels = fine_tuning_model(full_batch)
        embeddings.append(x_emb.cpu().numpy())  # Move to CPU and convert to numpy
        embeddings_allcls.append(x_emb_allcls.cpu().numpy())
        labels.append(label.cpu().numpy())
        embeddings_channels.append(x_channels)  # Append the channel names
        # Delete tensors to free GPU memory
        del full_batch, label, x_emb, x_emb_allcls
        torch.cuda.empty_cache()
        if i % 100 == 0:
            gc.collect()
    embeddings = np.vstack(embeddings)  # shape (n_samples, encoder_embed_dim)
    labels = np.array(labels)
    return embeddings, labels, embeddings_allcls, embeddings_channels


finetune_models = {
    "Classification": {
        "XGBoost": xgb.XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
    },
    "Regression": {
        "XGBoost": xgb.XGBRegressor(
            objective="reg:squarederror",
            use_label_encoder=False,
            eval_metric="rmse",
            random_state=42,
        )
    },
}

scores = {
    "Classification": {
        "Accuracy": accuracy_score,
        "Balanced Accuracy": balanced_accuracy_score,
        "Precision": partial(precision_score, zero_division=np.nan),
        "Recall": partial(recall_score, zero_division=np.nan),
        "F1 Score": partial(f1_score, zero_division=np.nan),
        "ROC AUC": roc_auc_score,
        "Confusion Matrix": confusion_matrix,
    },
    "Regression": {
        "MAE": mean_absolute_error,
        "RMSE": lambda y_true, y_pred: mean_squared_error(
            y_true, y_pred, squared=False
        ),  # Using lambda for RMSE
        "R-squared": r2_score,
        "MAPE": mean_absolute_percentage_error,
    },
}


########################################################################################################################

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

models_dir = (
    f"/itet-stor/maxihuber/net_scratch/finetune_models/{class_name}/{task_name}"
)
scores_dir = (
    f"/itet-stor/maxihuber/net_scratch/finetune_scores/{class_name}/{task_name}"
)
embeds_dir = (
    f"/itet-stor/maxihuber/net_scratch/finetune_embeddings/{class_name}/{task_name}"
)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(scores_dir, exist_ok=True)
os.makedirs(embeds_dir, exist_ok=True)

# Step 7: Extract embeddings for training data
train_embeddings, y_train, train_embeddings_allcls, train_channels = extract_embeddings(
    train_loader
)

# Step 8: Extract embeddings for test data
test_embeddings, y_test, test_embeddings_allcls, test_channels = extract_embeddings(
    test_loader
)

# Step 9: Store the embeddings to a file
train_data_tuple = (train_embeddings, y_train, train_embeddings_allcls, train_channels)
train_data_path = os.path.join(embeds_dir, f"train_data_{timestamp}")
test_data_tuple = (test_embeddings, y_test, test_embeddings_allcls, test_channels)
test_data_path = os.path.join(embeds_dir, f"test_data_{timestamp}")
with open(train_data_path, "wb") as f:
    pickle.dump(train_data_tuple, f)
    print(f"Stored train embeddings at {train_data_path}")
with open(test_data_path, "wb") as f:
    pickle.dump(test_data_tuple, f)
    print(f"Stored train embeddings at {test_data_path}")

from collections import Counter

print("Training set class distribution:", Counter(y_train))
print("Test set class distribution:", Counter(y_test))

# Initialize a list to store the scores
scores_list = []

for name, clf in finetune_models[task_type].items():
    print("=" * 10 + f"{name} Model" + "=" * 100, file=sys.stderr)

    if name == "XGBoost":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_embeddings)
        X_test_scaled = scaler.transform(test_embeddings)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
    else:
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(train_embeddings, y_train)
        y_pred = clf.predict(test_embeddings)

    # Store the fitted predictor for later use
    if name == "XGBoost":
        model_path = os.path.join(models_dir, f"{name}_model_{timestamp}.json")
        clf.save_model(model_path)
    else:
        model_path = os.path.join(models_dir, f"{name}_model_{timestamp}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

    model_scores = {"Model": name}
    for score_name, score_func in scores[task_type].items():
        score = score_func(y_test, y_pred)
        model_scores[score_name] = score

    # Store the scores already
    scores_df = pd.DataFrame(scores_list)
    scores_path = os.path.join(scores_dir, f"model_scores_{timestamp}.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"Stored scores at {scores_path}")

    # Store the scores for later use
    scores_list.append(model_scores)

    # Clean up to save memory
    del clf, y_pred
    gc.collect()

# Convert the scores list to a DataFrame and save it
scores_df = pd.DataFrame(scores_list)
scores_path = os.path.join(scores_dir, f"model_scores_{timestamp}.csv")
scores_df.to_csv(scores_path, index=False)
print(f"Stored scores at {scores_path}")
