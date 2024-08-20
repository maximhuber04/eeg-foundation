print("Hello world")

# Add custom path
import sys

sys.path.append("/home/maxihuber/eeg-foundation/")

# Standard library imports
import os
import gc
import glob
import json

# Third-party library imports
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt

# Torch/Lightning
import torch
import lightning.pytorch as L

# Sklearn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
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
from mne.preprocessing import Xdawn

mne.set_log_level("warning")

# Custom imports
from src.utils.preloading.utils import load_edf_to_dataframe
from src.finetune.v2.parallel_helpers import (
    load_sample,
    resample_signal,
    pad_or_truncate_signal,
)

# Seed everything
L.seed_everything(42)

print("Bye world")

###################################################################################################################

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

used_class = yc_class
# used_class = cli_class
# used_class = mi_class
# used_class = erp_class
# used_class = eye_class
#
# used_task = tuab
used_task = epilepsy
# used_task = age
# used_task = depression
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


def filter_channels_by_indices(data, channels, common_channels):
    common_channel_set = set(common_channels)  # Use a set for faster lookups
    for k, signals in tqdm(data.items(), desc="Filter for common channels"):
        common_channel_indices = [
            i for i, ch in enumerate(channels[k]) if ch in common_channel_set
        ]
        common_channel_indices = np.array(common_channel_indices)
        data[k] = signals[common_channel_indices, :].clone()  # Ensure data is copied
        # Free memory for the signal
        del signals
        if k % 10 == 0:
            gc.collect()
    return data  # Returning data is optional as the changes are in-place


# Parallel resampling function
def parallel_resample_signals(data, srs, target_sfreq, pool):
    resample_func = partial(resample_signal, target_sfreq=target_sfreq)
    inputs = [(idx, signal, srs[idx]) for idx, signal in data.items()]
    results = list(
        tqdm(
            pool.starmap(resample_func, inputs),
            total=len(inputs),
            desc="Resampling signals",
            position=0,
            leave=True,
        )
    )
    gc.collect()
    for idx, resampled_signal in results:
        if resampled_signal is not None:
            data[idx] = resampled_signal
    # Trigger final garbage collection after all tasks are done
    gc.collect()
    return data


def parallel_pad_or_truncate_signals(data, common_length, pool):
    pad_truncate_func = partial(pad_or_truncate_signal, common_length=common_length)
    inputs = [(idx, signal) for idx, signal in data.items()]
    results = list(
        tqdm(
            pool.starmap(pad_truncate_func, inputs),
            total=len(inputs),
            desc="Pad/Truncate signals",
            position=0,
            leave=True,
        )
    )
    gc.collect()
    for idx, padded_signal in results:
        if padded_signal is not None:
            data[idx] = padded_signal
    # Trigger final garbage collection after all tasks are done
    gc.collect()
    return data


def create_epochs(data, outputs, channels, sfreq=1000, is_classification=True):
    event_id = {}
    epochs_data = []
    events = []
    for idx, signal in tqdm(data.items(), desc="Creating epochs"):
        signal_numpy = signal.numpy()
        epochs_data.append(signal_numpy)
        if is_classification:
            if outputs[idx] not in event_id:
                event_id[outputs[idx]] = len(event_id) + 1
            events.append([idx, 0, event_id[outputs[idx]]])
        else:
            events.append([idx, 0, 1])
        # Free memory for the signal
        del signal, signal_numpy
        if idx % 10 == 0:
            gc.collect()
    events = np.array(events, dtype=int)
    epochs_data = np.array(epochs_data)
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(
        epochs_data,
        info,
        events=events,
        event_id=event_id if is_classification else None,
    )
    # Free memory for the intermediate variables
    del events, info, epochs_data
    gc.collect()

    return epochs


########################################################################################################################

L.seed_everything(42)
sys.path.append("/home/maxihuber/eeg-foundation/src/models/components/Baselines")

# Filter data by common channels
train_data_filtered = filter_channels_by_indices(
    train_data, train_channels, common_channels
)
del train_data, train_channels
test_data_filtered = filter_channels_by_indices(
    test_data, test_channels, common_channels
)
del test_data, test_channels

print("Start resampling signals...", file=sys.stderr)

# Resample signals
target_sfreq = int(max(list(train_sr.values()) + list(test_sr.values())))
print(f"Target Sampling Frequency: {target_sfreq}", file=sys.stderr)

train_data_resampled = parallel_resample_signals(
    train_data_filtered, train_sr, target_sfreq, pool
)
print("Resampled train data", file=sys.stderr)
test_data_resampled = parallel_resample_signals(
    test_data_filtered, test_sr, target_sfreq, pool
)
print("Resampled test data", file=sys.stderr)
del train_data_filtered, test_data_filtered  # Free memory
gc.collect()  # Explicitly invoke garbage collection

# Get duration and channel counts for padding/truncating
durs = [
    signals_tensor.shape[1] for idx, signals_tensor in train_data_resampled.items()
] + [signals_tensor.shape[1] for idx, signals_tensor in test_data_resampled.items()]
dur_90 = int(np.percentile(durs, 90))
common_length = dur_90

# Pad or truncate signals
train_data_padded = parallel_pad_or_truncate_signals(
    train_data_resampled, common_length, pool
)
print("Padded train data", file=sys.stderr)
test_data_padded = parallel_pad_or_truncate_signals(
    test_data_resampled, common_length, pool
)
print("Padded test data", file=sys.stderr)
del train_data_resampled, test_data_resampled  # Free memory
gc.collect()  # Explicitly invoke garbage collection

is_classification = True if task_type == "Classification" else False

print("Start creating epochs...", file=sys.stderr)
epochs_train = create_epochs(
    train_data_padded,
    train_outputs,
    list(common_channels),
    target_sfreq,
    is_classification,
)
print("Created train epochs", file=sys.stderr)
epochs_test = create_epochs(
    test_data_padded,
    test_outputs,
    list(common_channels),
    target_sfreq,
    is_classification,
)
print("Created test epochs", file=sys.stderr)
del train_data_padded, test_data_padded  # Free memory
gc.collect()  # Explicitly invoke garbage collection

print("Start fitting Xdawn...", file=sys.stderr)

xdawn = Xdawn(n_components=2, correct_overlap=False, reg=0.1)
xdawn.fit(epochs_train)

########################################################################################################################


def get_labels(train_outputs, test_outputs, is_classification):
    if is_classification:
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(list(train_outputs.values()))
        y_test = label_encoder.transform(list(test_outputs.values()))
    else:
        y_train = np.array(list(train_outputs.values()))
        y_test = np.array(list(test_outputs.values()))
    return y_train, y_test


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

simple_models = {
    "Classification": {
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "Nearest Neighbors": KNeighborsClassifier(3),
        "Linear SVM": SVC(kernel="linear", C=0.025, random_state=42),
        "RBF SVM": SVC(gamma=2, C=1, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
    },
    "Regression": {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=3),
        "Support Vector Regression (Linear)": SVR(kernel="linear", C=1.0),
        "Support Vector Regression (RBF)": SVR(kernel="rbf", C=1.0, gamma=0.1),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            objective="reg:squarederror",
            use_label_encoder=False,
            eval_metric="rmse",
            random_state=42,
        ),
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

models_dir = (
    f"/itet-stor/maxihuber/net_scratch/finetune_models/{class_name}/{task_name}"
)
scores_dir = (
    f"/itet-stor/maxihuber/net_scratch/finetune_scores/{class_name}/{task_name}"
)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(scores_dir, exist_ok=True)

# Transform the data using xDAWN
X_train_xdawn = xdawn.transform(epochs_train)
X_test_xdawn = xdawn.transform(epochs_test)

# Flatten the transformed data for LDA input
n_epochs_train, n_components, n_times = X_train_xdawn.shape
X_train_xdawn = X_train_xdawn.reshape(n_epochs_train, n_components * n_times)
n_epochs_test, n_components, n_times = X_test_xdawn.shape
X_test_xdawn = X_test_xdawn.reshape(n_epochs_test, n_components * n_times)

y_train, y_test = get_labels(train_outputs, test_outputs, is_classification)

from collections import Counter

print("Training set class distribution:", Counter(y_train))
print("Test set class distribution:", Counter(y_test))

# Initialize a list to store the scores
scores_list = []

for name, clf in simple_models[task_type].items():
    print("=" * 10 + f"{name} Model" + "=" * 100, file=sys.stderr)

    if name == "XGBoost":
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_xdawn)
        X_test_scaled = scaler.transform(X_test_xdawn)

        # Fit the model
        clf.fit(X_train_scaled, y_train)

        # Store the fitted model
        model_path = os.path.join(models_dir, f"{name}_model.json")
        clf.save_model(model_path)

        # Predict the test data
        y_pred = clf.predict(X_test_scaled)
    else:
        # Create a pipeline with scaling and the classifier
        clf_pipeline = make_pipeline(StandardScaler(), clf)
        clf_pipeline.fit(X_train_xdawn, y_train)

        # Store the fitted model
        model_path = os.path.join(models_dir, f"{name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf_pipeline, f)

        # Predict the test data
        y_pred = clf_pipeline.predict(X_test_xdawn)

    y_pred = clf.predict(X_test_xdawn)

    model_scores = {"Model": name}
    for score_name, score_func in scores[task_type].items():
        score = score_func(y_test, y_pred)
        model_scores[score_name] = score

    # Convert the scores list to a DataFrame and save it
    scores_df = pd.DataFrame(scores_list)
    scores_path = os.path.join(scores_dir, "model_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"Stored scores at {scores_path}")

    # Store the scores for later use
    scores_list.append(model_scores)

    # Clean up to save memory
    del clf, y_pred
    gc.collect()

# Convert the scores list to a DataFrame and save it
scores_df = pd.DataFrame(scores_list)
scores_path = os.path.join(scores_dir, "model_scores.csv")
scores_df.to_csv(scores_path, index=False)
print(f"Stored scores at {scores_path}")
