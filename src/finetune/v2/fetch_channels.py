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
import concurrent.futures
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from natsort import natsorted
from functools import partial
import matplotlib.pyplot as plt
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

# Seed everything
L.seed_everything(42)

print("Bye world")

####################################################################################################

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
        _file = os.path.basename(file)
        if _file in filename_to_nodepath:
            adjusted_files.append(filename_to_nodepath[_file])
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


def find_and_store_channels(
    data_index, filename_to_nodepath, load_mode, class_name, task_name, prefix_filepath
):
    task_channels = set()

    for sample in data_index:
        try:
            input_files = get_full_paths(
                sample["input"], prefix_filepath, filename_to_nodepath
            )

            file = input_files[0]
            if load_mode == 2:
                data = mne.io.read_raw_edf(file, preload=True)
                ch_names = data.ch_names
            else:
                data = pd.read_pickle(file)
                ch_names = list(data.columns)

            ch_names = [get_generic_channel_name(ch_name) for ch_name in ch_names]
            task_channels.update(ch_names)

            del data

        except Exception as e:
            print(f"Failed to process sample: {sample}. Error: {e}", file=sys.stderr)

    with open(
        f'/itet-stor/maxihuber/net_scratch/finetune_files/channels/{class_name.replace(" ", "_")}_{task_name}.json',
        "w",
    ) as f:
        json.dump(list(task_channels), f)


print(f"Preparing local paths...")
index_patterns = ["/dev/shm/mae/index_*.json", "/scratch/mae/index_*.json"]
node_index = get_node_index(index_patterns=index_patterns)
filename_to_nodepath = {
    os.path.basename(ie["origin_path"]): ie["new_path"]
    for trial_idx, ie in node_index.items()
}
print(f"Prepared local paths. {len(filename_to_nodepath)} files found on node.")

truncate = False
num_keep = 100

classes = {
    "YC": [yc_class, yc_tasks],
    "Clinical": [cli_class, cli_tasks],
    "MI": [mi_class, mi_tasks],
    "ERP": [erp_class, erp_tasks],
    # "EyeNet": [eye_class, eye_tasks],
}

########################################################################################################################


def process_task(class_name, used_class, used_task):
    class_name = used_class["class_name"]
    time_col = used_class["time_col"]
    prefix_filepath = used_class["prefix_filepath"]
    load_mode = used_class["load_mode"]
    task_name = used_task["task_name"]
    task_type = used_task["task_type"]
    json_path = used_task["json_path"]
    out_dim = used_task["out_dim"]
    short_mode = used_task["short_mode"] if "short_mode" in used_task else False

    print(f"Processing task: {class_name} - {task_name}")

    if load_mode != 1:
        train_index, test_index = load_index0(json_path)
        data_index = train_index + test_index
        find_and_store_channels(
            data_index,
            filename_to_nodepath,
            load_mode,
            class_name,
            task_name,
            prefix_filepath,
        )
    else:
        train_index, val_index, test_index = load_index1(json_path)
        data_index = train_index + val_index + test_index
        find_and_store_channels(
            data_index,
            filename_to_nodepath,
            load_mode,
            class_name,
            task_name,
            prefix_filepath,
        )


# Create a list of tasks
tasks = []
for class_name, [used_class, tasks_list] in classes.items():
    for used_task in tasks_list:
        tasks.append((class_name, used_class, used_task))

# Execute tasks in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(process_task, class_name, used_class, used_task)
        for class_name, used_class, used_task in tasks
    ]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()  # To raise exceptions if any occurred
        except Exception as e:
            print(f"Error occurred: {e}")

print("All tasks completed.")
