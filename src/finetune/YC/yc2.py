import sys

sys.path.append("/home/maxihuber/eeg-foundation/")

import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler

import os
import pickle
from tqdm import tqdm
import lightning as L
import torch.nn as nn
from torch.utils.data import random_split
from lightning.pytorch.callbacks import ModelCheckpoint
import random
from collections import Counter
from collections import defaultdict

import os
import numpy as np
import mne
import torch
from tqdm import tqdm
from mne.preprocessing import Xdawn
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

import torchaudio
from src.data.transforms import (
    crop_spg,
    custom_fft,
    normalize_spg,
)

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchmetrics

from functools import partial
from sklearn.metrics import balanced_accuracy_score
from src.models.mae_rope_encoder import EncoderViTRoPE
from src.models.components.vit_rope import (
    Flexible_RoPE_Layer_scale_init_Block,
    FlexibleRoPEAttention,
    compute_axial_cis,
    select_freqs_cis,
)
from timm.models.vision_transformer import Mlp as Mlp
from torch.nn import TransformerEncoderLayer
from src.models.components.SimpleTransformer import SimpleTransformer

mne.set_log_level("warning")

L.seed_everything(42)

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
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/cli/tuab.json",
    "out_dim": 2,
}

epilepsy = {
    "task_name": "Epilepsy",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/cli/epilepsy.json",
    "out_dim": 2,
}

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
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_tasks/age.json",
    "out_dim": 1,
}

depression = {
    "task_name": "Depression",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_tasks/cli_depression.json",
    "out_dim": 2,
}

parkinsons = {
    "task_name": "Parkinsons",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_tasks/cli_parkinsons.json",
    "out_dim": 2,
}

schizophrenia = {
    "task_name": "Schizophrenia",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_tasks/cli_schizophrenia.json",
    "out_dim": 2,
}

sex = {
    "task_name": "Sex",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_tasks/sex.json",
    "out_dim": 2,
}


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
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/eye_open_closed.json",
    "out_dim": 2,
    "outputs": set(["eye open", "eye closed"]),
    "short_mode": False,
}

eye_vh = {
    "task_name": "EyeVH",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/eye_vh.json",
    "out_dim": 2,
    "outputs": set(["vertical", "horizontal"]),
    "short_mode": False,
}

flexion_extension_imaginary = {
    "task_name": "FlexionExtensionImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/flexion_extension_imaginary.json",
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
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/flexion_extension_real.json",
    "out_dim": 2,
    "outputs": set(["hand movement elbow extension", "hand movement elbow flexion"]),
    "short_mode": False,
}

grasp_imaginary = {
    "task_name": "GraspImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/grasp_imaginary.json",
    "out_dim": 2,
    "outputs": set(["imagined palmar grasp", "imagined lateral grasp"]),
    "short_mode": False,
}

grasp_real = {
    "task_name": "GraspReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/grasp_real.json",
    "out_dim": 2,
    "outputs": set(["movement palmar grasp", "movement lateral grasp"]),
    "short_mode": False,
}

lr_imaginary = {
    "task_name": "LRImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/lr_imaginary.json",
    "out_dim": 2,
    "outputs": set(["left hand imagined movement", "right hand imagined movement"]),
    "short_mode": True,
}

lr_real = {
    "task_name": "LRReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/lr_real.json",
    "out_dim": 2,
    "outputs": set(["right hand movement", "left hand movement"]),
    "short_mode": True,
}

mi_task_body_parts_real = {
    "task_name": "BodyPartsReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/mi_task_body_parts.json",
    "out_dim": 4,
    "outputs": set(
        ["rest", "right hand movement", "foot movement", "left hand movement"]
    ),
    "short_mode": True,
}

mi_task_body_parts_imagined = {
    "task_name": "BodyPartsImagined",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/mi_task_body_parts.json",
    "out_dim": 4,
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

pronation_supination_real = {
    "task_name": "PronationSupinationReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/pronation_supination_real.json",
    "out_dim": 2,
    "outputs": set(["movement supination", "movement pronation"]),
    "short_mode": False,
}

pronation_supination_imaginary = {
    "task_name": "PronationSupinationImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/pronation_supination_imaginary.json",
    "out_dim": 2,
    "outputs": set(["imagined supination", "imagined pronation"]),
    "short_mode": False,
}

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
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/erp/erp_all.json",
    "out_dim": 5,
    "outputs": set(
        [
            "Participant is in resting state",
            "with event-related potential",
            "Participant is in interval between two flashes",
            "without event-related potential",
            "Participant keeps closing eyes",
        ]
    ),
}

errp = {
    "task_name": "ERRP",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/erp/errp_all.json",
    "out_dim": 7,
    "outputs": set(
        [
            "Target is located in the right",
            "without error-related potential",
            "The cursor moves to the left",
            "The feedback consisted in the selected item is presented on the screen",
            "The cursor moves to the right",
            "with error-related potential",
            "Target is located in the left",
        ]
    ),
}

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

########################################################################################################################

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

task_channels = set(
    [
        "p4",
        "c3",
        "pz",
        "fp2",
        "t6",
        "fz",
        "f4",
        "f8",
        "t4",
        "t3",
        "p3",
        "a2",
        "oz",
        "a1",
        "t5",
        "t1",
        "cz",
        "c4",
        "fp1",
        "o1",
        "o2",
        "f3",
        "f7",
    ]
)


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


dataset_dict = {
    "ERP_ERP_ANA": 0,
    "RS_RS_ALPHA": 1,
    "ERP_ERP_BISC": 2,
    "ERP_ERP_BBI": 3,
    "ERP_ERP_BICF": 4,
    "ERP_ERP_BICD": 5,
    "RS_RS_SPIS": 6,
    "MI_MI_HGD": 7,
    "MI_MI_SCP": 8,
    "ErrP_ErrP_MERP": 9,
    "MI_MI_ULM": 10,
    "MI_MI_VEP": 11,
    "MI_MI_LR": 12,
    "MI_BBCI_IV_Graz_b": 13,
    "MI_MI_EB": 14,
    "MI_BBCI_IV_Graz_a": 15,
    "MI_MI_GVH_V": 16,
    "MI_MI_GAL": 17,
    "MI_MI_Two": 18,
    "MI_MI_GVH_H": 19,
    "MI_MI_II": 20,
    "ErrP_ErrP_BCI": 21,
    "MI_MI_GVH_G": 22,
    "MI_MI_Limb": 23,
    "MI_MI_SCI": 24,
    "MI_BBCI_IV_Berlin": 25,
    "MI_eegmmidb": 26,
    "ERP_ERP_FHD": 27,
    "RS_RS_EID": 28,
}


def extract_dataset_name(file_path, dataset_dict):
    for name in dataset_dict.keys():
        if name in file_path:
            return name
    return "Unknown"


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


def load_edf_to_dataframe(file_path):
    eeg_data = mne.io.read_raw_edf(file_path, preload=True)
    channel_data_dict = {}

    for channel in eeg_data.ch_names:
        idx = eeg_data.ch_names.index(channel)
        channel = get_generic_channel_name(channel)
        data, times = eeg_data[idx, :]
        channel_data_dict[channel] = data.flatten()

    df = pd.DataFrame(channel_data_dict)
    df["Time in Seconds"] = times.flatten()
    return df


def load_file_data(data_index, task_channels):
    num_samples = 0
    data = {}
    outputs = {}
    srs = {}
    durs = {}
    channels = {}
    datasets = {}
    failed_samples = []

    for sample in tqdm(data_index, desc="Loading data", position=0, leave=True):
        try:
            # Load and concatenate dataframe
            input_files = sample["input"]

            if load_mode == 2:
                file = (
                    prefix_filepath + input_files[0]
                    if "/itet-stor" not in input_files[0]
                    else input_files[0]
                )
                df = load_edf_to_dataframe(file)
                datasets[num_samples] = "TUEG"
            else:
                df = pd.DataFrame()
                for file in input_files:
                    if load_mode != 1:
                        file = prefix_filepath + file
                    else:
                        file = file.replace("/itet-stor/kard", "/itet-stor/maxihuber")
                    with open(file, "rb") as f:
                        df_new = pd.read_pickle(f)
                        df = pd.concat([df, df_new], axis=0)
                dataset_name = extract_dataset_name(file, dataset_dict)
                datasets[num_samples] = dataset_name

            start = int(sample["start"])
            length = int(sample["length"]) if "length" in sample else int(sample["end"])
            if load_mode != 1:
                df = df.iloc[start:length, :]
                if short_mode:
                    df = df.iloc[: int(len(df) * 0.5), :]
            else:
                df = df.loc[start : start + length, :]

            # Add metadata
            if len(df) <= 1:
                assert False
            sr = int(
                1 / float(float(df[time_col].iloc[1]) - float(df[time_col].iloc[0]))
            )
            if load_mode != 1:
                outputs[num_samples] = (
                    sample["output"] if "output" in sample else sample["label"]
                )
            else:
                if task_name == "EyeNetPosition":
                    outputs[num_samples] = list(sample["output"].values())
                else:
                    outputs[num_samples] = list(sample["output"].values())[0]
            srs[num_samples] = sr
            durs[num_samples] = len(df) / sr
            channels[num_samples] = list(set(df.columns) & task_channels)
            df = df[channels[num_samples]].astype(float)
            signals = torch.tensor(df.to_numpy(), dtype=torch.float32).T
            data[num_samples] = signals
            num_samples += 1

        except Exception as e:
            print(f"Failed to process sample: {sample}. Error: {e}", file=sys.stderr)
            failed_samples.append(sample)

    return data, outputs, srs, durs, channels, datasets


if load_mode != 1:
    print(json_path, file=sys.stderr)
    train_index, test_index = load_index0(json_path)
else:
    train_index, val_index, test_index = load_index1(json_path)

print(f"Full train size: {len(train_index)}", file=sys.stderr)
print(f"Full test size: {len(test_index)}", file=sys.stderr)

truncate = """
if load_mode != 1:
    train_index = train_index[:100] + train_index[-100:]
    test_index = test_index[:50] + test_index[-50:]
else:
    train_index = train_index
    val_index = val_index
    test_index = test_index
"""

if load_mode == 0 or load_mode == 2:
    print("=" * 10 + "Load train data" + "=" * 100)
    train_data, train_outputs, train_sr, train_dur, train_channels, train_datasets = (
        load_file_data(train_index, task_channels)
    )
    print("=" * 10 + "Load test data" + "=" * 100)
    test_data, test_outputs, test_sr, test_dur, test_channels, test_datasets = (
        load_file_data(test_index, task_channels)
    )
elif load_mode == 1:
    train_data, train_outputs, train_sr, train_dur, train_channels, train_datasets = (
        load_file_data(train_index, task_channels)
    )
    val_data, val_outputs, val_sr, val_dur, val_channels, val_datasets = load_file_data(
        val_index, task_channels
    )
    test_data, test_outputs, test_sr, test_dur, test_channels, test_datasets = (
        load_file_data(test_index, task_channels)
    )
else:
    pass


# Label Encoder & Class Weights
from sklearn.preprocessing import LabelEncoder

if isinstance(list(train_outputs.values())[0], str):
    all_outputs = list(set(list(train_outputs.values()) + list(test_outputs.values())))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_outputs)

    print(f"Train classes: {set(train_outputs.values())}", file=sys.stderr)
    print(f"Test classes: {set(test_outputs.values())}", file=sys.stderr)

    # Encode the train and test outputs
    encoded_train_outputs = {
        k: label_encoder.transform([v])[0] for k, v in train_outputs.items()
    }
    encoded_test_outputs = {
        k: label_encoder.transform([v])[0] for k, v in test_outputs.items()
    }

    # Create the output counts map
    train_output_counts = defaultdict(int)
    for output in encoded_train_outputs.values():
        train_output_counts[output] += 1

    test_output_counts = defaultdict(int)
    for output in encoded_test_outputs.values():
        test_output_counts[output] += 1

    full_output_counts = train_output_counts.copy()
    for output, count in test_output_counts.items():
        full_output_counts[output] += count

    print("Full Output Counts:", full_output_counts, file=sys.stderr)

    # Calculate class weights
    total_count = sum(full_output_counts.values())
    class_weights = {
        output: total_count / count for output, count in full_output_counts.items()
    }

    # Convert class weights to a tensor
    weight_tensor = torch.tensor(
        [class_weights[i] for i in range(len(class_weights))], dtype=torch.float
    )
else:
    label_encoder = None
    weight_tensor = None

########################################################################################################################

L.seed_everything(42)

ckpt_path = "/itet-stor/maxihuber/net_scratch/checkpoints/980473/epoch=7-step=239317-val_loss=130.45-lr.ckpt"
ckpt_path = "/itet-stor/maxihuber/net_scratch/checkpoints/977598/epoch=0-step=32807-val_loss=133.55.ckpt"


#########################################################################################################
class FinetuneDataset(Dataset):
    def __init__(
        self,
        data,
        outputs,
        srs,
        durs,
        channels,
        datasets,
        task_type,
        label_encoder=None,
    ):
        self.data = data
        self.outputs = outputs
        self.srs = srs
        self.durs = durs
        self.channels = channels
        self.datasets = datasets
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
        dataset = self.datasets[idx]

        if self.task_type == "Classification" and self.label_encoder is not None:
            output = self.label_encoder.transform([output])[
                0
            ]  # Encode the output label
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
            "dataset": dataset,
        }


if load_mode != 1:
    full_train_dataset = FinetuneDataset(
        train_data,
        train_outputs,
        train_sr,
        train_dur,
        train_channels,
        train_datasets,
        task_type=task_type,
        label_encoder=label_encoder,
    )
    test_dataset = FinetuneDataset(
        test_data,
        test_outputs,
        test_sr,
        test_dur,
        test_channels,
        test_datasets,
        task_type=task_type,
        label_encoder=label_encoder,
    )
    # Define the split ratio
    train_ratio = 0.85
    val_ratio = 0.15
    # Calculate lengths for train and validation sets
    total_size = len(full_train_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    # Split the dataset
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )
elif load_mode == 1:
    train_dataset = FinetuneDataset(
        train_data,
        train_outputs,
        train_sr,
        train_dur,
        train_channels,
        train_datasets,
        task_type=task_type,
        label_encoder=label_encoder,
    )
    val_dataset = FinetuneDataset(
        val_data,
        val_outputs,
        val_sr,
        val_dur,
        val_channels,
        val_datasets,
        task_type=task_type,
        label_encoder=label_encoder,
    )
    test_dataset = FinetuneDataset(
        test_data,
        test_outputs,
        test_sr,
        test_dur,
        test_channels,
        test_datasets,
        task_type=task_type,
        label_encoder=label_encoder,
    )
else:
    pass

#########################################################################################################
# DataLoaders
self_win_shifts = [0.25, 0.5, 1, 2, 4, 8]
self_patch_size = 16
self_win_shift_factor = 0.25
self_max_win_shift = self_win_shifts[-1]
self_max_y_datapoints = 4_000
max_nr_patches = 8_500


def get_nr_y_patches(win_size, sr):
    return int((sr / 2 * win_size + 1) / self_patch_size)


def get_nr_x_patches(win_size, dur):
    win_shift = win_size * self_win_shift_factor
    x_datapoints_per_second = 1 / win_shift
    x_datapoints = dur * x_datapoints_per_second + 1
    return int(x_datapoints // self_patch_size)


channel_name_map_path = (
    "/home/maxihuber/eeg-foundation/src/data/components/channels_to_id.json"
)
with open(channel_name_map_path, "r") as file:
    self_channel_name_map = json.load(file)


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


def self_encode_mean(mean, win_size):
    y_datapoints = mean.shape[0]
    encoded_mean = torch.zeros(self_max_y_datapoints)
    step_size = int(self_max_win_shift // win_size)
    end_idx = step_size * y_datapoints
    indices = torch.arange(0, end_idx, step_size)
    encoded_mean[indices] = mean.squeeze_().float()
    encoded_mean.unsqueeze_(1)
    return encoded_mean


#########################################################################################################
# collate_fn
# make batches as the pre-trained network expects (channel tokens, means, standard deviation etc.)
def sample_collate_fn(batch):

    signals, output, sr, dur, channels, dataset = (
        batch[0]["signals"],
        batch[0]["output"],
        batch[0]["sr"],
        batch[0]["dur"],
        batch[0]["channels"],
        batch[0]["dataset"],
    )

    if dur > 1_000:
        dur = 1_000
        signals = signals[:, : 1_000 * sr]

    # only choose one win_shift, but keep it in a list
    valid_win_shifts = [
        [
            win_shift
            for win_shift in self_win_shifts
            if get_nr_y_patches(win_shift, sr) >= 1
            and get_nr_x_patches(win_shift, dur) >= 1
        ][0]
    ]

    # list holding assembled tensors for varying window shifts
    full_batch = {}

    for win_size in valid_win_shifts:

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
            mean = self_encode_mean(mean, win_size)
            std = self_encode_mean(std, win_size)

            spg_list.append(spg)
            chn_list.append(channel)
            mean_list.append(mean)
            std_list.append(std)

        win_batch = torch.stack(spg_list)
        win_channels = torch.stack(chn_list)
        win_means = torch.stack(mean_list)
        win_stds = torch.stack(std_list)

        win_batch.unsqueeze_(1)
        win_channels = win_channels.flatten(1)
        win_means = win_means.transpose(1, 2)
        win_stds = win_stds.transpose(1, 2)

        full_batch[win_size] = {
            "batch": win_batch,
            "channels": win_channels,
            "means": win_means,
            "stds": win_stds,
        }
        # print(f"[collate_fn] win_size={win_size}: {win_batch.shape}")

    # == Finished iterating over all possible window shifts
    print("collate_fn")
    return full_batch, output, dataset


train_loader = DataLoader(
    train_dataset, batch_size=1, collate_fn=sample_collate_fn, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=sample_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=sample_collate_fn)

print(len(train_loader), len(val_loader), len(test_loader))


#########################################################################################################
# Model
# == Metrics ==
def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


class SingleTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(SingleTransformerEncoderLayer, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead)

    def forward(self, src):
        return self.encoder_layer(src)


def mean_aggregation(tokens):
    return torch.mean(torch.stack(tokens), dim=0)


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

        # Finetuning network
        self.finetune_time_transformer = SimpleTransformer(
            embed_size=384, max_len=8_5000
        )

        self.finetune_channel_transformer = SimpleTransformer(
            embed_size=384,
            max_len=200,
        )

        # Modular aggregation method on channel tokens
        self.win_shift_aggregation = mean_aggregation

        if task_type == "Regression":
            self.head = nn.Linear(encoder.encoder_embed_dim, 1)
            self.criterion = nn.MSELoss()
        else:
            self.head = nn.Linear(encoder.encoder_embed_dim, 1)
            self.criterion = nn.BCEWithLogitsLoss()

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, full_x):
        x_embeds = {}
        H_W = {}

        print(f"[FT.forward] win_shifts: {full_x.keys()}")

        for win_size, x_win in full_x.items():
            spgs = x_win["batch"]
            channels = x_win["channels"]
            means = x_win["means"]
            stds = x_win["stds"]
            B, C, H, W = spgs.shape
            x_emb, _, _, nr_meta_patches = self.encoder(
                x=spgs,
                means=means,
                stds=stds,
                channels=channels,
                win_size=win_size,
                mask_ratio=self.mask_ratio,
            )
            x_embeds[win_size] = x_emb
            H_W[win_size] = (H, W)
            print(f"[FT.forward, after self.encoder] x_emb.shape: {x_emb.shape}")

        # Pass through time-transformer
        for win_size, x_emb in x_embeds.items():
            print(
                f"[FT.forward, before self.time_transformer] x_emb.shape: {x_emb.shape}"
            )
            x_emb = self.finetune_time_transformer(x_emb)
            x_emb = x_emb[:, 0]
            print(f"[FT.forward, after time-token] x_emb.shape: {x_emb.shape}")
            x_embeds[win_size] = x_emb

        # Pass through channel-transformer
        tokens = []
        for win_size, x_emb in x_embeds.items():
            x_emb = x_emb.unsqueeze(0)
            print(f"[FT.forward, before channel-token] x_emb.shape: {x_emb.shape}")
            x_emb = self.finetune_channel_transformer(x_emb)
            x_emb = x_emb[0, 0]
            print(f"[FT.forward, after channel-token] x_emb.shape: {x_emb.shape}")
            tokens.append(x_emb)

        # Average over all window shifts
        smart_token = self.win_shift_aggregation(tokens)

        # Pass through head
        y_hat = self.head(smart_token).squeeze()

        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, dataset = batch
        y_hat = self(x)
        loss = self.criterion(input=y_hat, target=y.float())
        self.log("train_loss", loss, prog_bar=True)

        if self.task_type == "Classification":
            y_pred = (torch.sigmoid(y_hat) >= 0.5).float()
            self.train_step_outputs.append((y.cpu(), y_pred.cpu(), dataset))
        elif self.task_type == "Regression":
            self.train_step_outputs.append((y.cpu(), y_hat.cpu(), dataset))

        return loss

    def on_train_epoch_end(self):
        self.compute_metrics(self.train_step_outputs, "train")
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y, dataset = batch
        y_hat = self(x)
        loss = self.criterion(input=y_hat, target=y.float())
        self.log("val_loss", loss, prog_bar=True)

        if self.task_type == "Classification":
            y_pred = (torch.sigmoid(y_hat) >= 0.5).float()
            self.validation_step_outputs.append((y.cpu(), y_pred.cpu(), dataset))
        elif self.task_type == "Regression":
            self.validation_step_outputs.append((y.cpu(), y_hat.cpu(), dataset))

        return loss

    def on_validation_epoch_end(self):
        self.compute_metrics(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y, dataset = batch
        y_hat = self(x)
        loss = self.criterion(input=y_hat, target=y.float())
        self.log("test_loss", loss, prog_bar=True)

        if self.task_type == "Classification":
            y_pred = (torch.sigmoid(y_hat) >= 0.5).float()
            self.test_step_outputs.append((y.cpu(), y_pred.cpu(), dataset))
        elif self.task_type == "Regression":
            self.test_step_outputs.append((y.cpu(), y_hat.cpu(), dataset))

        return loss

    def on_test_epoch_end(self):
        self.compute_metrics(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def compute_metrics(self, outputs, stage):
        y_true_all = defaultdict(list)
        y_pred_all = defaultdict(list)

        for y_true, y_pred, dataset in outputs:
            y_true_all[dataset].append(y_true)
            y_pred_all[dataset].append(y_pred)

        overall_y_true = []
        overall_y_pred = []

        for dataset in y_true_all.keys():
            y_true_cat = torch.stack(y_true_all[dataset])
            y_pred_cat = torch.stack(y_pred_all[dataset])

            overall_y_true.append(y_true_cat)
            overall_y_pred.append(y_pred_cat)

            if self.task_type == "Classification":
                balanced_acc = balanced_accuracy_score(y_true_cat, y_pred_cat)
                self.log(
                    f"{stage}_balanced_accuracy_{dataset}", balanced_acc, prog_bar=True
                )
            elif self.task_type == "Regression":
                rmse_value = rmse(y_true_cat, y_pred_cat)
                self.log(f"{stage}_rmse_{dataset}", rmse_value, prog_bar=True)

        # Compute overall metrics
        overall_y_true = torch.cat(overall_y_true, dim=0)
        overall_y_pred = torch.cat(overall_y_pred, dim=0)

        if self.task_type == "Classification":
            balanced_acc = balanced_accuracy_score(overall_y_true, overall_y_pred)
            self.log(f"{stage}_balanced_accuracy", balanced_acc, prog_bar=True)
        elif self.task_type == "Regression":
            rmse_value = rmse(overall_y_true, overall_y_pred)
            self.log(f"{stage}_rmse", rmse_value, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.head.parameters(), lr=self.learning_rate)

    def on_train_epoch_start(self):
        if self.trainer.current_epoch == 1:
            self.unfreeze_encoder()
            print(f"Unfroze encoder at epoch {self.trainer.current_epoch}")

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


#########################################################################################################
# Load the checkpoint
chkpt_path = ckpt_path
checkpoint = torch.load(chkpt_path, map_location=torch.device("cpu"))
state_dict = checkpoint["state_dict"]
state_dict = {
    k.replace("net.encoder.", ""): v
    for k, v in state_dict.items()
    if "net.encoder." in k
}

# Initialize the encoder and load the state dict
encoder = EncoderViTRoPE(channel_name_map_path)
encoder.load_state_dict(state_dict)

# Instantiate the fine-tuning model
fine_tuning_model = FineTuningModel(
    encoder=encoder,
    frozen_encoder=True,
    out_dim=out_dim,
    task_name=task_name,
    task_type=task_type,
    learning_rate=0.01,
    mask_ratio=0,
)

#########################################################################################################
# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=f"/itet-stor/maxihuber/deepeye_storage/finetune_ckpts/{task_name}",
    filename="{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    monitor="val_loss",
    mode="min",
)

# Train the model
trainer = L.Trainer(
    max_epochs=5,
    callbacks=[checkpoint_callback],
    log_every_n_steps=1,
    num_sanity_val_steps=0,
)

print(f"Class: {class_name}")
print(f"Task: {task_name} ({task_type})")
trainer.fit(
    fine_tuning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
)

trainer.test(model=fine_tuning_model, dataloaders=test_loader)
final_checkpoint_path = (
    f"/itet-stor/maxihuber/net_scratch/finetune_ckpts/{task_name}/final_model.ckpt"
)
trainer.save_checkpoint(final_checkpoint_path)
