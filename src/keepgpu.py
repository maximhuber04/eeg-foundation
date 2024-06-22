import time

while True:
    print("Keep GPU running...")
    time.sleep(60)

# import os
# import sys
# import json
# import logging
# from math import ceil
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# import re
# import mne

# mne.set_log_level("WARNING")


# def create_raw(
#     data,
#     ch_names1,
#     sr,
#     ch_names2=None,
# ):
#     if ch_names2 == None:
#         ch_names2 = ch_names1
#     ch_types = ["eeg" for _ in range(len(ch_names1))]
#     info = mne.create_info(ch_names2, ch_types=ch_types, sfreq=sr)
#     eeg_data = np.array(data[ch_names1].T, dtype="float") / 1_000_000
#     raw = mne.io.RawArray(eeg_data, info)
#     return raw


# def avg_channel(raw):
#     avg = raw.copy().add_reference_channels(ref_channels="AVG_REF")
#     avg = avg.set_eeg_reference(ref_channels="average")
#     return avg


# def load_from_path(path, channels, sr):
#     if path.endswith("edf"):
#         eeg_data = mne.io.read_raw_edf(
#             path,
#             include=channels,
#             preload=True,
#         )
#     elif path.endswith("pkl"):
#         # Load DataFrame from pickle
#         with open(path, "rb") as file:
#             df = pd.read_pickle(file)
#             eeg_data = create_raw(
#                 data=df,
#                 ch_names1=channels,
#                 sr=sr,
#             )
#     else:
#         assert False, "Invalid path"

#     # Add average reference
#     eeg_data = avg_channel(eeg_data)

#     # Datastructure to access data for each channel
#     channel_data_dict = {}

#     # Note: channel_data_dict also includes the AVG_REF channel
#     for channel in eeg_data.ch_names:
#         idx = eeg_data.ch_names.index(channel)
#         data, times = eeg_data[idx, :]
#         # Flatten the data to 1D if required
#         channel_data_dict[channel] = data.flatten()

#     df = pd.DataFrame(channel_data_dict)
#     df["Time in Seconds"] = times.flatten()

#     return df


# import json
# import matplotlib.pyplot as plt

# tueg_path_new = "/itet-stor/maxihuber/deepeye_storage/index_files/new_tueg_index.json"

# with open(
#     "/itet-stor/maxihuber/deepeye_storage/index_files/full_tueg_index.json", "r"
# ) as f:
#     index = json.load(f)

# prefix_path = "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf"

# for i, ie in tqdm(enumerate(index), "Recomputing durations", position=0, leave=True):
#     file = prefix_path + ie["path"]
#     # load
#     df = load_from_path(file, ie["channels"], ie["sr"])
#     ie["duration"] = len(df) / ie["sr"]
#     del df

#     if i % 1_000 == 0:
#         with open(tueg_path_new, "w") as f:
#             json.dump(index, f, indent=4)

# with open(tueg_path_new, "w") as f:
#     json.dump(index, f, indent=4)
