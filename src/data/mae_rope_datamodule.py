import glob
import os
import random
import sys
import time
import shutil
import gc
import json

from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning import LightningDataModule
import torchaudio
from tqdm import tqdm
import wandb

from src.data.mae_rope_dataset import LocalStorageDataset, RawDataset
from src.data.mae_rope_distributedsampler import (
    ByChannelDistributedSampler,
    PresampledDistributedSampler,
)
from src.data.transforms import (
    create_raw,
    crop_spg,
    custom_fft,
    normalize_spg,
)

from src.utils.preloading.utils import (
    filter_index_simple,
    info_from_index,
    load_edf_to_dataframe,
    load_trials,
    prepare_info_to_load,
)

from src.data.generate_batches import BatchGenerator


class TrainDataModule(LightningDataModule):
    def __init__(
        self,
        # Preloading
        source_indices,
        path_prefix="",
        min_duration=0,
        max_duration=86_400,
        split_duration=3_600,
        discard_sr=[],
        discard_datasets=[],
        load_data=True,
        index_patterns=["/dev/shm/mae/index_*.json", "/scratch/mae/index_*.json"],
        # Network
        channel_name_map_path="/home/maxihuber/eeg-foundation/src/data/components/channels_to_id3.json",
        recompute_freq=1,
        patch_size=16,
        max_nr_patches=8_500,
        win_shifts=[0.25, 0.5, 1, 2, 4, 8],
        win_shift_factor=0.25,
        none_channel_probability=0.2,
        # Dataset
        train_val_split=[0.9, 0.1],
        # Dataloader
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
    ):
        super().__init__()

        self.source_indices = source_indices
        self.path_prefix = path_prefix
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.split_duration = split_duration
        self.discard_sr = discard_sr
        self.discard_datasets = discard_datasets
        self.load_data = load_data
        self.index_patterns = index_patterns

        with open(channel_name_map_path, "r") as file:
            self.channel_name_map = json.load(file)

        self.train_val_split = train_val_split

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        self.recompute_freq = recompute_freq
        self.patch_size = patch_size
        self.max_nr_patches = max_nr_patches
        self.win_shifts = win_shifts
        self.max_win_shift = win_shifts[-1]
        self.max_nr_y_patches = int(500 * self.max_win_shift // self.patch_size)
        self.max_y_datapoints = self.max_nr_y_patches * self.patch_size

        self.win_shift_factor = win_shift_factor
        self.none_channel_probability = none_channel_probability

        self.save_hyperparameters(logger=False)

    # == Setup ==========================================================================================================================

    def setup(self, stage=None):

        if self.load_data:
            index = filter_index_simple(
                index_paths=self.source_indices,
                path_prefix=self.path_prefix,
                min_duration=self.min_duration,
                max_duration=self.max_duration,
                discard_sr=self.discard_sr,
                discard_datasets=self.discard_datasets,
            )
            trial_info_index = info_from_index(
                index_chunk=index,
                min_duration=self.min_duration,
                max_duration=self.max_duration,
                split_duration=self.split_duration,
            )
        else:
            index_paths = []
            for pattern in self.index_patterns:  # regex the index_patterns
                index_paths.extend(glob.glob(pattern))
            num_trials = 0
            trial_info_index = {}
            for index_path in index_paths:
                with open(index_path, "r") as f:
                    new_trial_info_index = json.load(f)
                    for trial_info in new_trial_info_index.values():
                        trial_info_index[num_trials] = trial_info
                        num_trials += 1
            print(f"[setup] # Trials = {num_trials}", file=sys.stderr)

        full_channel_index = {}
        num_signals = 0
        data_seconds = 0
        nr_trials_excluded = 0

        for trial_idx, trial_info in trial_info_index.items():
            for chn in trial_info["channels"]:
                full_channel_index[num_signals] = {
                    "channel": chn,
                    "channels": trial_info["channels"],
                    "sr": trial_info["sr"],
                    "dur": trial_info["dur"],
                    "path": (
                        trial_info["new_path"]
                        if "new_path" in trial_info
                        else trial_info["origin_path"]
                    ),
                    "trial_idx": trial_idx,
                    "Dataset": trial_info["Dataset"],
                    "SubjectID": trial_info["SubjectID"],
                }
                num_signals += 1
                data_seconds += trial_info["dur"]

        print(
            f"[setup] We have data from {len(trial_info_index)} trials.",
            file=sys.stderr,
        )
        print(
            f"[setup] This is {int(data_seconds)} seconds (single-channel).",
            file=sys.stderr,
        )
        print(f"[setup] We excluded {nr_trials_excluded} test trials.", file=sys.stderr)

        # TODO: quick fix, only cause it's accessed in the collate_fn
        # this definition actually belongs within the next "if self.load_data" branch
        self.batch_generator = BatchGenerator(
            patch_size=self.patch_size,
            win_shifts=self.win_shifts,
            win_shift_factor=self.win_shift_factor,
            max_nr_patches=self.max_nr_patches - 500,
            seed=0,
            epoch=0,
        )

        if self.load_data:

            # Group channels in subset_indices by (subject, trial)
            id_to_sr_to_trial_to_channels = {}
            for channel_idx, channel_info in full_channel_index.items():
                subject_id = channel_info["SubjectID"]
                sr = channel_info["sr"]
                trial_idx = channel_info["trial_idx"]
                if subject_id not in id_to_sr_to_trial_to_channels:
                    id_to_sr_to_trial_to_channels[subject_id] = {
                        sr: {trial_idx: [channel_idx]}
                    }
                elif sr not in id_to_sr_to_trial_to_channels[subject_id]:
                    id_to_sr_to_trial_to_channels[subject_id][sr] = {
                        trial_idx: [channel_idx]
                    }
                elif trial_idx not in id_to_sr_to_trial_to_channels[subject_id][sr]:
                    id_to_sr_to_trial_to_channels[subject_id][sr][trial_idx] = [
                        channel_idx
                    ]
                else:
                    id_to_sr_to_trial_to_channels[subject_id][sr][trial_idx].append(
                        channel_idx
                    )

            print(
                f"[setup] # Full-Subjects: {len(id_to_sr_to_trial_to_channels)}",
                file=sys.stderr,
            )

            start_time = time.time()
            batch_indices = self.batch_generator.generate_batches(
                id_to_sr_to_trial_to_channels, full_channel_index
            )
            print(
                f"[generate_batches] # GenerateBatches took: {round(time.time() - start_time,2)}s",
                file=sys.stderr,
            )
            print(f"[setup] # Batches (total): {len(batch_indices)}", file=sys.stderr)

            slurm_rank = int(os.getenv("SLURM_PROCID", "0"))
            slurm_size = int(os.getenv("SLURM_NPROCS", "1"))

            batches_per_process = len(batch_indices) // slurm_size
            new_total_batches = batches_per_process * slurm_size
            batch_indices = batch_indices[:new_total_batches]

            start_index = slurm_rank * batches_per_process
            end_index = start_index + batches_per_process

            batch_indices = batch_indices[start_index:end_index]

            # Now, load into memory all data that this process will access
            trial_idxs = sorted(
                set(
                    [
                        full_channel_index[channel_idx]["trial_idx"]
                        for batch in batch_indices
                        for (channel_idx, _, _) in batch
                    ]
                )
            )
            trunc_trial_info_index = {
                trial_idx: trial_info_index[trial_idx] for trial_idx in trial_idxs
            }
            trial_index = load_trials(trunc_trial_info_index)

            # Train/Test split of the batches
            keys = list(range(len(batch_indices)))
            train_keys, val_keys = train_test_split(
                keys,
                train_size=self.train_val_split[0],
                test_size=self.train_val_split[1],
                random_state=42,
            )
            train_batches = [batch_indices[key] for key in train_keys]
            val_batches = [batch_indices[key] for key in val_keys]
            print(
                f"[setup] # Train/Test Batches [rank={slurm_rank}]: {len(train_batches)}, {len(val_batches)}",
                file=sys.stderr,
            )

            self.train_dataset = RawDataset(
                channel_index=full_channel_index, trial_index=trial_index
            )
            self.val_dataset = RawDataset(
                channel_index=full_channel_index, trial_index=trial_index
            )

            self.train_sampler = PresampledDistributedSampler(
                mode="train",
                dataset=self.train_dataset,
                batch_indices=train_batches,
                shuffle=True,
                seed=0,
            )

            self.val_sampler = PresampledDistributedSampler(
                mode="val",
                dataset=self.val_dataset,
                batch_indices=val_batches,
                shuffle=False,
                seed=0,
            )

        else:
            keys = list(full_channel_index.keys())
            train_keys, val_keys = train_test_split(
                keys,
                train_size=self.train_val_split[0],
                test_size=self.train_val_split[1],
                random_state=42,
            )

            train_channel_index = {key: full_channel_index[key] for key in train_keys}
            val_channel_index = {key: full_channel_index[key] for key in val_keys}

            self.train_dataset = LocalStorageDataset(train_channel_index)
            self.val_dataset = LocalStorageDataset(val_channel_index)

            self.train_sampler = ByChannelDistributedSampler(
                mode="train",
                dataset=self.train_dataset,
                keys=train_keys,
                recompute_freq=self.recompute_freq,
                patch_size=self.patch_size,
                max_nr_patches=self.max_nr_patches - 500,
                win_shifts=self.win_shifts,
                win_shift_factor=self.win_shift_factor,
                shuffle=True,
                seed=0,
                keep_all=False,
            )

            self.val_sampler = ByChannelDistributedSampler(
                mode="val",
                dataset=self.val_dataset,
                keys=val_keys,
                recompute_freq=self.recompute_freq,
                patch_size=self.patch_size,
                max_nr_patches=self.max_nr_patches - 500,
                win_shifts=self.win_shifts,
                win_shift_factor=self.win_shift_factor,
                shuffle=False,
                seed=0,
                keep_all=False,
            )

    # == Collate Functions ===============================================================================================================

    def snake_collate_fn(self, batch):

        batch_len = len(batch)

        srs = [sample["sr"] for sample in batch]
        assert all(
            sr == srs[0] for sr in srs
        ), f"[snake_collate_fn] Differing sampling rates within batch, srs={srs}"

        sr = srs[0]
        durs = [sample["dur"] for sample in batch]
        min_dur = min(durs)

        # Randomly sample a win_size for the STFT
        valid_win_shifts = [
            win_shift
            for win_shift in self.win_shifts
            if self.batch_generator.get_nr_y_patches(win_shift, sr) >= 1
            and self.batch_generator.get_nr_x_patches(win_shift, min_dur) >= 1
        ]
        assert (
            len(valid_win_shifts) > 0
        ), f"[snake_collate_fn] No valid valid_win_shifts found, {[sample['sr'] for sample in batch]}, {[sample['dur'] for sample in batch]}"

        # print("[snake_collate_fn] Sampling set:", valid_win_shifts, file=sys.stderr)
        win_size = random.choice(valid_win_shifts)

        fft = torchaudio.transforms.Spectrogram(
            n_fft=int(sr * win_size),
            win_length=int(sr * win_size),
            hop_length=int(sr * win_size * self.win_shift_factor),
            normalized=True,
        )

        spgs = []  # Spectrograms of this batch
        chns = []  # List to store channel tensors for each patch
        spgs_w = set()  # Widths of the spectrograms

        H, W = 0, 0

        for i, sample in enumerate(batch):

            signal = sample["signal"]
            # print(len(signal) / sr, sample["dur"], file=sys.stderr)

            channel_name = self.get_generic_channel_name(sample["channel"])
            if channel_name in self.channel_name_map:
                p = random.random()  # Sample p from a uniform distribution over [0, 1)
                channel = (
                    self.channel_name_map[channel_name]
                    if p > self.none_channel_probability
                    else self.channel_name_map["None"]
                )
            else:
                channel = self.channel_name_map["None"]

            # STFT
            spg = fft(signal)
            spg = spg**2
            spg = crop_spg(spg, self.patch_size)

            H_new, W_new = spg.shape[0], spg.shape[1]
            h_new, w_new = H_new // self.patch_size, W_new // self.patch_size

            # Create a tensor filled with the current channel value
            channel_tensor = torch.full((h_new, w_new), channel, dtype=torch.float32)

            spgs.append(spg)
            chns.append(channel_tensor)
            spgs_w.add(w_new)

            H = H_new
            W += W_new

        total_patches = H * W // (self.patch_size**2)
        h, w = H // self.patch_size, W // self.patch_size

        # Now, given the H, W information, we can create a tensor of desired shape (B, H_new, W_new)

        # Now that all signals in the batch had the same length (same trial),
        #  we can just use the batch_len as the batch size
        batch_size = batch_len

        spgs_rows = []
        chns_rows = []
        means_rows = []
        stds_rows = []

        row_size = W // batch_size

        cur_spgs = []
        cur_chns = []
        cur_means = []
        cur_stds = []
        cur_W = 0

        for spg, chn in zip(spgs, chns):
            if cur_W + spg.shape[1] <= row_size:
                # Can use full spg
                spg, mean, std = normalize_spg(spg)
                mean = self.encode_mean(mean, win_size)
                std = self.encode_mean(std, win_size)
                cur_spgs.append(spg)
                cur_chns.append(chn)
                cur_means.append(mean)
                cur_stds.append(std)
                cur_W += spg.shape[1]
                if cur_W == row_size:
                    cur_W = 0
                    spgs_rows.append(cur_spgs)
                    chns_rows.append(cur_chns)
                    means_rows.append(cur_means)
                    stds_rows.append(cur_stds)
                    cur_spgs = []
                    cur_chns = []
                    cur_means = []
                    cur_stds = []
            else:
                # Need to split this spg to multiple rows
                #  in a while loop
                spg_W = spg.shape[1]
                spg_W_taken = 0
                while spg_W > 0:
                    take_W = min(spg_W, row_size - cur_W)
                    spg_chunk = spg[:, spg_W_taken : spg_W_taken + take_W]
                    chn_chunk = chn[
                        :,
                        (spg_W_taken // self.patch_size) : (
                            (spg_W_taken + take_W) // self.patch_size
                        ),
                    ]
                    spg_chunk, mean_chunk, std_chunk = normalize_spg(spg_chunk)
                    mean_chunk = self.encode_mean(mean_chunk, win_size)
                    std_chunk = self.encode_mean(std_chunk, win_size)
                    cur_spgs.append(spg_chunk)
                    cur_chns.append(chn_chunk)
                    cur_means.append(mean_chunk)
                    cur_stds.append(std_chunk)
                    spg_W -= take_W
                    cur_W += take_W
                    spg_W_taken += take_W
                    cur_W %= row_size
                    if cur_W == 0:
                        spgs_rows.append(cur_spgs)
                        chns_rows.append(cur_chns)
                        means_rows.append(cur_means)
                        stds_rows.append(cur_stds)
                        cur_spgs = []
                        cur_chns = []
                        cur_means = []
                        cur_stds = []

        # Concatenate rows
        final_batch = torch.stack([torch.cat(row, dim=-1) for row in spgs_rows])
        channels = torch.stack([torch.cat(row, dim=-1) for row in chns_rows])

        max_nr_mean_patches = max([len(means) for means in means_rows])
        means_rows = [torch.cat(means, dim=-1) for means in means_rows]
        means_rows = [
            F.pad(
                means,
                (0, max_nr_mean_patches - means.shape[1]),
                mode="constant",
                value=0,
            )
            for means in means_rows
        ]
        means = torch.stack(means_rows)

        stds_rows = [torch.cat(std, dim=-1) for std in stds_rows]
        stds_rows = [
            F.pad(
                stds,
                (0, max_nr_mean_patches - stds.shape[1]),
                mode="constant",
                value=0,
            )
            for stds in stds_rows
        ]
        stds = torch.stack(stds_rows)

        assert (
            final_batch.shape[1] == channels.shape[1] * self.patch_size
        ), f"Batch shape: {final_batch.shape[1]}, Channels shape: {channels.shape[1]*self.patch_size}"
        assert (
            final_batch.shape[2] == channels.shape[2] * self.patch_size
        ), f"Batch shape: {final_batch.shape[2]}, Channels shape: {channels.shape[2]*self.patch_size}"

        final_batch.unsqueeze_(1)

        # Flatten the channels tensor, the batch will be automatically by the network
        channels = channels.flatten(1)
        means = means.transpose(1, 2)
        stds = stds.transpose(1, 2)

        B, C, H, W = final_batch.shape

        # Send the constructed batch to the network
        # print(f"[return] final_batch.shape: {final_batch.shape}", file=sys.stderr)
        # print(f"[return] channels.shape: {channels.shape}", file=sys.stderr)
        # print(f"[return] means.shape: {means.shape}", file=sys.stderr)
        # print(f"[return] win_size: {win_size}", file=sys.stderr)

        return {
            "batch": final_batch,
            "channels": channels,
            "means": means,
            "stds": stds,
            "win_size": win_size,
        }

    def multi_trial_batch_collafe_fn(self, batch):

        print("[custom_collate_fn] Batch size:", len(batch), file=sys.stderr)

        # Batch: list of (signals, channels, win_size, trial_info["sr"], trial_info["duration"]) tuples
        win_size = batch[0][2]
        sr = batch[0][3]
        print("[custom_collate_fn] Window size:", win_size, file=sys.stderr)
        print("[custom_collate_fn] Sampling rate:", sr, file=sys.stderr)

        spgs = {}
        total_dur = 0

        # Transform signals to spectrograms
        for signals, chn_list, _, _, dur in batch:
            # Fourier transform => spectrograms
            fft = custom_fft(
                window_seconds=win_size,
                window_shift=win_size / 4,
                sr=sr,
                cuda=False,
            )

            for chn, signal in zip(chn_list, signals):
                chn = chn.lower()
                spg = fft(signal)
                spg = crop_spg(spg)
                spg = normalize_spg(spg)
                if chn not in spgs:
                    spgs[chn] = [spg]
                else:
                    spgs[chn].append(spg)

            total_dur += dur

        channel_to_len = {chn: sum([spg.shape[1] for spg in spgs[chn]]) for chn in spgs}
        max_len = max(max(channel_to_len.values()), self.patch_size)

        # concatenate along time axis, padding up to max_len
        spgs_cat_pad_by_channel = []
        channels = []
        for chn, spectros in spgs.items():

            # Concatenate signals from the same channel along the time dimension
            concat_spectros = torch.cat(spectros, dim=1)

            # Determine the amount of padding needed
            padding_length = max_len - concat_spectros.shape[1]
            if padding_length > 0:
                # Apply zero-padding
                padding = torch.zeros((concat_spectros.shape[0], padding_length))
                padded_spectros = torch.cat((concat_spectros, padding), dim=1)
            else:
                # No padding needed if concat_spectros is already of length max_len or more
                padded_spectros = concat_spectros

            spgs_cat_pad_by_channel.append(padded_spectros)
            channels.append(chn)

        batch = torch.stack(spgs_cat_pad_by_channel)
        batch.unsqueeze_(1)

        print(
            "[custom_collate_fn] Batch shape:",
            batch.shape,
            "(B, C, H, W)",
            file=sys.stderr,
        )

        return {
            "batch": batch,
            "chn_list": channels,
            "win_size": win_size,
        }

    def single_trial_batch_collate_fn(self, batch):

        print("[custom_collate_fn] Batch size:", len(batch), file=sys.stderr)

        # Batch: list of (signals, channels, win_size, trial_info["sr"], trial_info["duration"]) tuples
        win_size = batch[0][2]
        sr = batch[0][3]
        print("[custom_collate_fn] Window size:", win_size, file=sys.stderr)
        print("[custom_collate_fn] Sampling rate:", sr, file=sys.stderr)

        spgs = {}
        total_dur = 0

        # Transform signals to spectrograms
        for signals, chn_list, _, _, dur in batch:
            # Fourier transform => spectrograms
            fft = custom_fft(
                window_seconds=win_size,
                window_shift=win_size / 4,
                sr=sr,
                cuda=False,
            )

            for chn, signal in zip(chn_list, signals):
                chn = chn.lower()
                spg = fft(signal)
                spg = crop_spg(spg)
                spg = normalize_spg(spg)
                if chn not in spgs:
                    spgs[chn] = [spg]
                else:
                    spgs[chn].append(spg)

            total_dur += dur

        print({chn: len(spgs[chn]) for chn in spgs}, file=sys.stderr)

        # Concatenate signals of the same channel
        spgs_cat_by_channel = []
        channels = []
        for chn, spectros in spgs.items():
            # spectros is a list of torch tensors
            # need to concatenate these along the time axis
            # TODO: add empty patches in between or something
            # spgs_cat_by_channel.append(torch.cat(spectros, dim=1))
            spgs_cat_by_channel.append(spectros[0])
            channels.append(chn)

        batch = torch.stack(spgs_cat_by_channel)
        batch.unsqueeze_(1)
        print(
            "[custom_collate_fn] Batch shape:",
            batch.shape,
            "(B, C, H, W)",
            file=sys.stderr,
        )

        return {
            "batch": batch,
            "chn_list": channels,
            "win_size": win_size,
            "sr": sr,
            "dur": total_dur,
        }

    def probe_max_patches(self, batch):
        return {
            "batch": torch.randn(1, 1, 4000, 496),
            "channels": torch.ones(1, 7750),
            "means": torch.rand(1, 1, 4_000),
            "stds": torch.rand(1, 1, 4_000),
            "win_size": 8,
        }

    # == Data Loaders ===================================================================================================================

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self.snake_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_sampler=self.val_sampler,
            collate_fn=self.snake_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    # == Helpers ========================================================================================================================

    def get_generic_channel_name(self, channel_name):
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

    def encode_mean(self, mean, win_size):
        y_datapoints = mean.shape[0]
        encoded_mean = torch.zeros(self.max_y_datapoints)
        step_size = int(self.max_win_shift // win_size)
        end_idx = step_size * y_datapoints
        indices = torch.arange(0, end_idx, step_size)
        encoded_mean[indices] = mean.squeeze_().float()
        encoded_mean.unsqueeze_(1)
        return encoded_mean
