import math
import os
import sys
import time
import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm


class PresampledDistributedSampler(DistributedSampler):
    def __init__(
        self,
        mode,
        dataset,
        batch_indices,
        patch_size=16,
        max_nr_patches=8_500,
        win_shifts=[0.25, 0.5, 1, 2, 4, 8],
        win_shift_factor=0.25,
        shuffle=True,
        seed=0,
    ):
        super().__init__(dataset=dataset)

        self.mode = mode
        self.dataset = dataset
        self.batch_indices = batch_indices
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch if self.shuffle else self.seed)
            indices = torch.randperm(len(self.batch_indices), generator=g).tolist()
        else:
            indices = list(range(len(self.batch_indices)))
        self.batch_indices = [self.batch_indices[i] for i in indices]
        return iter(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices)


class ByChannelDistributedSampler(DistributedSampler):
    def __init__(
        self,
        mode,
        dataset,
        keys,
        recompute_freq=1,
        drop_last=False,
        patch_size=16,
        max_nr_patches=8_500,
        win_shifts=[0.25, 0.5, 1, 2, 4, 8],
        win_shift_factor=0.25,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        keep_all=False,
    ):
        super().__init__(dataset=dataset, drop_last=drop_last)

        self.mode = mode

        self.dataset = dataset
        self.keys = keys
        self.recompute_freq = recompute_freq

        self.drop_last = drop_last

        # Group channels in subset_indices by (subject, trial)
        self.id_to_sr_to_trial_to_channels = self._group_channels_by_subject_and_trial()

        print(f"[Sampler] # {mode}-Subjects: {len(self.id_to_sr_to_trial_to_channels)}")

        self.patch_size = patch_size
        self.max_nr_patches = max_nr_patches
        self.win_shifts = win_shifts
        self.win_shift_factor = win_shift_factor

        self.num_replicas = (
            num_replicas if num_replicas is not None else dist.get_world_size()
        )
        self.rank = rank if rank is not None else dist.get_rank()

        self.shuffle = shuffle
        self.seed = seed
        self.keep_all = keep_all

        self.batch_indices = []
        self.epoch = 0
        self.resume_from_epoch = -1
        self.current_position = 0

        print(f"[Sampler] Initialized {mode}-sampler!")

    def _group_channels_by_subject_and_trial(self):
        id_to_sr_to_trial_to_channels = {}
        for idx, channel_idx in enumerate(self.keys):
            channel_info = self.dataset.channel_index[channel_idx]
            subject_id = channel_info["SubjectID"]
            sr = channel_info["sr"]
            trial_idx = channel_info["trial_idx"]
            if subject_id not in id_to_sr_to_trial_to_channels:
                id_to_sr_to_trial_to_channels[subject_id] = {sr: {trial_idx: [idx]}}
            elif sr not in id_to_sr_to_trial_to_channels[subject_id]:
                id_to_sr_to_trial_to_channels[subject_id][sr] = {trial_idx: [idx]}
            elif trial_idx not in id_to_sr_to_trial_to_channels[subject_id][sr]:
                id_to_sr_to_trial_to_channels[subject_id][sr][trial_idx] = [idx]
            else:
                id_to_sr_to_trial_to_channels[subject_id][sr][trial_idx].append(idx)
        return id_to_sr_to_trial_to_channels

    def get_nr_y_patches(self, win_size, sr):
        return int((sr / 2 * win_size + 1) / self.patch_size)

    def get_nr_x_patches(self, win_size, dur):
        win_shift = win_size * self.win_shift_factor
        x_datapoints_per_second = 1 / win_shift
        x_datapoints = dur * x_datapoints_per_second + 1
        return int(x_datapoints / self.patch_size)

    def get_nr_patches(self, win_size, sr, dur):
        return self.get_nr_y_patches(win_size=win_size, sr=sr) * (
            self.get_nr_x_patches(win_size=win_size, dur=dur)
        )

    def get_max_nr_patches(self, sr, dur):
        suitable_win_sizes = self.get_suitable_win_sizes(sr, dur)
        return max(
            [self.get_nr_patches(win_size, sr, dur) for win_size in suitable_win_sizes]
        )

    def get_suitable_win_sizes(self, sr, dur):
        return [
            win_shift
            for win_shift in self.win_shifts
            if self.get_nr_y_patches(win_shift, sr) >= 1
            and self.get_nr_x_patches(win_shift, dur) >= 1
        ]

    def get_suitable_win_size(self, sr, dur):
        win_sizes = self.get_suitable_win_sizes(sr, dur)
        return None if not win_sizes else win_sizes[0]

    def get_max_y_patches(self, sr, dur):
        max_win_shift = max(self.get_suitable_win_sizes(sr, dur))
        return self.get_nr_y_patches(win_size=max_win_shift, sr=sr)

    def generate_batches(self):
        self.batch_indices = []

        for (
            subject_id,
            sr_to_trial_to_channels,
        ) in tqdm(
            self.id_to_sr_to_trial_to_channels.items(),
            desc=f"Generating batches ({self.mode})",
            position=0,
            leave=True,
        ):

            for sr, trial_to_channels in sr_to_trial_to_channels.items():

                for trial_idx, channels in trial_to_channels.items():

                    current_batch = []
                    current_nr_patches = 0

                    # Create a generator for deterministic shuffling
                    g = torch.Generator()
                    g.manual_seed(self.seed + self.epoch if self.shuffle else self.seed)
                    indices = torch.randperm(len(channels), generator=g).tolist()

                    # Shuffle the channels list deterministically
                    channels = [channels[i] for i in indices]

                    # Assert all durations are the same
                    durs = [
                        int(self.dataset.channel_index[self.keys[idx]]["dur"])
                        for idx in channels
                    ]
                    assert all(
                        [dur == durs[0] for dur in durs]
                    ), f"Not all durations are the same: {durs}"

                    # Fixed duration for all channels in this trial
                    # no need to randomize the duration (shorten), there is enough diversity
                    # in the TUEG dataset
                    batch_dur = durs[0]

                    for idx in channels:

                        channel_idx = self.keys[idx]
                        channel_info = self.dataset.channel_index[channel_idx]

                        new_patches = self.get_max_nr_patches(
                            sr=channel_info["sr"],
                            dur=batch_dur,
                        )

                        assert (
                            channel_info["sr"] == sr
                        ), f"channel_info['sr'] ({channel_info['sr']}) != sr ({sr})"
                        assert (
                            new_patches != 0
                        ), f"new_patches == 0, sr={channel_info['sr']}, dur={batch_dur}"

                        # Also add the column of separation patches, be pessimistic
                        #  (we need to be pessimistic because we don't have a signal duration for just one patch,
                        #  so we get a different value for different win_shifts)
                        sep_patches = self.get_max_y_patches(
                            channel_info["sr"], min(batch_dur, batch_dur)
                        )

                        # If adding the new patches would exceed the maximum number of patches,
                        #  finalize this batch and start a new one
                        if current_nr_patches <= self.max_nr_patches and (
                            current_nr_patches + sep_patches + new_patches
                            > self.max_nr_patches
                            or current_nr_patches + sep_patches > self.max_nr_patches
                        ):
                            # Store current batch
                            if current_batch:
                                assert (
                                    current_nr_patches <= self.max_nr_patches
                                ), f"1: current_nr_patches={current_nr_patches} > max_nr_patches={self.max_nr_patches}"
                                self.batch_indices.append(current_batch)
                            # Start a new batch
                            if new_patches > self.max_nr_patches:
                                max_durs = [
                                    int(
                                        (
                                            (self.patch_size**2) * self.max_nr_patches
                                            - sr * win_shift / 2
                                            - 1
                                        )
                                        / (
                                            sr / self.win_shift_factor / 2
                                            + 1 / self.win_shift_factor / win_shift
                                        )
                                    )
                                    for win_shift in (
                                        self.win_shifts
                                        if sr >= 120
                                        else self.win_shifts[1:]
                                    )
                                ]
                                max_dur = min(max_durs)
                                max_dur = int(max_dur)

                                for start in range(0, channel_info["dur"], max_dur):
                                    if channel_info["dur"] - start < max_dur:
                                        break
                                    current_batch = [
                                        (
                                            channel_idx,
                                            start,
                                            max_dur,
                                        )
                                    ]
                                    self.batch_indices.append(current_batch)
                                current_batch = []
                                current_nr_patches = 0
                            else:
                                current_batch = [(channel_idx, 0, batch_dur)]
                                current_nr_patches = new_patches

                        else:
                            current_nr_patches += new_patches
                            if current_batch:
                                current_nr_patches += sep_patches
                            current_batch.append((channel_idx, 0, batch_dur))

                    # == finished all channels for this trial ==

                    if current_batch:
                        assert (
                            current_nr_patches <= self.max_nr_patches
                        ), f"2: current_nr_patches={current_nr_patches} > max_nr_patches={self.max_nr_patches}"
                        self.batch_indices.append(current_batch)

        self.total_size = len(self.batch_indices)
        self.num_samples = (
            self.total_size
            if self.keep_all
            else math.ceil(self.total_size / self.num_replicas)
        )

        print(
            f"[generate_batches ({self.mode})] # Batches = {self.total_size}, # Batches (on rank)= {self.num_samples}",
            file=sys.stderr,
        )

    def __iter__(self):
        if self.shuffle:
            self.generate_batches()
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch if self.shuffle else self.seed)
            indices = torch.randperm(len(self.batch_indices), generator=g).tolist()
        else:
            indices = list(range(len(self.batch_indices)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible by num_replicas
            padding_size = 0
            while (len(indices) + padding_size) % self.num_replicas != 0:
                padding_size += 1
            final_size = len(indices) + padding_size
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible by num_replicas
            final_size = int(self.total_size // self.num_replicas * self.num_replicas)
            indices = indices[:final_size]
        assert (
            len(indices) % self.num_replicas == 0
        ), f"len(indices)={len(indices)}, num_replicas={self.num_replicas}"

        # subsample
        if not self.keep_all:
            indices = indices[self.rank : final_size : self.num_replicas]
        else:
            assert (
                len(indices) == self.num_samples
            ), f"len(indices)={len(indices)}, num_samples={self.num_samples}"

        # This is what changed compared to the DistributedSampler super-class
        batch_indices = [self.batch_indices[i] for i in indices]

        print(
            f"[Sampler] # {int(os.getenv('SLURM_PROCID', '0'))}-Batches ({self.mode}): {len(batch_indices)}",
            file=sys.stderr,
        )

        # return self
        return iter(batch_indices)

    def __len__(self):
        if self.batch_indices == []:
            self.generate_batches()
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    # Need to make the Sampler stateful because one epoch takes longer than the SLURM job limit...
    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict):
        print(f"[Sampler ({self.mode})] Loading from state_dict")
        self.epoch = state_dict["epoch"]
