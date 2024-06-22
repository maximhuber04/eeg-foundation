import random
import torch
from tqdm import tqdm


class BatchGenerator:
    def __init__(
        self, patch_size, win_shifts, win_shift_factor, max_nr_patches, seed, epoch
    ):
        self.patch_size = patch_size
        self.win_shifts = win_shifts
        self.win_shift_factor = win_shift_factor
        self.max_nr_patches = max_nr_patches
        self.seed = seed
        self.epoch = epoch

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

    def generate_batches(self, id_to_sr_to_trial_to_channels, full_channel_index):
        batch_indices = []

        for (
            subject_id,
            sr_to_trial_to_channels,
        ) in tqdm(
            id_to_sr_to_trial_to_channels.items(),
            desc="Generating batches",
            position=0,
            leave=True,
        ):

            for sr, trial_to_channels in sr_to_trial_to_channels.items():

                for trial_idx, channels in trial_to_channels.items():

                    current_batch = []
                    current_nr_patches = 0

                    # Create a generator for deterministic shuffling
                    g = torch.Generator()
                    g.manual_seed(self.seed + self.epoch)
                    indices = torch.randperm(len(channels), generator=g).tolist()

                    # Shuffle the channels list deterministically
                    channels = [channels[i] for i in indices]

                    # Assert all durations are the same
                    durs = [
                        int(full_channel_index[channel]["dur"]) for channel in channels
                    ]
                    assert all(
                        [dur == durs[0] for dur in durs]
                    ), f"Not all durations are the same: {durs}"

                    # Fixed duration for all channels in this trial
                    # no need to randomize the duration (shorten), there is enough diversity
                    # in the TUEG dataset
                    batch_dur = durs[0]

                    for channel_idx in channels:

                        channel_info = full_channel_index[channel_idx]

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
                                batch_indices.append(current_batch)
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
                                    batch_indices.append(current_batch)
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
                        batch_indices.append(current_batch)

        self.epoch += 1
        return batch_indices
