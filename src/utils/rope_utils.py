from itertools import repeat
import math

import torch
import torch.nn as nn

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


# ====Patch Embedding==============================================================================================================


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=16, in_chans=1, embed_dim=384):
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ====Random Masking==============================================================================================================


def random_masking_smart(x, mask_ratio, nr_meta_tokens):
    B, N, D = x.shape

    assert N - nr_meta_tokens > 0, "N - nr_meta_tokens = {N - nr_meta_tokens}"
    num_tokens_to_keep = int(math.ceil((N - nr_meta_tokens) * (1 - mask_ratio)))

    # indices we keep
    rand_indices, _ = (
        torch.rand(B, N - nr_meta_tokens, device=x.device)
        .argsort(dim=1)[:, :num_tokens_to_keep]
        .sort(dim=1)
    )
    rand_indices += nr_meta_tokens

    # Add True values at positions we keep
    mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
    mask.scatter_(1, rand_indices, True)

    # Fill mask[:, :nr_meta_tokens] with True (always keep metadata tokens)
    mask[:, :nr_meta_tokens] = True

    # Indices to restore in decoder
    # TODO: possible to make faster / more memory efficient?
    kept_indices = torch.nonzero(mask, as_tuple=True)[1].reshape(B, -1)
    masked_indices = torch.nonzero(~mask, as_tuple=True)[1].reshape(B, -1)
    ids_restore = torch.cat([kept_indices, masked_indices], dim=1)

    return mask, ids_restore
