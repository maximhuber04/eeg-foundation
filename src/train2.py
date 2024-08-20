import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import psutil
from lightning import Callback, LightningDataModule, LightningModule
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.accelerators import find_usable_cuda_devices
from omegaconf import DictConfig

import lightning as L
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    setup_wandb,
)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    L.seed_everything(42, workers=True)

    print("At train.py entry", file=sys.stderr)
    print("RAM memory % used:", psutil.virtual_memory()[2], file=sys.stderr)
    print("RAM Used (GB):", psutil.virtual_memory()[3] / 1_000_000_000, file=sys.stderr)
    print("Usable cuda devices: ", find_usable_cuda_devices(), file=sys.stderr)

    # == Instantiate Loggers ==
    log.info("Instantiating loggers...")
    setup_wandb(cfg)

    # == Instantiate Callbacks ==
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.callbacks)

    # == Instantiate DataModule ==
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # == Instantiate LightningModule ==
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # == Instantiate Trainer ==
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        num_nodes=int(os.getenv("SLURM_JOB_NUM_NODES")),
        devices=(len(os.getenv("CUDA_VISIBLE_DEVICES").split(","))),
        callbacks=callbacks,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.paths.ckpt_path,
    )


if __name__ == "__main__":
    main()
