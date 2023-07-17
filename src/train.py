import argparse
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from dvc.repo import Repo
from dvclive.fastai import DVCLiveCallback
from fastai.data.all import Normalize, get_files
from fastai.metrics import DiceMulti
from fastai.vision.all import (
    Resize,
    SegmentationDataLoaders,
    imagenet_stats,
    models,
    unet_learner,
)
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def get_mask_path(x, train_data_dir):
    return Path(train_data_dir) / f"{Path(x).stem}.png"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--valid_pct", type=float)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--fine_tune_args.epochs", type=int, dest="epochs")
    parser.add_argument("--fine_tune_args.base_lr", type=float, dest="base_lr")
    
    # input data and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ.get("SM_MODEL_DIR", "models"))
    parser.add_argument('--train', type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data/train_data"))
    
    args, _ = parser.parse_known_args()
    return args

    
def train(params):

    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    train_data_dir = Path(params.train)

    data_loader = SegmentationDataLoaders.from_label_func(
        path=train_data_dir,
        fnames=get_files(train_data_dir, extensions=".jpg"),
        label_func=partial(get_mask_path, train_data_dir=train_data_dir),
        codes=["not-pool", "pool"],
        bs=params.batch_size,
        valid_pct=params.valid_pct,
        item_tfms=Resize(params.img_size),
        batch_tfms=[
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    model_names = [
        name
        for name in dir(models)
        if not name.startswith("_")
        and name.islower()
        and name not in ("all", "tvm", "unet", "xresnet")
    ]
    if params.arch not in model_names:
        raise ValueError(f"Unsupported model, must be one of:\n{model_names}")

    learn = unet_learner(
        data_loader, arch=getattr(models, params.arch), metrics=DiceMulti
    )

    learn.fine_tune(
        params.epochs, params.base_lr,
        cbs=[DVCLiveCallback(dir="results/train", report="md")],
    )
    models_dir = Path(params.model_dir)
    models_dir.mkdir(exist_ok=True)
    # save to fast ai format for evaluation.
    learn.export(fname=(models_dir / "model.pkl").absolute())
    # save to pytorch for deployment.
    learn.save(models_dir.absolute() / "model")

if __name__ == "__main__":
    params = parse_args()
    train(params)
