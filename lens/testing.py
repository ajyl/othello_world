"""
Module Doc String
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from fancy_einsum import einsum
#import tqdm.auto as tqdm
from tqdm import tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from mingpt.dataset import CharDataset
from torchtyping import TensorType as TT
from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
from lens.setup_utils import _convert_to_transformer_lens_format
from data import get_othello
from data.othello import permit, start_hands, OthelloBoardState

# from train_probe_othello import ProbingDataset


OTHELLO_HOME = "/home/repos/othello_world"


LOAD_AND_CONVERT_CHECKPOINT = False
if LOAD_AND_CONVERT_CHECKPOINT:
    synthetic_checkpoint = torch.load(
        "/workspace/othello_world/gpt_synthetic.ckpt"
    )
    for name, param in synthetic_checkpoint.items():
        if name.startswith("blocks.0") or not name.startswith("blocks"):
            print(name, param.shape)

    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)
    model.load_and_process_state_dict(
        _convert_to_transformer_lens_format(synthetic_checkpoint)
    )

torch.set_grad_enabled(False)


def _load_model():
    """
    Load Transformer-Lens version of Othello-GPT
    """
    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)

    sd = utils.download_file_from_hf(
        "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
    )
    model.load_state_dict(sd)
    return model


def get_othello_data():
    """
    Load Othello data
    """
    othello = get_othello(
        data_root=os.path.join(OTHELLO_HOME, "data/othello_championship")
    )

    train_dataset = CharDataset(othello)
    loader = DataLoader(
        train_dataset, shuffle=False, pin_memory=True, batch_size=1, num_workers=1
    )
    return loader


def main():
    """ Driver """
    model = _load_model()
    data_loader = get_othello_data()
    for x, y in tqdm(data_loader, total=len(data_loader)):

        logits = model(x)

        next_token_logits = logits[0, -1]
        next_token_prediction = next_token_logits.argmax()

        breakpoint()


if __name__ == "__main__":
    main()
