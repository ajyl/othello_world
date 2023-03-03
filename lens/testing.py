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
from src.othello import get as get_othello, permit, start_hands, OthelloBoardState

# from train_probe_othello import ProbingDataset


OTHELLO_HOME = "/home/ajyl/othello_world"


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


def load_lens_model():
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



#def print_board(moves) -> None:
#    """
#    Print board based on tensor.
#
#    Input: Tensor of shape (1, 59) for now.
#    """
#    assert moves.shape == (1, 59)
#    moves_list = moves.squeeze().tolist()
#
#    board = OthelloBoardState()
#    board.update(moves_list)
#    board.__print__()


def check_correct(moves, pred, i_to_s):
    """
    Check if prediction was valid.
    """
    board = OthelloBoardState()
    board.update(moves)

    x = board.get_valid_moves()
    y = pred.item()
    breakpoint()
    return y in x


def get_othello_data():
    """
    Load Othello data
    """
    othello = get_othello(
        data_root=os.path.join(OTHELLO_HOME, "data/othello_small")
    )

    train_dataset = CharDataset(othello)
    loader = DataLoader(
        train_dataset, shuffle=False, pin_memory=True, batch_size=1, num_workers=1
    )
    return loader, train_dataset.stoi, train_dataset.itos


def main():
    """ Driver """
    model = load_lens_model()
    data_loader, s_to_i, i_to_s = get_othello_data()
    correct = 0
    for x, y, z in tqdm(data_loader, total=len(data_loader)):

        logits = model(x)

        next_token_logits = logits[0, -1]
        next_token_prediction = next_token_logits.argmax()

        if next_token_prediction == y.squeeze()[-1].item():
            correct += 1

        #valid_until = z.index(-100) if -100 in z else 999
        #moves = z[:valid_until]

        #if check_correct(moves, next_token_prediction, i_to_s):

        #    correct += 1

        logits2, cache = model.run_with_cache(x)
        breakpoint()


    breakpoint()


if __name__ == "__main__":
    main()
