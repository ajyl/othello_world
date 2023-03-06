"""
Script for training probe.
"""

import numpy as np
import torch
#from torch import einsum
#import wandb
from fancy_einsum import einsum
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib.pyplot import imshow

from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import transformer_lens.utils as utils
from data.othello import OthelloBoardState
from mech_exp.tl_othello_utils import load_hooked_model, to_string, to_int
from mech_exp.tl_plot_utils import (
    plot_single_board,
    plot_board_log_probs,
    plot_board,
)


model = load_hooked_model("synthetic")

#plot_single_board(["D2", "C4"])
#plot_board_log_probs(
#    to_string(["D2", "C4"]), model(torch.tensor(to_int(["D2", "C4"])))
#)
#plot_board(["D2", "C4"])

board_seqs_int = torch.load("board_seqs_int.pth")
board_seqs_string = torch.load("board_seqs_string.pth")


def seq_to_state_stack(str_moves):
    if isinstance(str_moves, torch.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move in str_moves:
        board.umpire(move)
        states.append(np.copy(board.state))
    states = np.stack(states, axis=0)
    return states


state_stack = torch.tensor(
    np.stack([seq_to_state_stack(seq) for seq in board_seqs_string[:50, :-1]])
)
print(state_stack.shape)
# %%


def state_stack_to_one_hot(state_stack):
    one_hot = torch.zeros(
        2,  # blank vs color (mode)
        state_stack.shape[0],
        state_stack.shape[1],
        8,  # rows
        8,  # cols
        2,  # the two options
        device=state_stack.device,
        dtype=torch.int,
    )
    one_hot[0, ..., 0] = state_stack == 0
    one_hot[0, ..., 1] = 1 - one_hot[0, ..., 0]
    one_hot[1, :, 0::2, :, :, 0] = (state_stack == 1)[
        :, 0::2
    ]  # black to play because we start on move 3
    one_hot[1, :, 1::2, :, :, 0] = (state_stack == -1)[:, 1::2]
    one_hot[1, ..., 1] = 1 - one_hot[1, ..., 0]
    return one_hot


state_stack_one_hot = state_stack_to_one_hot(state_stack)
print(state_stack_one_hot.shape)
print((state_stack_one_hot[:, 0, 17, 4:9, 2:5]))
print((state_stack[0, 17, 4:9, 2:5]))

layer = 4
batch_size = 100
lr = 1e-4
wd = 0.01
pos_start = 3
pos_end = model.cfg.n_ctx - 3
length = pos_end - pos_start
options = 2
rows = 8
cols = 8
num_epochs = 1
num_games = 4500000
x = 0
y = 2
probe_name = "L4_blank_vs_color"
# The first mode is blank or not, the second mode is next or prev GIVEN that it is not blank
modes = 2
alternating = torch.tensor(
    [1 if i % 2 == 0 else -1 for i in range(length)], device="cuda"
)
# %%
linear_probe = torch.randn(
    modes,
    model.cfg.d_model,
    rows,
    cols,
    options,
    requires_grad=False,
    device="cuda",
) / np.sqrt(model.cfg.d_model)
linear_probe.requires_grad = True
optimiser = torch.optim.AdamW(
    [linear_probe], lr=lr, betas=(0.9, 0.99), weight_decay=wd
)

writer = SummaryWriter("runs/testing")

for epoch in range(num_epochs):
    full_train_indices = torch.randperm(num_games)
    for i in tqdm(range(0, num_games, batch_size)):
        indices = full_train_indices[i : i + batch_size]
        games_int = board_seqs_int[indices]
        games_str = board_seqs_string[indices]
        state_stack = torch.stack(
            [
                torch.tensor(seq_to_state_stack(games_str[i]))
                for i in range(batch_size)
            ]
        )
        state_stack = state_stack[:, pos_start:pos_end, :, :]

        state_stack_one_hot = state_stack_to_one_hot(state_stack).cuda()
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                games_int.cuda()[:, :-1], return_type=None
            )
            resid_post = cache["resid_post", layer][:, pos_start:pos_end]
        breakpoint()
        probe_out = einsum(
            "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
            resid_post,
            linear_probe,
        )
        # print(probe_out.shape)

        acc_blank = (
            (probe_out[0].argmax(-1) == state_stack_one_hot[0].argmax(-1))
            .float()
            .mean()
        )
        acc_color = (
            (probe_out[1].argmax(-1) == state_stack_one_hot[1].argmax(-1))
            * state_stack_one_hot[1].sum(-1)
        ).float().sum() / (state_stack_one_hot[1]).float().sum()

        probe_log_probs = probe_out.log_softmax(-1)
        probe_correct_log_probs = (
            einops.reduce(
                probe_log_probs * state_stack_one_hot,
                "modes batch pos rows cols options -> modes pos rows cols",
                "mean",
            )
            * options
        )  # Multiply to correct for the mean over options
        loss_blank = -probe_correct_log_probs[0, :].mean(0).sum()
        loss_color = -probe_correct_log_probs[1, :].mean(0).sum()

        loss = loss_blank + loss_color
        loss.backward()

        #wandb.log(
        #    {
        #        "loss_blank": loss_blank.item(),
        #        "acc_blank": acc_blank.item(),
        #        "loss_blank_E2": -probe_correct_log_probs[0, :]
        #        .mean(0)[4, 2]
        #        .item(),
        #        "loss_color": loss_color.item(),
        #        "acc_color": acc_color.item(),
        #        "loss_color_E2": -probe_correct_log_probs[1, :]
        #        .mean(0)[4, 2]
        #        .item(),
        #    }
        #)
        writer.add_scalar("loss_blank", loss_blank.item())
        writer.add_scalar("acc_black", acc_blank.item())
        writer.add_scalar("loss_blank_E2", -probe_correct_log_probs[0, :].mean(0)[4, 2].item())
        writer.add_scalar("loss_color", loss_color.item())
        writer.add_scalar("acc_color", acc_color.item())
        writer.add_scalar("loss_color_E2", -probe_correct_log_probs[1, :].mean(0)[4, 2].item())

        optimiser.step()
        optimiser.zero_grad()
torch.save(linear_probe, f"linear_probe_{probe_name}_v1.pth")
# %%
# %%
