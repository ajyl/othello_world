"""
Utility functions for plotting experiments.
"""

import numpy as np
import torch
from tqdm import tqdm
from plotly.express import imshow
from data.othello import OthelloBoardState
from mech_exp.tl_othello_utils import int_labels, to_string, stoi_indices, to_board_label, moves_to_state, to_label, to_int, ALPHA, to_numpy



def _make_plot_state(board):
    state = np.copy(board.state).flatten()
    valid_moves = board.get_valid_moves()
    next_move = board.get_next_hand_color()
    for move in valid_moves:
        state[move] = next_move - 0.5
    return state


def _add_counter(fig, position, color):
    is_black = color > 0
    row = position // 8
    col = position % 8
    fig.layout.shapes += (
        dict(
            type="circle",
            x0=col - 0.2,
            y0=row - 0.2,
            x1=col + 0.2,
            y1=row + 0.2,
            fillcolor="black" if is_black else "white",
            line_color="green",
            line_width=0.5,
        ),
    )
    return fig


def _add_ring(fig, position, color):
    is_black = color > 0
    row = position // 8
    col = position % 8
    offset = 0.3
    fig.layout.shapes += (
        dict(
            type="rect",
            x0=col - offset,
            y0=row - offset,
            x1=col + offset,
            y1=row + offset,
            line_color="black" if is_black else "red",
            line_width=5,
            fillcolor=None,
        ),
    )
    return fig


def _counter_shape(position, color, mode="normal"):
    is_black = color > 0
    row = position // 8
    col = position % 8
    shape = dict(
        type="circle",
        fillcolor="black" if is_black else "white",
    )
    if mode == "normal":
        shape.update(
            x0=col - 0.2,
            y0=row - 0.2,
            x1=col + 0.2,
            y1=row + 0.2,
            line_color="green",
            line_width=0.5,
        )
    elif mode == "flipped":
        shape.update(
            x0=col - 0.22,
            y0=row - 0.22,
            x1=col + 0.22,
            y1=row + 0.22,
            line_color="purple",
            line_width=3,
        )
    elif mode == "new":
        shape.update(
            line_color="red",
            line_width=4,
            x0=col - 0.25,
            y0=row - 0.25,
            x1=col + 0.25,
            y1=row + 0.25,
        )
    return shape


def plot_board(moves, return_fig=False):
    """
    Plot board-state given sequence of moves.
    """
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()
    if isinstance(moves[0], str):
        moves = to_string(moves)
    board = OthelloBoardState()
    states = []
    states.append(_make_plot_state(board))
    for move in moves:
        board.umpire(move)
        states.append(_make_plot_state(board))

    states = np.stack(states, axis=0)
    fig = imshow(
        states.reshape(-1, 8, 8),
        color_continuous_scale="Geyser",
        aspect="equal",
        #return_fig=True,
        animation_frame=0,
        y=["a", "b", "c", "d", "e", "f", "g", "h"],
        x=["0", "1", "2", "3", "4", "5", "6", "7"],
        #animation_index=[
        #    f"{i+1} ({'W' if i%2==0 else 'B'}) [{to_board_label(moves[i]) if i>=0 else 'X'} -> {to_board_label(moves[i+1]) if i<len(moves)-1 else 'X'}]"
        #    for i in range(-1, len(moves))
        #],
        #animation_name="Move",
    )
    fig.update_traces(
        text=[[str(i + 8 * j) for i in range(8)] for j in range(8)],
        texttemplate="%{text}",
    )
    for c, frame in enumerate(fig.frames):
        for i in range(64):
            if states[c].flatten()[i] == 1:
                frame = _add_counter(frame, i, True)
            elif states[c].flatten()[i] == -1:
                frame = _add_counter(frame, i, False)
    fig.layout.shapes = fig.frames[0].layout.shapes
    if return_fig:
        return fig

    fig.show()
    return None


def plot_board_log_probs(moves, logits, return_fig=False, use_counters=False):
    logits = logits.squeeze(0)
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()

    if isinstance(moves[0], str):
        moves = to_string(moves)

    assert len(moves) == len(logits)
    board = OthelloBoardState()
    states = []
    for move in moves:
        board.umpire(move)
        states.append(_make_plot_state(board))
    states = np.stack(states, axis=0)

    log_probs = logits.log_softmax(dim=-1)
    log_probs_template = torch.zeros((len(moves), 64)).cuda() - 100
    if log_probs.shape[-1] == 61:
        log_probs_template[:, stoi_indices] = log_probs[:, 1:]
    else:
        log_probs_template[:, stoi_indices] = log_probs[:, :]
    log_probs_template = log_probs_template.reshape(-1, 8, 8)
    log_probs_template = log_probs_template.cpu().detach()

    fig = imshow(
        log_probs_template,
        color_continuous_scale="Blues",
        zmin=-6.0,
        zmax=0.0,
        aspect="equal",
        #return_fig=True,
        animation_frame=0,
        y=["a", "b", "c", "d", "e", "f", "g", "h"],
        x=["0", "1", "2", "3", "4", "5", "6", "7"],
        #animation_index=[
        #    f"{i+1} ({'W' if i%2==0 else 'B'}) [{to_board_label(moves[i])} -> {to_board_label(moves[i+1]) if i<len(moves)-1 else 'X'}]"
        #    for i in range(len(moves))
        #],
        #animation_name="Move",
    )
    # fig.update_traces(text=[[str(i+8*j) for i in range(8)] for j in range(8)], texttemplate="%{text}")
    for c, frame in enumerate(tqdm(fig.frames)):
        text = []
        shapes = []
        for i in range(64):
            text.append("")
            counter_text = "O" if moves[c] != i else "X"
            if states[c].flatten()[i] == 1:
                if use_counters:
                    shapes.append(_counter_shape(i, True))
                else:
                    # black = red
                    text[
                        -1
                    ] = f"<b style='font-size: 24em; color: red; '>{counter_text}</b>"
            elif states[c].flatten()[i] == -1:
                if use_counters:
                    shapes.append(_counter_shape(i, False))
                else:
                    # white = green
                    text[
                        -1
                    ] = f"<b style='font-size: 24em; color: green;'>{counter_text}</b>"
            else:
                if states[c].flatten()[i] > 0.2:
                    text[
                        -1
                    ] = f"<span style='font-size: 12em; '>{to_board_label(i)}</span>"
                    # print(i, c, "b")
                    # frame = _add_ring(frame, i, True)
                elif states[c].flatten()[i] < -0.2:
                    text[
                        -1
                    ] = f"<span style='font-size: 12em; color: white'>{to_board_label(i)}</span>"
                    # print(i, c, "w")
                    # frame = _add_ring(frame, i, False)
        frame.layout.shapes = tuple(shapes)
        frame.data[0]["text"] = np.array(text).reshape(8, 8)
        frame.data[0]["texttemplate"] = "%{text}"
        frame.data[0][
            "hovertemplate"
        ] = "<b>%{y}%{x}</b><br>log prob: %{z}<br>prob=%{customdata}<extra></extra>"
        frame.data[0]["customdata"] = to_numpy(log_probs_template[c].exp())
    # print(states)
    fig.layout.shapes = fig.frames[0].layout.shapes
    fig.data[0]["text"] = fig.frames[0].data[0]["text"]
    fig.data[0]["texttemplate"] = fig.frames[0].data[0]["texttemplate"]
    fig.data[0]["customdata"] = fig.frames[0].data[0]["customdata"]
    fig.data[0]["hovertemplate"] = fig.frames[0].data[0]["hovertemplate"]
    if return_fig:
        return fig
    else:
        fig.show()


def plot_single_board(moves, model=None, return_fig=False, title=None):
    # moves is a list of string entries (ints)
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()
    if isinstance(moves[0], str):
        moves = to_string(moves)
    board = OthelloBoardState()
    if len(moves) > 1:
        board.update(moves[:-1])

    prev_state = np.copy(board.state)
    prev_player = board.next_hand_color
    prev_valid_moves = board.get_valid_moves()
    board.umpire(moves[-1])
    next_state = np.copy(board.state)
    next_player = board.next_hand_color
    next_valid_moves = board.get_valid_moves()

    empty = (prev_state == 0) & (next_state == 0)
    new = (prev_state == 0) & (next_state != 0)
    flipped = (prev_state != 0) & (next_state != prev_state) & (~new)
    prev_valid = moves_to_state(prev_valid_moves)
    next_valid = moves_to_state(next_valid_moves)

    state = np.copy(next_state)
    state[flipped] *= 0.9
    state[prev_valid] = 0.25 * prev_player
    state[next_valid] = 0.5 * next_player
    state[new] = 0.9 * prev_player
    if model is not None:
        logits = model(torch.tensor(to_int(moves)).cuda().unsqueeze(0)).cpu()
        log_probs = logits.log_softmax(-1)
        lps = torch.zeros(64) - 15.0
        lps[stoi_indices] = log_probs[0, -1, 1:]

    if title is None:
        title = f"Board State After {'Black' if prev_player==1 else 'White'} Plays {to_label(moves[-1], from_int=False)}"

    fig = imshow(
        state,
        color_continuous_scale="Geyser",
        title=title,
        y=[i for i in ALPHA],
        x=[str(i) for i in range(8)],
        aspect="equal",
        #return_fig=True,
    )
    fig.data[0][
        "hovertemplate"
    ] = "<b>%{y}%{x}</b><br>%{customdata}<extra></extra>"

    shapes = []
    texts = []
    for i in range(64):
        texts.append("")
        if empty.flatten()[i]:
            texts[-1] = to_label(i, from_int=False)
        elif flipped.flatten()[i]:
            shapes.append(_counter_shape(i, prev_player == 1, mode="flipped"))
        elif new.flatten()[i]:
            shapes.append(_counter_shape(i, prev_player == 1, mode="new"))
        elif prev_state.flatten()[i] != 0:
            shapes.append(
                _counter_shape(i, prev_state.flatten()[i] == 1, mode="normal")
            )
        else:
            raise ValueError(i)
    fig.layout.shapes = tuple(shapes)
    fig.data[0]["text"] = np.array(texts).reshape(8, 8)
    fig.data[0]["texttemplate"] = "%{text}"
    if model is not None:
        fig.data[0]["customdata"] = np.array(
            [
                f"LP:{lps[i].item():.4f}<br>I:{int_labels[i]}<br>S:{i}"
                for i in range(64)
            ]
        ).reshape(8, 8)
    else:
        fig.data[0]["customdata"] = np.array(
            [f"I:{int_labels[i]}<br>S:{i}" for i in range(64)]
        ).reshape(8, 8)

    if return_fig:
        return fig
    else:
        fig.show()
