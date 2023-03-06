"""
Utility functions for mech int experiments.
"""
from functools import partial
from tqdm import tqdm
import numpy as np
import torch
from data import get_othello
from data.othello import OthelloBoardState
from mingpt.dataset import CharDataset
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import transformer_lens.utils as utils


torch.set_grad_enabled(True)
STARTING_SQUARES = [27, 28, 35, 36]

# try:
#     othello
#     print("Othello dataset exists")

# except:
#     print("Making dataset")
#     othello = get_othello(ood_num=-1, data_root=None, wthor=True)
#     train_dataset = CharDataset(othello)


# Hmm, output board_seqs_int.pth file size is smaller.
othello = get_othello(ood_num=-1, data_root=None, wthor=True)
train_dataset = CharDataset(othello)
full_seqs = list(filter(lambda x: len(x) == 60, train_dataset.data.sequences))
print(len(full_seqs))
board_seqs = torch.tensor(full_seqs)
print(board_seqs.numel())


n = 4500000
board_seqs = torch.zeros((n, 60), dtype=int)
for c, seq in enumerate(tqdm(othello.sequences)):
    board_seqs[c, : len(seq)] = torch.tensor(seq)
    if c == n - 1:
        break

board_seqs_string = board_seqs
print(board_seqs_string.numel())
# # %%
board_seqs_int = board_seqs_string.clone()
board_seqs_int[board_seqs_string < 29] += 1
board_seqs_int[(board_seqs_string >= 29) & (board_seqs_string <= 34)] -= 1
board_seqs_int[(board_seqs_string > 34)] -= 3
rand = torch.randint(0, 1000000, (20,))
print(board_seqs_int.flatten()[rand])
print(board_seqs_string.flatten()[rand])
# torch.save(board_seqs, "board_seqs.pt")
# %%
indices = torch.randperm(len(board_seqs_int))
board_seqs_int = board_seqs_int[indices]
board_seqs_string = board_seqs_string[indices]
torch.save(board_seqs_int, "board_seqs_int.pth")
torch.save(board_seqs_string, "board_seqs_string.pth")


board_seqs_int = torch.load("board_seqs_int.pth")
board_seqs_string = torch.load("board_seqs_string.pth")
print(board_seqs_int.shape)
# imshow(board_seqs_int[:5], title="Board Seqs Int Test")
# imshow(board_seqs_string[:5], title="Board Seqs String Test")
# %%


def get_itos():
    """
    Build itos mapping.
    Handles 27, 28, 35, 36 squares (starting squares).
    """
    itos = {0: -100}
    for idx in range(1, 28):
        itos[idx] = idx - 1

    for idx in range(28, 34):
        itos[idx] = idx + 1

    for idx in range(34, 61):
        itos[idx] = idx + 3
    return itos


def get_stoi():
    """
    Build stoi mapping.
    Handles 27, 28, 35, 36 squares (starting squares).
    """
    _itos = get_itos()
    stoi = {y: x for x, y in _itos.items()}
    stoi[-1] = 0
    for sq in STARTING_SQUARES:
        assert sq not in stoi
    return stoi


itos = get_itos()
stoi = get_stoi()
stoi_indices = [x for x in range(64) if x not in STARTING_SQUARES]
ALPHA = "ABCDEFGH"


def to_board_label(idx):
    return f"{ALPHA[idx//8]}{idx%8}"


board_labels = list(map(to_board_label, stoi_indices))


def str_to_int(s):
    return stoi[s] - 1


def to_int(x):
    """
    Convert x (board cell) to 'int' representation (model's vocabulary).
    Calls itself recursively.
    """
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_int(x.item())

    if isinstance(x, (list, np.ndarray, torch.Tensor)):
        return [to_int(i) for i in x]

    if isinstance(x, int):
        return stoi[x]

    if isinstance(x, str):
        x = x.upper()
        return to_int(to_string(x))

    raise RuntimeError(f"Unknown type for x: {type(x)}.")


def to_string(x):
    """
    Confusingly, maps x (board cell)to an int, but a board position
    label not a token label.
    (token labels have 0 == pass, and middle board cells don't exist)
    """
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_string(x.item())

    if isinstance(x, (list, np.ndarray, torch.Tensor)):
        return [to_string(i) for i in x]

    if isinstance(x, int):
        return itos[x]

    if isinstance(x, str):
        x = x.upper()
        return 8 * ALPHA.index(x[0]) + int(x[1])

    raise RuntimeError(f"Unknown type for x: {type(x)}.")


def to_label(x, from_int=True):
    """
    Convert x (board cell) to 'label' representation.
    """
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_label(x.item(), from_int=from_int)

    if isinstance(x, (list, np.ndarray, torch.Tensor)):
        return [to_label(i, from_int=from_int) for i in x]

    if isinstance(x, int):
        if from_int:
            return to_board_label(to_string(x))
        return to_board_label(x)

    if isinstance(x, str):
        return x

    raise RuntimeError(f"Unknown type for x: {type(x)}.")


def to_numpy(x):
    """
    Convert tensor to numpy array.
    """
    return x.numpy()


int_to_label = to_label
string_to_label = partial(to_label, from_int=False)
str_to_label = string_to_label


def moves_to_state(moves):
    """
    :moves: a list of string entries (ints)
    """
    state = np.zeros((8, 8), dtype=bool)
    for move in moves:
        state[move // 8, move % 8] = 1.0
    return state


int_labels = (
    list(range(1, 28))
    + ["X", "X"]
    + list(range(28, 34))
    + ["X", "X"]
    + list(range(34, 61))
)


def get_valid_moves(sequence):
    """
    Get valid moves for a list of moves.
    """
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
    board = OthelloBoardState()
    return board.get_gt(sequence, "get_valid_moves")


def one_hot(list_of_ints, num_classes=64):
    out = torch.zeros((num_classes,), dtype=torch.float32)
    out[list_of_ints] = 1.
    return out


#offset = 4123456
#num_games = 2000
#games_int = board_seqs_int[offset:offset+num_games]
#games_str = board_seqs_string[offset:offset+num_games]
#big_states = np.zeros((num_games, 59, 8, 8), dtype=np.float32)
#big_valid_moves = torch.zeros((num_games, 59, 64), dtype=torch.float32)
#for i in tqdm(range(num_games)):
#    board = OthelloBoardState()
#    for j in range(59):
#        try:
#            board.umpire(games_str[i][j])
#        except:
#            breakpoint()
#        big_states[i][j] = board.state
#        big_valid_moves[i][j] = one_hot(board.get_valid_moves())
#big_valid_moves = einops.rearrange(big_valid_moves, "num_games pos (r c) -> num_games pos r c", r=8, c=8)
## %%
#big_othello_state_dict = {
#    "big_states": big_states,
#    "big_valid_moves": big_valid_moves,
#    "offset": offset,
#    "games_str": games_str,
#    "games_int": games_int,
#}
#torch.save(big_othello_state_dict, "/workspace/_scratch/big_othello_state_dict.pth")
#big_othello_state_dict = torch.load(
#    "/workspace/_scratch/big_othello_state_dict.pth"
#)
#big_states = big_othello_state_dict["big_states"]
#big_valid_moves = big_othello_state_dict["big_valid_moves"]
#offset = big_othello_state_dict["offset"]
#games_str = big_othello_state_dict["games_str"]
#games_int = big_othello_state_dict["games_int"]


def load_hooked_model(model_type):
    """
    Load a HookedTransformer model pulled from Huggingface.
    :model_type: must be either "synthetic" or "champsionship"
    """
    if model_type not in ["synthetic", "championship"]:
        raise ValueError(f"Invalid 'model_type': {model_type}.")

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

    if model_type == "synthetic":
        model_path = "synthetic_model.pth"
    else:
        model_path = "championship_model.pth"

    sd = utils.download_file_from_hf(
        "NeelNanda/Othello-GPT-Transformer-Lens", model_path
    )
    model.load_state_dict(sd)
    return model


#model = load_hooked_model("synthetic")
## %%
#with torch.inference_mode():
#    big_logits, big_cache = model.run_with_cache(games_int[:, :-1].cuda())
