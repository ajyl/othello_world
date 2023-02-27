"""
Module for trees.
"""

import time
from collections import deque
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from src.othello import OthelloBoardState, get as get_othello
from mingpt.dataset import CharDataset


class OthelloTree:
    def __init__(self, dataset: CharDataset, board_state: OthelloBoardState):
        """
        Initialize tree object.
        """
        self.dataset = dataset
        self.board = board_state

    def _single_rollout(self, moves):
        """
        Get single rollout?
        """

    @staticmethod
    def print_tree(tree, hash_table, root):
        """
        Print tree.
        """
        queue = [(root, None, None, 0)]
        rows = "abcdefgh"

        while len(queue) > 0:

            curr, last_move, last_player, depth = queue.pop()
            board_state = hash_table[curr]

            if last_move is not None:
                row, col = last_move // 8, last_move % 8 + 1
                last_player = {-1: "O", 1: "X"}[last_player]
                print(
                    " " * (depth * 4)
                    + "Last move: (%s, %d) played by %s" % (rows[row], col, last_player)
                )
            board_state.__print__(depth)

            for child, _last_move, _last_player in tree[curr]:
                queue.append((child, _last_move, _last_player, depth + 1))

    def roll_out(self, rollout, print_tree=False):
        """
        Build tree?
        """
        start_time = time.time()
        tree = {}
        hash_table = {}

        root_hash = self.board.get_hash()
        hash_table[root_hash] = self.board

        queue = deque([(self.board, rollout)])

        while len(queue) > 0:
            curr_board, _rollout = queue.pop()

            if len(tree) % 10000 == 0:
                print(
                    "Curr tree size: %d, roll out: %d, time taken: %s"
                    % (len(tree), _rollout, time.time() - start_time)
                )

            _hash = curr_board.get_hash()
            if _hash != root_hash:
                if _hash in hash_table:
                    assert _hash in tree
                    continue

            hash_table[_hash] = curr_board

            if _hash not in tree:
                tree[_hash] = []

            if _rollout <= 0:
                continue

            # TODO: Handle 'forfeit' moves.
            valid_moves, error_code = curr_board.get_valid_moves()
            if error_code != 0:
                continue

            next_player = curr_board.next_hand_color
            for move in valid_moves:
                next_board = deepcopy(curr_board)
                next_board.umpire(move)
                tree[_hash].append((next_board.get_hash(), move, next_player))

                queue.appendleft((next_board, _rollout - 1))

        # if len(tree) > 2:
        #    breakpoint()
        if print_tree:
            self.print_tree(tree, hash_table, root_hash)
        return tree, hash_table, root_hash


def build_entire_tree():
    """
    Build entire tree from very first node.
    """

    othello = get_othello(data_root="data/othello_championship")
    dataset = CharDataset(othello)

    init_board = OthelloBoardState()
    othello_tree = OthelloTree(dataset, init_board)

    start = time.time()
    tree, hash_table, root_hash = othello_tree.roll_out(10, print_tree=False)
    end = time.time()
    print("Time took:", end - start)

    breakpoint()


def debug():
    """Driver"""

    _othello = get_othello(data_root="data/othello_small")
    _dataset = CharDataset(_othello)

    loader = DataLoader(_dataset, batch_size=1)
    for x, y in tqdm(loader, total=len(loader)):
        tbf = [_dataset.itos[_] for _ in x.tolist()[0]]
        valid_until = tbf.index(-100) if -100 in tbf else 999

        _board = OthelloBoardState()
        _board.get_gt(tbf[:valid_until], "get_state")

        tree = OthelloTree(_dataset, _board)
        print(tree.board.get_valid_moves())
        x = tree.roll_out(3)


def main():
    """Driver"""
    build_entire_tree()


if __name__ == "__main__":
    main()
