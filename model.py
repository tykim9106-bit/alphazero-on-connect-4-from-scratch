"""
AlphaZero on Connect-4 from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_empty_board
import numpy as np

def make_empty_board():
    """Return a 6x7 integer NumPy array of zeros representing an empty Connect-4 board."""
    return np.zeros((6, 7), dtype=int)

# Step 2 - column_top_row
def column_top_row(board, column):
    """Return the lowest empty row in `column`, or -1 if the column is full."""
    for row in range(len(board) - 1, -1, -1):
        if board[row, column] == 0:
            return row

    return -1

# Step 3 - drop_piece
def drop_piece(board, column, player):
    # Place `player` in the lowest empty row of `column`
    # and return a new board.

    row = column_top_row(board, column)

    if row == -1:
        raise ValueError("Column is full.")

    new_board = board.copy()
    new_board[row, column] = player

    return new_board

# Step 4 - column_full
def column_full(board, column):
    """Return True if `column` has no empty rows left."""
    return column_top_row(board, column) == -1

# Step 5 - valid_moves
def valid_moves(board):
    """Return a list of column indices that still have at least one empty row."""
    return [
        column
        for column in range(board.shape[1])
        if not column_full(board, column)
    ]

# Step 6 - four_in_a_row_horizontal
def four_in_a_row_horizontal(board):
    """Return True if there are four matching non-zero pieces in a row horizontally."""
    rows, columns = board.shape

    for row in range(rows):
        for column in range(columns - 3):
            piece = board[row, column]

            if (
                piece != 0
                and board[row, column + 1] == piece
                and board[row, column + 2] == piece
                and board[row, column + 3] == piece
            ):
                return piece

    return 0

# Step 7 - four_in_a_row_vertical
def four_in_a_row_vertical(board):
    """Return True if there are four matching non-zero pieces in a column vertically."""
    rows, columns = board.shape

    for column in range(columns):
        for row in range(rows - 3):
            piece = board[row, column]

            if (
                piece != 0
                and board[row + 1, column] == piece
                and board[row + 2, column] == piece
                and board[row + 3, column] == piece
            ):
                return piece

    return 0

# Step 8 - four_in_a_row_diagonal_down_right
def four_in_a_row_diagonal_down_right(board):
    rows, columns = board.shape
    
    for column in range(columns - 3):
        for row in range(rows - 3):
            piece = board[row, column]

            if (
                piece != 0
                and board[row + 1, column + 1] == piece
                and board[row + 2, column + 2] == piece
                and board[row + 3, column + 3] == piece
            ):
                return piece

    return 0

# Step 9 - four_in_a_row_diagonal_up_right
def four_in_a_row_diagonal_up_right(board):
    rows, columns = board.shape
    
    for column in range(columns - 3):
        for row in range(3, rows):
            piece = board[row, column]

            if (
                piece != 0
                and board[row - 1, column + 1] == piece
                and board[row - 2, column + 2] == piece
                and board[row - 3, column + 3] == piece
            ):
                return piece

    return 0

# Step 10 - check_winner
def check_winner(board):
    checks = [
        four_in_a_row_horizontal,
        four_in_a_row_vertical,
        four_in_a_row_diagonal_up_right,
        four_in_a_row_diagonal_down_right,
    ]

    for check in checks:
        winner = check(board)
        if winner != 0:
            return winner

    return 0

# Step 11 - board_is_full
def board_is_full(board):
    return np.all(board[0, :] != 0)

# Step 12 - is_terminal
def is_terminal(board):
    winner = int(check_winner(board))

    if winner != 0:
        return True, winner

    if board_is_full(board):
        return True, 0

    return False, 0

# Step 13 - other_player
def other_player(player):
    return 3-player

# Step 14 - step_env
def step_env(board, column, player):
    new_board = drop_piece(board, column, player)
    done, winner = is_terminal(new_board)
    next_player = other_player(player)

    return new_board, done, winner, next_player

# Step 15 - encode_board
def encode_board(board, current_player):
    """Encode a 6x7 board as a (2, 6, 7) float32 tensor from current_player's view."""
    opponent = other_player(current_player)

    current_plane = (board == current_player).astype(np.float32)
    opponent_plane = (board == opponent).astype(np.float32)

    return np.stack([current_plane, opponent_plane], axis=0)

# Step 16 - board_to_torch_tensor
def board_to_torch_tensor(board, current_player):
    enc = encode_board(board, current_player)
    return torch.from_numpy(enc).unsqueeze(0).float()

# Step 17 - init_conv_backbone
import torch.nn as nn
def init_conv_backbone(in_channels=2, hidden_channels=16):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

# Step 18 - init_policy_head
import torch
import torch.nn as nn

def init_policy_head(hidden_channels=16, num_columns=7):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, num_columns) logits."""

    return nn.Sequential(
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
        nn.Flatten(start_dim=1),
        nn.Linear(6 * 7, num_columns)
    )

# Step 19 - init_value_head
import torch
import torch.nn as nn

def init_value_head(hidden_channels=16):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, 1) in (-1, 1)."""
    
    return nn.Sequential(
        nn.Conv2d(hidden_channels, 1, kernel_size=(6, 7)),
        nn.Flatten(start_dim=1),
        nn.Tanh()
    )

# Step 20 - build_policy_value_net
import torch
import torch.nn as nn

def build_policy_value_net(in_channels=2, hidden_channels=16, num_columns=7):
    """Compose backbone + policy head + value head into one nn.Module."""
        
    class PolicyValueNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.backbone = init_conv_backbone(
                in_channels=in_channels,
                hidden_channels=hidden_channels
            )

            self.policy_head = init_policy_head(
                hidden_channels=hidden_channels,
                num_columns=num_columns
            )

            self.value_head = init_value_head(
                hidden_channels=hidden_channels
            )

        def forward(self, x):
            features = self.backbone(x)

            policy_logits = self.policy_head(features)
            value = self.value_head(features)

            return policy_logits, value
        
    return PolicyValueNet()

# Step 21 - policy_value_forward
import torch
import torch.nn as nn

def policy_value_forward(net, encoded_board):
    """Run encoded_board (B,2,6,7) through net and return (logits, value)."""
    return net(encoded_board)

# Step 22 - action_mask
import numpy as np

def action_mask(board):
    mask = np.zeros(7, dtype=bool)

    for col in valid_moves(board):
        mask[col] = True

    return mask

# Step 23 - masked_policy_logits
import torch

def masked_policy_logits(logits, mask):
    """Set logits at illegal columns to -inf.

    logits: torch.Tensor of shape (..., 7)
    mask:   bool array/tensor of shape (7,), True = legal
    returns: torch.Tensor of same shape as logits
    """
    masked_logits = logits.clone()
    masked_logits[...,~mask] = -torch.inf
    return masked_logits

# Step 24 - masked_log_softmax (not yet solved)
# TODO: implement

# Step 25 - sample_action_from_policy (not yet solved)
# TODO: implement

# Step 26 - greedy_action_from_policy (not yet solved)
# TODO: implement

# Step 27 - make_mcts_node (not yet solved)
# TODO: implement

# Step 28 - node_q_value (not yet solved)
# TODO: implement

# Step 29 - ucb_score (not yet solved)
# TODO: implement

# Step 30 - select_best_child (not yet solved)
# TODO: implement

# Step 31 - select_leaf (not yet solved)
# TODO: implement

# Step 32 - evaluate_with_network (not yet solved)
# TODO: implement

# Step 33 - expand_node (not yet solved)
# TODO: implement

# Step 34 - backup_value (not yet solved)
# TODO: implement

# Step 35 - run_one_simulation (not yet solved)
# TODO: implement

# Step 36 - run_mcts (not yet solved)
# TODO: implement

# Step 37 - visit_count_policy (not yet solved)
# TODO: implement

# Step 38 - mcts_choose_action (not yet solved)
# TODO: implement

# Step 39 - record_self_play_step (not yet solved)
# TODO: implement

# Step 40 - play_self_play_game (not yet solved)
# TODO: implement

# Step 41 - assign_value_targets (not yet solved)
# TODO: implement

# Step 42 - generate_self_play_batch (not yet solved)
# TODO: implement

# Step 43 - value_loss_mse (not yet solved)
# TODO: implement

# Step 44 - policy_loss_cross_entropy (not yet solved)
# TODO: implement

# Step 45 - l2_regularization_loss (not yet solved)
# TODO: implement

# Step 46 - combined_loss (not yet solved)
# TODO: implement

# Step 47 - encode_batch_states (not yet solved)
# TODO: implement

# Step 48 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 49 - training_step (not yet solved)
# TODO: implement

# Step 50 - training_epoch (not yet solved)
# TODO: implement

# Step 51 - self_play_iteration (not yet solved)
# TODO: implement

# Step 52 - train_loop (not yet solved)
# TODO: implement

# Step 53 - random_policy_action (not yet solved)
# TODO: implement

# Step 54 - greedy_agent_action (not yet solved)
# TODO: implement

# Step 55 - play_one_match (not yet solved)
# TODO: implement

# Step 56 - match_win_rate (not yet solved)
# TODO: implement

# Step 57 - evaluate_against_random (not yet solved)
# TODO: implement

