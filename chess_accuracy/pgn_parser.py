"""PGN parser that converts PGN games to maia3 batch input tensors."""

import io
import math
from collections import deque

import chess
import chess.pgn
import numpy as np
import torch

from .maia3.dataset import get_historical_tokens, tokenize_board
from .maia3.utils import get_all_possible_moves, mirror_move

# Pre-computed move vocabulary (4352 moves)
ALL_MOVES = get_all_possible_moves()
ALL_MOVES_DICT = {m: i for i, m in enumerate(ALL_MOVES)}


def parse_pgn_to_positions(pgn_text: str) -> list[dict]:
    """Parse a PGN text and extract all positions with the moves played.

    Returns a list of dicts, one per move, each containing:
        - board: chess.Board state BEFORE the move
        - move: chess.Move that was played
        - is_white_turn: bool
        - clk_left_before: float (seconds, default 300)
        - clk_ponder: float (seconds, default 0)
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return []

    board = game.board()
    positions = []

    # Parse clock annotations from comments
    node = game
    clocks = {}  # ply -> (clk_before, clk_ponder)
    ply = 0
    while node.variations:
        next_node = node.variation(0)
        comment = node.comment
        clk_before = 300.0
        clk_ponder = 0.0

        # Extract [%clk H:MM:SS] from comment
        import re
        clk_match = re.search(r'\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]', comment)
        if clk_match:
            h, m, s = int(clk_match.group(1)), int(clk_match.group(2)), float(clk_match.group(3))
            clk_before = h * 3600 + m * 60 + s

        # Extract [%clk for opponent after ponder]
        clk_opp_match = re.search(r'\[%clk_opp\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]', comment)
        if clk_opp_match:
            h, m, s = int(clk_opp_match.group(1)), int(clk_opp_match.group(2)), float(clk_opp_match.group(3))
            clk_ponder = h * 3600 + m * 60 + s

        clocks[ply] = (clk_before, clk_ponder)

        move = next_node.move
        clk_before, clk_ponder = clocks.get(ply, (300.0, 0.0))

        positions.append({
            "board": board.copy(),
            "move": move,
            "is_white_turn": board.turn == chess.WHITE,
            "clk_left_before": clk_before,
            "clk_ponder": clk_ponder,
        })

        board.push(move)
        node = next_node
        ply += 1

    return positions


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Convert a chess.Move to the model's 4352-dimensional output index.

    The model always sees from white's perspective (board mirrored for black).
    Standard moves: from_sq * 64 + to_sq (indices 0-4095)
    Promotions: 4096 + from_file * 32 + to_file * 4 + piece_type (indices 4096-4351)
    """
    # Mirror for black to get the model's perspective
    if board.turn == chess.BLACK:
        move_uci = mirror_move(move.uci())
    else:
        move_uci = move.uci()

    # Fast path: look up in the pre-computed dictionary
    if move_uci in ALL_MOVES_DICT:
        return ALL_MOVES_DICT[move_uci]

    # Promotion fallback (shouldn't happen if ALL_MOVES is correct, but be safe)
    if len(move_uci) > 4:
        from_file = ord(move_uci[0]) - ord('a')
        to_file = ord(move_uci[2]) - ord('a')
        piece_map = {'q': 0, 'r': 1, 'b': 2, 'n': 3}
        piece_idx = piece_map[move_uci[4]]
        return 4096 + from_file * 32 + to_file * 4 + piece_idx

    # Standard move fallback
    from_idx = ord(move_uci[0]) - ord('a') + (int(move_uci[1]) - 1) * 8
    to_idx = ord(move_uci[2]) - ord('a') + (int(move_uci[3]) - 1) * 8
    return from_idx * 64 + to_idx


def _select_sample_indices(total_moves: int, n_sample: int) -> list[int]:
    """Select move indices to evaluate, preferring middlegame positions."""
    if n_sample <= 0 or n_sample >= total_moves:
        return list(range(total_moves))

    skip_open = min(8, total_moves // 6)
    skip_end = min(5, total_moves // 8)
    lo, hi = skip_open, total_moves - skip_end
    if hi - lo <= n_sample:
        return list(range(lo, hi))

    center = lo + (hi - lo) * 0.4
    spread = (hi - lo) / 3.0
    candidates = list(range(lo, hi))
    weights = [math.exp(-0.5 * ((i - center) / spread) ** 2) for i in candidates]

    chosen = []
    remaining = list(range(len(candidates)))
    for _ in range(n_sample):
        w_sum = sum(weights[i] for i in remaining)
        r = np.random.random() * w_sum
        cumul = 0.0
        for idx in remaining:
            cumul += weights[idx]
            if cumul >= r:
                chosen.append(candidates[idx])
                remaining.remove(idx)
                break

    return sorted(chosen)


def build_batch_tensors(
    positions: list[dict],
    elo_values: np.ndarray,
    cfg,
    n_sample: int = 0,
) -> dict:
    """Build batch tensors for N positions × M ELO values.

    Returns dict with:
        - tokens: (N*M, 64, D_in)
        - self_elos: (N*M,)
        - oppo_elos: (N*M,) — same as self_elos
        - human_moves: (N,) — move indices for match-rate computation
        - legal_masks: (N, 4352) — boolean legal move mask per position
        - n_positions: int (N)
        - n_elos: int (M)
    """
    from .maia3.dataset import get_legal_moves_mask

    n_elo = len(elo_values)

    # Select which positions to evaluate
    if n_sample > 0:
        sample_indices = _select_sample_indices(len(positions), n_sample)
    else:
        sample_indices = list(range(len(positions)))

    n_pos = len(sample_indices)

    # Build history deques for each sampled position
    board = chess.Board()
    history: deque[torch.Tensor] = deque(maxlen=cfg.history)

    sampled_set = set(sample_indices)

    all_tokens = []
    all_human_moves = []
    all_legal_masks = []

    for pos_idx, pos in enumerate(positions):
        token = tokenize_board(board)
        history.append(token)

        if pos_idx in sampled_set:
            hist_tokens = get_historical_tokens(
                history, cfg,
                base=300.0, inc=0.0,
                clk_left_before=pos["clk_left_before"],
                clk_ponder=pos["clk_ponder"],
            )
            all_tokens.append(hist_tokens)

            human_move_idx = move_to_index(pos["move"], board)
            all_human_moves.append(human_move_idx)

            # Compute legal move mask for this position
            legal_mask = get_legal_moves_mask(board, ALL_MOVES_DICT)
            all_legal_masks.append(legal_mask)

        board.push(pos["move"])

    tokens_n = torch.stack(all_tokens, dim=0)
    human_moves = np.array(all_human_moves, dtype=np.int64)
    legal_masks = torch.stack(all_legal_masks, dim=0)  # (N, 4352)

    if not cfg.include_time_info:
        tokens_n = tokens_n[:, :, :12 * cfg.history]

    tokens_batch = tokens_n.repeat_interleave(n_elo, dim=0)

    elo_t = torch.tensor(elo_values, dtype=torch.float32)
    self_elos = elo_t.repeat(n_pos)
    oppo_elos = self_elos.clone()

    return {
        "tokens": tokens_batch,
        "self_elos": self_elos,
        "oppo_elos": oppo_elos,
        "human_moves": human_moves,
        "legal_masks": legal_masks,
        "n_positions": n_pos,
        "n_elos": n_elo,
    }


def build_batch_tensors_2d(
    positions: list[dict],
    white_elo_values: np.ndarray,
    black_elo_values: np.ndarray,
    cfg,
    n_sample: int = 0,
) -> dict:
    """Build batch tensors for 2D ELO grid search (white_elo × black_elo).

    For white's turn positions: self_elo = white_elo, oppo_elo = black_elo
    For black's turn positions: self_elo = black_elo, oppo_elo = white_elo

    Returns dict with:
        - tokens: (N*W*B, 64, D_in)
        - self_elos: (N*W*B,)
        - oppo_elos: (N*W*B,)
        - human_moves: (N,) — move indices
        - legal_masks: (N, 4352)
        - is_white_turn: (N,) — boolean mask
        - white_elo_values: (W,)
        - black_elo_values: (B,)
        - n_positions: int (N)
        - n_white_elo: int (W)
        - n_black_elo: int (B)
    """
    from .maia3.dataset import get_legal_moves_mask

    n_w = len(white_elo_values)
    n_b = len(black_elo_values)
    n_grid = n_w * n_b  # total ELO combinations

    # Select which positions to evaluate
    if n_sample > 0:
        sample_indices = _select_sample_indices(len(positions), n_sample)
    else:
        sample_indices = list(range(len(positions)))

    n_pos = len(sample_indices)

    # Build history and collect position data
    board = chess.Board()
    history: deque[torch.Tensor] = deque(maxlen=cfg.history)
    sampled_set = set(sample_indices)

    all_tokens = []
    all_human_moves = []
    all_legal_masks = []
    all_is_white = []

    for pos_idx, pos in enumerate(positions):
        token = tokenize_board(board)
        history.append(token)

        if pos_idx in sampled_set:
            hist_tokens = get_historical_tokens(
                history, cfg,
                base=300.0, inc=0.0,
                clk_left_before=pos["clk_left_before"],
                clk_ponder=pos["clk_ponder"],
            )
            all_tokens.append(hist_tokens)
            all_human_moves.append(move_to_index(pos["move"], board))
            all_legal_masks.append(get_legal_moves_mask(board, ALL_MOVES_DICT))
            all_is_white.append(pos["is_white_turn"])

        board.push(pos["move"])

    tokens_n = torch.stack(all_tokens, dim=0)  # (N, 64, D_in)
    human_moves = np.array(all_human_moves, dtype=np.int64)
    legal_masks = torch.stack(all_legal_masks, dim=0)  # (N, 4352)
    is_white = np.array(all_is_white, dtype=bool)  # (N,)

    if not cfg.include_time_info:
        tokens_n = tokens_n[:, :, :12 * cfg.history]

    # Build ELO tensors for the 2D grid
    # Grid: for each position, replicate across all (W, B) combinations
    # White positions: self=white_elo[i], oppo=black_elo[j]
    # Black positions: self=black_elo[j], oppo=white_elo[i]
    w_t = torch.tensor(white_elo_values, dtype=torch.float32)
    b_t = torch.tensor(black_elo_values, dtype=torch.float32)

    # Create meshgrid: (W, B) -> flatten to (W*B,)
    w_grid, b_grid = torch.meshgrid(w_t, b_t, indexing="ij")
    w_flat = w_grid.reshape(-1)  # (W*B,)
    b_flat = b_grid.reshape(-1)  # (W*B,)

    # For each position, tile the grid: (N, W*B)
    # White positions: self=w_flat, oppo=b_flat
    # Black positions: self=b_flat, oppo=w_flat
    tokens_batch = tokens_n.repeat_interleave(n_grid, dim=0)  # (N*W*B, 64, D)

    self_elos_list = []
    oppo_elos_list = []
    for i in range(n_pos):
        if is_white[i]:
            self_elos_list.append(w_flat)
            oppo_elos_list.append(b_flat)
        else:
            self_elos_list.append(b_flat)
            oppo_elos_list.append(w_flat)

    self_elos = torch.cat(self_elos_list)  # (N*W*B,)
    oppo_elos = torch.cat(oppo_elos_list)  # (N*W*B,)

    return {
        "tokens": tokens_batch,
        "self_elos": self_elos,
        "oppo_elos": oppo_elos,
        "human_moves": human_moves,
        "legal_masks": legal_masks,
        "is_white_turn": is_white,
        "white_elo_values": white_elo_values,
        "black_elo_values": black_elo_values,
        "n_positions": n_pos,
        "n_white_elo": n_w,
        "n_black_elo": n_b,
    }
