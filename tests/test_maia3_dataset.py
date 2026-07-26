from types import SimpleNamespace

import chess
import torch

from chess_accuracy.maia3.dataset import (
    get_historical_tokens,
    get_legal_moves_mask,
    tokenize_board,
)


class TestTokenizeBoard:
    def test_starting_position_shape(self):
        board = chess.Board()
        tokens = tokenize_board(board)
        assert tokens.shape == (64, 12)

    def test_empty_square_all_zeros(self):
        board = chess.Board()
        tokens = tokenize_board(board)
        # Square a3 (index 16) should be empty in starting position
        assert tokens[16].sum() == 0

    def test_piece_present(self):
        board = chess.Board()
        tokens = tokenize_board(board)
        # Square e1 (index 4) has white king
        assert tokens[4].sum() == 1.0

    def test_black_perspective(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        tokens = tokenize_board(board)
        assert tokens.shape == (64, 12)


class TestGetLegalMovesMask:
    def test_initial_position(self):
        board = chess.Board()
        all_moves = {m: i for i, m in enumerate(["".join([
            chr(f + ord("a")), str(r + 1),
            chr(tf + ord("a")), str(tr + 1)
        ]) for r in range(8) for f in range(8)
            for tr in range(8) for tf in range(8)])}
        # Use the actual move dict from utils
        from chess_accuracy.maia3.utils import get_all_possible_moves
        all_moves = {m: i for i, m in enumerate(get_all_possible_moves())}
        mask = get_legal_moves_mask(board, all_moves)
        assert mask.dtype == torch.bool
        assert mask.sum() == 20  # 20 legal moves in starting position

    def test_check_position(self):
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        from chess_accuracy.maia3.utils import get_all_possible_moves
        all_moves = {m: i for i, m in enumerate(get_all_possible_moves())}
        mask = get_legal_moves_mask(board, all_moves)
        assert mask.sum() > 0

    def test_mask_shape(self):
        board = chess.Board()
        from chess_accuracy.maia3.utils import get_all_possible_moves
        all_moves = {m: i for i, m in enumerate(get_all_possible_moves())}
        mask = get_legal_moves_mask(board, all_moves)
        assert mask.shape == (4352,)


class TestGetHistoricalTokens:
    def test_full_history(self):
        cfg = SimpleNamespace(history=8, include_time_info=False)
        board_tokens = torch.zeros(64, 12)
        history = __import__("collections").deque([board_tokens] * 8, maxlen=8)
        result = get_historical_tokens(
            history,
            cfg, base=300.0, inc=0.0, clk_left_before=300.0, clk_ponder=0.0,
        )
        # 12*8 + 1 (clk_ponder column appended even without time info)
        assert result.shape == (64, 12 * 8 + 1)

    def test_short_history_padded(self):
        cfg = SimpleNamespace(history=8, include_time_info=False)
        board_tokens = torch.zeros(64, 12)
        history = __import__("collections").deque([board_tokens] * 3, maxlen=8)
        result = get_historical_tokens(history, cfg, base=300.0, inc=0.0, clk_left_before=300.0, clk_ponder=0.0)
        assert result.shape == (64, 12 * 8 + 1)

    def test_with_time_info(self):
        cfg = SimpleNamespace(history=2, include_time_info=True)
        board_tokens = torch.zeros(64, 12)
        history = __import__("collections").deque([board_tokens] * 2, maxlen=2)
        result = get_historical_tokens(history, cfg, base=300.0, inc=0.0, clk_left_before=300.0, clk_ponder=0.0)
        # 12*2 + 4 = 28
        assert result.shape == (64, 28)
