import chess

from chess_accuracy.pgn_parser import (
    _select_sample_indices,
    move_to_index,
    parse_pgn_to_positions,
)

SAMPLE_PGN = """\
[Event "Test Game"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O 1-0
"""


class TestParsePgnToPositions:
    def test_valid_pgn(self):
        positions = parse_pgn_to_positions(SAMPLE_PGN)
        assert len(positions) > 0

    def test_positions_have_required_keys(self):
        positions = parse_pgn_to_positions(SAMPLE_PGN)
        for pos in positions:
            assert "board" in pos
            assert "move" in pos
            assert "is_white_turn" in pos
            assert "clk_left_before" in pos
            assert "clk_ponder" in pos

    def test_board_is_chess_board(self):
        positions = parse_pgn_to_positions(SAMPLE_PGN)
        for pos in positions:
            assert isinstance(pos["board"], chess.Board)

    def test_move_is_chess_move(self):
        positions = parse_pgn_to_positions(SAMPLE_PGN)
        for pos in positions:
            assert isinstance(pos["move"], chess.Move)

    def test_empty_pgn(self):
        positions = parse_pgn_to_positions("")
        assert positions == []

    def test_invalid_pgn(self):
        positions = parse_pgn_to_positions("not a pgn")
        assert positions == []

    def test_white_turn_first(self):
        positions = parse_pgn_to_positions(SAMPLE_PGN)
        assert positions[0]["is_white_turn"] is True

    def test_alternating_turns(self):
        positions = parse_pgn_to_positions(SAMPLE_PGN)
        for i, pos in enumerate(positions):
            expected_white = i % 2 == 0
            assert pos["is_white_turn"] == expected_white


class TestMoveToIndex:
    def test_standard_move(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        idx = move_to_index(move, board)
        assert 0 <= idx < 4096

    def test_different_moves_different_indices(self):
        board = chess.Board()
        idx1 = move_to_index(chess.Move.from_uci("e2e4"), board)
        idx2 = move_to_index(chess.Move.from_uci("d2d4"), board)
        assert idx1 != idx2

    def test_black_move_mirrored(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        move = chess.Move.from_uci("e7e5")
        idx = move_to_index(move, board)
        assert 0 <= idx < 4352


class TestSelectSampleIndices:
    def test_zero_sample(self):
        indices = _select_sample_indices(100, 0)
        assert indices == list(range(100))

    def test_sample_exceeds_total(self):
        indices = _select_sample_indices(10, 20)
        assert indices == list(range(10))

    def test_sample_within_range(self):
        indices = _select_sample_indices(100, 10)
        assert len(indices) == 10
        assert all(0 <= i < 100 for i in indices)

    def test_indices_sorted(self):
        indices = _select_sample_indices(100, 15)
        assert indices == sorted(indices)

    def test_no_duplicates(self):
        indices = _select_sample_indices(100, 20)
        assert len(indices) == len(set(indices))
