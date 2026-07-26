import chess

from chess_accuracy.divider import faithful_division, heuristic_division


class TestHeuristicDivision:
    def test_short_game(self):
        div = heuristic_division(10)
        assert div.middle is None
        assert div.end is None
        assert div.plies == 10

    def test_medium_game(self):
        div = heuristic_division(40)
        assert div.middle == 20
        assert div.end is None
        assert div.plies == 40

    def test_long_game(self):
        div = heuristic_division(80)
        assert div.middle == 20
        assert div.end == 60
        assert div.plies == 80

    def test_exact_boundary_20(self):
        div = heuristic_division(20)
        assert div.middle is None
        assert div.end is None

    def test_exact_boundary_60(self):
        div = heuristic_division(60)
        assert div.middle == 20
        assert div.end is None

    def test_plies_matches_input(self):
        for n in [1, 15, 30, 50, 100]:
            div = heuristic_division(n)
            assert div.plies == n


class TestFaithfulDivision:
    def test_starting_position(self):
        board = chess.Board()
        boards = [board.copy()]
        div = faithful_division(boards)
        assert div.middle is None
        assert div.end is None

    def test_early_game(self):
        board = chess.Board()
        boards = [board.copy()]
        for _ in range(10):
            move = next(iter(board.legal_moves))
            board.push(move)
            boards.append(board.copy())
        div = faithful_division(boards)
        assert div.middle is None

    def test_endgame_few_pieces(self):
        board = chess.Board()
        board.clear()
        board.set_fen("8/4k3/8/8/8/8/4K3/8 w - - 0 1")
        boards = [board.copy()]
        div = faithful_division(boards)
        assert div.end is not None

    def test_plies_count(self):
        board = chess.Board()
        boards = [board.copy()]
        for _ in range(5):
            move = next(iter(board.legal_moves))
            board.push(move)
            boards.append(board.copy())
        div = faithful_division(boards)
        assert div.plies == 5
