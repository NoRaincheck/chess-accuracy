from chess_accuracy.maia3.utils import (
    get_all_possible_moves,
    mirror_move,
    mirror_square,
    parse_time_control,
)


class TestGetAllPossibleMoves:
    def test_length(self):
        moves = get_all_possible_moves()
        assert len(moves) == 4352

    def test_no_duplicates(self):
        moves = get_all_possible_moves()
        assert len(moves) == len(set(moves))

    def test_format_standard(self):
        moves = get_all_possible_moves()
        assert "a1b1" in moves
        assert "e2e4" in moves

    def test_format_promotion(self):
        moves = get_all_possible_moves()
        promos = [m for m in moves if len(m) == 5]
        assert len(promos) == 256
        assert "a7a8q" in promos
        assert "h7h8n" in promos


class TestMirrorSquare:
    def test_rank1_to_rank8(self):
        assert mirror_square("a1") == "a8"

    def test_rank8_to_rank1(self):
        assert mirror_square("h8") == "h1"

    def test_center(self):
        assert mirror_square("d4") == "d5"

    def test_rank4_to_rank5(self):
        assert mirror_square("e4") == "e5"


class TestMirrorMove:
    def test_standard_move(self):
        assert mirror_move("e2e4") == "e7e5"

    def test_promotion(self):
        assert mirror_move("a7a8q") == "a2a1q"

    def test_symmetry(self):
        move = "g1f3"
        mirrored = mirror_move(move)
        double = mirror_move(mirrored)
        assert double == move


class TestParseTimeControl:
    def test_with_increment(self):
        base, inc = parse_time_control("180+2")
        assert base == 180.0
        assert inc == 2.0

    def test_no_increment(self):
        base, inc = parse_time_control("300")
        assert base == 300.0
        assert inc == 0.0

    def test_zero_increment(self):
        base, inc = parse_time_control("60+0")
        assert base == 60.0
        assert inc == 0.0
