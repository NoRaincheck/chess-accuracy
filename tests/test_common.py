import pytest

from chess_accuracy.common import (
    Division,
    accuracy_from_win_percentage,
    game_accuracy,
    phase_accuracy,
    win_percentage_from_white_cp,
)


class TestWinPercentageFromWhiteCp:
    def test_zero_cp(self):
        assert win_percentage_from_white_cp(0) == pytest.approx(50.0)

    def test_large_positive_cp(self):
        assert win_percentage_from_white_cp(1000) > 90.0

    def test_large_negative_cp(self):
        assert win_percentage_from_white_cp(-1000) < 10.0

    def test_symmetry(self):
        pos = win_percentage_from_white_cp(300)
        neg = win_percentage_from_white_cp(-300)
        assert pos + neg == pytest.approx(100.0)

    def test_monotonic(self):
        vals = [win_percentage_from_white_cp(cp) for cp in [-500, -100, 0, 100, 500]]
        assert vals == sorted(vals)


class TestAccuracyFromWinPercentage:
    def test_no_drop(self):
        assert accuracy_from_win_percentage(80.0, 80.0) == 100.0
        assert accuracy_from_win_percentage(80.0, 90.0) == 100.0

    def test_small_drop(self):
        acc = accuracy_from_win_percentage(80.0, 79.0)
        assert acc > 90.0

    def test_large_drop(self):
        acc = accuracy_from_win_percentage(90.0, 10.0)
        assert acc < 20.0

    def test_bounds(self):
        acc = accuracy_from_win_percentage(100.0, 0.0)
        assert 0.0 <= acc <= 100.0


class TestGameAccuracy:
    def test_short_game(self):
        w, b = game_accuracy([])
        assert w == 100.0
        assert b == 100.0

    def test_one_move(self):
        w, b = game_accuracy([50.0])
        assert 0.0 <= w <= 100.0
        assert 0.0 <= b <= 100.0

    def test_as_winpcts(self):
        win_pcts = [60.0, 55.0, 65.0, 50.0]
        w, b = game_accuracy(win_pcts, as_winpcts=True)
        assert 0.0 <= w <= 100.0
        assert 0.0 <= b <= 100.0

    def test_start_color_black(self):
        cps = [100, -100, 50, -50]
        w, b = game_accuracy(cps, start_color="black")
        assert 0.0 <= w <= 100.0
        assert 0.0 <= b <= 100.0

    def test_constant_eval(self):
        cps = [0] * 20
        w, b = game_accuracy(cps)
        assert w == pytest.approx(100.0)
        assert b == pytest.approx(100.0)


class TestPhaseAccuracy:
    def test_no_division(self):
        cps = [0] * 10
        div = Division(middle=None, end=None, plies=10)
        phases = phase_accuracy(cps, div)
        assert "opening" in phases
        assert "middlegame" not in phases
        assert "endgame" not in phases

    def test_full_division(self):
        cps = [0] * 80
        div = Division(middle=20, end=60, plies=80)
        phases = phase_accuracy(cps, div)
        assert "opening" in phases
        assert "middlegame" in phases
        assert "endgame" in phases

    def test_only_middle(self):
        cps = [0] * 40
        div = Division(middle=20, end=None, plies=40)
        phases = phase_accuracy(cps, div)
        assert "opening" in phases
        assert "middlegame" in phases
        assert "endgame" not in phases

    def test_middle_beyond_length(self):
        cps = [0] * 10
        div = Division(middle=20, end=30, plies=10)
        phases = phase_accuracy(cps, div)
        assert "opening" in phases
        assert len(phases) == 1
