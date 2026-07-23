from collections import namedtuple

import numpy as np

Division = namedtuple("Division", ["middle", "end", "plies"])

WIN_PCT_A = 0.00368208

ACC_A = 103.1668100711649
ACC_K = 0.04354415386753951
ACC_B = -3.166924740191411


def win_percentage_from_white_cp(centipawns):
    return 50 + 50 * (2 / (1 + np.exp(-WIN_PCT_A * centipawns)) - 1)


def accuracy_from_win_percentage(before, after):
    if after >= before:
        return 100.0
    win_diff = before - after
    raw = ACC_A * np.exp(-ACC_K * win_diff) + ACC_B
    return float(np.clip(raw + 1, 0, 100))


def game_accuracy(white_pov_cps, start_color="white", as_winpcts=False):
    n = len(white_pov_cps)
    if as_winpcts:
        win_pcts = [50.0] + [float(wp) for wp in white_pov_cps]
    else:
        win_pcts = [50.0] + [
            float(win_percentage_from_white_cp(cp)) for cp in white_pov_cps
        ]

    window_size = max(2, min(8, n // 10))
    if len(win_pcts) < 2:
        return (100.0, 100.0)
    actual_window = min(window_size, len(win_pcts))

    windows = [win_pcts[:actual_window]] * max(0, actual_window - 2)
    windows += [
        win_pcts[i : i + actual_window]
        for i in range(len(win_pcts) - actual_window + 1)
    ]

    weights = []
    for w in windows:
        std = float(np.std(w, ddof=1)) if len(w) >= 2 else 0.0
        weights.append(max(0.5, min(12.0, std)))

    white_moves = []
    black_moves = []
    white_weighted = []
    black_weighted = []

    start_is_white = start_color == "white"

    for i in range(n):
        prev_pct = win_pcts[i]
        next_pct = win_pcts[i + 1]
        weight = weights[i]

        is_white = (i % 2 == 0) == start_is_white

        if is_white:
            acc = accuracy_from_win_percentage(prev_pct, next_pct)
            white_moves.append(acc)
            white_weighted.append((acc, weight))
        else:
            acc = accuracy_from_win_percentage(next_pct, prev_pct)
            black_moves.append(acc)
            black_weighted.append((acc, weight))

    def color_score(moves, weighted):
        if len(moves) == 0:
            return 100.0
        total_w = sum(w for _, w in weighted)
        if total_w > 0:
            weighted_mean = sum(a * w for a, w in weighted) / total_w
        else:
            weighted_mean = float(np.mean(moves))
        if all(m > 0 for m in moves):
            harmonic_mean = float(len(moves) / sum(1.0 / m for m in moves))
        else:
            harmonic_mean = weighted_mean
        return (weighted_mean + harmonic_mean) / 2.0

    white_acc = color_score(white_moves, white_weighted)
    black_acc = color_score(black_moves, black_weighted)

    return white_acc, black_acc


def phase_accuracy(white_pov_cps, division, start_color="white", as_winpcts=False):
    middle_ply = division.middle
    end_ply = division.end

    def segment_start(ply_offset):
        if ply_offset % 2 == 1:
            return "black" if start_color == "white" else "white"
        return start_color

    phases = {}

    if middle_ply is None or middle_ply >= len(white_pov_cps):
        phases["opening"] = game_accuracy(
            white_pov_cps, start_color, as_winpcts=as_winpcts
        )
        return phases

    opening_cps = white_pov_cps[:middle_ply]
    phases["opening"] = game_accuracy(opening_cps, start_color, as_winpcts=as_winpcts)

    if end_ply is None or end_ply >= len(white_pov_cps):
        middle_cps = white_pov_cps[middle_ply:]
        phases["middlegame"] = game_accuracy(
            middle_cps, segment_start(middle_ply), as_winpcts=as_winpcts
        )
        return phases

    middle_cps = white_pov_cps[middle_ply:end_ply]
    phases["middlegame"] = game_accuracy(
        middle_cps, segment_start(middle_ply), as_winpcts=as_winpcts
    )

    end_cps = white_pov_cps[end_ply:]
    phases["endgame"] = game_accuracy(
        end_cps, segment_start(end_ply), as_winpcts=as_winpcts
    )

    return phases
