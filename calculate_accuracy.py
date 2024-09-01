import numpy as np
from tqdm import tqdm
from scipy.stats import hmean
import chess
import chess.engine
import chess.pgn
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

engine = chess.engine.SimpleEngine.popen_uci(Path("bin/fruit/fruit_21.exe"))
engine_limit_config = {"nodes": 3, "time": 1, "depth": 20}


def win_probability_from_pov_score(pov_score):
    centipawns = pov_score.relative.score()
    win_percentage = 50 + 50 * (2 / (1 + np.exp(-0.00368208 * centipawns)) - 1)
    return float(win_percentage)


def annotate_game_with_eval(pgn_path, engine, limit_config):
    annotated_game = chess.pgn.Game()
    game = chess.pgn.read_game(open(pgn_path))
    annotated_game.headers = game.headers
    annotated_node = None
    pov_scores = []
    for move in tqdm(game.mainline_moves()):
        annotated_node = (
            annotated_game.add_variation(move)
            if annotated_node is None
            else annotated_node.add_variation(move)
        )
        game = game.next()
        pov_score = engine.analyse(game.board(), chess.engine.Limit(**limit_config))[
            "score"
        ]
        annotated_node.set_eval(pov_score)
        pov_scores.append(pov_score)
    return annotated_game, pov_scores


def accuracy_from_win_percentage(
    win_percentage_before, win_percentage_after, *, is_black=False
):
    if is_black:
        win_percentage_before = 100 - win_percentage_before
        win_percentage_after = 100 - win_percentage_after
    if win_percentage_after > win_percentage_before:
        return 100
    accuracy = (
        103.1668 * np.exp(-0.04354 * (win_percentage_before - win_percentage_after))
        - 3.1669
        + 1
    )
    return np.clip(accuracy, 0, 100)


def game_accuracy(pov_scores):
    win_probas = [50] + [win_probability_from_pov_score(x) for x in pov_scores]
    print(win_probas)
    accuracy = [
        (win_proba_before, win_proba_after)
        for win_proba_before, win_proba_after in zip(win_probas[:-1], win_probas[1:])
    ]

    # accuracy_from_win_percentage
    white_accuracy = np.array([accuracy_from_win_percentage(*x) for x in accuracy[::2]])
    black_accuracy = np.array(
        [accuracy_from_win_percentage(*x, is_black=True) for x in accuracy[1::2]]
    )

    # smoothing - sim. to lichess
    white_accuracy_smooth = np.mean(
        sliding_window_view(white_accuracy.copy(), min(white_accuracy.shape[0], 10)), 1
    )
    black_accuracy_smooth = np.mean(
        sliding_window_view(black_accuracy.copy(), min(black_accuracy.shape[0], 10)), 1
    )

    # weights via std
    white_weights = np.clip(
        np.std(
            sliding_window_view(
                white_accuracy.copy(), min(white_accuracy.shape[0], 10)
            ),
            1,
        ),
        0.5,
        12,
    )
    black_weights = np.clip(
        np.std(
            sliding_window_view(
                black_accuracy.copy(), min(black_accuracy.shape[0], 10)
            ),
            1,
        ),
        0.5,
        12,
    )

    white_weighted_accuracy = np.mean(
        white_accuracy_smooth
        * ((white_weights * len(white_weights)) / np.sum(white_weights))
    )
    black_weights_accuracy = np.mean(
        black_accuracy_smooth
        * ((black_weights * len(black_weights)) / np.sum(black_weights))
    )
    return float((white_weighted_accuracy + hmean(white_accuracy)) / 2), float(
        (black_weights_accuracy + hmean(black_accuracy)) / 2
    )


annotated_game, pov_scores = annotate_game_with_eval(
    Path("example2.pgn"), engine, engine_limit_config
)
accuracy = game_accuracy(pov_scores)
annotated_game.headers["Accuracy"] = (
    f"W {accuracy[0]:.2f}% B {accuracy[1]:.2f}% accuracy"
)
print(annotated_game)

engine.quit()
