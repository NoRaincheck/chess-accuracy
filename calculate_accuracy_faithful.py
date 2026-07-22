import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm

from chess_accuracy import (
    game_accuracy,
    phase_accuracy,
    faithful_division,
)

engine = chess.engine.SimpleEngine.popen_uci(Path("bin/stockfish/stockfish"))
engine_limit_config = {"nodes": 3, "time": 1, "depth": 20}


def annotate_game(pgn_path):
    game = chess.pgn.read_game(open(pgn_path))
    annotated = chess.pgn.Game()
    annotated.headers = game.headers.copy()
    annotated.setup(game.board())
    annotated_node = None
    white_pov_cps = []
    boards = [game.board().copy()]

    for move in tqdm(game.mainline_moves()):
        annotated_node = (
            annotated.add_variation(move)
            if annotated_node is None
            else annotated_node.add_variation(move)
        )
        game = game.next()
        board = game.board().copy()
        boards.append(board)
        info = engine.analyse(board, chess.engine.Limit(**engine_limit_config))
        score = info["score"]
        cp = score.white().score()
        if cp is None:
            cp = 0
        white_pov_cps.append(cp)
        annotated_node.set_eval(score)

    return annotated, white_pov_cps, boards


def format_acc(label, w, b):
    return f"{label}: W {w:.2f}%  B {b:.2f}%"


pgn_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("example2.pgn")
annotated_game, white_pov_cps, boards = annotate_game(pgn_path)

division = faithful_division(boards)
middle_str = str(division.middle) if division.middle is not None else "None"
end_str = str(division.end) if division.end is not None else "None"
print(f"Division: opening {middle_str} plies, "
      f"endgame from ply {end_str}")

game_w, game_b = game_accuracy(white_pov_cps)
print(format_acc("Game", game_w, game_b))

phases = phase_accuracy(white_pov_cps, division)
for phase_name in ("opening", "middlegame", "endgame"):
    if phase_name in phases:
        w, b = phases[phase_name]
        print(format_acc(f"  {phase_name.title():12s}", w, b))

annotated_game.headers["Accuracy"] = (
    f"W {game_w:.2f}% B {game_b:.2f}% accuracy"
)
print()
print(annotated_game)

engine.quit()
