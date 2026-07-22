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
from lczerolens import LczeroBoard, LczeroModel

MODEL_ID = "lczerolens/t1-256x10-distilled-swa-2432500"

model = LczeroModel.from_hf(MODEL_ID)
model.eval()


def wdl_to_white_cp(wdl, is_white_turn):
    wdl = wdl.detach()
    q_stm = wdl[0] + 0.5 * wdl[1]
    q_white = q_stm if is_white_turn else 1.0 - q_stm
    v = float(2 * q_white - 1)
    cp = 290 * v / (1 - 1.1 * v * v) if abs(v) < 0.99 else 290 * v / 0.01
    return cp


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
        lz_board = LczeroBoard(fen=board.fen())
        output = model.forward(lz_board)
        cp = wdl_to_white_cp(output["wdl"].squeeze(), board.turn == chess.WHITE)
        white_pov_cps.append(cp)
        pov_score = chess.engine.PovScore(chess.engine.Cp(int(round(cp))), chess.WHITE)
        annotated_node.set_eval(pov_score)

    return annotated, white_pov_cps, boards


def format_acc(label, w, b):
    return f"{label}: W {w:.2f}%  B {b:.2f}%"


pgn_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("example2.pgn")
annotated_game, white_pov_cps, boards = annotate_game(pgn_path)

division = faithful_division(boards)
middle_str = str(division.middle) if division.middle is not None else "None"
end_str = str(division.end) if division.end is not None else "None"
print(f"Division: opening {middle_str} plies, endgame from ply {end_str}")

game_w, game_b = game_accuracy(white_pov_cps)
print(format_acc("Game", game_w, game_b))

phases = phase_accuracy(white_pov_cps, division)
for phase_name in ("opening", "middlegame", "endgame"):
    if phase_name in phases:
        w, b = phases[phase_name]
        print(format_acc(f"  {phase_name.title():12s}", w, b))

annotated_game.headers["Accuracy"] = f"W {game_w:.2f}% B {game_b:.2f}% accuracy"
print()
print(annotated_game)
