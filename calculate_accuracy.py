import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm

from chess_accuracy import (
    game_accuracy,
    phase_accuracy,
    heuristic_division,
)
from lczerolens import LczeroBoard, LczeroModel

MODEL_ID = "lczerolens/t1-256x10-distilled-swa-2432500"
MODEL_ID = "lczerolens/BT3-768x15x24h-swa-2790000"

model = LczeroModel.from_hf(MODEL_ID)
model.eval()


def wdl_to_white_cp_and_winpct(wdl, is_white_turn):
    wdl = wdl.detach()
    q_stm = wdl[0] + 0.5 * wdl[1]
    q_white = q_stm if is_white_turn else 1.0 - q_stm
    win_pct = float(q_white * 100)
    v = float(2 * q_white - 1)
    cp = 290 * v / (1 - 1.1 * v * v) if abs(v) < 0.99 else 290 * v / 0.01
    return cp, win_pct


def annotate_game(pgn_path):
    game = chess.pgn.read_game(open(pgn_path))
    annotated = chess.pgn.Game()
    annotated.headers = game.headers.copy()
    annotated.setup(game.board())
    annotated_node = None
    white_pov_cps = []
    white_pov_winpcts = []
    lz_board = LczeroBoard()

    for move in tqdm(game.mainline_moves()):
        annotated_node = (
            annotated.add_variation(move)
            if annotated_node is None
            else annotated_node.add_variation(move)
        )
        lz_board.push(move)
        game = game.next()
        board = game.board()
        output = model.forward(lz_board)
        cp, win_pct = wdl_to_white_cp_and_winpct(
            output["wdl"].squeeze(), board.turn == chess.WHITE
        )
        white_pov_cps.append(cp)
        white_pov_winpcts.append(win_pct)
        pov_score = chess.engine.PovScore(chess.engine.Cp(int(round(cp))), chess.WHITE)
        annotated_node.set_eval(pov_score)

    return annotated, white_pov_cps, white_pov_winpcts


def format_acc(label, w, b):
    return f"{label}: W {w:.2f}%  B {b:.2f}%"


pgn_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("example2.pgn")
annotated_game, white_pov_cps, white_pov_winpcts = annotate_game(pgn_path)

division = heuristic_division(len(white_pov_cps))
print(f"Division: opening {division.middle} plies, endgame from ply {division.end}")

game_w, game_b = game_accuracy(white_pov_winpcts, as_winpcts=True)
print(format_acc("Game", game_w, game_b))

phases = phase_accuracy(white_pov_winpcts, division, as_winpcts=True)
for phase_name in ("opening", "middlegame", "endgame"):
    if phase_name in phases:
        w, b = phases[phase_name]
        print(format_acc(f"  {phase_name.title():12s}", w, b))

annotated_game.headers["Accuracy"] = f"W {game_w:.2f}% B {game_b:.2f}% accuracy"
print()
print(annotated_game)
