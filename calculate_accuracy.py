import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from lczerolens import LczeroBoard, LczeroModel
from tqdm import tqdm

from chess_accuracy import (
    game_accuracy,
    heuristic_division,
    phase_accuracy,
)
from chess_accuracy.lc0_utils import MODEL_ID, format_acc, wdl_to_white_cp_and_winpct

model = LczeroModel.from_hf(MODEL_ID)
model.eval()


def annotate_game(pgn_path):
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)
    assert game is not None
    annotated = chess.pgn.Game()
    annotated.headers = game.headers.copy()
    annotated.setup(game.board())
    annotated_node = None
    white_pov_cps = []
    white_pov_winpcts = []
    lz_board = LczeroBoard()

    for move in tqdm(game.mainline_moves()):
        annotated_node = annotated.add_variation(move) if annotated_node is None else annotated_node.add_variation(move)
        lz_board.push(move)
        game = game.next()
        assert game is not None
        board = game.board()
        output = model.forward(lz_board)
        cp, win_pct = wdl_to_white_cp_and_winpct(output["wdl"].squeeze(), board.turn == chess.WHITE)
        white_pov_cps.append(cp)
        white_pov_winpcts.append(win_pct)
        pov_score = chess.engine.PovScore(chess.engine.Cp(round(cp)), chess.WHITE)
        annotated_node.set_eval(pov_score)

    return annotated, white_pov_cps, white_pov_winpcts


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
