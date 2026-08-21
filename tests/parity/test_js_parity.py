"""Cross-language parity tests: browser JS implementation (docs/js) vs Python (chess_accuracy).

The Python side generates JSON inputs (PGNs, FENs, ELO grids, synthetic logits),
runs them through docs/js via ``js_runner.mjs`` under Node, and compares the JS
outputs against the Python reference implementations.

Run with: uv run pytest tests/parity -v
"""

import json
import math
import shutil
import subprocess
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.pgn
import numpy as np
import pytest

from chess_accuracy.batch_inference import BatchMaia3Inference, _compute_score, _marginal_stats, joint_posterior_2d
from chess_accuracy.maia3.dataset import (
    get_historical_tokens,
    get_legal_moves_mask,
    tokenize_board,
)
from chess_accuracy.maia3.model_registry import resolve_model_spec
from chess_accuracy.maia3.utils import get_all_possible_moves, mirror_move
from chess_accuracy.pgn_parser import (
    build_batch_tensors,
    build_batch_tensors_2d,
    cap_indices,
    informative_indices,
    move_to_index,
    parse_pgn_to_positions,
)

HERE = Path(__file__).parent
RUNNER = HERE / "js_runner.mjs"
MODEL_PATH = HERE.parent.parent / "docs" / "models" / "maia3-5m.onnx"

CFG = SimpleNamespace(**resolve_model_spec("maia3-5m").config)

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required for JS parity tests")


# ── PGN / position fixtures ──────────────────────────────────────────────────


def build_pgn(uci_moves, clock_after=None, extra_comment_after=None, result="1-0"):
    """Build a valid PGN from a legal UCI sequence, optionally annotating clocks."""
    board = chess.Board()
    tokens = []
    for i, uci in enumerate(uci_moves):
        move = chess.Move.from_uci(uci)
        if board.turn == chess.WHITE:
            tokens.append(f"{board.fullmove_number}.")
        tokens.append(board.san(move))
        bits = []
        if clock_after and i in clock_after:
            h, m, s = clock_after[i]
            bits.append(f"[%clk {h}:{m:02d}:{s:05.2f}]")
        if extra_comment_after and i in extra_comment_after:
            bits.append(extra_comment_after[i])
        if bits:
            tokens.append("{" + " ".join(bits) + "}")
        board.push(move)
    tokens.append(result)
    headers = '[Event "Parity"]\n[White "W"]\n[Black "B"]\n[Result "%s"]' % result
    return headers + "\n\n" + " ".join(tokens)


SIMPLE_MOVES = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"]
CASTLE_MOVES = [
    "d2d4", "d7d5", "g1f3", "b8c6", "c1f4", "c8f5", "e2e3", "e7e6",
    "f1d3", "f8d6", "d1e2", "d8d7", "e1g1", "e8c8",
]


def _ucis_from_pgn_file(path):
    with open(path) as f:
        game = chess.pgn.read_game(f)
    board = game.board()
    ucis = []
    for move in game.mainline_moves():
        ucis.append(move.uci())
        board.push(move)
    return ucis


# A real ~58-ply game (the bundled example), guaranteed legal.
LONG_CLOCKED_MOVES = _ucis_from_pgn_file(HERE.parent.parent / "example2.pgn")

SIMPLE_PGN = build_pgn(SIMPLE_MOVES)
CASTLE_PGN = build_pgn(CASTLE_MOVES)
CLOCK_PGN = build_pgn(
    SIMPLE_MOVES[:4],
    clock_after={0: (0, 3, 0.0), 1: (0, 2, 55.0), 2: (0, 2, 58.0), 3: (0, 2, 50.0)},
)
CLK_OPP_PGN = build_pgn(
    SIMPLE_MOVES[:4],
    clock_after={1: (0, 2, 55.0)},
    extra_comment_after={1: "[%clk_opp 0:02:40]", 3: "[%clk_opp 0:02:30]"},
)
LONG_CLOCK_PGN = build_pgn(
    LONG_CLOCKED_MOVES,
    clock_after={i: (0, 3, 60.0 - i) for i in range(len(LONG_CLOCKED_MOVES))},
)
PROMO_PGN = (
    '[Event "Promo"]\n[White "W"]\n[Black "B"]\n[SetUp "1"]'
    '\n[FEN "8/P6k/8/8/8/8/7K/8 w - - 0 1"]\n\n'
    "1. a8=Q Kh6 2. Qa6+ *"
)


def replay_fens(uci_moves):
    board = chess.Board()
    fens = [board.fen()]
    for uci in uci_moves:
        board.push(chess.Move.from_uci(uci))
        fens.append(board.fen())
    return fens


def collect_test_fens():
    fens = []
    fens.extend(replay_fens([]))
    fens.extend(replay_fens(SIMPLE_MOVES[:5]))
    fens.extend(replay_fens(["f2f3", "e7e5", "g2g4", "d8h4"]))  # checkmate
    fens.extend(replay_fens(["e2e4", "a7a6", "e4e5", "d7d5"]))  # en passant available
    fens.extend(replay_fens(["g1f3", "g8f6", "g2g3", "g7g6", "f1h1", "f8h8"]))  # partial castling rights
    promo = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    fens.append(promo.fen())
    promo.push(chess.Move.from_uci("a7a8q"))
    fens.append(promo.fen())  # black to move, white queen on a8
    promo_b = chess.Board("8/7k/8/8/8/8/p6K/8 b - - 0 1")
    fens.append(promo_b.fen())
    promo_b.push(chess.Move.from_uci("a2a1n"))
    fens.append(promo_b.fen())  # white to move, black knight on a1
    return fens


MIRROR_UCIS = ["a1a2", "h8h1", "e2e4", "e7e5", "d7d8q", "b1b3", "g2h1q"]


def _posterior_cases():
    """Synthetic score surfaces for posterior/CI parity checks."""

    def gaussian_bump(n_w, n_b):
        w = np.linspace(300.0, 3000.0, n_w)
        b = np.linspace(300.0, 3000.0, n_b)
        ww, bb = np.meshgrid(w, b, indexing="ij")
        return (
            -((ww - 1700.0) ** 2) / (2 * 400.0**2) - ((bb - 1300.0) ** 2) / (2 * 350.0**2),
            w,
            b,
        )

    surf1, w1, b1 = gaussian_bump(4, 4)  # stage-B-like fine window
    surf2, w2, b2 = gaussian_bump(7, 5)
    rng = np.random.default_rng(7)
    surf3 = rng.normal(size=(6, 9))  # noisy surface
    return [
        {
            "surface": surf1.ravel().tolist(),
            "whiteElos": w1.tolist(),
            "blackElos": b1.tolist(),
        },
        {
            "surface": surf2.ravel().tolist(),
            "whiteElos": w2.tolist(),
            "blackElos": b2.tolist(),
        },
        {
            "surface": surf3.ravel().tolist(),
            "whiteElos": np.linspace(800.0, 2200.0, 6).tolist(),
            "blackElos": np.linspace(900.0, 2400.0, 9).tolist(),
        },
        # Degenerate single-cell surface.
        {"surface": [3.5], "whiteElos": [1500.0], "blackElos": [1600.0]},
    ]


# ── Harness plumbing ─────────────────────────────────────────────────────────


def run_node(mode, payload, tmp_dir):
    in_path = Path(tmp_dir) / f"{mode}_inputs.json"
    out_path = Path(tmp_dir) / f"{mode}_outputs.json"
    in_path.write_text(json.dumps(payload))
    proc = subprocess.run(
        ["node", str(RUNNER), mode, str(in_path), str(out_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"js_runner failed:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(out_path.read_text())


def py_batch(pgn, elo_values):
    positions = parse_pgn_to_positions(pgn)
    assert positions, f"python failed to parse fixture PGN: {pgn[:60]}..."
    return build_batch_tensors(positions, np.array(elo_values, dtype=np.float32), CFG, n_sample=0)


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("parity")

    games = [
        {"pgn": LONG_CLOCK_PGN, "eloValues": [800, 1200, 1600]},
        {"pgn": SIMPLE_PGN, "eloValues": [300, 1500, 3000]},
        {"pgn": CASTLE_PGN, "eloValues": [800, 1200]},
    ]
    inputs = {
        "mirrorUcis": MIRROR_UCIS,
        "fens": collect_test_fens(),
        "histories": [
            {"fens": replay_fens(SIMPLE_MOVES)[:2]},
            {"fens": replay_fens(SIMPLE_MOVES)[:3]},
            {"fens": replay_fens(SIMPLE_MOVES)},  # full history (8 frames)
            {"fens": replay_fens(LONG_CLOCKED_MOVES)},  # > 8 frames trims deque
        ],
        "games": games,
        "singleColorGames": [
            {"pgn": LONG_CLOCK_PGN, "eloValues": [1500], "colorIsWhite": True, "opponentElo": 1700},
            {"pgn": LONG_CLOCK_PGN, "eloValues": [1600], "colorIsWhite": False, "opponentElo": 1400},
        ],
        "samplingCases": [
            {"totalMoves": 10, "nSample": 0},
            {"totalMoves": 10, "nSample": 10},
            {"totalMoves": 10, "nSample": 99},
            {"totalMoves": 58, "nSample": 20},  # random branch: only shape-checked
        ],
        "capCases": [
            {"indices": list(range(10)), "maxPositions": 0},
            {"indices": list(range(10)), "maxPositions": 10},
            {"indices": list(range(10)), "maxPositions": 3},
            {"indices": list(range(7)), "maxPositions": 1},
            {"indices": [5, 9, 13, 21, 40, 57, 58, 60, 77, 91], "maxPositions": 4},
        ],
        "jointGames": [
            {
                "pgn": LONG_CLOCK_PGN,
                "whiteElos": [800, 1500, 2200],
                "blackElos": [1000, 1800],
                "opts": {"skipOpening": 8, "minLegalMoves": 2, "maxPositions": 0},
            },
            {
                "pgn": CASTLE_PGN,
                "whiteElos": [1200, 2000],
                "blackElos": [900, 1600, 2400],
                "opts": {"skipOpening": 8, "minLegalMoves": 2, "maxPositions": 0},
            },
            {
                # cap kicks in: long game capped to 6 positions
                "pgn": LONG_CLOCK_PGN,
                "whiteElos": [1300],
                "blackElos": [1700],
                "opts": {"skipOpening": 8, "minLegalMoves": 2, "maxPositions": 6},
            },
        ],
        "gridCases": [
            {"lo": 300, "hi": 3000, "step": 700, "center": 1500, "margin": 300, "nPoints": 4},
            {"lo": 300, "hi": 3000, "step": 550, "center": 2900, "margin": 400, "nPoints": 7},
            {"lo": 300, "hi": 3000, "step": 2700, "center": 300, "margin": 50, "nPoints": 1},
        ],
        "upsampleCases": [
            {
                "src": [1.0, 2.0, 0.5, -1.0, 3.0, 2.5, 0.0, 1.5, 4.0],
                "srcW": 3,
                "srcB": 3,
                "dstW": 7,
                "dstB": 9,
            },
            {"src": [2.0], "srcW": 1, "srcB": 1, "dstW": 5, "dstB": 5},
            {"src": [0.0, 1.0, 4.0, 9.0], "srcW": 2, "srcB": 2, "dstW": 4, "dstB": 1},
        ],
        "posteriorCases": _posterior_cases(),
        "pgns": [SIMPLE_PGN, CLOCK_PGN, CLK_OPP_PGN, PROMO_PGN, CASTLE_PGN, "", "not a pgn"],
    }
    outputs = run_node("run", inputs, tmp_dir)
    return inputs, outputs, tmp_dir


# ── A/B. Move vocabulary & mirroring ─────────────────────────────────────────


class TestMoveVocabulary:
    def test_all_moves_identical(self, parity):
        _, outs, _ = parity
        assert outs["allMoves"] == get_all_possible_moves()
        assert len(outs["allMoves"]) == 4352

    def test_mirror_move(self, parity):
        _, outs, _ = parity
        for uci in MIRROR_UCIS:
            assert outs["mirror"][uci] == mirror_move(uci), uci


# ── C. Move indexing ─────────────────────────────────────────────────────────


class TestMoveIndex:
    def test_every_legal_move_maps_identically(self, parity):
        _, outs, _ = parity
        for fen in collect_test_fens():
            board = chess.Board(fen)
            expected = {}
            for move in board.legal_moves:
                expected[move.uci()] = move_to_index(move, board)
            got = outs["moveIndex"][fen]
            assert got == expected, f"move index mismatch at {fen}"


# ── D/E. Tokenization & legal masks ──────────────────────────────────────────


class TestTokenizeBoard:
    def test_tokenization_identical(self, parity):
        _, outs, _ = parity
        for fen in collect_test_fens():
            expected = tokenize_board(chess.Board(fen)).numpy().flatten()
            got = np.array(outs["tokenize"][fen], dtype=np.float32)
            np.testing.assert_array_equal(got, expected, err_msg=f"tokenize mismatch at {fen}")


class TestLegalMovesMask:
    def test_masks_identical(self, parity):
        _, outs, _ = parity
        all_moves_dict = {m: i for i, m in enumerate(get_all_possible_moves())}
        for fen in collect_test_fens():
            expected = get_legal_moves_mask(chess.Board(fen), all_moves_dict)
            got = np.array(outs["legalMask"][fen], dtype=np.bool_)
            np.testing.assert_array_equal(got, expected.numpy(), err_msg=f"mask mismatch at {fen}")


# ── F. Historical tokens ─────────────────────────────────────────────────────


class TestHistoricalTokens:
    def test_historical_tokens_identical(self, parity):
        _, outs, _ = parity
        for i, hist in enumerate(parity[0]["histories"]):
            dq = deque(maxlen=CFG.history)
            for fen in hist["fens"]:
                dq.append(tokenize_board(chess.Board(fen)))
            expected = get_historical_tokens(
                dq, CFG, base=300.0, inc=0.0, clk_left_before=300.0, clk_ponder=12.0
            )
            # include_time_info=False appends a clk_ponder channel that the model
            # never sees (build_batch_tensors slices it off) — slice here too.
            expected = expected[:, : 12 * CFG.history].numpy().flatten()
            got = np.array(outs["historical"][i], dtype=np.float32)
            np.testing.assert_array_equal(got, expected, err_msg=f"history {i} mismatch")


# ── G. Batch tensor construction ─────────────────────────────────────────────


class TestBuildBatchTensors:
    def test_batches_identical(self, parity):
        inputs, outs, _ = parity
        for i, game in enumerate(inputs["games"]):
            expected = py_batch(game["pgn"], game["eloValues"])
            got = outs["batches"][i]
            assert got is not None, f"JS returned null batch for game {i}"
            n = expected["n_positions"] * expected["n_elos"]

            assert got["nPositions"] == expected["n_positions"]
            assert got["nElos"] == expected["n_elos"]
            np.testing.assert_array_equal(
                np.array(got["humanMoves"], dtype=np.int64), expected["human_moves"]
            )
            np.testing.assert_array_equal(
                np.array(got["selfElos"], dtype=np.float32), expected["self_elos"].numpy()
            )
            np.testing.assert_array_equal(
                np.array(got["oppoElos"], dtype=np.float32), expected["oppo_elos"].numpy()
            )
            tokens = np.array(got["tokens"], dtype=np.float32).reshape(n, 64, 96)
            np.testing.assert_array_equal(tokens, expected["tokens"].numpy())
            masks = np.array(got["legalMasks"], dtype=np.bool_).reshape(expected["n_positions"], 4352)
            np.testing.assert_array_equal(masks, expected["legal_masks"].numpy())


# ── I. PGN parsing ───────────────────────────────────────────────────────────


class TestParsePgn:
    def _py_positions(self, pgn):
        positions = parse_pgn_to_positions(pgn)
        return [
            {
                "moveUci": p["move"].uci(),
                "isWhiteTurn": p["is_white_turn"],
                "clkLeftBefore": p["clk_left_before"],
                "clkPonder": p["clk_ponder"],
            }
            for p in positions
        ]

    def test_positions_identical(self, parity):
        _, outs, _ = parity
        pgns = parity[0]["pgns"]
        names = ["simple", "clock", "clk_opp", "promo", "castle", "empty", "invalid"]
        for name, pgn, got in zip(names, pgns, outs["pgns"]):
            expected = self._py_positions(pgn)
            if not expected:
                assert got in (None,) or got.get("error"), f"{name}: JS should reject, got {got}"
                continue
            assert isinstance(got, dict) and "positions" in got, f"{name}: JS failed: {got}"
            assert len(got["positions"]) == len(expected), (
                f"{name}: JS parsed {len(got['positions'])} positions, python {len(expected)}"
            )
            for j, (g, e) in enumerate(zip(got["positions"], expected)):
                assert g["moveUci"] == e["moveUci"], f"{name} pos {j}: {g['moveUci']} != {e['moveUci']}"
                assert g["isWhiteTurn"] == e["isWhiteTurn"], f"{name} pos {j}: turn mismatch"
                assert g["clkLeftBefore"] == pytest.approx(e["clkLeftBefore"]), (
                    f"{name} pos {j}: clkLeftBefore {g['clkLeftBefore']} != {e['clkLeftBefore']}"
                )
                assert g["clkPonder"] == pytest.approx(e["clkPonder"]), (
                    f"{name} pos {j}: clkPonder {g['clkPonder']} != {e['clkPonder']}"
                )


# ── G2. Single-color batch tensors ───────────────────────────────────────────


class TestSingleColorBatch:
    def test_matches_2d_builder_restricted_to_color(self, parity):
        inputs, outs, _ = parity
        pgn = inputs["singleColorGames"][0]["pgn"]
        positions = parse_pgn_to_positions(pgn)
        for i, game in enumerate(inputs["singleColorGames"]):
            color_is_white = game["colorIsWhite"]
            self_elo = game["eloValues"][0]
            oppo_elo = game["opponentElo"]

            # W=B=1 grid: white-turn rows have self=white elo / oppo=black elo,
            # black-turn rows the reverse — exactly what the JS single-color
            # builder produces for the matching color.
            w = np.array([self_elo if color_is_white else oppo_elo], dtype=np.float32)
            b = np.array([oppo_elo if color_is_white else self_elo], dtype=np.float32)
            py2d = build_batch_tensors_2d(positions, w, b, CFG, n_sample=0)
            keep = py2d["is_white_turn"] if color_is_white else ~py2d["is_white_turn"]

            got = outs["singleColorBatches"][i]
            assert got is not None, f"single-color batch {i} was null"
            n_keep = int(keep.sum())
            assert got["nPositions"] == n_keep
            np.testing.assert_array_equal(
                np.array(got["humanMoves"], dtype=np.int64), py2d["human_moves"][keep]
            )
            tokens = np.array(got["tokens"], dtype=np.float32).reshape(n_keep, 64, 96)
            np.testing.assert_array_equal(tokens, py2d["tokens"].numpy()[keep])
            np.testing.assert_array_equal(
                np.array(got["selfElos"], dtype=np.float32), py2d["self_elos"].numpy()[keep]
            )
            np.testing.assert_array_equal(
                np.array(got["oppoElos"], dtype=np.float32), py2d["oppo_elos"].numpy()[keep]
            )
            masks = np.array(got["legalMasks"], dtype=np.bool_).reshape(n_keep, 4352)
            np.testing.assert_array_equal(masks, py2d["legal_masks"].numpy()[keep])


class TestSamplingIndices:
    def test_deterministic_branches_identical(self, parity):
        import math

        from chess_accuracy.pgn_parser import _select_sample_indices

        _, outs, _ = parity
        for case, got in zip(parity[0]["samplingCases"], outs["sampling"]):
            total, n = case["totalMoves"], case["nSample"]
            expected = _select_sample_indices(total, n)
            skip_open = min(8, total // 6)
            skip_end = min(5, total // 8)
            lo, hi = skip_open, total - skip_end
            is_random_branch = 0 < n < total and hi - lo > n
            if is_random_branch:
                # RNG streams differ between languages; only check shape/validity.
                assert len(got) == n
                assert got == sorted(got)
                assert all(0 <= i < total for i in got)
            else:
                assert got == expected, f"sampling mismatch for {case}: {got} != {expected}"


# ── K. Deterministic position cap ────────────────────────────────────────────


class TestCapIndices:
    def test_caps_identical(self, parity):
        _, outs, _ = parity
        for case, got in zip(parity[0]["capCases"], outs["capIndices"]):
            expected = cap_indices(case["indices"], case["maxPositions"])
            assert got == expected, f"cap mismatch for {case}: {got} != {expected}"


class TestJointBatch:
    def test_matches_2d_builder_with_filtered_capped_indices(self, parity):
        inputs, outs, _ = parity
        for i, game in enumerate(inputs["jointGames"]):
            positions = parse_pgn_to_positions(game["pgn"])
            opts = game["opts"]
            indices = cap_indices(
                informative_indices(positions, skip_opening=opts["skipOpening"], min_legal_moves=opts["minLegalMoves"]),
                opts["maxPositions"],
            )
            w = np.array(game["whiteElos"], dtype=np.float32)
            b = np.array(game["blackElos"], dtype=np.float32)
            expected = build_batch_tensors_2d(positions, w, b, CFG, n_sample=0, indices=indices)

            got = outs["jointBatches"][i]
            assert got is not None, f"joint batch {i} was null"
            n_grid = len(w) * len(b)
            n = expected["n_positions"] * n_grid

            assert got["nPositions"] == expected["n_positions"]
            assert got["nElos"] == n_grid
            np.testing.assert_array_equal(
                np.array(got["humanMoves"], dtype=np.int64), expected["human_moves"]
            )
            np.testing.assert_array_equal(
                np.array(got["selfElos"], dtype=np.float32), expected["self_elos"].numpy()
            )
            np.testing.assert_array_equal(
                np.array(got["oppoElos"], dtype=np.float32), expected["oppo_elos"].numpy()
            )
            tokens = np.array(got["tokens"], dtype=np.float32).reshape(n, 64, 96)
            np.testing.assert_array_equal(tokens, expected["tokens"].numpy())
            masks = np.array(got["legalMasks"], dtype=np.bool_).reshape(expected["n_positions"], 4352)
            np.testing.assert_array_equal(masks, expected["legal_masks"].numpy())


# ── M. Surface helpers (grid construction + bilinear upsample) ───────────────


def _bilinear_ref(src, src_w, src_b, dst_w, dst_b):
    out = np.zeros((dst_w, dst_b), dtype=np.float64)
    x_max = max(src_w - 2, 0)
    y_max = max(src_b - 2, 0)
    for i in range(dst_w):
        x = 0.0 if dst_w == 1 else i * (src_w - 1) / (dst_w - 1)
        x0 = min(max(int(np.floor(x)), 0), x_max)
        x1 = min(x0 + 1, src_w - 1)
        fx = x - x0
        for j in range(dst_b):
            y = 0.0 if dst_b == 1 else j * (src_b - 1) / (dst_b - 1)
            y0 = min(max(int(np.floor(y)), 0), y_max)
            y1 = min(y0 + 1, src_b - 1)
            fy = y - y0
            out[i, j] = (
                src[x0, y0] * (1 - fx) * (1 - fy)
                + src[x0, y1] * (1 - fx) * fy
                + src[x1, y0] * fx * (1 - fy)
                + src[x1, y1] * fx * fy
            )
    return out


class TestSurfaceHelpers:
    def _make_grid_ref(self, lo, hi, step):
        n = math.ceil((hi - lo) / step)
        grid = lo + step * np.arange(n + 1, dtype=np.float64)
        return np.clip(grid, lo, hi).tolist()

    def test_grids_match_python(self, parity):
        _, outs, _ = parity
        for case, got in zip(parity[0]["gridCases"], outs["grids"]):
            expected_make = self._make_grid_ref(case["lo"], case["hi"], case["step"])
            assert got["make"] == pytest.approx(expected_make), f"makeGrid mismatch: {case}"
            expected_window = self._window_grid_ref(
                case["center"], case["margin"], case["nPoints"], case["lo"], case["hi"]
            )
            assert got["window"] == pytest.approx(expected_window), f"windowGrid mismatch: {case}"

    def _window_grid_ref(self, center, margin, n_points, lo, hi):
        start = max(lo, center - margin)
        end = min(hi, center + margin)
        if n_points <= 1:
            return [start]
        return [start + (end - start) * k / (n_points - 1) for k in range(n_points)]

    def test_upsample_identical(self, parity):
        _, outs, _ = parity
        for case, got in zip(parity[0]["upsampleCases"], outs["upsamples"]):
            src = np.array(case["src"], dtype=np.float64).reshape(case["srcW"], case["srcB"])
            expected = _bilinear_ref(src, case["srcW"], case["srcB"], case["dstW"], case["dstB"])
            dst = np.array(got["dst"], dtype=np.float64).reshape(case["dstW"], case["dstB"])
            np.testing.assert_allclose(dst, expected, rtol=1e-12, atol=1e-12)

            flat = expected
            row_sums = flat.sum(axis=1)
            col_sums = flat.sum(axis=0)
            assert got["wi"] == int(np.argmax(row_sums))
            assert got["bi"] == int(np.argmax(col_sums))


class TestPosteriors:
    def test_joint_posterior_and_marginal_stats_match_python(self, parity):
        inputs, outs, _ = parity
        for case, got in zip(inputs["posteriorCases"], outs["posteriors"]):
            w_vals = np.array(case["whiteElos"], dtype=np.float64)
            b_vals = np.array(case["blackElos"], dtype=np.float64)
            surface = np.array(case["surface"], dtype=np.float64).reshape(len(w_vals), len(b_vals))

            expected = joint_posterior_2d(surface, w_vals, b_vals)
            np.testing.assert_allclose(
                np.array(got["posterior"]), expected.ravel(), rtol=1e-12, atol=1e-15
            )

            for color, post, vals, mean_key, std_key, ci_key in (
                ("w", expected.sum(axis=1), w_vals, "wMean", "wStd", "wCi"),
                ("b", expected.sum(axis=0), b_vals, "bMean", "bStd", "bCi"),
            ):
                mean, std, (ci_lo, ci_hi) = _marginal_stats(post, vals)
                assert got[mean_key] == pytest.approx(mean), f"{color} mean: {case}"
                assert got[std_key] == pytest.approx(std), f"{color} std: {case}"
                assert got[ci_key] == pytest.approx([ci_lo, ci_hi]), f"{color} ci: {case}"


# ── H/J. Scoring & end-to-end through the real ONNX model ────────────────────


def _random_scoring_case(seed=42, n_pos=7, n_elo=4, vocab=4352):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n_pos, n_elo, vocab))
    masks = rng.random((n_pos, vocab)) < 0.7
    human_moves = np.empty(n_pos, dtype=np.int64)
    for p in range(n_pos):
        legal = np.flatnonzero(masks[p])
        human_moves[p] = rng.choice(legal)
        masks[p, human_moves[p]] = True
    masked = np.where(masks[:, None, :], logits, -np.inf)
    return logits, masks, human_moves, masked


class TestComputeScore:
    def test_scores_identical_on_synthetic_logits(self, parity, tmp_path):
        _, _, shared_dir = parity
        logits, masks, human_moves, masked = _random_scoring_case()
        expected = _compute_score(masked, human_moves, len(human_moves), alpha=0.6)

        payload = {
            "logits": logits.flatten().tolist(),
            "humanMoves": human_moves.tolist(),
            "legalMasks": masks.flatten().astype(np.uint8).tolist(),
            "nPos": int(len(human_moves)),
            "nElo": int(logits.shape[1]),
        }
        got = run_node("score", payload, shared_dir)["scores"]
        np.testing.assert_allclose(np.array(got), expected, rtol=1e-12, atol=1e-12)


class TestEndToEnd:
    def test_full_pipeline_through_real_model(self, parity):
        inputs, outs, shared_dir = parity
        game = inputs["games"][0]  # long clocked game
        elos = np.array(game["eloValues"], dtype=np.float32)

        py_b = py_batch(game["pgn"], game["eloValues"])
        js_b = outs["batches"][0]

        engine = BatchMaia3Inference(str(MODEL_PATH))

        def scores_for(tokens, self_elos, oppo_elos, human_moves, legal_masks):
            n_pos, n_elo = py_b["n_positions"], py_b["n_elos"]
            logits_move, _, _ = engine.predict(tokens, self_elos, oppo_elos)
            logits = logits_move.reshape(n_pos, n_elo, -1)
            masked = np.where(legal_masks.numpy()[:, np.newaxis, :], logits, -np.inf)
            return _compute_score(masked, human_moves, n_pos, alpha=0.6), logits

        scores_py, logits_py = scores_for(
            py_b["tokens"], py_b["self_elos"], py_b["oppo_elos"], py_b["human_moves"], py_b["legal_masks"]
        )

        # Same model, JS-built tensors: must give bit-identical scores.
        scores_js_tensors, _ = scores_for(
            torch_from_flat(js_b["tokens"], (py_b["n_positions"] * py_b["n_elos"], 64, 96)),
            torch_from_flat(js_b["selfElos"], (py_b["n_positions"] * py_b["n_elos"],)),
            torch_from_flat(js_b["oppoElos"], (py_b["n_positions"] * py_b["n_elos"],)),
            np.array(js_b["humanMoves"], dtype=np.int64),
            torch_from_flat(js_b["legalMasks"], (py_b["n_positions"], 4352)).bool(),
        )
        np.testing.assert_array_equal(scores_js_tensors, scores_py)

        # JS scoring function fed the Python model's logits: must agree too.
        payload = {
            "logits": logits_py.astype(np.float64).flatten().tolist(),
            "humanMoves": np.array(js_b["humanMoves"]).tolist(),
            "legalMasks": np.array(js_b["legalMasks"]).flatten().astype(np.uint8).tolist(),
            "nPos": int(py_b["n_positions"]),
            "nElo": int(py_b["n_elos"]),
        }
        scores_js_scoring = np.array(run_node("score", payload, shared_dir)["scores"])
        np.testing.assert_allclose(scores_js_scoring, scores_py, rtol=1e-9, atol=1e-9)

        best_py = int(np.argmax(scores_py))
        best_js = int(np.argmax(scores_js_scoring))
        assert best_py == best_js


def torch_from_flat(flat, shape):
    import torch

    return torch.tensor(np.array(flat, dtype=np.float32).reshape(shape))
