// PGN parsing using chess.js
// Mirrors chess_accuracy/pgn_parser.py:parse_pgn_to_positions()

const DEFAULT_CLK_BEFORE = 300;
const DEFAULT_CLK_PONDER = 0;

// Extract [%clk H:MM:SS] and [%clk_opp H:MM:SS] from a PGN comment string,
// falling back to the same defaults as the Python parser.
function parseClockComment(comment) {
  let clkBefore = DEFAULT_CLK_BEFORE;
  let clkPonder = DEFAULT_CLK_PONDER;
  if (!comment) return { clkBefore, clkPonder };

  const clkMatch = comment.match(/\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]/);
  if (clkMatch) {
    clkBefore =
      parseInt(clkMatch[1], 10) * 3600 + parseInt(clkMatch[2], 10) * 60 + parseFloat(clkMatch[3]);
  }

  const clkOppMatch = comment.match(/\[%clk_opp\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]/);
  if (clkOppMatch) {
    clkPonder =
      parseInt(clkOppMatch[1], 10) * 3600 +
      parseInt(clkOppMatch[2], 10) * 60 +
      parseFloat(clkOppMatch[3]);
  }

  return { clkBefore, clkPonder };
}

function parsePgnToPositions(pgnText) {
  const Chess = window.Chess;
  const game = new Chess();

  // python-chess tolerates invalid PGNs (yielding no positions); mirror that
  // instead of throwing.
  try {
    game.loadPgn(pgnText);
  } catch (err) {
    return null;
  }

  const headers = game.header();
  const history = game.history({ verbose: true });

  if (history.length === 0) {
    return null;
  }

  // chess.js keys each comment by the FEN of the position it appears after —
  // the same node comment python-chess associates with the following ply.
  const commentsByFen = {};
  for (const { fen, comment } of game.getComments()) {
    commentsByFen[fen] = comment;
  }

  const positions = [];

  for (let i = 0; i < history.length; i++) {
    const move = history[i];
    // FEN of the position the move is played from (respects a PGN [FEN]
    // header); also the key chess.js files comments under.
    const fenBefore = move.before;

    // Clocks for this position come from the comment attached to the
    // position before the move is played.
    const clockInfo = parseClockComment(commentsByFen[fenBefore]);

    // Compute move index for model output
    const isBlackTurn = fenBefore.split(' ')[1] === 'b';
    const moveUci = move.from + move.to + (move.promotion || '');
    const idx = window.__moveIndex(moveUci, isBlackTurn);

    positions.push({
      moveSan: move.san,
      moveUci: moveUci,
      moveIndex: idx,
      isWhiteTurn: !isBlackTurn,
      fenBefore: fenBefore,
      clkLeftBefore: clockInfo.clkBefore,
      clkPonder: clockInfo.clkPonder,
    });
  }

  return {
    positions,
    headers,
    nMoves: history.length,
  };
}

export { parsePgnToPositions };
