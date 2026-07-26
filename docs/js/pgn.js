// PGN parsing using chess.js
// Mirrors chess_accuracy/pgn_parser.py:parse_pgn_to_positions()

function parsePgnToPositions(pgnText) {
  const Chess = window.Chess;
  const game = new Chess();

  game.loadPgn(pgnText);

  const headers = game.header();
  const history = game.history({ verbose: true });

  if (history.length === 0) {
    return null;
  }

  // Parse clock annotations from PGN text
  const clockMap = parseClockAnnotations(pgnText);

  const positions = [];
  const replayBoard = new Chess();

  for (let i = 0; i < history.length; i++) {
    const move = history[i];

    // Get clock info
    const clockInfo = clockMap[i] || { clkBefore: 300, clkPonder: 0 };

    // Compute move index for model output
    const isBlackTurn = replayBoard.turn() === 'b';
    const moveUci = move.from + move.to + (move.promotion || '');
    const idx = window.__moveIndex(moveUci, isBlackTurn);

    positions.push({
      moveSan: move.san,
      moveUci: moveUci,
      moveIndex: idx,
      isWhiteTurn: !isBlackTurn,
      clkLeftBefore: clockInfo.clkBefore,
      clkPonder: clockInfo.clkPonder,
    });

    replayBoard.move(move.san);
  }

  return {
    positions,
    headers,
    nMoves: history.length,
  };
}

function parseClockAnnotations(pgnText) {
  const clockMap = {};

  // Find the move text section (after headers)
  let moveTextStart = 0;
  const lines = pgnText.split('\n');
  let pastHeaders = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('[')) {
      pastHeaders = true;
      continue;
    }
    if (pastHeaders && line !== '') {
      moveTextStart = pgnText.indexOf(line);
      break;
    }
  }

  const moveText = pgnText.substring(moveTextStart);

  // Match clock annotations: {[%clk H:MM:SS]} and {[%clk_opp H:MM:SS]}
  // These appear before the move they correspond to
  const regex = /\{\s*\[%clk(?:_opp)?\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]\s*\}/g;

  // Parse move numbers and associate clock times
  // Split into tokens: move numbers and moves
  const tokens = moveText.split(/\s+/).filter(t => t.length > 0);

  let currentPly = 0;
  let currentClockBefore = 300;
  let currentClockPonder = 0;
  let pendingClockBefore = null;
  let pendingClockPonder = null;

  for (const token of tokens) {
    // Skip move numbers like "1." "12..."
    if (/^\d+\.+$/.test(token)) continue;

    // Check for clock annotation in the token or surrounding text
    const clkMatch = token.match(/\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]/);
    const clkOppMatch = token.match(/\[%clk_opp\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]/);

    if (clkMatch) {
      const h = parseInt(clkMatch[1]);
      const m = parseInt(clkMatch[2]);
      const s = parseFloat(clkMatch[3]);
      pendingClockBefore = h * 3600 + m * 60 + s;
    }

    if (clkOppMatch) {
      const h = parseInt(clkOppMatch[1]);
      const m = parseInt(clkOppMatch[2]);
      const s = parseFloat(clkOppMatch[3]);
      pendingClockPonder = h * 3600 + m * 60 + s;
    }

    // Check if this token looks like a chess move
    if (isChessMove(token)) {
      clockMap[currentPly] = {
        clkBefore: pendingClockBefore !== null ? pendingClockBefore : currentClockBefore,
        clkPonder: pendingClockPonder !== null ? pendingClockPonder : currentClockPonder,
      };

      if (pendingClockBefore !== null) currentClockBefore = pendingClockBefore;
      if (pendingClockPonder !== null) currentClockPonder = pendingClockPonder;
      pendingClockBefore = null;
      pendingClockPonder = null;

      currentPly++;
    }
  }

  return clockMap;
}

function isChessMove(token) {
  // Remove result markers and annotations
  const cleaned = token.replace(/[#!?+]+$/, '').replace(/\{.*\}/, '');
  if (cleaned.length === 0) return false;

  // Castling
  if (cleaned === 'O-O' || cleaned === 'O-O-O' || cleaned === '0-0' || cleaned === '0-0-0') return true;

  // Standard move: piece? x? file rank (=piece)? 
  if (/^[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?$/.test(cleaned)) return true;

  return false;
}

export { parsePgnToPositions };
