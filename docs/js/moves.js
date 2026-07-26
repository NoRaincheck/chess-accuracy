// Move vocabulary for Maia3 model (4352 moves)
// Mirrors chess_accuracy/maia3/utils.py:get_all_possible_moves()

const FILES = 'abcdefgh';
const RANKS = '12345678';

function squareName(file, rank) {
  return FILES[file] + RANKS[rank];
}

function mirrorSquare(sq) {
  return sq[0] + String(9 - parseInt(sq[1]));
}

function mirrorMove(moveUci) {
  const isPromo = moveUci.length > 4;
  const start = moveUci.slice(0, 2);
  const end = moveUci.slice(2, 4);
  const promo = isPromo ? moveUci.slice(4) : '';
  return mirrorSquare(start) + mirrorSquare(end) + promo;
}

// Generate all 4352 possible moves (same order as Python)
const ALL_MOVES = [];
for (let rank = 0; rank < 8; rank++) {
  for (let file = 0; file < 8; file++) {
    const sq = squareName(file, rank);
    for (let tRank = 0; tRank < 8; tRank++) {
      for (let tFile = 0; tFile < 8; tFile++) {
        ALL_MOVES.push(sq + squareName(tFile, tRank));
      }
    }
  }
}

// Promotions: rank 7 to rank 8 (from white's perspective, after mirroring)
const PROMO_PIECES = ['q', 'r', 'b', 'n'];
for (const fileFrom of FILES) {
  for (const fileTo of FILES) {
    for (const piece of PROMO_PIECES) {
      ALL_MOVES.push(`${fileFrom}7${fileTo}8${piece}`);
    }
  }
}

const ALL_MOVES_DICT = new Map(ALL_MOVES.map((m, i) => [m, i]));

function moveIndex(moveUci, isBlackTurn) {
  const uci = isBlackTurn ? mirrorMove(moveUci) : moveUci;
  const idx = ALL_MOVES_DICT.get(uci);
  if (idx !== undefined) return idx;

  // Fallback for promotions
  if (uci.length > 4) {
    const fromFile = uci.charCodeAt(0) - 97; // 'a' = 97
    const toFile = uci.charCodeAt(2) - 97;
    const pieceMap = { q: 0, r: 1, b: 2, n: 3 };
    return 4096 + fromFile * 32 + toFile * 4 + pieceMap[uci[4]];
  }

  // Standard move fallback
  const fromIdx = (uci.charCodeAt(0) - 97) + (parseInt(uci[1]) - 1) * 8;
  const toIdx = (uci.charCodeAt(2) - 97) + (parseInt(uci[3]) - 1) * 8;
  return fromIdx * 64 + toIdx;
}

function getLegalMovesMask(board, isBlackTurn) {
  const mask = new Uint8Array(4352);
  const legalMoves = board.moves({ verbose: true });
  for (const move of legalMoves) {
    const uci = move.from + move.to + (move.promotion || '');
    const idx = moveIndex(uci, isBlackTurn);
    mask[idx] = 1;
  }
  return mask;
}

export { ALL_MOVES, ALL_MOVES_DICT, mirrorSquare, mirrorMove, moveIndex, getLegalMovesMask };
