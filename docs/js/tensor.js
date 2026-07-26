// Board tokenization and tensor construction for Maia3
// Mirrors chess_accuracy/maia3/dataset.py + pgn_parser.py

import { getLegalMovesMask } from './moves.js';

// Piece type to channel index
const PIECE_CHANNEL = {
  p: 0, n: 1, b: 2, r: 3, q: 4, k: 5,
  P: 0, N: 1, B: 2, R: 3, Q: 4, K: 5,
};

const HISTORY_LEN = 8;
const CHANNELS_PER_FRAME = 12;
const D_IN = HISTORY_LEN * CHANNELS_PER_FRAME; // 96 (model input dim when include_time_info=False)

// Tokenize board into (64, 12) one-hot encoding
// Board is mirrored if it's black's turn so model always sees from white's perspective
function tokenizeBoard(board) {
  const tokens = new Float32Array(64 * CHANNELS_PER_FRAME);
  const fenParts = board.fen().split(' ');
  const position = fenParts[0];
  const isBlackTurn = fenParts[1] === 'b';

  let fenSquare = 0;
  for (const ch of position) {
    if (ch === '/') continue;
    if (ch >= '1' && ch <= '8') {
      fenSquare += parseInt(ch);
    } else {
      // FEN position goes rank 8->1, file a->h
      const rank = 7 - Math.floor(fenSquare / 8);
      const file = fenSquare % 8;

      // Mirror if black's turn
      const targetSquare = isBlackTurn ? (7 - rank) * 8 + file : rank * 8 + file;

      const colorOffset = (ch >= 'A' && ch <= 'Z') ? 0 : 6; // white=0-5, black=6-11
      const pieceIdx = PIECE_CHANNEL[ch];

      tokens[targetSquare * CHANNELS_PER_FRAME + colorOffset + pieceIdx] = 1;
      fenSquare++;
    }
  }

  return tokens;
}

// Build historical tokens: concatenation of last 8 board tokenizations
// Returns (64, 96) tensor: 64 squares x 8 history frames x 12 channels
// Time channel is NOT included (include_time_info=False)
function getHistoricalTokens(history) {
  const result = new Float32Array(64 * D_IN);
  const padCount = HISTORY_LEN - history.length;

  for (let sq = 0; sq < 64; sq++) {
    for (let h = 0; h < HISTORY_LEN; h++) {
      const srcIdx = Math.max(0, h - padCount);
      const srcOffset = srcIdx * 64 * CHANNELS_PER_FRAME + sq * CHANNELS_PER_FRAME;
      const dstOffset = h * CHANNELS_PER_FRAME;
      for (let c = 0; c < CHANNELS_PER_FRAME; c++) {
        result[sq * D_IN + dstOffset + c] = history[srcOffset + c];
      }
    }
  }

  return result;
}

// Position sampling: prefer middlegame positions (same logic as Python _select_sample_indices)
function selectSampleIndices(totalMoves, nSample) {
  if (nSample <= 0 || nSample >= totalMoves) {
    return Array.from({ length: totalMoves }, (_, i) => i);
  }

  const skipOpen = Math.min(8, Math.floor(totalMoves / 6));
  const skipEnd = Math.min(5, Math.floor(totalMoves / 8));
  const lo = skipOpen;
  const hi = totalMoves - skipEnd;

  if (hi - lo <= nSample) {
    return Array.from({ length: hi - lo }, (_, i) => lo + i);
  }

  const center = lo + (hi - lo) * 0.4;
  const spread = (hi - lo) / 3.0;
  const candidates = [];
  for (let i = lo; i < hi; i++) candidates.push(i);

  const weights = candidates.map(i => Math.exp(-0.5 * ((i - center) / spread) ** 2));

  const chosen = [];
  const remaining = candidates.map((_, i) => i);
  for (let n = 0; n < nSample; n++) {
    const wSum = remaining.reduce((s, i) => s + weights[i], 0);
    let r = Math.random() * wSum;
    let cumul = 0;
    for (let j = 0; j < remaining.length; j++) {
      cumul += weights[remaining[j]];
      if (cumul >= r) {
        chosen.push(candidates[remaining[j]]);
        remaining.splice(j, 1);
        break;
      }
    }
  }

  return chosen.sort((a, b) => a - b);
}

// Build batch tensors for N positions x M ELO values
// Returns { tokens, selfElos, oppoElos, humanMoves, legalMasks, nPositions, nElos }
function buildBatchTensor(positions, eloValues, nSample) {
  const nElo = eloValues.length;
  const sampleIndices = selectSampleIndices(positions.length, nSample);
  const nPos = sampleIndices.length;

  const sampledSet = new Set(sampleIndices);
  const history = []; // Array of Float32Array(64*12)

  const allTokens = [];
  const allHumanMoves = [];
  const allLegalMasks = [];

  // Replay game to build history
  const Chess = window.Chess;
  const gameBoard = new Chess();

  for (let posIdx = 0; posIdx < positions.length; posIdx++) {
    const pos = positions[posIdx];
    const token = tokenizeBoard(gameBoard);
    history.push(token);

    // Keep only last 8
    while (history.length > HISTORY_LEN) history.shift();

    if (sampledSet.has(posIdx)) {
      allTokens.push(getHistoricalTokens(history));

      const isBlackTurn = gameBoard.turn() === 'b';
      allHumanMoves.push(pos.moveIndex);
      allLegalMasks.push(getLegalMovesMask(gameBoard, isBlackTurn));
    }

    gameBoard.move(pos.moveSan, { sloppy: true });
  }

  if (nPos === 0) return null;

  // Stack tokens: (N, 64, 96)
  const tokensN = new Float32Array(nPos * 64 * D_IN);
  for (let i = 0; i < nPos; i++) {
    tokensN.set(allTokens[i], i * 64 * D_IN);
  }

  // Tile tokens for all ELO values: (N*M, 64, 96)
  const tokensBatch = new Float32Array(nPos * nElo * 64 * D_IN);
  for (let i = 0; i < nPos; i++) {
    for (let e = 0; e < nElo; e++) {
      const srcOffset = i * 64 * D_IN;
      const dstOffset = (i * nElo + e) * 64 * D_IN;
      tokensBatch.set(tokensN.subarray(srcOffset, srcOffset + 64 * D_IN), dstOffset);
    }
  }

  // ELO arrays: (N*M,)
  const selfElos = new Float32Array(nPos * nElo);
  const oppoElos = new Float32Array(nPos * nElo);
  for (let i = 0; i < nPos; i++) {
    for (let e = 0; e < nElo; e++) {
      selfElos[i * nElo + e] = eloValues[e];
      oppoElos[i * nElo + e] = eloValues[e];
    }
  }

  // Legal masks: (N, 4352)
  const legalMasks = new Uint8Array(nPos * 4352);
  for (let i = 0; i < nPos; i++) {
    legalMasks.set(allLegalMasks[i], i * 4352);
  }

  return {
    tokens: tokensBatch,
    selfElos,
    oppoElos,
    humanMoves: new Int32Array(allHumanMoves),
    legalMasks,
    nPositions: nPos,
    nElos,
  };
}

// Build batch tensors for a single color's positions (for per-color refinement)
function buildBatchTensorSingleColor(positions, eloValues, colorIsWhite, opponentElo, nSample) {
  const Chess = window.Chess;
  const nElo = eloValues.length;

  const colorPositions = positions.filter(p => p.isWhiteTurn === colorIsWhite);
  if (colorPositions.length === 0) return null;

  const sampleIndices = nSample > 0 && nSample < colorPositions.length
    ? selectSampleIndices(colorPositions.length, nSample)
    : Array.from({ length: colorPositions.length }, (_, i) => i);

  const sampledCount = sampleIndices.length;
  const sampledSet = new Set(sampleIndices);

  const history = [];
  const allTokens = [];
  const allHumanMoves = [];
  const allLegalMasks = [];

  const gameBoard = new Chess();
  let targetSeen = 0;

  for (let posIdx = 0; posIdx < positions.length; posIdx++) {
    const pos = positions[posIdx];
    const token = tokenizeBoard(gameBoard);
    history.push(token);
    while (history.length > HISTORY_LEN) history.shift();

    if (pos.isWhiteTurn === colorIsWhite) {
      if (sampledSet.has(targetSeen)) {
        allTokens.push(getHistoricalTokens(history));
        allHumanMoves.push(pos.moveIndex);
        allLegalMasks.push(getLegalMovesMask(gameBoard, gameBoard.turn() === 'b'));
      }
      targetSeen++;
    }

    gameBoard.move(pos.moveSan, { sloppy: true });
  }

  if (sampledCount === 0) return null;

  const tokensBatch = new Float32Array(sampledCount * nElo * 64 * D_IN);
  for (let i = 0; i < sampledCount; i++) {
    for (let e = 0; e < nElo; e++) {
      const src = i * 64 * D_IN;
      const dst = (i * nElo + e) * 64 * D_IN;
      tokensBatch.set(allTokens[i].subarray(src, src + 64 * D_IN), dst);
    }
  }

  const selfElos = new Float32Array(sampledCount * nElo);
  const oppoElos = new Float32Array(sampledCount * nElo);
  for (let i = 0; i < sampledCount; i++) {
    for (let e = 0; e < nElo; e++) {
      selfElos[i * nElo + e] = eloValues[e];
      oppoElos[i * nElo + e] = opponentElo;
    }
  }

  const legalMasks = new Uint8Array(sampledCount * 4352);
  for (let i = 0; i < sampledCount; i++) {
    legalMasks.set(allLegalMasks[i], i * 4352);
  }

  return {
    tokens: tokensBatch,
    selfElos,
    oppoElos,
    humanMoves: new Int32Array(allHumanMoves),
    legalMasks,
    nPositions: sampledCount,
    nElos,
  };
}

export { tokenizeBoard, getHistoricalTokens, selectSampleIndices, buildBatchTensor, buildBatchTensorSingleColor, D_IN };
