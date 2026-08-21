// Top-k move prediction at fixed ELO levels for a single position.
// Pure logic (no DOM) so it can be unit-tested in Node.

import { ALL_MOVES, mirrorMove } from './moves.js';

// Fixed ELO range shown in the viewer: 750 to 2500 in steps of 250
const VIEW_ELOS = [750, 1000, 1250, 1500, 1750, 2000, 2250, 2500];

const N_MOVES = 4352;

// Softmax over legal moves only. Returns Float64Array(4352) with zeros at
// illegal moves; computed via log-softmax for numerical stability.
function softmaxMasked(logits, legalMask) {
  let maxLogit = -Infinity;
  for (let m = 0; m < N_MOVES; m++) {
    if (legalMask[m] && logits[m] > maxLogit) maxLogit = logits[m];
  }

  let total = 0;
  const probs = new Float64Array(N_MOVES);
  for (let m = 0; m < N_MOVES; m++) {
    if (!legalMask[m]) continue;
    const p = Math.exp(logits[m] - maxLogit);
    probs[m] = p;
    total += p;
  }
  if (total > 0) {
    for (let m = 0; m < N_MOVES; m++) probs[m] /= total;
  }
  return probs;
}

// Model output index -> real UCI (model vocabulary is white-perspective;
// black-turn positions use mirrored squares).
function indexToUci(idx, isBlackTurn) {
  const uci = ALL_MOVES[idx];
  return isBlackTurn ? mirrorMove(uci) : uci;
}

// UCI -> SAN using the verbose legal-move list of the current board.
function uciToSan(board, uci) {
  for (const move of board.moves({ verbose: true })) {
    if (move.from + move.to + (move.promotion || '') === uci) return move.san;
  }
  return null;
}

// Select the top-k legal moves from one row of model logits.
// Returns [{ san, uci, prob, isPlayed }] sorted by descending probability,
// where isPlayed marks the move actually played in the game (if any).
function selectTopK(logits, legalMask, board, isBlackTurn, playedUci, k) {
  const probs = softmaxMasked(logits, legalMask);

  // Partial selection of the k largest probabilities
  const candidates = [];
  for (let m = 0; m < N_MOVES; m++) {
    if (probs[m] <= 0) continue;
    if (candidates.length < k) {
      candidates.push(m);
      candidates.sort((a, b) => probs[a] - probs[b]); // ascending
    } else if (probs[m] > probs[candidates[0]]) {
      candidates[0] = m;
      candidates.sort((a, b) => probs[a] - probs[b]);
    }
  }
  candidates.reverse();

  const results = [];
  for (const idx of candidates) {
    const uci = indexToUci(idx, isBlackTurn);
    const san = uciToSan(board, uci);
    if (san === null) continue;
    results.push({ san, uci, prob: probs[idx], isPlayed: uci === playedUci });
  }
  return results;
}

export { VIEW_ELOS, softmaxMasked, indexToUci, uciToSan, selectTopK };
