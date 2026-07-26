// Scoring and ELO optimization
// Mirrors chess_accuracy/batch_inference.py:_compute_score and estimate_elo.py

const ALPHA = 0.6;
const MIN_ELO = 300;
const MAX_ELO = 3000;
const FIDELITY = 50;

// Compute blended top-1 accuracy + MRR score
// logitsMasked: (N, M, 4352) - masked logits
// humanMoves: (N,) - human move indices
// Returns: (M,) - score per ELO value
function computeScore(logitsMasked, humanMoves, nPos) {
  if (nPos === 0) {
    return new Float32Array(logitsMasked.length / (4352));
  }

  const nElo = logitsMasked.shape[1];
  const scores = new Float32Array(nElo);

  for (let e = 0; e < nElo; e++) {
    let top1Correct = 0;
    let mrrSum = 0;

    for (let p = 0; p < nPos; p++) {
      const baseIdx = (p * nElo + e) * 4352;
      const humanLogit = logitsMasked.data[baseIdx + humanMoves[p]];

      // Find argmax (top-1)
      let maxLogit = -Infinity;
      let maxIdx = -1;
      for (let m = 0; m < 4352; m++) {
        const val = logitsMasked.data[baseIdx + m];
        if (val > maxLogit) {
          maxLogit = val;
          maxIdx = m;
        }
      }

      if (maxIdx === humanMoves[p]) top1Correct++;

      // Count rank of human move
      let rank = 0;
      for (let m = 0; m < 4352; m++) {
        if (logitsMasked.data[baseIdx + m] >= humanLogit) rank++;
      }
      mrrSum += 1.0 / rank;
    }

    const top1Acc = top1Correct / nPos;
    const mrr = mrrSum / nPos;
    scores[e] = ALPHA * top1Acc + (1 - ALPHA) * mrr;
  }

  return scores;
}

// Compute score without full (N, M, 4352) materialization
// Processes one ELO at a time to save memory
function computeScoreStreaming(logitsAllElos, humanMoves, legalMasks, nPos, nElo) {
  const scores = new Float32Array(nElo);
  if (nPos === 0) return scores;

  for (let e = 0; e < nElo; e++) {
    let top1Correct = 0;
    let mrrSum = 0;

    for (let p = 0; p < nPos; p++) {
      const logitBase = (p * nElo + e) * 4352;
      const maskBase = p * 4352;
      const humanIdx = humanMoves[p];

      // Apply legal mask
      let maxLogit = -Infinity;
      let maxIdx = -1;
      let humanLogit = -Infinity;

      for (let m = 0; m < 4352; m++) {
        if (!legalMasks[maskBase + m]) continue;
        const val = logitsAllElos[logitBase + m];
        if (val > maxLogit) {
          maxLogit = val;
          maxIdx = m;
        }
        if (m === humanIdx) humanLogit = val;
      }

      if (maxIdx === humanIdx) top1Correct++;

      // Rank of human move among legal moves
      let rank = 0;
      for (let m = 0; m < 4352; m++) {
        if (!legalMasks[maskBase + m]) continue;
        if (logitsAllElos[logitBase + m] >= humanLogit) rank++;
      }
      mrrSum += 1.0 / rank;
    }

    scores[e] = ALPHA * (top1Correct / nPos) + (1 - ALPHA) * (mrrSum / nPos);
  }

  return scores;
}

// 1D ELO sweep: find best ELO assuming both players have the same rating
function estimateElo1D(scores, eloValues) {
  let bestIdx = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > bestScore) {
      bestScore = scores[i];
      bestIdx = i;
    }
  }
  return { bestElo: eloValues[bestIdx], bestScore, bestIdx };
}

// Round helpers
function roundUp(x) { return Math.ceil(x / 100) * 100; }
function roundDown(x) { return Math.floor(x / 100) * 100; }

export { computeScore, computeScoreStreaming, estimateElo1D, roundUp, roundDown, MIN_ELO, MAX_ELO, FIDELITY };
