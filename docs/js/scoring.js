// Scoring and ELO optimization
// Mirrors chess_accuracy/batch_inference.py:_compute_score and estimate_elo.py

const ALPHA = 0.6;
const MIN_ELO = 300;
const MAX_ELO = 3000;
const FIDELITY = 50;

// Compute blended top-1 accuracy + MRR score
// Processes one ELO at a time to save memory
function computeScoreStreaming(logitsAllElos, humanMoves, legalMasks, nPos, nElo) {
  // Float64 to match numpy float64 output of batch_inference.py:_compute_score
  const scores = new Float64Array(nElo);
  if (nPos === 0) return scores;

  for (let e = 0; e < nElo; e++) {
    let top1Correct = 0;
    let mrrSum = 0;

    for (let p = 0; p < nPos; p++) {
      const logitBase = (p * nElo + e) * 4352;
      const maskBase = p * 4352;
      const humanIdx = humanMoves[p];

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

      // 1-indexed rank: count legal moves with logit >= human logit, plus 1
      let rank = 1;
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

export { computeScoreStreaming, estimateElo1D, roundUp, roundDown, MIN_ELO, MAX_ELO, FIDELITY };
