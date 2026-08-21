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

// Population prior over ELO (truncated Gaussian, unnormalized log).
// Mirrors chess_accuracy/batch_inference.py:elo_log_prior
const PRIOR_MEAN = 1500;
const PRIOR_STD = 350;

function eloLogPrior(eloValues, mean = PRIOR_MEAN, std = PRIOR_STD) {
  const out = new Float64Array(eloValues.length);
  for (let i = 0; i < eloValues.length; i++) {
    const z = (eloValues[i] - mean) / std;
    out[i] = -0.5 * z * z;
  }
  return out;
}

// Normalize a 2D score surface into P(w, b | game) ∝ exp(surface)·prior(w)·prior(b).
// Mirrors chess_accuracy/batch_inference.py:joint_posterior_2d
function jointPosterior2D(surface, whiteElos, blackElos) {
  const nW = whiteElos.length;
  const nB = blackElos.length;
  const priorW = eloLogPrior(whiteElos);
  const priorB = eloLogPrior(blackElos);

  let maxVal = -Infinity;
  const logPost = new Float64Array(nW * nB);
  for (let i = 0; i < nW; i++) {
    for (let j = 0; j < nB; j++) {
      const v = surface[i * nB + j] + priorW[i] + priorB[j];
      logPost[i * nB + j] = v;
      if (v > maxVal) maxVal = v;
    }
  }

  let total = 0;
  const post = new Float64Array(nW * nB);
  for (let k = 0; k < logPost.length; k++) {
    const e = Math.exp(logPost[k] - maxVal);
    post[k] = e;
    total += e;
  }
  for (let k = 0; k < post.length; k++) post[k] /= total;
  return post;
}

// Sum a joint posterior over the black axis (w marginal) / white axis (b marginal).
function sumMarginals(post, nW, nB) {
  const w = new Float64Array(nW);
  const b = new Float64Array(nB);
  for (let i = 0; i < nW; i++) {
    for (let j = 0; j < nB; j++) {
      w[i] += post[i * nB + j];
      b[j] += post[i * nB + j];
    }
  }
  return { w, b };
}

// Linear interpolation with numpy.interp edge clamping.
function interp(x, xs, ys) {
  const n = xs.length;
  if (n === 0) return NaN;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  if (xs[hi] === xs[lo]) return ys[lo];
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

// Posterior mean, std, and central 95% credible interval over a 1D grid.
// Mirrors chess_accuracy/batch_inference.py:_marginal_stats
function marginalStats(posterior, values) {
  let mean = 0;
  for (let i = 0; i < posterior.length; i++) mean += posterior[i] * values[i];
  let variance = 0;
  for (let i = 0; i < posterior.length; i++) {
    const d = values[i] - mean;
    variance += posterior[i] * d * d;
  }
  const std = Math.sqrt(Math.max(variance, 0));

  const cdf = new Float64Array(posterior.length);
  let acc = 0;
  for (let i = 0; i < posterior.length; i++) {
    acc += posterior[i];
    cdf[i] = acc;
  }

  return {
    mean,
    std,
    ci: [interp(0.025, cdf, values), interp(0.975, cdf, values)],
  };
}

export {
  computeScoreStreaming,
  estimateElo1D,
  roundUp,
  roundDown,
  eloLogPrior,
  jointPosterior2D,
  sumMarginals,
  marginalStats,
  MIN_ELO,
  MAX_ELO,
  FIDELITY,
  PRIOR_MEAN,
  PRIOR_STD,
};
