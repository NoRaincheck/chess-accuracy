// Main entry point - orchestrates PGN parsing, inference, and UI
import { moveIndex, getLegalMovesMask } from './moves.js';
import { parsePgnToPositions } from './pgn.js';
import { buildBatchTensorJoint } from './tensor.js';
import { loadModel, isModelLoaded, predict } from './inference.js';
import { computeScoreStreaming, jointPosterior2D, sumMarginals, marginalStats, MIN_ELO, MAX_ELO } from './scoring.js';
import { makeGrid, windowGrid, upsampleBilinear, marginalModes } from './surface.js';
import { initViewer, setViewerGame } from './viewer.js';

// Anchor-grid search configuration (mirrors estimate_elo.py defaults)
const ANCHOR_STEP = 700;   // stage A full-range joint anchor grid spacing
const DENSE_N = 55;        // interpolated surface resolution per axis (localization only)
const FINE_MARGIN = 300;   // stage B window half-width around per-color modes
const FINE_N = 4;          // stage B grid points per axis
const MAX_POSITIONS = 80;  // deterministic cap on evaluated positions

// Expose moveIndex globally for pgn.js fallback
window.__moveIndex = moveIndex;

// UI Elements
const pgnInput = document.getElementById('pgn-input');
const estimateBtn = document.getElementById('estimate-btn');
const exampleBtn = document.getElementById('example-btn');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const resultsPanel = document.getElementById('results-panel');
const modelStatus = document.getElementById('model-status');

// Result elements
const whiteName = document.getElementById('white-name');
const whiteElo = document.getElementById('white-elo');
const whiteEloRange = document.getElementById('white-elo-range');
const blackName = document.getElementById('black-name');
const blackElo = document.getElementById('black-elo');
const blackEloRange = document.getElementById('black-elo-range');
const matchRate = document.getElementById('match-rate');
const headerInfo = document.getElementById('header-info');

// Initialize
async function init() {
  initViewer();

  // Load model in background
  loadModelStatus('Loading model...');
  try {
    await loadModel('./models/maia3-5m.onnx', (msg) => loadModelStatus(msg));
    loadModelStatus('Model ready');
    estimateBtn.disabled = false;
  } catch (err) {
    loadModelStatus('Model load failed: ' + err.message);
    console.error('Model load error:', err);
  }
}

function loadModelStatus(msg) {
  if (modelStatus) modelStatus.textContent = msg;
}

function setProgress(text, percent) {
  progressContainer.style.display = text ? 'block' : 'none';
  progressText.textContent = text || '';
  progressBar.style.width = (percent || 0) + '%';
}

// ── Live estimate display ────────────────────────────────────────────────────
// While the search runs, each row shows the current estimate with a range
// (the search window), pulsing. Once settled, the range becomes the 95%
// credible interval from the Stage B posterior and the pulse stops.

const numberState = {};

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function animateNumber(el, key, to, fmt, dur = 450) {
  const from = numberState[key];
  numberState[key] = to;
  cancelAnimationFrame(el.__tweenRaf);
  if (from === undefined || prefersReducedMotion()) {
    el.textContent = fmt(to);
    return;
  }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = fmt(from + (to - from) * eased);
    if (t < 1) el.__tweenRaf = requestAnimationFrame(step);
  };
  el.__tweenRaf = requestAnimationFrame(step);
}

function setPulsing(pulsing) {
  for (const el of [whiteElo, whiteEloRange, blackElo, blackEloRange]) {
    el.classList.toggle('pulsing', pulsing);
  }
}

function fmtRange(lo, hi) {
  return `${Math.round(lo)}–${Math.round(hi)}`;
}

function fmtCi(ci) {
  return `${Math.round(ci[0])}–${Math.round(ci[1])} · 95% CI`;
}

function resetLiveEstimates() {
  for (const key of ['wPt', 'bPt']) delete numberState[key];
  whiteElo.textContent = '—';
  blackElo.textContent = '—';
}

function startLiveEstimates() {
  resultsPanel.classList.add('has-results');
  headerInfo.style.display = 'none';
  matchRate.textContent = '—';
  resetLiveEstimates();
  // Full search range until the anchor grid localizes the modes.
  whiteEloRange.textContent = fmtRange(MIN_ELO, MAX_ELO);
  blackEloRange.textContent = fmtRange(MIN_ELO, MAX_ELO);
  setPulsing(true);
}

function updateLiveEstimates(wCenter, bCenter) {
  animateNumber(whiteElo, 'wPt', wCenter, (v) => String(Math.round(v)));
  animateNumber(blackElo, 'bPt', bCenter, (v) => String(Math.round(v)));
  whiteEloRange.textContent = fmtRange(
    Math.max(MIN_ELO, wCenter - FINE_MARGIN),
    Math.min(MAX_ELO, wCenter + FINE_MARGIN),
  );
  blackEloRange.textContent = fmtRange(
    Math.max(MIN_ELO, bCenter - FINE_MARGIN),
    Math.min(MAX_ELO, bCenter + FINE_MARGIN),
  );
}

function showResults(result) {
  whiteName.textContent = result.white || 'White';
  blackName.textContent = result.black || 'Black';
  animateNumber(whiteElo, 'wPt', result.estWhiteElo, (v) => String(Math.round(v)));
  animateNumber(blackElo, 'bPt', result.estBlackElo, (v) => String(Math.round(v)));
  matchRate.textContent = (result.peakRate * 100).toFixed(1) + '%';

  whiteEloRange.textContent = result.whiteCi ? fmtCi(result.whiteCi) : '—';
  blackEloRange.textContent = result.blackCi ? fmtCi(result.blackCi) : '—';
  setPulsing(false);

  // Show header ELO comparison
  if (result.whiteEloHdr && result.blackEloHdr) {
    headerInfo.textContent = `PGN headers: W ${result.whiteEloHdr} / B ${result.blackEloHdr}`;
    headerInfo.style.display = 'block';
  } else {
    headerInfo.style.display = 'none';
  }

  resultsPanel.classList.add('has-results');
}

function hideResults() {
  resultsPanel.classList.remove('has-results');
  setPulsing(false);
  resetLiveEstimates();
  whiteEloRange.textContent = '—';
  blackEloRange.textContent = '—';
}

// Main estimation function
async function estimateElo() {
  const pgnText = pgnInput.value.trim();
  if (!pgnText) {
    alert('Please enter a PGN game.');
    return;
  }

  if (!isModelLoaded()) {
    alert('Model is still loading. Please wait.');
    return;
  }

  estimateBtn.disabled = true;
  hideResults();
  setProgress('Parsing PGN...', 5);

  await sleep(50);

  // Parse PGN
  const game = parsePgnToPositions(pgnText);
  if (!game) {
    alert('Could not parse PGN. Please check the format.');
    estimateBtn.disabled = false;
    setProgress('', 0);
    return;
  }

  window.__currentGame = game;

  // Load the game into the board viewer (board + move navigation + top-k)
  setViewerGame(game);

  startLiveEstimates();

  setProgress(`Parsed ${game.nMoves} moves. Building tensors...`, 15);
  await sleep(50);

  // Stage A: sparse joint (white, black) anchor grid over the full range.
  // The score surface is smooth in (w, b), so bilinear interpolation of the
  // anchor surface localizes per-color modes cheaply; a fine grid around the
  // modes then produces the final estimates from real model evaluations.
  const anchors = makeGrid(MIN_ELO, MAX_ELO, ANCHOR_STEP);
  const nA = anchors.length;

  setProgress(`Stage A: joint anchor grid (${nA}x${nA})...`, 20);
  await sleep(50);

  const batchA = buildBatchTensorJoint(game.positions, anchors, anchors, { maxPositions: MAX_POSITIONS });
  if (!batchA) {
    alert('No informative positions found to evaluate.');
    estimateBtn.disabled = false;
    hideResults();
    setProgress('', 0);
    return;
  }

  setProgress(`Running inference on ${batchA.nPositions} positions x ${batchA.nElos} ELO pairs...`, 30);
  await sleep(50);

  const resultA = await predict(batchA.tokens, batchA.selfElos, batchA.oppoElos);
  const scoresA = computeScoreStreaming(resultA.logitsMove, batchA.humanMoves, batchA.legalMasks, batchA.nPositions, batchA.nElos);

  setProgress('Interpolating score surface...', 55);
  await sleep(50);

  const dense = upsampleBilinear(scoresA, nA, nA, DENSE_N, DENSE_N);
  const denseVals = makeGrid(MIN_ELO, MAX_ELO, (MAX_ELO - MIN_ELO) / (DENSE_N - 1));
  let { wi, bi } = marginalModes(dense, DENSE_N, DENSE_N);
  let bestW = denseVals[wi];
  let bestB = denseVals[bi];

  updateLiveEstimates(bestW, bestB);

  setProgress(`Modes: W=${Math.round(bestW)}, B=${Math.round(bestB)}. Refining...`, 65);
  await sleep(50);

  // Stage B: joint fine grid around the modes. If a mode lands on the window
  // edge the true mode may lie just outside — slide the window there once.
  let peakRate = 0;
  let wFine = null;
  let bFine = null;
  let scoresB = null;
  for (let attempt = 0; attempt < 2; attempt++) {
    wFine = windowGrid(bestW, FINE_MARGIN, FINE_N, MIN_ELO, MAX_ELO);
    bFine = windowGrid(bestB, FINE_MARGIN, FINE_N, MIN_ELO, MAX_ELO);

    setProgress(`Stage B${attempt ? ' (re-centered)' : ''}: fine grid ${wFine.length}x${bFine.length}...`, 70 + attempt * 10);
    await sleep(50);

    const batchB = buildBatchTensorJoint(game.positions, wFine, bFine, { maxPositions: MAX_POSITIONS });
    if (!batchB) break;

    const resultB = await predict(batchB.tokens, batchB.selfElos, batchB.oppoElos);
    scoresB = computeScoreStreaming(resultB.logitsMove, batchB.humanMoves, batchB.legalMasks, batchB.nPositions, batchB.nElos);

    let bestIdx = 0;
    for (let k = 1; k < scoresB.length; k++) {
      if (scoresB[k] > scoresB[bestIdx]) bestIdx = k;
    }
    const wiBest = Math.floor(bestIdx / batchB.nB);
    const biBest = bestIdx % batchB.nB;
    peakRate = scoresB[bestIdx];

    const onEdge = wiBest === 0 || wiBest === wFine.length - 1 || biBest === 0 || biBest === bFine.length - 1;
    if (attempt === 0 && onEdge) {
      bestW = wFine[wiBest];
      bestB = bFine[biBest];
      updateLiveEstimates(bestW, bestB);
      continue;
    }

    bestW = wFine[wiBest];
    bestB = bFine[biBest];
    break;
  }

  setProgress('Done!', 100);
  await sleep(200);

  // 95% credible intervals from the Stage B posterior marginals.
  let whiteCi = null;
  let blackCi = null;
  if (scoresB && wFine && bFine) {
    const posterior = jointPosterior2D(scoresB, wFine, bFine);
    const { w: wPost, b: bPost } = sumMarginals(posterior, wFine.length, bFine.length);
    whiteCi = marginalStats(wPost, wFine).ci;
    blackCi = marginalStats(bPost, bFine).ci;
  }

  // Show results
  showResults({
    white: game.headers.White || 'White',
    black: game.headers.Black || 'Black',
    whiteEloHdr: game.headers.WhiteElo,
    blackEloHdr: game.headers.BlackElo,
    estWhiteElo: bestW,
    estBlackElo: bestB,
    peakRate: peakRate,
    whiteCi: whiteCi,
    blackCi: blackCi,
  });

  setProgress('', 0);
  estimateBtn.disabled = false;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Event listeners
estimateBtn.addEventListener('click', estimateElo);

exampleBtn.addEventListener('click', async () => {
  try {
    const resp = await fetch('example2.pgn');
    if (resp.ok) {
      pgnInput.value = await resp.text();
    } else {
      pgnInput.value = `[Event "Live Chess"]
[Site "Chess.com"]
[Date "2024.08.31"]
[White "Hikaru"]
[Black "DanielNaroditsky"]
[Result "0-1"]
[WhiteElo "3225"]
[BlackElo "3151"]

1. c4 e5 2. Nc3 Bb4 3. g3 Bxc3 4. bxc3 d6 5. Bg2 Nf6 6. d3 O-O 7. Nf3 Re8 8. O-O
e4 9. Nd4 Nbd7 10. h3 Nc5 11. Be3 Bd7 12. Nb3 Na4 13. Qd2 c5 14. Rae1 Bc6 15.
Qc2 h6 16. Kh2 exd3 17. exd3 Bxg2 18. Kxg2 Qd7 19. f3 Re6 20. Nd2 Rae8 21. Ne4
d5 22. Nxf6+ gxf6 23. Bf2 Rxe1 24. Rxe1 Rxe1 25. Bxe1 dxc4 26. dxc4 Nb6 27. Bf2
Nxc4 28. Bxc5 b6 29. Bd4 Qxd4 0-1`;
    }
  } catch {
    pgnInput.value = '[Event "Example"]\n[White "Player1"]\n[Black "Player2"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0';
  }
});

// Init on load
init();
