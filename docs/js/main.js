// Main entry point - orchestrates PGN parsing, inference, and UI
import { moveIndex, getLegalMovesMask } from './moves.js';
import { parsePgnToPositions } from './pgn.js';
import { tokenizeBoard, getHistoricalTokens, buildBatchTensor, buildBatchTensorSingleColor } from './tensor.js';
import { loadModel, isModelLoaded, predict } from './inference.js';
import { computeScoreStreaming, estimateElo1D, roundUp, roundDown, MIN_ELO, MAX_ELO, FIDELITY } from './scoring.js';

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
const blackName = document.getElementById('black-name');
const blackElo = document.getElementById('black-elo');
const matchRate = document.getElementById('match-rate');
const headerInfo = document.getElementById('header-info');

// Initialize
async function init() {
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

function showResults(result) {
  whiteName.textContent = result.white || 'White';
  blackName.textContent = result.black || 'Black';
  whiteElo.textContent = Math.round(result.estWhiteElo);
  blackElo.textContent = Math.round(result.estBlackElo);
  matchRate.textContent = (result.peakRate * 100).toFixed(1) + '%';

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

  setProgress(`Parsed ${game.nMoves} moves. Building tensors...`, 15);
  await sleep(50);

  // Stage 1: 1D sweep
  const eloValues = [];
  for (let e = MIN_ELO; e <= MAX_ELO; e += 200) eloValues.push(e);

  setProgress(`Stage 1: 1D sweep (${eloValues.length} values)...`, 20);
  await sleep(50);

  const batch1 = buildBatchTensor(game.positions, eloValues, 0);
  if (!batch1) {
    alert('No positions found to evaluate.');
    estimateBtn.disabled = false;
    setProgress('', 0);
    return;
  }

  setProgress(`Running inference on ${batch1.nPositions} positions x ${eloValues.length} ELOs...`, 30);
  await sleep(50);

  const result1 = await predict(batch1.tokens, batch1.selfElos, batch1.oppoElos);

  setProgress('Computing scores...', 50);
  await sleep(50);

  const scores1 = computeScoreStreaming(result1.logitsMove, batch1.humanMoves, batch1.legalMasks, batch1.nPositions, batch1.nElos);
  const { bestElo: stage1Elo } = estimateElo1D(scores1, eloValues);

  setProgress(`Stage 1 result: ${stage1Elo}. Starting per-color refinement...`, 55);
  await sleep(50);

  // Stage 2: Per-color refinement
  let bestW = stage1Elo;
  let bestB = stage1Elo;
  let margin = Math.min(400, Math.floor((MAX_ELO - MIN_ELO) / 2));
  const nAxis = Math.ceil(Math.sqrt(eloValues.length));
  let opponentElo = bestW < 1500 ? roundUp(bestW + 50) : roundDown(bestW - 50);

  let roundNum = 0;
  while (true) {
    const step = (margin * 2) / Math.max(nAxis - 1, 1);
    if (step < FIDELITY) break;

    roundNum++;
    const wLo = Math.max(MIN_ELO, bestW - margin);
    const wHi = Math.min(MAX_ELO, bestW + margin);
    const bLo = Math.max(MIN_ELO, bestB - margin);
    const bHi = Math.min(MAX_ELO, bestB + margin);

    const whiteElos = linspace(wLo, wHi, nAxis);
    const blackElos = linspace(bLo, bHi, nAxis);

    setProgress(`Round ${roundNum}: White sweep...`, 60 + roundNum * 5);
    await sleep(50);

    const batchW = buildBatchTensorSingleColor(game.positions, whiteElos, true, opponentElo, 0);
    if (batchW && batchW.nPositions > 0) {
      const resultW = await predict(batchW.tokens, batchW.selfElos, batchW.oppoElos);
      const scoresW = computeScoreStreaming(resultW.logitsMove, batchW.humanMoves, batchW.legalMasks, batchW.nPositions, batchW.nElos);
      const { bestElo } = estimateElo1D(scoresW, whiteElos);
      bestW = bestElo;
    }

    setProgress(`Round ${roundNum}: Black sweep...`, 65 + roundNum * 5);
    await sleep(50);

    const batchB = buildBatchTensorSingleColor(game.positions, blackElos, false, opponentElo, 0);
    if (batchB && batchB.nPositions > 0) {
      const resultB = await predict(batchB.tokens, batchB.selfElos, batchB.oppoElos);
      const scoresB = computeScoreStreaming(resultB.logitsMove, batchB.humanMoves, batchB.legalMasks, batchB.nPositions, batchB.nElos);
      const { bestElo } = estimateElo1D(scoresB, blackElos);
      bestB = bestElo;
    }

    margin = Math.floor(margin / 2);
  }

  setProgress('Done!', 100);
  await sleep(200);

  // Show results
  const peakRate = Math.max(...scores1);
  showResults({
    white: game.headers.White || 'White',
    black: game.headers.Black || 'Black',
    whiteEloHdr: game.headers.WhiteElo,
    blackEloHdr: game.headers.BlackElo,
    estWhiteElo: bestW,
    estBlackElo: bestB,
    peakRate: peakRate,
  });

  setProgress('', 0);
  estimateBtn.disabled = false;
}

function linspace(lo, hi, n) {
  if (n <= 1) return [lo];
  const step = (hi - lo) / (n - 1);
  return Array.from({ length: n }, (_, i) => lo + i * step);
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
