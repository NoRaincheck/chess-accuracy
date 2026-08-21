// Game viewer: board with move navigation and per-ELO top-k move predictions.
// Owns the board panel DOM; pure logic lives in topk.js / tensor.js.

import { renderBoard } from './board.js';
import { buildPositionInput } from './tensor.js';
import { predict, isModelLoaded } from './inference.js';
import { getLegalMovesMask } from './moves.js';
import { VIEW_ELOS, selectTopK } from './topk.js';

const TOP_K = 5;

// UI elements
let boardEl;
let plyLabelEl;
let startBtn;
let prevBtn;
let nextBtn;
let endBtn;
let moveListEl;
let topkBodyEl;

// Viewer state
let positions = null; // parsed game positions
let headers = null;
let currentPly = 0;
let predictRequest = 0;

function initViewer() {
  boardEl = document.getElementById('board');
  plyLabelEl = document.getElementById('ply-label');
  startBtn = document.getElementById('nav-start');
  prevBtn = document.getElementById('nav-prev');
  nextBtn = document.getElementById('nav-next');
  endBtn = document.getElementById('nav-end');
  moveListEl = document.getElementById('move-list');
  topkBodyEl = document.getElementById('topk-body');

  startBtn.addEventListener('click', () => goTo(0));
  prevBtn.addEventListener('click', () => goTo(currentPly - 1));
  nextBtn.addEventListener('click', () => goTo(currentPly + 1));
  endBtn.addEventListener('click', () => goTo(positions ? positions.length : 0));

  document.addEventListener('keydown', (e) => {
    if (!positions) return;
    if (e.target && (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT')) return;
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      goTo(currentPly - 1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      goTo(currentPly + 1);
    }
  });
}

// Load a parsed game into the viewer (called after successful PGN parse).
function setViewerGame(game) {
  positions = game.positions;
  headers = game.headers;
  currentPly = 0;

  buildMoveList();
  goTo(0);
}

function buildMoveList() {
  moveListEl.innerHTML = '';
  for (let i = 0; i < positions.length; i += 2) {
    const num = document.createElement('span');
    num.className = 'move-number';
    num.textContent = (i / 2 + 1) + '.';
    moveListEl.appendChild(num);

    for (let j = i; j < Math.min(i + 2, positions.length); j++) {
      const san = document.createElement('span');
      san.className = 'move-san';
      san.textContent = positions[j].moveSan;
      san.dataset.ply = String(j + 1);
      san.addEventListener('click', () => goTo(j + 1));
      moveListEl.appendChild(san);
    }
  }
}

function updateNav() {
  const maxPly = positions.length;
  startBtn.disabled = prevBtn.disabled = currentPly === 0;
  nextBtn.disabled = endBtn.disabled = currentPly === maxPly;

  const white = headers.White || 'White';
  const black = headers.Black || 'Black';
  if (currentPly === 0) {
    plyLabelEl.textContent = 'Start';
  } else {
    const moveNo = Math.ceil(currentPly / 2);
    const mover = currentPly % 2 === 1 ? white : black;
    const san = positions[currentPly - 1].moveSan;
    plyLabelEl.textContent = `${moveNo}${currentPly % 2 === 1 ? '.' : '...'} ${san} (${mover})`;
  }

  for (const el of moveListEl.querySelectorAll('.move-san')) {
    el.classList.toggle('active', Number(el.dataset.ply) === currentPly);
  }
  const active = moveListEl.querySelector('.move-san.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

async function goTo(ply) {
  if (!positions) return;
  const clamped = Math.max(0, Math.min(ply, positions.length));
  currentPly = clamped;

  updateNav();

  // Render board from the position's FEN; highlight the squares of the
  // previously played move.
  const fenBefore = clamped < positions.length ? positions[clamped].fenBefore : finalFen(positions);
  let highlight = null;
  if (clamped > 0) {
    const uci = positions[clamped - 1].moveUci;
    highlight = { from: uci.slice(0, 2), to: uci.slice(2, 4) };
  }
  renderBoard(boardEl, fenBefore, highlight);

  await updateTopK(clamped);
}

// The FEN after the final move is not stored in positions; replay it.
function finalFen(positions) {
  const Chess = window.Chess;
  const board = new Chess();
  if (positions.length > 0 && positions[0].fenBefore) {
    board.load(positions[0].fenBefore);
  }
  for (const pos of positions) board.move(pos.moveSan, { sloppy: true });
  return board.fen();
}

function placementMap(fen) {
  const map = {};
  const rows = fen.split(' ')[0].split('/');
  const files = 'abcdefgh';
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') file += parseInt(ch, 10);
      else map[files[file++] + String(8 - r)] = ch;
    }
  }
  return map;
}

// Run the model at each fixed ELO level for the position at `ply` and fill
// the top-k table. Guards against stale results when navigating quickly.
async function updateTopK(ply) {
  const requestId = ++predictRequest;

  if (ply >= positions.length) {
    topkBodyEl.innerHTML = '<div class="topk-empty">Game over — no moves to predict.</div>';
    return;
  }

  if (!isModelLoaded()) {
    topkBodyEl.innerHTML = '<div class="topk-empty">Model not loaded.</div>';
    return;
  }

  topkBodyEl.innerHTML = '<div class="topk-empty">Predicting...</div>';

  const input = buildPositionInput(positions, ply);
  const legalMask = getLegalMovesMask(input.board, input.isBlackTurn);
  const playedUci = positions[ply].moveUci;

  const nElo = VIEW_ELOS.length;
  const seqLen = input.tokens.length; // 64 squares x 96 channels
  const tokensBatch = new Float32Array(nElo * seqLen);
  const selfElos = new Float32Array(nElo);
  const oppoElos = new Float32Array(nElo);
  for (let e = 0; e < nElo; e++) {
    tokensBatch.set(input.tokens, e * seqLen);
    selfElos[e] = VIEW_ELOS[e];
    oppoElos[e] = VIEW_ELOS[e];
  }

  let logits;
  try {
    const result = await predict(tokensBatch, selfElos, oppoElos);
    logits = result.logitsMove;
  } catch (err) {
    if (requestId === predictRequest) {
      topkBodyEl.innerHTML = '<div class="topk-empty">Prediction failed: ' + err.message + '</div>';
    }
    return;
  }

  if (requestId !== predictRequest) return;

  topkBodyEl.innerHTML = '';
  for (let e = 0; e < nElo; e++) {
    const rowLogits = logits.subarray(e * 4352, (e + 1) * 4352);
    const topK = selectTopK(rowLogits, legalMask, input.board, input.isBlackTurn, playedUci, TOP_K);

    const row = document.createElement('div');
    row.className = 'topk-row';

    const eloCell = document.createElement('span');
    eloCell.className = 'topk-elo';
    eloCell.textContent = String(VIEW_ELOS[e]);
    row.appendChild(eloCell);

    const movesCell = document.createElement('span');
    movesCell.className = 'topk-moves';
    for (const pick of topK) {
      const chip = document.createElement('span');
      chip.className = 'topk-chip' + (pick.isPlayed ? ' played' : '');
      chip.title = `${pick.san} (${(pick.prob * 100).toFixed(1)}%)`;
      chip.innerHTML =
        `<b>${pick.san}</b> <i>${(pick.prob * 100).toFixed(0)}%</i>`;
      movesCell.appendChild(chip);
    }
    row.appendChild(movesCell);

    topkBodyEl.appendChild(row);
  }
}

export { initViewer, setViewerGame };
