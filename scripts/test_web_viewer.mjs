// Smoke tests for the docs/ web viewer's pure logic modules.
// Run with: node scripts/test_web_viewer.mjs
//
// These run in Node (no browser): window is shimmed and chess.js is loaded
// directly. DOM-dependent code (board rendering, viewer wiring) is covered
// by tests/test_web_viewer_e2e.py (Playwright).

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

// Shim window before importing modules that reference it
globalThis.window = globalThis;

const { Chess } = await import(join(root, 'docs/js/chess.esm.js'));
window.Chess = Chess;

const { parsePgnToPositions } = await import(join(root, 'docs/js/pgn.js'));
const { moveIndex, getLegalMovesMask, ALL_MOVES, mirrorMove } = await import(
  join(root, 'docs/js/moves.js')
);
// pgn.js resolves the move index via window.__moveIndex (set by main.js)
window.__moveIndex = moveIndex;
const { buildPositionInput, tokenizeBoard, D_IN } = await import(join(root, 'docs/js/tensor.js'));
const { VIEW_ELOS, softmaxMasked, indexToUci, uciToSan, selectTopK } = await import(
  join(root, 'docs/js/topk.js')
);
const { fenToSquares } = await import(join(root, 'docs/js/board.js'));

let failures = 0;
function check(name, cond) {
  if (cond) {
    console.log(`  ok  ${name}`);
  } else {
    console.error(`FAIL  ${name}`);
    failures++;
  }
}

// ── Fixed ELO range ──────────────────────────────────────────────────────────
check('VIEW_ELOS is 750..2500 step 250', JSON.stringify(VIEW_ELOS) === JSON.stringify([750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]));

// ── PGN parsing ──────────────────────────────────────────────────────────────
const pgnText = readFileSync(join(root, 'docs/example2.pgn'), 'utf8');
const game = parsePgnToPositions(pgnText);
check('example2.pgn parses to positions', game !== null && game.nMoves > 0);
console.log(`      (${game.nMoves} plies)`);

const nMoves = game.positions.length;
for (const [i, pos] of game.positions.entries()) {
  if (!pos.fenBefore || !pos.moveUci || !Number.isInteger(pos.moveIndex)) {
    check(`position ${i} has required fields`, false);
    break;
  }
}
check('all positions have fenBefore/moveUci/moveIndex', true);

// ── buildPositionInput replay correctness ────────────────────────────────────
const start = buildPositionInput(game.positions, 0);
check('ply 0 board is the initial position', start.board.fen().startsWith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w'));
check('ply 0 tokens shape is 64*96', start.tokens.length === 64 * D_IN && D_IN === 96);
check('ply 0 is white to play', start.isBlackTurn === false);

let replayOk = true;
for (const ply of [1, 2, 5, Math.floor(nMoves / 2), nMoves - 1]) {
  const input = buildPositionInput(game.positions, ply);
  if (input.board.fen() !== game.positions[ply].fenBefore) replayOk = false;
}
check('replayed FEN matches positions[ply].fenBefore at sampled plies', replayOk);

const finalInput = buildPositionInput(game.positions, nMoves);
check('final ply exists and alternates turn', finalInput !== null);
check('out-of-range ply returns null', buildPositionInput(game.positions, -1) === null && buildPositionInput(game.positions, nMoves + 1) === null);

// History frames: ply >= 7 must have a non-empty last frame equal to the
// tokenization of the current board.
{
  const input = buildPositionInput(game.positions, 10);
  const currentTokens = tokenizeBoard(input.board);
  let lastFrameMatches = true;
  // layout is [square][historyFrame][channel]; compare frame 7 of every square
  for (let sq = 0; sq < 64; sq++) {
    for (let c = 0; c < 12; c++) {
      if (input.tokens[sq * D_IN + 7 * 12 + c] !== currentTokens[sq * 12 + c]) {
        lastFrameMatches = false;
      }
    }
  }
  check('history last frame equals tokenized current board', lastFrameMatches);
}

// ── softmaxMasked ────────────────────────────────────────────────────────────
{
  const mask = new Uint8Array(4352);
  mask[0] = 1; mask[5] = 1; mask[4351] = 1;
  const logits = new Float32Array(4352);
  logits[0] = 1; logits[5] = 2; logits[4351] = 3;
  const probs = softmaxMasked(logits, mask);
  const sum = probs[0] + probs[5] + probs[4351];
  check('softmax over legal moves sums to 1', Math.abs(sum - 1) < 1e-9);
  check('illegal moves get zero probability', [...probs].every((p, i) => mask[i] === 1 || p === 0));
  check('higher logit gets higher probability', probs[4351] > probs[5] && probs[5] > probs[0]);
}

// ── index <-> UCI mapping incl. black-turn mirroring ────────────────────────
check('indexToUci white turn is identity on vocabulary', indexToUci(moveIndex('e2e4', false), false) === 'e2e4');
check('indexToUci black turn mirrors back', indexToUci(moveIndex('e7e5', true), true) === 'e7e5');
check('mirrorMove round trip', mirrorMove(mirrorMove('d2d4')) === 'd2d4');

// ── selectTopK with synthetic logits ─────────────────────────────────────────
{
  // White to move: force e2e4 to dominate.
  const input = buildPositionInput(game.positions, 0);
  const mask = getLegalMovesMask(input.board, false);
  const idx = moveIndex('e2e4', false);
  const logits = new Float32Array(4352).fill(-1);
  logits[idx] = 50;

  const top = selectTopK(logits, mask, input.board, false, 'e2e4', 5);
  check('dominant legal move ranks first', top.length === 5 && top[0].san === 'e4' && top[0].uci === 'e2e4');
  check('top-1 probability ~1', top[0].prob > 0.999);
  check('played move flagged', top[0].isPlayed === true);
  const sanSet = new Set(top.map((t) => t.san));
  check('all returned moves are legal SANs', [...sanSet].every((s) => s && s.length > 0));

  // Black to move: same position after 1. e4, force e7e5 via mirrored index.
  const blackInput = buildPositionInput(game.positions, 1);
  const blackMask = getLegalMovesMask(blackInput.board, true);
  const bIdx = moveIndex('e7e5', true);
  const bLogits = new Float32Array(4352).fill(-1);
  bLogits[bIdx] = 50;
  const bTop = selectTopK(bLogits, blackMask, blackInput.board, true, 'e7e5', 3);
  check('black-turn dominant move maps to e5', bTop[0].san === 'e5' && bTop[0].uci === 'e7e5');

  // k larger than the number of legal moves must not crash or duplicate.
  const tiny = new Chess('7k/8/8/8/8/8/8/K7 w - - 0 1');
  const tinyMask = getLegalMovesMask(tiny, false);
  const tinyLogits = new Float32Array(4352).fill(0);
  const tinyTop = selectTopK(tinyLogits, tinyMask, tiny, false, null, 5);
  check('k clamped to number of legal moves', tinyTop.length === 3);
}

// ── uciToSan / castling ──────────────────────────────────────────────────────
{
  const board = new Chess();
  for (const m of ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6']) board.move(m);
  check('uciToSan resolves O-O', uciToSan(board, 'e1g1') === 'O-O');
}

// ── fenToSquares ─────────────────────────────────────────────────────────────
{
  const squares = fenToSquares('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  check('fenToSquares a1=R h1=R', squares[0] === 'R' && squares[7] === 'R');
  check('fenToSquares e2=P e7=p e8=k', squares[12] === 'P' && squares[52] === 'p' && squares[60] === 'k');
  check('fenToSquares empty middle', squares.slice(16, 48).every((s) => s === null));
}

console.log(failures === 0 ? '\nAll viewer logic tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
