// Chessboard renderer: draws a position from a FEN into a container as an
// 8x8 CSS grid with Unicode piece glyphs. Highlights the last move squares.

const GLYPHS = {
  K: '\u2654', Q: '\u2655', R: '\u2656', B: '\u2657', N: '\u2658', P: '\u2659',
  k: '\u265A', q: '\u265B', r: '\u265C', b: '\u265D', n: '\u265E', p: '\u265F',
};

const FILES = 'abcdefgh';

// Parse the placement field of a FEN into a 64-entry array (a1..h8) of
// piece chars or null.
function fenToSquares(fen) {
  const squares = new Array(64).fill(null);
  const rows = fen.split(' ')[0].split('/');
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') {
        file += parseInt(ch, 10);
      } else {
        // FEN rows go rank 8 -> rank 1; square array is a1..h8
        squares[file + (7 - r) * 8] = ch;
        file++;
      }
    }
  }
  return squares;
}

// Render the board. lastMove is { from, to } in algebraic coords or null.
function renderBoard(container, fen, lastMove) {
  container.innerHTML = '';
  container.classList.add('board');

  const squares = fenToSquares(fen);

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const cell = document.createElement('div');
      const light = (row + col) % 2 === 0;
      cell.className = 'square ' + (light ? 'light' : 'dark');

      const name = FILES[col] + String(8 - row);
      if (lastMove && (name === lastMove.from || name === lastMove.to)) {
        cell.classList.add('last-move');
      }

      const piece = squares[col + (7 - row) * 8];
      if (piece) {
        const glyph = document.createElement('span');
        glyph.className = 'piece ' + (piece === piece.toUpperCase() ? 'white-piece' : 'black-piece');
        glyph.textContent = GLYPHS[piece];
        cell.appendChild(glyph);
      }

      container.appendChild(cell);
    }
  }
}

export { renderBoard, fenToSquares };
