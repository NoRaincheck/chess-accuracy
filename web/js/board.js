// Chess board renderer using canvas
// Renders a chess board with SVG piece images

const PIECE_MAP = {
  wk: 'wK', wq: 'wQ', wr: 'wR', wb: 'wB', wn: 'wN', wp: 'wP',
  bk: 'bK', bq: 'bQ', br: 'bR', bb: 'bB', bn: 'bN', bp: 'bP',
};

const FILES = 'abcdefgh';
const RANKS = '87654321'; // Top to bottom

class ChessBoard {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.canvas = document.createElement('canvas');
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');

    this.squareSize = options.squareSize || 64;
    this.lightColor = options.lightColor || '#f0d9b5';
    this.darkColor = options.darkColor || '#b58863';
    this.highlightColor = options.highlightColor || 'rgba(255, 255, 0, 0.4)';
    this.lastMoveFrom = null;
    this.lastMoveTo = null;
    this.flipped = false;

    this.pieceImages = {};
    this.piecesLoaded = false;

    this.resize();
    this.loadPieces();
  }

  resize() {
    const size = this.squareSize * 8;
    this.canvas.width = size + 40; // Extra for coordinates
    this.canvas.height = size + 40;
    this.canvas.style.width = (size + 40) + 'px';
    this.canvas.style.height = (size + 40) + 'px';
  }

  async loadPieces() {
    const pieces = Object.keys(PIECE_MAP);
    const promises = pieces.map(piece => {
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
          this.pieceImages[piece] = img;
          resolve();
        };
        img.onerror = () => resolve(); // Graceful fallback
        img.src = `assets/pieces/${PIECE_MAP[piece]}.svg`;
      });
    });

    await Promise.all(promises);
    this.piecesLoaded = true;
  }

  setFlipped(flipped) {
    this.flipped = flipped;
  }

  setLastMove(fromSquare, toSquare) {
    this.lastMoveFrom = fromSquare;
    this.lastMoveTo = toSquare;
  }

  render(board) {
    if (!this.piecesLoaded) return;

    const ctx = this.ctx;
    const size = this.squareSize * 8;
    const offsetX = 20;
    const offsetY = 20;

    // Clear
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw squares
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const displayRow = this.flipped ? 7 - row : row;
        const displayCol = this.flipped ? 7 - col : col;

        const x = offsetX + displayCol * this.squareSize;
        const y = offsetY + displayRow * this.squareSize;
        const isLight = (row + col) % 2 === 0;

        ctx.fillStyle = isLight ? this.lightColor : this.darkColor;
        ctx.fillRect(x, y, this.squareSize, this.squareSize);

        // Highlight last move
        const file = FILES[col];
        const rank = RANKS[row];
        const sq = file + rank;

        if (sq === this.lastMoveFrom || sq === this.lastMoveTo) {
          ctx.fillStyle = this.highlightColor;
          ctx.fillRect(x, y, this.squareSize, this.squareSize);
        }
      }
    }

    // Draw pieces
    if (board) {
      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
          const displayRow = this.flipped ? 7 - row : row;
          const displayCol = this.flipped ? 7 - col : col;

          const x = offsetX + displayCol * this.squareSize;
          const y = offsetY + displayRow * this.squareSize;

          const file = FILES[col];
          const rank = RANKS[row];
          const square = file + rank;
          const piece = board.get(square);

          if (piece) {
            const color = piece.color === 'w' ? 'w' : 'b';
            const type = piece.type;
            const pieceKey = color + type;
            const img = this.pieceImages[pieceKey];

            if (img) {
              const padding = 4;
              ctx.drawImage(img, x + padding, y + padding,
                this.squareSize - padding * 2, this.squareSize - padding * 2);
            }
          }
        }
      }
    }

    // Draw coordinates
    ctx.fillStyle = '#666';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';

    for (let i = 0; i < 8; i++) {
      const displayCol = this.flipped ? 7 - i : i;
      const x = offsetX + displayCol * this.squareSize + this.squareSize / 2;
      ctx.fillText(FILES[i], x, offsetY + 8 * this.squareSize + 14);
    }

    ctx.textAlign = 'right';
    for (let i = 0; i < 8; i++) {
      const displayRow = this.flipped ? 7 - i : i;
      const y = offsetY + displayRow * this.squareSize + this.squareSize / 2 + 4;
      ctx.fillText(RANKS[i], offsetX - 4, y);
    }
  }

  // Get square name from pixel coordinates
  getSquareFromPixel(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left - 20;
    const y = clientY - rect.top - 20;

    let col = Math.floor(x / this.squareSize);
    let row = Math.floor(y / this.squareSize);

    if (this.flipped) {
      col = 7 - col;
      row = 7 - row;
    }

    if (col < 0 || col > 7 || row < 0 || row > 7) return null;
    return FILES[col] + RANKS[row];
  }
}

export { ChessBoard };
