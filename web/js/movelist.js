// Move list panel with click-to-navigate

class MoveList {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.moves = [];
    this.currentMoveIndex = -1;
    this.onMoveClick = null;

    this.render();
  }

  setMoves(moves) {
    this.moves = moves;
    this.currentMoveIndex = -1;
    this.render();
  }

  setCurrentMove(index) {
    this.currentMoveIndex = index;
    this.highlightCurrent();
  }

  render() {
    this.container.innerHTML = '';
    if (this.moves.length === 0) return;

    const table = document.createElement('div');
    table.className = 'move-table';

    let moveNum = 1;
    let row = null;

    for (let i = 0; i < this.moves.length; i++) {
      const isWhite = i % 2 === 0;

      if (isWhite) {
        row = document.createElement('div');
        row.className = 'move-row';

        const numEl = document.createElement('span');
        numEl.className = 'move-number';
        numEl.textContent = moveNum + '.';
        row.appendChild(numEl);
      }

      const moveEl = document.createElement('span');
      moveEl.className = 'move-san';
      moveEl.textContent = this.moves[i].san;
      moveEl.dataset.index = i;
      moveEl.addEventListener('click', () => {
        this.setCurrentMove(parseInt(moveEl.dataset.index));
        if (this.onMoveClick) this.onMoveClick(parseInt(moveEl.dataset.index));
      });
      row.appendChild(moveEl);

      if (!isWhite) {
        table.appendChild(row);
        moveNum++;
      }
    }

    // Odd number of moves
    if (this.moves.length % 2 !== 0) {
      table.appendChild(row);
    }

    this.container.appendChild(table);
  }

  highlightCurrent() {
    const allMoves = this.container.querySelectorAll('.move-san');
    allMoves.forEach((el, i) => {
      el.classList.toggle('active', i === this.currentMoveIndex);
    });

    // Scroll into view
    const activeEl = this.container.querySelector('.move-san.active');
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }
}

export { MoveList };
