# Chess Accuracy — Technical Reference

## Overview

Lichess's accuracy metric quantifies how well a player performed in a game by measuring how much their winning chances dropped with each move. This document details every step of the computation, including details that go beyond [Lichess's blog post](https://lichess.org/page/accuracy).

---

## 1. Win% — Centipawns to Winning Probability

The Stockfish evaluation in centipawns is mapped to a winning probability via a logistic curve fitted to real game data:

```
Win% = 50 + 50 × (2 / (1 + exp(-0.00368208 × cp)) - 1)
```

where `cp` is the evaluation in centipawns from **white's perspective** (positive = good for white).

All positions use white-perspective evaluations. This is critical — when computing black's move accuracy, the before/after win percentages are swapped rather than subtracted from 100.

The constant `0.00368208` was derived by fitting the curve to games among ~2300-rated players on Lichess.

---

## 2. Move Accuracy% — Comparing Before and After

Once each position has a Win%, the accuracy of a single move is:

```
WinDiff = WinPercentBefore - WinPercentAfter

Accuracy% = 103.1668100711649 × exp(-0.04354415386753951 × WinDiff) + -3.166924740191411
```

If `WinPercentAfter >= WinPercentBefore`, accuracy is 100% (the player didn't lose any winning chances).

### Uncertainty Bonus

A +1 is added to the raw accuracy value (before clamping to [0, 100]). This accounts for imperfect engine analysis — the best move may not always be found, so a small penalty buffer prevents over-penalizing near-optimal moves.

### Full-Precision Constants

The constants in the Lichess source code are full double-precision values from an `scipy.optimize.curve_fit` fit:

```python
# Sample data used for fitting:
xs = [   0,   5,  10,  20,  40,  60,   80,  90, 100]  # WinDiff
ys = [ 100,  75,  60,  42,  20,   5,    0,   0,   0]  # Accuracy
sigma = [0.005, 1, 1, 1, 1, 1, 0.005, 1, 1]           # error stdev

# Result (from lichess source):
a = 103.1668100711649
k = 0.04354415386753951
b = -3.166924740191411
```

### Per-Move Color Handling

All evaluations are stored as white-perspective centipawns. For a white move:
- `before = win% before white's move`
- `after = win% after white's move`

For a black move, since win% is from white's perspective and black wants white's win% to drop:
- `before = win% after white's previous move` (which is the state before black moves)
- `after = win% before white's previous move` (which is the state after black's move)
- This is equivalent to swapping: `fromWinPercents(next, prev)` instead of `fromWinPercents(prev, next)`

---

## 3. Game Accuracy — Aggregating Individual Moves

Game accuracy is **not** a simple average of move accuracies. It combines two measures:

### 3a. Window Construction

```
windowSize = clamp(num_moves / 10, 2, 8)
```

The window size adapts to game length:
- Games with ≤20 plies: window size = 2
- Games with 20-80 plies: window size scales from 2 to 8
- Games with ≥80 plies: window size = 8

Windows are constructed from the sequence of Win% values (including the initial 50% position):
1. Pad with `(windowSize - 2)` copies of the first window (so there are as many windows as moves)
2. Append all `sliding(windowSize)` windows from the full Win% sequence

```
Example with 5 Win% values, windowSize = 3:
  padding copies: [W[0:3], W[0:3]]  (windowSize - 2 = 1 copy)
  sliding:        [W[0:3], W[1:4], W[2:5]]
  total windows:  4  (same as number of moves)
```

### 3b. Volatility Weights

For each window, compute the sample standard deviation of its Win% values:

```
weight = clamp(std(WinPercent_window), 0.5, 12)
```

Windows with higher volatility (large swing in winning chances) get higher weight — positions where accuracy is more meaningful are emphasized.

### 3c. Per-Color Aggregation

Moves are assigned to colors based on move index parity and the starting color:
- Start color white: even indices → white moves, odd indices → black moves
- Start color black: even indices → black moves, odd indices → white moves

Each move has a `(accuracy, weight)` pair.

### 3d. Final Score

For each color:

```
weightedMean = sum(accuracy_i × weight_i) / sum(weight_i)
harmonicMean = num_moves / sum(1 / accuracy_i)
gameAccuracy = (weightedMean + harmonicMean) / 2
```

---

## 4. Game Phase Accuracy

Accuracy is also computed per game phase by splitting the move list and running `gameAccuracy` on each segment independently.

### Phase Definitions

The game is divided into three phases:

| Phase     | Condition                        |
|-----------|----------------------------------|
| Opening   | ply < `midGamePly`               |
| Middlegame | `midGamePly` ≤ ply < `endGamePly` |
| Endgame   | ply ≥ `endGamePly`               |

If a phase boundary is `None` (not reached), that phase and all later phases are absent.

### Segment Start Color

When computing `gameAccuracy` on a phase segment, the effective start color must account for how many plies preceded the segment:
- If `segment_start_ply` is even: the segment starts with the same color as the overall game
- If `segment_start_ply` is odd: the segment starts with the opposite color

### 4a. Heuristic Division

Fixed ply ranges:
- Opening: plies 0–19 (moves 1–10)
- Middlegame: plies 20–59 (moves 11–30)
- Endgame: plies 60+ (moves 31+)

### 4b. Faithful Division (Lichess Divider)

Reproduces the exact algorithm from [scalachess/Divider.scala](https://github.com/lichess-org/scalachess/blob/master/core/src/main/scala/Divider.scala). The Divider scans board positions sequentially to find:

**Middlegame start** — first ply where any of these is true:
1. **`majorsAndMinors ≤ 10`**: Total non-king, non-pawn pieces (queens, rooks, bishops, knights) ≤ 10
2. **`backrankSparse`**: Either side has < 4 own pieces on its back rank (rank 1 for white, rank 8 for black), indicating development
3. **`mixedness > 150`**: Pieces are spread across the board. The mixedness score sums over 49 overlapping 3×2-tile regions across the 8×8 board, with a scoring table that rewards intermingled pieces:

```
Score table for (white_count, black_count) in a region:
  (0,0) -> 0
  (1,0) -> 1 + (8 - y)
  (2,0) -> if y > 2: 2 + (y - 2) else 0
  (3,0) -> if y > 1: 3 + (y - 1) else 0
  (4,0) -> if y > 1: 3 + (y - 1) else 0
  (0,1) -> 1 + y
  (1,1) -> 5 + |4 - y|
  (2,1) -> 4 + (y - 1)
  (3,1) -> 5 + (y - 1)
  (0,2) -> if y < 6: 2 + (6 - y) else 0
  (1,2) -> 4 + (7 - y)
  (2,2) -> 7
  (0,3) -> if y < 7: 3 + (7 - y) else 0
  (1,3) -> 5 + (7 - y)
  (0,4) -> if y < 7: 3 + (7 - y) else 0
  otherwise -> 0
```

where `y` is the region's vertical position (1–7, with 1 = top/rank 1).

**Endgame start** — first ply where `majorsAndMinors ≤ 6` (after middlegame has started).

If no middlegame start is found, there is no endgame either (short game → all opening).

---

## 5. Implementation Notes

### Eval Consistency

All evaluations must use white-perspective centipawns. In python-chess, `PovScore.white().score()` returns the white-perspective centipawn value. Using `PovScore.relative().score()` would give alternating perspectives and produce incorrect results with this implementation.

### Edge Cases

- **Mate scores**: When the position is checkmate or the engine finds a forced mate, `score()` returns `None`. These are treated as 0 centipawns (or could be converted to ±∞ for the win% formula, but 0 is the safer default for the accuracy computation).
- **Short games**: If a game has fewer than 2 moves, game accuracy is 100% for all players.
- **Zero-weight moves**: If all moves in a phase have zero weight (unlikely with the [0.5, 12] clamp), falls back to simple mean.
- **Zero accuracy moves**: Harmonic mean requires all values > 0; if any move has 0% accuracy (shouldn't happen due to the +1 bonus), falls back to weighted mean.

### Color Alignment

The implementation aligns moves as follows:
```
Index 0 → first move of the game (made by `start_color`)
Index 1 → second move (made by the opponent)
Index 2 → third move (back to `start_color`)
...
```

When the game starts with white, indices 0, 2, 4, ... are white moves and 1, 3, 5, ... are black moves. When it starts with black, the parity is flipped.

### Relation to Lichess Source

The implementation maps directly to `AccuracyPercent.scala`:
| Scala function                       | Python equivalent                   |
|--------------------------------------|-------------------------------------|
| `fromWinPercents(before, after)`     | `accuracy_from_win_percentage`      |
| `WinPercent.fromCentiPawns(cp)`      | `win_percentage_from_white_cp`      |
| `gameAccuracy(startColor, cps)`      | `game_accuracy(cps, start_color)`   |
| `phaseAccuracies(div, analysis)`     | `phase_accuracy(cps, division, ...)` |

---

## 6. Comparison: Lichess vs Other Platforms

| Aspect                | Lichess                          | Chess.com                         |
|-----------------------|----------------------------------|-----------------------------------|
| Engine                | Stockfish NNUE (latest)          | Stockfish (version varies)        |
| Analysis depth        | Deeper server-side               | Usually shallower                 |
| Win% formula          | Logistic fit to 2300-rated games | Different proprietary formula     |
| Game accuracy formula | Volatility-weighted + harmonic   | Simple average or other method    |
| Phase breakdown       | Opening/Middlegame/Endgame       | Usually not provided              |

Different engine versions, analysis depths, and formulas mean accuracy scores are **not comparable across platforms**.
