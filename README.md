# Chess Accuracy

The purpose of this repository is implement and test chess accuracy using different engines. Even though there are notes from [Lichess](https://lichess.org/page/accuracy), the equations and tuning are based on their chosen engine, and may not be applicable if different UCI compatible engines are used.

This includes replicating the win-percentage formula:

$$
ProbabilityWin \approx 50 + 50 * (2 / (1 + \exp(-0.00368208 * Centipawns)) - 1)
$$

Move-by-move accuracy:

$$
Accuracy = 103.1668 * \exp(-0.04354 * (WinPercentBefore - WinPercentAfter)) - 3.1669
$$

Game level accuracy is then calculated by:

$$
GameAccuracy = (WeightedAverageAccuracy + HarmonicMeanAccuracy)/2
$$

Where `Weighted Average Accuracy` is weighted by the standard deviation over a sliding window (window size adapts to game length).

### Game phase accuracy

Accuracy is also computed per game phase: **opening**, **middlegame**, and **endgame**. This is done by splitting the move list into phases (using either a heuristic or the Lichess Divider algorithm) and computing `gameAccuracy` on each segment's moves independently.

Two division strategies are provided:
- **Heuristic** (`calculate_accuracy.py`): fixed ply boundaries (opening: 0-20, middlegame: 21-60, endgame: 61+)
- **Faithful** (`calculate_accuracy_faithful.py`): Lichess's Divider algorithm using piece-count and piece-placement heuristics from [scalachess/Divider.scala](https://github.com/lichess-org/scalachess/blob/master/core/src/main/scala/Divider.scala)

---

This repository uses `uv`. Set it up by running:

```sh
uv sync
```

The example script is shown in `calculate_accuracy.py`, the current implementation is based on [Lichess notes](https://lichess.org/page/accuracy) and their [Scala implementation](https://github.com/lichess-org/lila/blob/master/modules/analyse/src/main/AccuracyPercent.scala).

The repo bundles [Stockfish](https://stockfishchess.org/) for portability, but will work with any UCI-compatible engine.

## Example

The example script calculates move-by-move and game-level accuracy for a sample PGN game using Stockfish as the evaluation engine. It demonstrates both the heuristic phase division (fixed ply boundaries) and the faithful Lichess Divider algorithm.

```sh
uv run calculate_accuracy.py
```

Output:

```sh
Division: opening 20 plies, endgame from ply None
Game: W 97.92%  B 99.22%
  Opening     : W 100.00%  B 100.00%
  Middlegame  : W 96.24%  B 98.76%

[Event "Live Chess"]
[Site "Chess.com"]
[Date "2024.08.31"]
[Round "?"]
[White "Hikaru"]
[Black "DanielNaroditsky"]
[Result "0-1"]
...
[Accuracy "W 97.92% B 99.22% accuracy"]

1. c4 { [%eval 0.22] } 1... e5 { [%eval -0.15] } ...
```

With the faithful Lichess Divider:

```sh
uv run calculate_accuracy_faithful.py
```

```sh
Division: opening 25 plies, endgame from ply 47
Game: W 91.71%  B 97.61%
  Opening     : W 94.43%  B 93.68%
  Middlegame  : W 90.04%  B 100.00%
  Endgame     : W 93.24%  B 97.30%
```

### Estimating ELO

Estimates the ELO rating of a chess engine (e.g., Maia) by comparing its move choices against a known PGN game. It uses a ternary search over the ELO range, evaluating how well the engine's play matches the game's actual moves at each rating level. A heuristic sample of middlegame positions is used for faster estimation.

```sh
uv run estimate_elo.py example2.pgn --sample 30
```

```sh
Game: Hikaru vs DanielNaroditsky
WhiteElo: 3225, BlackElo: 3151

Maia3 estimate:        2480  (peak rate 56.7%)
PGN reference:       W   3225   B   3151

Method: maia3 is queried at each ELO level. A heuristic sample
of middlegame positions is used for faster estimation.
A ternary search narrows the ELO range to the peak match rate
(fidelity ±50 ELO).
```