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
uv run estimate_elo.py example2.pgn
```

```sh
Loading maia3 ONNX model (maia3-5m)...
Stage 1: 1D sweep (55 values, step=50)...
  -> 1D estimate: 2700 (rate=0.6207)
Round 1: 2D refinement (8x8, margin=±1350, step=386)...
  -> best: W=3000, B=2764 (rate=0.6379)
Round 2: 2D refinement (8x8, margin=±675, step=193)...
  -> best: W=2904, B=2610 (rate=0.6379)
Round 3: 2D refinement (8x8, margin=±337, step=96)...
  -> best: W=2876, B=2658 (rate=0.6379)
Final: W=2876, B=2658 (rate=0.6379)

Game: Hikaru vs DanielNaroditsky
WhiteElo: 3225, BlackElo: 3151

Estimated:  W   2876   B   2658  (rate 63.8%)
PGN ref:    W   3225   B   3151
```

**Hold Out Results**

```
$ ./estimate_all.sh
  [  5s] 1000-1400_huEchdBz.pgn                W:   1362 -> 2066.7 (+704.7)  B:   1170 -> 1666.7 (+496.7)
  [  2s] 1000-1400_mk9moDDq.pgn                W:   1248 -> 1866.7 (+618.7)  B:   1190 -> 1200.0 (+10.0)
  [  7s] 1000-1400_yMk3fTsK.pgn                W:   1157 ->  666.7 (-490.3)  B:   1240 ->  533.3 (-706.7)
  [  5s] 1700-2100_QK5egQTl.pgn                W:   1842 -> 2066.7 (+224.7)  B:   1857 -> 2133.3 (+276.3)
  [  4s] 1700-2100_ROmEhCmX.pgn                W:   1721 -> 2200.0 (+479.0)  B:   1725 -> 2000.0 (+275.0)
  [  5s] 1700-2100_foe2ahdY.pgn                W:   2063 -> 1600.0 (-463.0)  B:   1821 -> 1533.3 (-287.7)
  [  5s] 2100+_BMwcT27N.pgn                    W:   2367 -> 2466.7 (+99.7)  B:   2245 -> 2666.7 (+421.7)
  [  9s] 2100+_Q5mCQ4jR.pgn                    W:   2259 -> 2200.0 (-59.0)  B:   2351 -> 2200.0 (-151.0)
  [  4s] 2100+_jv6QQCbT.pgn                    W:   2152 -> 2466.7 (+314.7)  B:   2276 -> 1400.0 (-876.0)
  [  5s] u1000_2b0kEVul.pgn                    W:    889 ->  800.0 (-89.0)  B:    990 -> 1000.0 (+10.0)
  [  4s] u1000_dSJPzhNR.pgn                    W:    970 -> 1200.0 (+230.0)  B:    871 ->  466.7 (-404.3)
  [  7s] u1000_fzpcPioo.pgn                    W:    891 ->  600.0 (-291.0)  B:    938 ->  733.3 (-204.7)

Done. Processed 12 file(s) in 62s total.

===== Alignment Summary =====
Games: 12
Mean Absolute Error (white): 338.6
Mean Absolute Error (black): 343.3
Mean Absolute Error (overall): 341.0
Avg wall time: 5s per game

CSV written to: /Users/crn/dev/projects/chess-accuracy/elo_results.csv
```