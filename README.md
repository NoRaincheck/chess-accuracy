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

Estimates per-color ELO ratings from a PGN game by scoring how well the maia3
model explains the game's actual moves. The default scorer ("loglik") sums the
log-likelihood of each human move under the model's policy across a grid of
candidate ELOs, then converts the likelihood curve into a Bayesian posterior
(with a Gaussian population prior) over white and black ELO:

- Stage 1: 1D sweep assuming both players share one ELO (localization).
- Stage 2a: joint (white, black) coarse grid over the full range — catches
  lopsided games that a shared-ELO assumption would clip.
- Stage 2b: joint fine grid around the per-color modes; marginal posteriors
  give a point estimate (parabola-refined), a standard deviation, and a 95%
  credible interval per color.

The original top-1+MRR argmax sweeps are still available via `--scorer legacy`.

```sh
uv run estimate_elo.py example2.pgn
```

```sh
Loading maia3 ONNX model (maia3-5m)...
Stage 1: 1D likelihood sweep (15 values, step=200)...
  -> game center: 2523 (σ=148)
Stage 2a: joint coarse grid (12x12, step=250)...
  -> modes: W=2050, B=2550
Stage 2b: joint fine grid (7x7, ±300 @ step=100)...
Final: W=2063 ± 163, B=2598 ± 141 (rate=0.5862)

Game: Hikaru vs DanielNaroditsky
WhiteElo: 3225, BlackElo: 3151

Estimated:  W   2063 ± 163   B   2598 ± 141  (rate 58.6%)
95% CI:     W [1750, 2314]   B [2255, 2811]
PGN ref:    W   3225   B   3151
```

**Hold Out Results**

Default loglik scorer (12 games from `data/`):

```
$ ./estimate_all.sh
  [ 31s] 1000-1400_huEchdBz.pgn                W:   1362 -> 1826.9 (+464.9)  B:   1170 -> 1367.8 (+197.8)
  [ 14s] 1000-1400_mk9moDDq.pgn                W:   1248 -> 1820.2 (+572.2)  B:   1190 -> 1203.5 (+13.5)
  [ 47s] 1000-1400_yMk3fTsK.pgn                W:   1157 ->  914.3 (-242.7)  B:   1240 ->  701.4 (-538.6)
  [ 43s] 1700-2100_foe2ahdY.pgn                W:   2063 -> 1956.6 (-106.4)  B:   1821 -> 1762.3 (-58.7)
  [ 31s] 1700-2100_QK5egQTl.pgn                W:   1842 -> 1982.8 (+140.8)  B:   1857 -> 1872.0 (+15.0)
  [ 33s] 1700-2100_ROmEhCmX.pgn                W:   1721 -> 1876.1 (+155.1)  B:   1725 -> 2003.4 (+278.4)
  [ 38s] 2100+_BMwcT27N.pgn                    W:   2367 -> 2297.2 (-69.8)  B:   2245 -> 2355.3 (+110.3)
  [ 21s] 2100+_jv6QQCbT.pgn                    W:   2152 -> 2287.9 (+135.9)  B:   2276 -> 1862.7 (-413.3)
  [ 69s] 2100+_Q5mCQ4jR.pgn                    W:   2259 -> 2082.0 (-177.0)  B:   2351 -> 2259.2 (-91.8)
  [ 35s] u1000_2b0kEVul.pgn                    W:    889 ->  626.2 (-262.8)  B:    990 -> 1148.3 (+158.3)
  [ 27s] u1000_dSJPzhNR.pgn                    W:    970 -> 1252.1 (+282.1)  B:    871 -> 1213.2 (+342.2)
  [ 43s] u1000_fzpcPioo.pgn                    W:    891 ->  781.2 (-109.8)  B:    938 -> 1109.2 (+171.2)

Done. Processed 12 file(s) in 432s total.

===== Alignment Summary =====
Games: 12
Mean Absolute Error (white): 226.6
Mean Absolute Error (black): 199.1
Mean Absolute Error (overall): 212.9
Avg wall time: 36s per game

CSV written to: /Users/crn/dev/projects/chess-accuracy/elo_results.csv
```