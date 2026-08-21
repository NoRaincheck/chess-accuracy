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
log-likelihood of each human move under the model's policy across candidate
ELOs, then converts the likelihood surface into a Bayesian posterior
(with a Gaussian population prior) over white and black ELO:

- Stage A: sparse joint (white, black) anchor grid over the full range. The
  log-likelihood surface is very smooth in `(w, b)`, so bicubic interpolation
  of the anchor surface localizes the per-color modes at a fraction of the
  cost of a dense sweep.
- Stage B: joint fine grid around the modes; marginal posteriors give a point
  estimate (parabola-refined), a standard deviation, and a 95% credible
  interval per color. All reported statistics come from real model
  evaluations — interpolated values are used for localization only.

Position budget: book openings (first 8 plies) and forced positions carry no
ELO signal and are skipped; long games are thinned to an even-spacing cap of
80 positions (`--max-positions`, deterministic).

The original top-1+MRR argmax sweeps are still available via `--scorer legacy`.

```sh
uv run estimate_elo.py example2.pgn
```

```sh
Loading maia3 ONNX model (maia3-5m)...
Stage A: joint anchor grid (6x6, step=550)...
  -> game center: 2238, modes: W=2050, B=2500
Stage B: joint fine grid (7x7, ±300 @ step=100)...
Final: W=2054 ± 172, B=2508 ± 156 (rate=0.5800)

Game: Hikaru vs DanielNaroditsky
WhiteElo: 3225, BlackElo: 3151

Estimated:  W   2054 ± 172   B   2508 ± 156  (rate 58.0%)
95% CI:     W [1750, 2319]   B [2200, 2759]
PGN ref:    W   3225   B   3151
```

**Hold Out Results**

Default loglik scorer (12 games from `data/`):

```
$ ./estimate_all.sh
  [ 11s] 1000-1400_huEchdBz.pgn                W:   1362 -> 1791.5 (+429.5)  B:   1170 -> 1559.4 (+389.4)
  [  5s] 1000-1400_mk9moDDq.pgn                W:   1248 -> 1821.4 (+573.4)  B:   1190 -> 1256.6 (+66.6)
  [ 18s] 1000-1400_yMk3fTsK.pgn                W:   1157 ->  880.7 (-276.3)  B:   1240 ->  670.5 (-569.5)
  [ 11s] 1700-2100_QK5egQTl.pgn                W:   1842 -> 1974.7 (+132.7)  B:   1857 -> 1809.9 (-47.1)
  [ 12s] 1700-2100_ROmEhCmX.pgn                W:   1721 -> 1862.0 (+141.0)  B:   1725 -> 2023.7 (+298.7)
  [ 16s] 1700-2100_foe2ahdY.pgn                W:   2063 -> 2007.1 (-55.9)   B:   1821 -> 1801.8 (-19.2)
  [ 14s] 2100+_BMwcT27N.pgn                    W:   2367 -> 2293.1 (-73.9)   B:   2245 -> 2339.7 (+94.7)
  [ 18s] 2100+_Q5mCQ4jR.pgn                    W:   2259 -> 1966.1 (-292.9)  B:   2351 -> 2285.8 (-65.2)
  [  7s] 2100+_jv6QQCbT.pgn                    W:   2152 -> 2162.4 (+10.4)   B:   2276 -> 1849.0 (-427.0)
  [ 13s] u1000_2b0kEVul.pgn                    W:    889 ->  651.2 (-237.8)  B:    990 -> 1161.2 (+171.2)
  [ 10s] u1000_dSJPzhNR.pgn                    W:    970 -> 1256.5 (+286.5)  B:    871 -> 1077.2 (+206.2)
  [ 16s] u1000_fzpcPioo.pgn                    W:    891 ->  786.0 (-105.0)  B:    938 -> 1173.3 (+235.3)

Done. Processed 12 file(s) in 151s total.

===== Alignment Summary =====
Games: 12
Mean Absolute Error (white): 217.9
Mean Absolute Error (black): 215.8
Mean Absolute Error (overall): 216.9
Avg wall time: 12s per game

CSV written to: /Users/crn/dev/projects/chess-accuracy/elo_results.csv
```

Accuracy is unchanged within noise versus the previous dense-grid pipeline
(MAE overall 212.9 → 216.9) while running ~3x faster (37s → 12s per game).