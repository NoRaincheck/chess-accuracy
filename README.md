# Chess Accuracy

The purpose of this repository is implement and test chess accuracy using different engines. Even though there are notes from [Lichess](https://lichess.org/page/accuracy), the equations and tuning are based on their chosen engine, and may not be applicable if different UCI compatible engines are used. 

This includes replicating the win-percentage formula:

$$
Pr(Win) = 50 + 50 * (2 / (1 + \exp(-0.00368208 * centipawns)) - 1)
$$

Move-by-move accuracy:

$$
Accuracy = 103.1668 * exp(-0.04354 * (winPercentBefore - winPercentAfter)) - 3.1669
$$

Game level accuracy is then calculated by:

$$
Game Accuracy = (Weighted Average Accuracy + Harmonic Mean Accuracy)/2
$$

Where `Weighted Average Accuracy` is weighted by the standard deviation over a smoothing window (they those a window of 10 in the Lichess code). 

With this high number of parameters in the equation it may be worth having a reproducible way to tune these (arbitrary) metrics

---

This repository is created via `uv`. And can be setup firstly by running:

```sh
pipx install uv
uv install
```

The example script is shown in `calculate_accuracy.py`, the current implementation is based on [Lichess notes](https://lichess.org/page/accuracy) and their [Scala implementation](https://github.com/lichess-org/lila/blob/master/modules/analyse/src/main/AccuracyPercent.scala#L38-L44). My current thought process is to package it as a standalone application using `pyinstaller` using `pydeargui` as a GUI. 

This repository is partly to learn how to build GUI applications and package them for Windows.

It will come bundled with the [Fruit v2.1 open source](https://github.com/Warpten/Fruit-2.1) chess engine for portability, but will work with any UCI compatible engine such as Lc0 or Stockfish.

## Example

```sh
uv run calculate_accuracy.py
```

Output:

```sh
[Event "Live Chess"]
[Site "Chess.com"]
[Date "2024.08.31"]
[Round "?"]
[White "Hikaru"]
[Black "DanielNaroditsky"]
[Result "0-1"]
[ECO "A21"]
[WhiteElo "3225"]
[BlackElo "3151"]
[TimeControl "180"]
[EndTime "8:20:28 PDT"]
[Termination "DanielNaroditsky won by resignation"]
[Annotator "W 90.2/78 B 95.9/97"]
[Accuracy "W 98.63% B 98.84% accuracy"]

1. c4 { [%eval 0.11] } 1... e5 { [%eval -0.10] } 2. Nc3 { [%eval -0.11] } 2... Bb4 { [%eval 0.13] } 3. g3 { [%eval -0.37] } 3... Bxc3 { [%eval -0.23] } 4. bxc3 { [%eval -0.07] } 4... d6 { [%eval 0.05] } 5. Bg2 { [%eval 0.07] } 5... Nf6 { [%eval 0.00] } 6. d3 { [%eval -0.04] } 6... O-O { [%eval -0.01] } 7. Nf3 { [%eval -0.13] } 7... Re8 { [%eval 0.07] } 8. O-O { [%eval 0.10] } 8... e4 { [%eval 0.02] } 9. Nd4 { [%eval 0.08] } 9... Nbd7 { [%eval 0.27] } 10. h3 { [%eval -0.04] } 10... Nc5 { [%eval -0.01] } 11. Be3 { [%eval -0.05] } 11... Bd7 { [%eval -0.17] } 12. Nb3 { [%eval -0.04] } 12... Na4 { [%eval 0.00] } 13. Qd2 { [%eval -0.04] } 13... c5 { [%eval 0.00] } 14. Rae1 { [%eval 0.00] } 14... Bc6 { [%eval -0.09] } 15. Qc2 { [%eval -0.24] } 15... h6 { [%eval -0.07] } 16. Kh2 { [%eval -0.39] } 16... exd3 { [%eval -0.41] } 17. exd3 { [%eval -0.45] } 17... Bxg2 { [%eval -0.41] } 18. Kxg2 { [%eval -0.39] } 18... Qd7 { [%eval -0.42] } 19. f3 { [%eval -0.52] } 19... Re6 { [%eval -0.57] } 20. Nd2 { [%eval -0.49] } 20... Rae8 { [%eval -0.18] } 21. Ne4 { [%eval -0.27] } 21... d5 { [%eval 0.00] } 22. Nxf6+ { [%eval 0.04] } 22... gxf6 { [%eval -0.06] } 23. Bf2 { [%eval -0.03] } 23... Rxe1 { [%eval -0.06] } 24. Rxe1 { [%eval -0.08] } 24... Rxe1 { [%eval -0.10] } 25. Bxe1 { [%eval -0.10] } 25... dxc4 { [%eval -0.12] } 26. dxc4 { [%eval -0.12] } 26... Nb6 { [%eval 0.12] } 27. Bf2 { [%eval 0.14] } 27... Nxc4 { [%eval 0.04] } 28. Bxc5 { [%eval 0.00] } 28... b6 { [%eval 0.00] } 29. Bd4 { [%eval -3.04] } 29... Qxd4 { [%eval -3.16] } 0-1
```

