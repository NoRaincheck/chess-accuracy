This repository is created via `uv`. And can be setup firstly by running:

```sh
pipx install uv
uv install
```

The example script is shown in `calculate_accuracy.py`, the current implementation is based on [Lichess notes and their Scala implementation](https://lichess.org/page/accuracy). My current thought process is to package it as a standalone application using `pyinstaller` using `pydeargui` as a GUI. 

This repository is partly to learn how to build GUI applications and package them for Windows.

It will come bundled with the Fruit v2.1 open source chess engine for portability, but will work with any UCI compatible engine such as Lc0 or Stockfish.

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
[Accuracy "W 98.75% B 98.67% accuracy"]

1. c4 { [%eval 0.11] } 1... e5 { [%eval -0.10] } 2. Nc3 { [%eval -0.11] } 2... Bb4 { [%eval 0.36] } 3. g3 { [%eval -0.19] } 3... Bxc3 { [%eval -0.15] } 4. bxc3 { [%eval -0.07] } 4... d6 { [%eval -0.32] } 5. Bg2 { [%eval 0.06] } 5... Nf6 { [%eval 0.06] } 6. d3 { [%eval -0.08] } 6... O-O { [%eval -0.06] } 7. Nf3 { [%eval -0.12] } 7... Re8 { [%eval 0.07] } 8. O-O { [%eval 0.10] } 8... e4 { [%eval 0.08] } 9. Nd4 { [%eval 0.08] } 9... Nbd7 { [%eval 0.15] } 10. h3 { [%eval -0.04] } 10... Nc5 { [%eval -0.01] } 11. Be3 { [%eval -0.05] } 11... Bd7 { [%eval 0.00] } 12. Nb3 { [%eval -0.04] } 12... Na4 { [%eval 0.00] } 13. Qd2 { [%eval -0.04] } 13... c5 { [%eval 0.00] } 14. Rae1 { [%eval -0.03] } 14... Bc6 { [%eval -0.05] } 15. Qc2 { [%eval -0.22] } 15... h6 { [%eval -0.08] } 16. Kh2 { [%eval -0.39] } 16... exd3 { [%eval -0.36] } 17. exd3 { [%eval -0.42] } 17... Bxg2 { [%eval -0.41] } 18. Kxg2 { [%eval -0.43] } 18... Qd7 { [%eval -0.45] } 19. f3 { [%eval -0.51] } 19... Re6 { [%eval -0.56] } 20. Nd2 { [%eval -0.49] } 20... Rae8 { [%eval -0.18] } 21. Ne4 { [%eval -0.17] } 21... d5 { [%eval 0.04] } 22. Nxf6+ { [%eval -0.06] } 22... gxf6 { [%eval -0.03] } 23. Bf2 { [%eval -0.05] } 23... Rxe1 { [%eval -0.06] } 24. Rxe1 { [%eval -0.08] } 24... Rxe1 { [%eval -0.10] } 25. Bxe1 { [%eval -0.12] } 25... dxc4 { [%eval -0.12] } 26. dxc4 { [%eval -0.12] } 26... Nb6 { [%eval 0.12] } 27. Bf2 { [%eval 0.14] } 27... Nxc4 { [%eval 0.04] } 28. Bxc5 { [%eval 0.00] } 28... b6 { [%eval 0.00] } 29. Bd4 { [%eval -3.05] } 29... Qxd4 { [%eval -3.16] } 0-1
```

