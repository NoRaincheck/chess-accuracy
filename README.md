This repository is created via `uv`. And can be setup firstly by running:

```sh
pipx install uv
uv install
```

The example script is shown in `calculate_accuracy.py` but there is a lot more work that is needed. My current thought process is to package it as a standalone application using `pyinstaller` using `pydeargui` as a GUI. 

This repository is partly to learn how to build GUI applications and package them for Windows.

It will come bundled with the Fruit v2.1 open source chess engine for portability, but will work with any UCI compatible engine such as Lc0 or Stockfish.
