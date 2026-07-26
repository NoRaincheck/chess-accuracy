"""Shared helpers for lc0-based accuracy scripts."""

MODEL_ID = "lczerolens/BT3-768x15x24h-swa-2790000"


def wdl_to_white_cp_and_winpct(wdl, is_white_turn):
    wdl = wdl.detach()
    q_stm = wdl[0] + 0.5 * wdl[1]
    q_white = q_stm if is_white_turn else 1.0 - q_stm
    win_pct = float(q_white * 100)
    v = float(2 * q_white - 1)
    cp = 290 * v / (1 - 1.1 * v * v) if abs(v) < 0.99 else 290 * v / 0.01
    return cp, win_pct


def format_acc(label, w, b):
    return f"{label}: W {w:.2f}%  B {b:.2f}%"
