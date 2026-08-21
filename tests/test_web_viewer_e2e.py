"""End-to-end tests for the web viewer (docs/) using Playwright.

Serves docs/ over a local HTTP server and drives a headless Chromium through
the real UI: PGN input -> Estimate ELO -> board with move navigation ->
top-k predicted moves at fixed ELO levels (750..2500 step 250).

Run with: uv run pytest tests/test_web_viewer_e2e.py -v
"""

import http.server
import socketserver
import threading
from functools import partial
from pathlib import Path

import pytest
from playwright.sync_api import expect

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

VIEW_ELOS = ["750", "1000", "1250", "1500", "1750", "2000", "2250", "2500"]


@pytest.fixture(scope="module")
def http_server():
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS_DIR))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
        server.shutdown()


@pytest.fixture(scope="module")
def loaded_page(http_server, browser):
    """A page with the model loaded and the example game estimated."""
    context = browser.new_context()
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda err: errors.append(err))
    page._viewer_errors = errors

    page.goto(http_server)
    # Model must load before anything is clickable
    expect(page.locator("#estimate-btn")).to_be_enabled(timeout=120_000)

    page.click("#example-btn")
    expect(page.locator("#pgn-input")).not_to_have_value("", timeout=10_000)
    page.click("#estimate-btn")

    # Board loads as soon as the PGN parses (before heavy inference finishes)
    expect(page.locator("#board .square")).to_have_count(64, timeout=30_000)
    expect(page.locator("#move-list .move-san").first).to_be_visible()

    yield page
    context.close()


def test_board_renders_initial_position(loaded_page):
    page = loaded_page
    squares = page.locator("#board .square")
    expect(squares).to_have_count(64)
    # Initial position: 32 pieces rendered
    expect(page.locator("#board .piece")).to_have_count(32)
    expect(page.locator("#ply-label")).to_have_text("Start")


def test_topk_table_structure(loaded_page):
    page = loaded_page
    # Navigate to ply 1 so predictions run
    page.click("#nav-next")
    rows = page.locator(".topk-row")
    expect(rows).to_have_count(len(VIEW_ELOS), timeout=60_000)

    # One row per fixed ELO level, in ascending order
    labels = [rows.nth(i).locator(".topk-elo").inner_text() for i in range(len(VIEW_ELOS))]
    assert labels == VIEW_ELOS

    # Each row lists k=5 moves with a percentage
    for i in range(rows.count()):
        chips = rows.nth(i).locator(".topk-chip")
        expect(chips).to_have_count(5)
        for j in range(5):
            text = chips.nth(j).inner_text()
            assert "%" in text, f"row {labels[i]} chip {j} missing probability: {text!r}"


def test_played_move_highlighted_at_ply_1(loaded_page):
    """The first move of the example (c4) should be flagged in the table."""
    page = loaded_page
    page.click("#nav-start")
    page.click("#nav-next")
    rows = page.locator(".topk-row")
    expect(rows).to_have_count(len(VIEW_ELOS), timeout=60_000)
    # The played move is a quiet opening move every Maia level plays often;
    # require it highlighted in at least one ELO row.
    played = page.locator(".topk-chip.played")
    expect(played.first).to_be_visible(timeout=10_000)


def test_navigation_buttons_and_keyboard(loaded_page):
    page = loaded_page
    page.click("#nav-start")
    expect(page.locator("#ply-label")).to_have_text("Start")

    # Next twice -> 1. c4 e5
    page.click("#nav-next")
    expect(page.locator("#ply-label")).to_contain_text("c4")
    page.click("#nav-next")
    expect(page.locator("#ply-label")).to_contain_text("e5")

    # Keyboard arrows
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#ply-label")).to_contain_text("c4")
    page.keyboard.press("ArrowRight")
    expect(page.locator("#ply-label")).to_contain_text("e5")

    # Start/end buttons
    page.click("#nav-end")
    expect(page.locator("#ply-label")).to_contain_text("Qxd4")
    expect(page.locator(".topk-empty")).to_contain_text("Game over")
    page.click("#nav-start")
    expect(page.locator("#ply-label")).to_have_text("Start")


def test_move_list_click_jumps_to_ply(loaded_page):
    page = loaded_page
    moves = page.locator("#move-list .move-san")
    moves.nth(9).click()  # 5th move pair, black's move -> ply 10
    label = page.locator("#ply-label").inner_text()
    assert label.startswith("5...")
    active = page.locator("#move-list .move-san.active")
    expect(active).to_have_count(1)


def test_last_move_squares_highlighted(loaded_page):
    page = loaded_page
    page.click("#nav-next")  # after 1. c4 -> e2/c4... actually c2->c4
    highlighted = page.locator("#board .square.last-move")
    expect(highlighted).to_have_count(2)


def test_no_uncaught_page_errors(loaded_page):
    errors = getattr(loaded_page, "_viewer_errors", [])
    assert errors == [], f"Uncaught JS errors: {errors}"
