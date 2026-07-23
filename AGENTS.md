# AGENTS.md

This repo uses **uv** for Python dependency management and virtual environments.

## Key Commands

- `uv sync` — Install all dependencies from `pyproject.toml` and `uv.lock`
- `uv add <package>` — Add a dependency and update lockfile
- `uv run <script>` — Run a script in the project's virtual environment
- `uv venv` — Create a virtual environment (`.venv`)

## Environment

- Python: `>=3.10,<3.13`
- Virtual environment: `.venv` (created by `uv`)
- Lockfile: `uv.lock`
