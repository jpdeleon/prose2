# prose2
prose2 is an extension of upstream prose photometry pipeline.
The main feature is end-to-end photometry using run_photometry script.

# Python Package Management with conda

Use conda env prose exclusively for Python package management in this project.

## Package management with pip

- All Python dependencies **must be installed, synchronized, and locked** using pip inside prose conda env

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync environment: `uv sync`
- Lock dependencies: `uv lock`

## Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools with `uv run <tool>` (e.g. `uv run pytest`, `uv run ruff`, `uv run mypy`, `uv run pre-commit`)
- Launch a Python REPL with `uv run python`


## Linting and formatting

This project uses Ruff for both linting and formatting. Do not call Black,
flake8, isort, or pylint.

- Lint: `uv run ruff check .`
- Lint and auto-fix: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Check formatting without writing: `uv run ruff format --check .`
- Always invoke Ruff through `uv run` so it resolves to the project's
  virtual environment.

Ruff configuration lives in `pyproject.toml` under `[tool.ruff]`. Do not
add a separate `ruff.toml` or `.ruff.toml`. Do not add inline `# noqa`
comments without a rule code.

## Create test, Update readme, and Git commit for each new feature
Create a unit tests. Add tests as new features are added. Organize the tests accordingly.
Update readme.md.
Create a detailed git commit after accomplishing a unique feature.

## gitignore
Do not track large files locally and in git history. Update .gitignore if needed.

## Session limit
When the prompt starts `session limit`, it means we hit a session limit and restarted. 
The goal is marked with `goal`. Based on git status, the modified files are marked with
`file1`. Review these local files, check the current implementation, and tell me what the immediate next step should be.

## Prompt
Ask questions for clarifications if prompt is vague or confusing.
Verify non-obvious assumptions before implementing edit.
