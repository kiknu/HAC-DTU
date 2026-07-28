set minimum-version := '1.55.0'
set default-list := true

install: update-submodules
    uv sync -v

# Format Python files with Ruff.
format:
    uv run ruff check --select I --fix --exclude "*.ipynb" .
    uv run ruff format --exclude "*.ipynb" .

# Update the submodules
update-submodules:
    git submodule update --init --recursive
