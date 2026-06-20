# List available targets.
default:
    @just --list

# Format Python files with Ruff.
format:
    uv run ruff format .
