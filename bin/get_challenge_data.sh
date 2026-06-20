#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd "$SCRIPT_DIR/.." && pwd -P)"
DATA_DIR="${DSDIR:-$PROJECT_ROOT/challenge_data}"

if [ "$(basename "$DATA_DIR")" != "challenge_data" ]; then
  DATA_DIR="$DATA_DIR/challenge_data"
fi

if [ -e "$DATA_DIR" ] && [ ! -d "$DATA_DIR" ]; then
  {
    echo "Error: challenge data path points to a file, not a directory: $DATA_DIR"
    echo "Set DSDIR to a writable directory path or unset it to use the project default."
  } >&2
  exit 1
fi

if ! mkdir -p "$DATA_DIR"; then
  {
    echo "Error: Could not create or access challenge data directory: $DATA_DIR"
    echo "Check path permissions and try again."
  } >&2
  exit 1
fi

OUTPUT_DIR="$(CDPATH= cd "$DATA_DIR" && pwd -P)"

# Ensure unzip exists
if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip command is not available"
  echo "Please install unzip and try again."
  echo "On Debian/Ubuntu: sudo apt update && sudo apt-get install unzip"
  exit 1
fi

ARCHIVE="$(dirname "$OUTPUT_DIR")/challenge_data.zip"
echo "Challenge data will be downloaded to $OUTPUT_DIR"

URL="https://www.dropbox.com/scl/fo/gcqw0ffbt2xa6vfzlhu51/AESiKo8nIYUKRU_6agUcF1g?rlkey=4o0tiegpeiepxf9ikvnnzfh2s&e=1&dl=1"

echo "Downloading challenge data archive to $ARCHIVE..."
wget --show-progress --progress=bar:force:noscroll -O "$ARCHIVE" "$URL"

echo "Extracting $ARCHIVE to $OUTPUT_DIR (overwriting existing files)..."
unzip -o "$ARCHIVE" -d "$OUTPUT_DIR"

echo "Removing temporary archive $ARCHIVE..."
rm "$ARCHIVE"

echo "Challenge data is ready in $OUTPUT_DIR."
