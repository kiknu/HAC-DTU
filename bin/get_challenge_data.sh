#!/bin/bash

# Ensure unzip exists
if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is not installed. Installing it now..."
  sudo apt update
  sudo apt install -y unzip
fi

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd "$SCRIPT_DIR/.." && pwd -P)"
CURRENT_DIR="$(pwd -P)"

if [ "$CURRENT_DIR" != "$PROJECT_ROOT" ]; then
  {
    echo "Error: get_challenge_data.sh must be run from the project root."
    echo
    echo "Current directory: $CURRENT_DIR"
    echo "Project root:      $PROJECT_ROOT"
    echo
    echo "Run:"
    echo "  cd \"$PROJECT_ROOT\""
    echo "  ./bin/get_challenge_data.sh"
  } >&2
  exit 1
fi

ARCHIVE="challenge_data.zip"
OUTPUT_DIR="challenge_data"
URL="https://www.dropbox.com/scl/fo/gcqw0ffbt2xa6vfzlhu51/AESiKo8nIYUKRU_6agUcF1g?rlkey=4o0tiegpeiepxf9ikvnnzfh2s&e=1&dl=1"

echo "Downloading challenge data archive to $ARCHIVE..."
wget --show-progress --progress=bar:force:noscroll -O "$ARCHIVE" "$URL"

echo "Extracting $ARCHIVE to $OUTPUT_DIR..."
unzip "$ARCHIVE" -d "$OUTPUT_DIR"

echo "Removing temporary archive $ARCHIVE..."
rm "$ARCHIVE"

echo "Challenge data is ready in $OUTPUT_DIR."
