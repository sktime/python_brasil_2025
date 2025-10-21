#!/usr/bin/env bash

# Converts every .qmd file under book/content/pt/ into an equivalent .ipynb
# notebook, preserving the relative folder structure inside the notebooks/
# directory. Also copies image assets referenced by the notebooks so inline
# media continues to work. Requires Quarto to be installed and available on
# the PATH.

set -euo pipefail

INPUT_ROOT="book/content/pt"
OUTPUT_ROOT="notebooks"

if ! command -v quarto >/dev/null 2>&1; then
  echo "Error: quarto command not found. Please install Quarto." >&2
  exit 1
fi

if [ ! -d "$INPUT_ROOT" ]; then
  echo "Error: input directory '$INPUT_ROOT' does not exist." >&2
  exit 1
fi

find "$INPUT_ROOT" -type f -name '*.qmd' -print0 |
while IFS= read -r -d '' qmd_file; do
  rel_path="${qmd_file#$INPUT_ROOT/}"
  rel_dir="$(dirname "$rel_path")"
  base_name="$(basename "$rel_path" .qmd)"
  target_dir="$OUTPUT_ROOT/$rel_dir"
  target_file="$target_dir/$base_name.ipynb"

  mkdir -p "$target_dir"
  quarto convert "$qmd_file" --output "$target_file"
  echo "Converted $qmd_file -> $target_file"
done

# Copy supporting image assets; extend patterns below if needed.
find "$INPUT_ROOT" -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.gif' \
     -o -iname '*.svg' -o -iname '*.webp' -o -iname '*.bmp' \) -print0 |
while IFS= read -r -d '' asset_file; do
  rel_path="${asset_file#$INPUT_ROOT/}"
  target_path="$OUTPUT_ROOT/$rel_path"
  target_dir="$(dirname "$target_path")"

  mkdir -p "$target_dir"
  cp -p "$asset_file" "$target_path"
  echo "Copied asset $asset_file -> $target_path"
done
