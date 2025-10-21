#!/usr/bin/env bash

# Converts every .qmd file under book/content/pt/ into an equivalent .ipynb
# notebook, preserving the relative folder structure inside the notebooks/
# directory. Requires Quarto to be installed and available on the PATH.

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

while IFS= read -r -d '' qmd_file; do
  rel_path="${qmd_file#$INPUT_ROOT/}"
  rel_dir="$(dirname "$rel_path")"
  base_name="$(basename "$rel_path" .qmd)"
  target_dir="$OUTPUT_ROOT/$rel_dir"
  target_file="$target_dir/$base_name.ipynb"

  mkdir -p "$target_dir"
  quarto convert "$qmd_file" --output "$target_file"
  echo "Converted $qmd_file -> $target_file"
done < <(find "$INPUT_ROOT" -type f -name '*.qmd' -print0)
