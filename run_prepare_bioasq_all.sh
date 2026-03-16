#!/usr/bin/env bash
set -euo pipefail

INPUT_PATTERN="${1:-data/10B*_golden.json}"
OUT_DIR="${2:-data/processed}"
CHUNK_SIZE="${3:-20}"
NUM_CHUNKS="${4:-10}"
SEED="${5:-42}"

mkdir -p "$OUT_DIR"

python3 prepare_bioasq.py \
  --input-pattern "$INPUT_PATTERN" \
  --chunk-size "$CHUNK_SIZE" \
  --num-chunks "$NUM_CHUNKS" \
  --seed "$SEED" \
  --out "$OUT_DIR/bioasq_chunked.json"

python3 prepare_bioasq_gold_context.py \
  --input-pattern "$INPUT_PATTERN" \
  --chunk-size "$CHUNK_SIZE" \
  --num-chunks "$NUM_CHUNKS" \
  --seed "$SEED" \
  --out "$OUT_DIR/bioasq_chunked_gold_context.json"
