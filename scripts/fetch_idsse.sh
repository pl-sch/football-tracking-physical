#!/usr/bin/env bash
# Download the 7 Bundesliga 2022/23 matches of the IDSSE open dataset (DFL / Sportec Solutions,
# CC-BY 4.0, Bassek et al. 2025, https://doi.org/10.1038/s41597-025-04505-y) into data/idsse/.
# Run from the repo root:  bash scripts/fetch_idsse.sh
set -euo pipefail
BASE="https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
OUT="data/idsse"
mkdir -p "$OUT"
# match:competition pairs (macOS ships bash 3.2, no associative arrays)
for pair in J03WMX:DFL-COM-000001 J03WN1:DFL-COM-000001 J03WOH:DFL-COM-000002 \
            J03WOY:DFL-COM-000002 J03WPY:DFL-COM-000002 J03WQQ:DFL-COM-000002 \
            J03WR9:DFL-COM-000002; do
  m="${pair%%:*}"; c="${pair#*:}"
  for f in "DFL_02_01_matchinformation_${c}_DFL-MAT-${m}.xml" \
           "DFL_03_02_events_raw_${c}_DFL-MAT-${m}.xml" \
           "DFL_04_03_positions_raw_observed_${c}_DFL-MAT-${m}.xml"; do
    if [ -s "$OUT/$f" ]; then echo "already there: $f"; continue; fi
    echo "downloading $f"
    curl -fL --retry 3 -o "$OUT/$f.part" "$BASE/$f" && mv "$OUT/$f.part" "$OUT/$f"
  done
done
echo "done:"; ls -lh "$OUT"
