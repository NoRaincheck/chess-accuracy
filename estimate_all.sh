#!/usr/bin/env bash
# Run estimate_elo.py on all PGN files in data/ using maia3-79m with no sampling.
# Outputs estimate_all.json (NDJSON) and elo_results.csv with per-game ELO, mean, and difference.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
JSON_FILE="$SCRIPT_DIR/estimate_all.json"
CSV_FILE="$SCRIPT_DIR/elo_results.csv"

# Start fresh
> "$JSON_FILE"

count=0
for pgn in "$DATA_DIR"/*.pgn; do
    [ -f "$pgn" ] || continue
    count=$((count + 1))
    echo "============================================"
    echo "Processing: $(basename "$pgn")"
    echo "============================================"
    python3 "$SCRIPT_DIR/estimate_elo.py" "$pgn" --json >> "$JSON_FILE"
done

echo "Done. Processed $count file(s)."
echo ""

# Filter to only JSON object lines
JSONLINES=$(grep '^{' "$JSON_FILE" || true)

if [ -z "$JSONLINES" ]; then
    echo "Error: no JSON results found" >&2
    exit 1
fi

# Build CSV: one JSON object per line → CSV with mean & diff
# First pass: compute mean corrected_elo across all games
mean_elo=$(echo "$JSONLINES" | jq -s '[.[].corrected_elo] | add / length')

# Second pass: write CSV with mean and per-game difference
{
    echo "game_file,white,black,white_elo_hdr,black_elo_hdr,raw_elo,corrected_elo,peak_rate,n_evaluations,sampled,diff_from_mean"
    echo "$JSONLINES" | jq -r --arg mean "$mean_elo" '
        . as $r |
        ($r.corrected_elo - ($mean | tonumber)) as $diff |
        [
            ($r.white + " vs " + $r.black),
            $r.white,
            $r.black,
            ($r.white_elo_hdr | tostring),
            ($r.black_elo_hdr | tostring),
            ($r.raw_elo | tostring),
            ($r.corrected_elo | tostring),
            ($r.peak_rate | tostring),
            (if $r.n_evaluations then ($r.n_evaluations | tostring) else "null" end),
            (if $r.sampled then "true" else "false" end),
            ($diff | tostring)
        ] | @csv
    '
} > "$CSV_FILE"

echo "JSON written to: $JSON_FILE"
echo "CSV written to:  $CSV_FILE"
echo "Mean corrected ELO: $mean_elo"
echo ""
head -5 "$CSV_FILE"
echo "..."
echo "(total $count games)"
