#!/usr/bin/env bash
# Run estimate_elo.py on all PGN files in data/ and report speed + alignment with header ELO.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
CSV_FILE="$SCRIPT_DIR/elo_results.csv"

count=0
total_seconds=0
sum_abs_w=0
sum_abs_b=0
n_w=0
n_b=0
n_total=0

# CSV header
echo "game_file,white,black,white_elo_hdr,black_elo_hdr,est_white_elo,est_black_elo,white_std,black_std,white_ci95,black_ci95,peak_rate,n_evaluations,n_positions,n_moves,sampled,wall_seconds" > "$CSV_FILE"

for pgn in "$DATA_DIR"/*.pgn; do
    [ -f "$pgn" ] || continue
    count=$((count + 1))
    fname=$(basename "$pgn")

    SECONDS=0
    result=$(uv run python3 "$SCRIPT_DIR/estimate_elo.py" "$pgn" --json --quiet 2>/dev/null)
    elapsed=$SECONDS
    total_seconds=$((total_seconds + elapsed))

    # Extract fields
    white=$(echo "$result" | jq -r '.white')
    black=$(echo "$result" | jq -r '.black')
    wh=$(echo "$result" | jq -r '.white_elo_hdr')
    bh=$(echo "$result" | jq -r '.black_elo_hdr')
    ew=$(echo "$result" | jq -r '.est_white_elo')
    eb=$(echo "$result" | jq -r '.est_black_elo')
    pr=$(echo "$result" | jq -r '.peak_rate')
    ne=$(echo "$result" | jq -r '.n_evaluations')
    nm=$(echo "$result" | jq -r '.n_moves')
    sa=$(echo "$result" | jq -r '.sampled')
    ws=$(echo "$result" | jq -r '.white_std // ""')
    bs=$(echo "$result" | jq -r '.black_std // ""')
    wci=$(echo "$result" | jq -r 'if .white_ci95 then "[\(.white_ci95[0]|round),\(.white_ci95[1]|round)]" else "" end')
    bci=$(echo "$result" | jq -r 'if .black_ci95 then "[\(.black_ci95[0]|round),\(.black_ci95[1]|round)]" else "" end')
    np_=$(echo "$result" | jq -r '.n_positions // ""')

    # Write CSV row
    printf '"%s vs %s","%s","%s","%s","%s",%s,%s,%s,%s,"%s","%s",%s,%s,%s,%s,%s,%s\n' \
        "$white" "$black" "$white" "$black" "$wh" "$bh" "$ew" "$eb" "$ws" "$bs" "$wci" "$bci" "$pr" "$ne" "$np_" "$nm" "$sa" "$elapsed" \
        >> "$CSV_FILE"

    # Accumulate alignment stats (header ELO can be numeric or "?"; skip non-numeric)
    w_diff=""
    b_diff=""
    if [[ "$wh" =~ ^[0-9]+$ ]]; then
        w_diff=$(python3 -c "print(round(abs($ew - $wh), 1))")
        sum_abs_w=$(python3 -c "print($sum_abs_w + $w_diff)")
        n_w=$((n_w + 1))
    fi
    if [[ "$bh" =~ ^[0-9]+$ ]]; then
        b_diff=$(python3 -c "print(round(abs($eb - $bh), 1))")
        sum_abs_b=$(python3 -c "print($sum_abs_b + $b_diff)")
        n_b=$((n_b + 1))
    fi
    n_total=$((n_total + 1))

    # Compute signed diffs for display
    w_signed="${w_diff:-?}"
    b_signed="${b_diff:-?}"
    if [[ -n "$w_diff" ]]; then
        w_signed=$(python3 -c "d=$ew - ($wh); print(f'+{d:.1f}' if d >= 0 else f'{d:.1f}')")
    fi
    if [[ -n "$b_diff" ]]; then
        b_signed=$(python3 -c "d=$eb - ($bh); print(f'+{d:.1f}' if d >= 0 else f'{d:.1f}')")
    fi

    printf "  [%3ds] %-36s  W: %6s -> %6s (%s)  B: %6s -> %6s (%s)\n" \
        "$elapsed" "$fname" "$wh" "$ew" "$w_signed" "$bh" "$eb" "$b_signed"
done

echo ""
echo "Done. Processed $count file(s) in ${total_seconds}s total."
echo ""
echo "===== Alignment Summary ====="
echo "Games: $count"

if [ "$n_w" -gt 0 ]; then
    mae_w=$(python3 -c "print(round($sum_abs_w / $n_w, 1))")
    echo "Mean Absolute Error (white): $mae_w"
fi
if [ "$n_b" -gt 0 ]; then
    mae_b=$(python3 -c "print(round($sum_abs_b / $n_b, 1))")
    echo "Mean Absolute Error (black): $mae_b"
fi
if [ "$n_w" -gt 0 ] && [ "$n_b" -gt 0 ]; then
    mae_all=$(python3 -c "print(round(($sum_abs_w + $sum_abs_b) / ($n_w + $n_b), 1))")
    echo "Mean Absolute Error (overall): $mae_all"
fi

avg=$((total_seconds / count))
echo "Avg wall time: ${avg}s per game"
echo ""
echo "CSV written to: $CSV_FILE"
