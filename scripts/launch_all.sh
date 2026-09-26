#!/bin/bash
# scripts/launch_all.sh -- set up the first-wave arms (R0..R7, one GPU each) and print (or, with
# AUTO_SUBMIT=yes, run) the eight launch commands. The PROTO arms are a second wave and are not
# included here; run scripts/run_forcing_suite.sh R1_PROTO / R0_PROTO by hand.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_SUBMIT="${AUTO_SUBMIT:-no}"
FIRST_WAVE="R0 R1 R2 R3 R4 R5 R6 R7"

CMDS=()
for arm in $FIRST_WAVE; do
    bash "$HERE/run_forcing_suite.sh" "$arm"
    exp_dir="${BASE_DIR:-/data1/vanderbc/test_dinov2_swaraj}/forcing_ViT-S_${arm}"
    CMDS+=("$(cat "$exp_dir/LAUNCH_CMD.txt")")
done

echo ""
echo "========================================"
if [ "$AUTO_SUBMIT" = "yes" ]; then
    echo "AUTO_SUBMIT=yes: submitting ${#CMDS[@]} arms"
    for c in "${CMDS[@]}"; do echo "  $c"; bash -c "$c"; done
else
    echo "AUTO_SUBMIT=$AUTO_SUBMIT: NOT submitting. Run these ${#CMDS[@]} commands by hand:"
    for c in "${CMDS[@]}"; do echo "  $c"; done
fi
echo "========================================"
