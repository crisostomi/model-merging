#!/bin/bash
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/model-merging
#SBATCH --job-name=autoresearch
#SBATCH --output=./slurm/autoresearch-%j.out
#SBATCH --error=./slurm/autoresearch-%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=30000M
#SBATCH --partition=lrd_all_serial
#
# ── Self-resubmitting autoresearch launcher ────────────────────────────
#
# Runs Claude Code on the budget-free serial partition (4h limit).
# At 3h45m, sends SIGTERM and submits a continuation job.
# Continuation jobs use --continue to resume the conversation.
#
# Usage:
#   sbatch slurm/launch_autoresearch.sh                         # uses autoresearch.md
#   sbatch slurm/launch_autoresearch.sh autoresearch_general.md  # uses general template
#
# To stop the chain:  touch .stop_autoresearch
# To resume:          rm .stop_autoresearch && sbatch slurm/launch_autoresearch.sh
# To reset (fresh):   rm .autoresearch_active && sbatch slurm/launch_autoresearch.sh
# ───────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────
PROMPT_FILE="${1:-autoresearch.md}"
SCRIPT_PATH="slurm/launch_autoresearch.sh"
TIMEOUT=13500           # 3h45m — leaves 15min margin before SLURM kills at 4h
STOP_FILE=".stop_autoresearch"
SESSION_MARKER=".autoresearch_active"

# Extra claude flags — customize as needed:
#   --dangerously-skip-permissions   for fully autonomous (no approval prompts)
#   --model opus                     to pin a specific model
#   --max-budget-usd 10              to cap API spend per session
CLAUDE_EXTRA_FLAGS="${CLAUDE_EXTRA_FLAGS:-}"

# Push notifications via ntfy.sh (free, no signup)
# 1. Install ntfy app on your phone (iOS/Android)
# 2. Subscribe to your topic in the app
# 3. Set the topic here (pick something unique/unguessable):
NTFY_TOPIC="${NTFY_TOPIC:-}"
# Example: NTFY_TOPIC="dcrisost-autoresearch-abc123"

# ── notify helper ─────────────────────────────────────────────────────
notify() {
    local title="$1" msg="$2" priority="${3:-default}" tags="${4:-}"
    [ -z "$NTFY_TOPIC" ] && return 0
    curl -sf \
        -H "Title: $title" \
        -H "Priority: $priority" \
        ${tags:+-H "Tags: $tags"} \
        -d "$msg" \
        "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

# ── environment ───────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

# ── stop check ────────────────────────────────────────────────────────
if [ -f "$STOP_FILE" ]; then
    echo "Stop file found ($STOP_FILE). Remove it to resume autoresearch."
    rm -f "$SESSION_MARKER"
    exit 0
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: prompt file '$PROMPT_FILE' not found"
    exit 1
fi

mkdir -p slurm

echo "== Autoresearch job ${SLURM_JOB_ID} started at $(date) on $(hostname) =="
echo "   Prompt: $PROMPT_FILE"
echo "   Timeout: ${TIMEOUT}s ($(( TIMEOUT / 3600 ))h$(( (TIMEOUT % 3600) / 60 ))m)"
echo "   Extra flags: ${CLAUDE_EXTRA_FLAGS:-<none>}"
echo ""

# ── run claude ────────────────────────────────────────────────────────
EXIT_CODE=0

if [ -f "$SESSION_MARKER" ]; then
    # Continuation: resume the most recent conversation in this directory
    echo "Continuing previous session (--continue)..."
    timeout --signal=SIGTERM --kill-after=60 "$TIMEOUT" \
        claude --continue \
        -p "Previous session was interrupted by the SLURM time limit. Continue the autoresearch. Check flywheel state (flywheel_summarize_node_tree) and git log for any progress since the prompt file was last updated." \
        $CLAUDE_EXTRA_FLAGS \
        2>&1 || EXIT_CODE=$?
else
    # First run: start a fresh session with the full prompt
    echo "Starting new session..."
    touch "$SESSION_MARKER"
    timeout --signal=SIGTERM --kill-after=60 "$TIMEOUT" \
        claude -p "$(cat "$PROMPT_FILE")" \
        $CLAUDE_EXTRA_FLAGS \
        2>&1 || EXIT_CODE=$?
fi

echo ""
echo "== Autoresearch ended at $(date), exit code $EXIT_CODE =="

# ── resubmit on timeout ──────────────────────────────────────────────
if [ "$EXIT_CODE" -eq 124 ] && [ ! -f "$STOP_FILE" ]; then
    echo "Time limit approaching — resubmitting continuation job..."
    NEXT_JOB=$(sbatch --parsable "$SCRIPT_PATH" "$PROMPT_FILE")
    echo "Submitted job $NEXT_JOB"
    notify "Autoresearch resubmitted" "Job ${SLURM_JOB_ID} timed out. Continuation: $NEXT_JOB" "low" "arrows_counterclockwise"
else
    # Natural exit or stop requested — clean up marker
    rm -f "$SESSION_MARKER"
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "Session completed successfully."
        notify "Autoresearch finished" "Job ${SLURM_JOB_ID} completed successfully." "high" "white_check_mark"
    else
        echo "Session exited with code $EXIT_CODE. No resubmission."
        notify "Autoresearch stopped" "Job ${SLURM_JOB_ID} exited with code $EXIT_CODE." "urgent" "warning"
    fi
fi
