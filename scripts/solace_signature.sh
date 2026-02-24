#!/usr/bin/env bash
# Solace — bench signature helper
#
# Captures or verifies the bench node count (the "signature") for the Solace
# binary in baseline mode (SolaceAggressionMode=Off), ensuring that search
# changes do not silently alter perft-equivalent node counts.
#
# Usage:
#   # Capture and print signature:
#   tests/solace_signature.sh <binary>
#
#   # Verify against a known reference:
#   tests/solace_signature.sh <binary> <reference_signature>
#
#   # Update the pinned signature file:
#   tests/solace_signature.sh <binary> --update tests/solace_bench_sig.txt
#
# Environment:
#   SOLACE_BIG_NET    path to big NNUE net (required for an actual bench run)
#   SOLACE_SMALL_NET  path to small NNUE net
#   If nets are absent the script exits 2 (SKIP) — suitable for CI without nets.
#
# Exit codes:
#   0  signature matched (or captured successfully)
#   1  signature mismatch or bench failed
#   2  nets absent — bench skipped (CI-safe)
#
# GPLv3 — part of the Solace project (Stockfish fork).

set -euo pipefail

BINARY="${1:?Usage: solace_signature.sh <binary> [reference|--update <file>]}"
MODE="${2:-}"
UPDATE_FILE="${3:-}"

BIG_NET="${SOLACE_BIG_NET:-}"
SMALL_NET="${SOLACE_SMALL_NET:-}"

STDOUT="$(mktemp)"
STDERR="$(mktemp)"
trap 'rm -f "$STDOUT" "$STDERR"' EXIT

# ── Net check ────────────────────────────────────────────────────────────────
if [[ -z "$BIG_NET" || -z "$SMALL_NET" || ! -f "$BIG_NET" || ! -f "$SMALL_NET" ]]; then
    echo "[sig] SKIP: SOLACE_BIG_NET / SOLACE_SMALL_NET not set or files missing."
    echo "[sig] Set env vars to enable bench signature testing."
    exit 2
fi

# ── Run bench in Off mode (baseline-identical) ───────────────────────────────
BENCH_CMD=$(printf \
    "setoption name EvalFile value %s\nsetoption name EvalFileSmall value %s\nsetoption name SolaceAggressionMode value Off\nbench\nquit\n" \
    "$BIG_NET" "$SMALL_NET")

echo "[sig] Running bench (mode=Off) on $BINARY ..."
echo "$BENCH_CMD" | timeout 180 "$BINARY" > "$STDOUT" 2> "$STDERR" || {
    echo "[sig] FAIL: bench command failed."
    echo "=== STDERR ===" ; cat "$STDERR"
    exit 1
}

SIGNATURE=$(grep "Nodes searched  :" "$STDERR" | awk '{print $4}' || true)
if [[ -z "$SIGNATURE" ]]; then
    echo "[sig] FAIL: could not extract 'Nodes searched' from bench output."
    echo "=== STDERR ===" ; cat "$STDERR"
    exit 1
fi

echo "[sig] Signature: $SIGNATURE"

# ── Compare / capture ────────────────────────────────────────────────────────
if [[ "$MODE" == "--update" && -n "$UPDATE_FILE" ]]; then
    echo "$SIGNATURE" > "$UPDATE_FILE"
    echo "[sig] Updated reference: $UPDATE_FILE"
    exit 0
fi

if [[ -n "$MODE" && "$MODE" != "--update" ]]; then
    # MODE is the reference value
    REFERENCE="$MODE"
    if [[ "$SIGNATURE" == "$REFERENCE" ]]; then
        echo "[sig] OK: signature matches reference $REFERENCE"
        exit 0
    else
        echo "[sig] FAIL: expected $REFERENCE got $SIGNATURE"
        exit 1
    fi
fi

# No reference provided — just print
echo "$SIGNATURE"
exit 0
