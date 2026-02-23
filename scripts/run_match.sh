#!/usr/bin/env bash
# Solace — Elo match runner
#
# Runs a gauntlet between solace (engine under test) and a reference stockfish
# binary using cutechess-cli or fastchess. Writes a timestamped PGN and a
# result JSON that analyze_match.py can consume.
#
# Usage:
#   run_match.sh [options]
#
# Required:
#   --solace    <path>   Path to solace binary
#   --ref       <path>   Path to reference Stockfish binary
#   --big-net   <path>   Big NNUE net for BOTH engines (baseline)
#   --small-net <path>   Small NNUE net for BOTH engines (baseline)
#   --out-dir   <path>   Directory for PGN + JSON output
#
# Optional:
#   --games     <N>      Number of games (default: 200)
#   --tc        <str>    Time control in cutechess format (default: 10+0.1)
#   --concurrency <N>    Parallel games (default: 1)
#   --mode      <str>    Solace aggression mode: Off|Param|NNUE (default: Off)
#   --aggr-level <N>     SolaceAggressionLevel when mode=Param (default: 75)
#   --solace-net <path>  SolaceAggressionNet path when mode=NNUE
#   --openings  <path>   EPD/PGN opening book (default: built-in 8 positions)
#   --runner    <path>   Explicit path to cutechess-cli or fastchess binary
#   --hash      <MB>     Hash size for each engine in MB (default: 64)
#   --threads   <N>      Threads per engine (default: 1)
#
# Dependencies: cutechess-cli OR fastchess (auto-detected on PATH)
#
# GPLv3 — part of the Solace project (Stockfish fork).

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SOLACE_BIN=""
REF_BIN=""
BIG_NET=""
SMALL_NET=""
OUT_DIR="match_results"
GAMES=200
TC="10+0.1"
CONCURRENCY=1
AGGR_MODE="Off"
AGGR_LEVEL=75
SOLACE_NET=""
OPENINGS_FILE=""
RUNNER_BIN=""
HASH_MB=64
THREADS=1

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solace)     SOLACE_BIN="$2";    shift 2 ;;
        --ref)        REF_BIN="$2";       shift 2 ;;
        --big-net)    BIG_NET="$2";       shift 2 ;;
        --small-net)  SMALL_NET="$2";     shift 2 ;;
        --out-dir)    OUT_DIR="$2";       shift 2 ;;
        --games)      GAMES="$2";         shift 2 ;;
        --tc)         TC="$2";            shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --mode)       AGGR_MODE="$2";     shift 2 ;;
        --aggr-level) AGGR_LEVEL="$2";   shift 2 ;;
        --solace-net) SOLACE_NET="$2";    shift 2 ;;
        --openings)   OPENINGS_FILE="$2"; shift 2 ;;
        --runner)     RUNNER_BIN="$2";    shift 2 ;;
        --hash)       HASH_MB="$2";       shift 2 ;;
        --threads)    THREADS="$2";       shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
for arg_name in SOLACE_BIN REF_BIN BIG_NET SMALL_NET; do
    if [[ -z "${!arg_name}" ]]; then
        echo "ERROR: --${arg_name//_/-} is required." >&2
        exit 1
    fi
done

for f in "${SOLACE_BIN}" "${REF_BIN}" "${BIG_NET}" "${SMALL_NET}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: file not found: ${f}" >&2
        exit 1
    fi
done

# ── Locate match runner ───────────────────────────────────────────────────────
if [[ -n "${RUNNER_BIN}" ]]; then
    RUNNER="${RUNNER_BIN}"
elif command -v fastchess    &>/dev/null; then RUNNER="$(command -v fastchess)"
elif command -v cutechess-cli &>/dev/null; then RUNNER="$(command -v cutechess-cli)"
else
    cat >&2 << 'EOF'
ERROR: neither cutechess-cli nor fastchess found on PATH.

Install one of:
  cutechess-cli: https://github.com/cutechess/cutechess
  fastchess:     https://github.com/Disservin/fastchess
    apt install fastchess   # Ubuntu 24.04+
    brew install fastchess  # macOS

Or specify the binary with --runner <path>.
EOF
    exit 1
fi

RUNNER_NAME="$(basename "${RUNNER}")"
echo "[run_match] runner     : ${RUNNER} (${RUNNER_NAME})"

# ── Detect runner CLI style ───────────────────────────────────────────────────
# fastchess uses -engine ... -pgnout; cutechess-cli uses same flags but
# slightly different -each syntax. Both are largely compatible.
IS_FASTCHESS=0
[[ "${RUNNER_NAME}" == "fastchess" ]] && IS_FASTCHESS=1

# ── Build Solace UCI option string ───────────────────────────────────────────
SOLACE_OPTION_STR="option.EvalFile=${BIG_NET} option.EvalFileSmall=${SMALL_NET} option.Hash=${HASH_MB} option.Threads=${THREADS} option.SolaceAggressionMode=${AGGR_MODE}"

if [[ "${AGGR_MODE}" == "Param" ]]; then
    SOLACE_OPTION_STR="${SOLACE_OPTION_STR} option.SolaceAggressionLevel=${AGGR_LEVEL}"
fi
if [[ "${AGGR_MODE}" == "NNUE" && -n "${SOLACE_NET}" ]]; then
    if [[ ! -f "${SOLACE_NET}" ]]; then
        echo "ERROR: solace net not found: ${SOLACE_NET}" >&2; exit 1
    fi
    SOLACE_OPTION_STR="${SOLACE_OPTION_STR} option.SolaceAggressionNet=${SOLACE_NET}"
fi

REF_OPTION_STR="option.EvalFile=${BIG_NET} option.EvalFileSmall=${SMALL_NET} option.Hash=${HASH_MB} option.Threads=${THREADS}"

# ── Output paths ─────────────────────────────────────────────────────────────
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUT_DIR}"
PGN_FILE="${OUT_DIR}/match_${TIMESTAMP}.pgn"
JSON_FILE="${OUT_DIR}/match_${TIMESTAMP}.json"
LOG_FILE="${OUT_DIR}/match_${TIMESTAMP}.log"

echo "[run_match] solace     : ${SOLACE_BIN}"
echo "[run_match] reference  : ${REF_BIN}"
echo "[run_match] big net    : ${BIG_NET}"
echo "[run_match] mode       : ${AGGR_MODE} (level=${AGGR_LEVEL})"
echo "[run_match] games      : ${GAMES}"
echo "[run_match] tc         : ${TC}"
echo "[run_match] concurrency: ${CONCURRENCY}"
echo "[run_match] pgn        : ${PGN_FILE}"
echo "[run_match] json       : ${JSON_FILE}"

# ── Opening book: built-in 8 EPD positions when none provided ────────────────
TEMP_OPENINGS=""
if [[ -z "${OPENINGS_FILE}" ]]; then
    TEMP_OPENINGS="$(mktemp /tmp/solace_openings.XXXX.epd)"
    cat > "${TEMP_OPENINGS}" << 'EOF'
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 bm e5; id "Open";
rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1 bm d5; id "Closed";
rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2 bm Nf3; id "Open_reply";
rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2 bm c4; id "QGD_start";
r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 bm Bb5; id "Ruy_Lopez";
r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 3 bm exd4; id "Center_Attack";
rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 bm Nxe5; id "Petrov";
rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2 bm d4; id "French";
EOF
    OPENINGS_FILE="${TEMP_OPENINGS}"
    echo "[run_match] openings   : built-in 8 EPD positions"
else
    echo "[run_match] openings   : ${OPENINGS_FILE}"
fi

# Detect opening file format
OPENING_FMT="epd"
if [[ "${OPENINGS_FILE}" == *.pgn ]]; then OPENING_FMT="pgn"; fi

# ── Build command ─────────────────────────────────────────────────────────────
CMD_ARGS=(
    -engine name=Solace    cmd="${SOLACE_BIN}"  ${SOLACE_OPTION_STR}
    -engine name=Stockfish cmd="${REF_BIN}"     ${REF_OPTION_STR}
    -each    tc="${TC}"
    -rounds  "${GAMES}"
    -concurrency "${CONCURRENCY}"
    -openings file="${OPENINGS_FILE}" format="${OPENING_FMT}" order=random
    -pgnout "${PGN_FILE}"
    -recover
    -repeat
)

if [[ "${IS_FASTCHESS}" -eq 0 ]]; then
    CMD_ARGS+=(-resign movecount=4 score=600)
    CMD_ARGS+=(-draw movenumber=40 movecount=8 score=10)
fi

echo "[run_match] starting match ..."
"${RUNNER}" "${CMD_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

# ── Cleanup temp openings ─────────────────────────────────────────────────────
[[ -n "${TEMP_OPENINGS}" && -f "${TEMP_OPENINGS}" ]] && rm -f "${TEMP_OPENINGS}"

# ── Extract result counts from log ────────────────────────────────────────────
SCORE_LINE="$(grep -iE 'Score of Solace|Final result|Elo difference' "${LOG_FILE}" | tail -5 || true)"
W=0; D=0; L=0; TOTAL=0

# cutechess-cli: "Score of Solace vs Stockfish: W D L  [...]"
if echo "${SCORE_LINE}" | grep -qiE 'Score of Solace'; then
    RAW="$(echo "${SCORE_LINE}" | grep -iE 'Score of Solace' | grep -oE '[0-9]+ - [0-9]+ - [0-9]+' | head -1 || echo '0 - 0 - 0')"
    W="$(echo "${RAW}" | cut -d' ' -f1)"
    D="$(echo "${RAW}" | cut -d' ' -f3)"
    L="$(echo "${RAW}" | cut -d' ' -f5)"
fi
# fastchess: "Wins: W Losses: L Draws: D"
if echo "${SCORE_LINE}" | grep -qiE 'Wins:'; then
    W="$(echo "${SCORE_LINE}" | grep -oP 'Wins: \K[0-9]+'   || echo 0)"
    L="$(echo "${SCORE_LINE}" | grep -oP 'Losses: \K[0-9]+' || echo 0)"
    D="$(echo "${SCORE_LINE}" | grep -oP 'Draws: \K[0-9]+'  || echo 0)"
fi

TOTAL=$(( W + D + L ))

echo "[run_match] result: W=${W} D=${D} L=${L} total=${TOTAL}"

# ── JSON summary ──────────────────────────────────────────────────────────────
cat > "${JSON_FILE}" << EOF
{
  "timestamp":    "${TIMESTAMP}",
  "solace_bin":   "${SOLACE_BIN}",
  "ref_bin":      "${REF_BIN}",
  "aggr_mode":    "${AGGR_MODE}",
  "aggr_level":   ${AGGR_LEVEL},
  "solace_net":   "${SOLACE_NET}",
  "tc":           "${TC}",
  "games":        ${GAMES},
  "wins":         ${W},
  "draws":        ${D},
  "losses":       ${L},
  "total_played": ${TOTAL},
  "pgn_file":     "${PGN_FILE}",
  "log_file":     "${LOG_FILE}"
}
EOF

echo "[run_match] JSON : ${JSON_FILE}"
cat "${JSON_FILE}"
echo "[run_match] done. Run analyze_match.py ${JSON_FILE} for Elo + LOS."
exit "${EXIT_CODE}"
