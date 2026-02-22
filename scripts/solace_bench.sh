#!/usr/bin/env bash
# Solace — baseline benchmark harness
#
# Usage:
#   solace_bench.sh <binary> <log-dir> [big-net.nnue] [small-net.nnue]
#
# When net paths are omitted, bench and perft are skipped and the script
# records only the UCI identity (still useful for CI identity regression).
# Nets can be downloaded from:
#   https://github.com/official-stockfish/networks
#
# Outputs:
#   <log-dir>/bench_<timestamp>.log   — raw engine output
#   <log-dir>/bench_<timestamp>.json  — structured summary for comparison
#
# GPLv3 — part of the Solace project (Stockfish fork).

set -euo pipefail

SOLACE="${1:?Usage: solace_bench.sh <binary> <log-dir> [big.nnue] [small.nnue]}"
LOG_DIR="${2:?Usage: solace_bench.sh <binary> <log-dir> [big.nnue] [small.nnue]}"
BIG_NET="${3:-}"
SMALL_NET="${4:-}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/bench_${TIMESTAMP}.log"
JSON_FILE="${LOG_DIR}/bench_${TIMESTAMP}.json"
PERFT_EXPECTED=4865609

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

echo "[solace_bench] binary    : ${SOLACE}"
echo "[solace_bench] log       : ${LOG_FILE}"
echo "[solace_bench] big net   : ${BIG_NET:-<none>}"
echo "[solace_bench] small net : ${SMALL_NET:-<none>}"

# ── 1. UCI identity check (always runs; no net needed) ───────────────────────
ID_OUTPUT="$(printf 'uci\nquit\n' | "${SOLACE}" 2>&1 | tee -a "${LOG_FILE}")"
ID_NAME="$(echo "${ID_OUTPUT}"   | grep '^id name'   | sed 's/^id name //')"
ID_AUTHOR="$(echo "${ID_OUTPUT}" | grep '^id author' | sed 's/^id author //')"
UCI_OK="$(echo "${ID_OUTPUT}"    | grep -c '^uciok' || true)"

echo "[solace_bench] id name   : ${ID_NAME}"
echo "[solace_bench] id author : ${ID_AUTHOR}"
echo "[solace_bench] uciok     : ${UCI_OK}"

BENCH_NPS=0
BENCH_HASH=0
PERFT_NODES=0
PERFT_STATUS="SKIPPED"
BENCH_STATUS="SKIPPED"

if [[ -z "${BIG_NET}" || -z "${SMALL_NET}" ]]; then
    echo "[solace_bench] WARNING: net files not provided — skipping bench and perft."
    echo "[solace_bench] Download nets from: https://github.com/official-stockfish/networks"
else
    # ── 2. bench ─────────────────────────────────────────────────────────────
    echo "[solace_bench] running bench ..."
    SET_NETS="setoption name EvalFile value ${BIG_NET}\nsetoption name EvalFileSmall value ${SMALL_NET}\n"
    BENCH_OUTPUT="$(printf "${SET_NETS}bench\nquit\n" | timeout 120 "${SOLACE}" 2>&1 | tee -a "${LOG_FILE}")"
    BENCH_NPS="$(echo "${BENCH_OUTPUT}"  | grep -i 'Nodes/second' | grep -oE '[0-9]+' | tail -1 || echo 0)"
    BENCH_HASH="$(echo "${BENCH_OUTPUT}" | grep -i 'hashfull'     | grep -oE '[0-9]+' | tail -1 || echo 0)"
    if echo "${BENCH_OUTPUT}" | grep -q 'Nodes/second'; then
        BENCH_STATUS="PASS"
    else
        BENCH_STATUS="FAIL"
    fi
    echo "[solace_bench] bench NPS    : ${BENCH_NPS}"
    echo "[solace_bench] bench status : ${BENCH_STATUS}"

    # ── 3. perft depth-5 startpos ────────────────────────────────────────────
    echo "[solace_bench] running perft 5 ..."
    PERFT_OUTPUT="$(printf "${SET_NETS}position startpos\ngo perft 5\nquit\n" | timeout 60 "${SOLACE}" 2>&1 | tee -a "${LOG_FILE}")"
    PERFT_NODES="$(echo "${PERFT_OUTPUT}" | grep -i 'Nodes searched' | grep -oE '[0-9]+' | tail -1 || echo 0)"
    if [[ "${PERFT_NODES}" -eq "${PERFT_EXPECTED}" ]]; then
        PERFT_STATUS="PASS"
    elif [[ "${PERFT_NODES}" -eq 0 ]]; then
        PERFT_STATUS="FAIL"
    else
        PERFT_STATUS="FAIL (got ${PERFT_NODES}, expected ${PERFT_EXPECTED})"
    fi
    echo "[solace_bench] perft 5 nodes  : ${PERFT_NODES}"
    echo "[solace_bench] perft 5 status : ${PERFT_STATUS}"
fi

# ── 4. Structured JSON summary ────────────────────────────────────────────────
cat > "${JSON_FILE}" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "binary": "${SOLACE}",
  "id_name": "${ID_NAME}",
  "id_author": "${ID_AUTHOR}",
  "uciok": ${UCI_OK},
  "big_net": "${BIG_NET}",
  "small_net": "${SMALL_NET}",
  "bench_nps": ${BENCH_NPS},
  "bench_status": "${BENCH_STATUS}",
  "perft5_nodes": ${PERFT_NODES},
  "perft5_expected": ${PERFT_EXPECTED},
  "perft5_status": "${PERFT_STATUS}"
}
EOF

echo "[solace_bench] JSON summary : ${JSON_FILE}"
cat "${JSON_FILE}"
echo "[solace_bench] done."
