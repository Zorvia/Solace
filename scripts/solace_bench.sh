#!/usr/bin/env bash
# Solace — benchmark harness
#
# Usage:
#   solace_bench.sh <binary> <log-dir> [big.nnue] [small.nnue] [solace_net.nnue]
#
# Arguments:
#   binary          Path to the solace executable
#   log-dir         Directory to write timestamped logs and JSON
#   big.nnue        Stockfish baseline big net (nn-*.nnue from official-stockfish/networks)
#   small.nnue      Stockfish baseline small net
#   solace_net.nnue Solace aggression-trained net (from nnue_export.py)
#                   When provided, a second bench run with SolaceAggressionMode=NNUE
#                   is executed and aggression stats are captured.
#
# Outputs:
#   <log-dir>/bench_<timestamp>.log   — raw engine stdout
#   <log-dir>/bench_<timestamp>.json  — structured summary for regression comparison
#
# GPLv3 — part of the Solace project (Stockfish fork).

set -euo pipefail

SOLACE="${1:?Usage: solace_bench.sh <binary> <log-dir> [big.nnue] [small.nnue] [solace_net.nnue]}"
LOG_DIR="${2:?Usage: solace_bench.sh <binary> <log-dir> [big.nnue] [small.nnue] [solace_net.nnue]}"
BIG_NET="${3:-}"
SMALL_NET="${4:-}"
SOLACE_NET="${5:-}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/bench_${TIMESTAMP}.log"
JSON_FILE="${LOG_DIR}/bench_${TIMESTAMP}.json"
PERFT_EXPECTED=4865609

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

echo "[solace_bench] binary      : ${SOLACE}"
echo "[solace_bench] log         : ${LOG_FILE}"
echo "[solace_bench] big net     : ${BIG_NET:-<none>}"
echo "[solace_bench] small net   : ${SMALL_NET:-<none>}"
echo "[solace_bench] solace net  : ${SOLACE_NET:-<none>}"

# ── 1. UCI identity (always runs — no net needed) ────────────────────────────
ID_OUTPUT="$(printf 'uci\nquit\n' | "${SOLACE}" 2>&1 | tee -a "${LOG_FILE}")"
ID_NAME="$(echo "${ID_OUTPUT}"   | grep '^id name'   | sed 's/^id name //'   || true)"
ID_AUTHOR="$(echo "${ID_OUTPUT}" | grep '^id author' | sed 's/^id author //' || true)"
UCI_OK="$(echo "${ID_OUTPUT}"    | grep -c '^uciok'  || true)"

echo "[solace_bench] id name     : ${ID_NAME}"
echo "[solace_bench] id author   : ${ID_AUTHOR}"
echo "[solace_bench] uciok       : ${UCI_OK}"

SOLACE_OPTS_OK=0
echo "${ID_OUTPUT}" | grep -q "SolaceAggressionMode"  && \
echo "${ID_OUTPUT}" | grep -q "SolaceAggressionLevel" && \
echo "${ID_OUTPUT}" | grep -q "SolaceAggressionNet"   && SOLACE_OPTS_OK=1
echo "[solace_bench] solace opts : ${SOLACE_OPTS_OK}"

# ── Result variables ─────────────────────────────────────────────────────────
BENCH_NPS=0; BENCH_STATUS="SKIPPED"
PERFT_NODES=0; PERFT_STATUS="SKIPPED"
AGGR_BENCH_NPS=0; AGGR_BENCH_STATUS="SKIPPED"
AGGR_TOTAL_MOVES=0; AGGR_SACRIFICES=0; AGGR_SAC_PER_1K=0
AGGR_KING_ATTACKS=0; AGGR_KING_PER_1K=0; AGGR_DRAW_VICINITY=0

if [[ -z "${BIG_NET}" || -z "${SMALL_NET}" ]]; then
    echo "[solace_bench] WARNING: baseline nets not provided — skipping bench/perft."
    echo "[solace_bench]   Download from: https://github.com/official-stockfish/networks"
else
    NET_OPTS="setoption name EvalFile value ${BIG_NET}
setoption name EvalFileSmall value ${SMALL_NET}"

    # ── 2. Baseline bench ────────────────────────────────────────────────────
    echo "[solace_bench] running baseline bench ..."
    BENCH_OUTPUT="$(printf '%s\nbench\nquit\n' "${NET_OPTS}" | timeout 120 "${SOLACE}" 2>&1 | tee -a "${LOG_FILE}")"
    BENCH_NPS="$(echo "${BENCH_OUTPUT}" | grep -i 'Nodes/second' | grep -oE '[0-9]+' | tail -1 || echo 0)"
    echo "${BENCH_OUTPUT}" | grep -q 'Nodes/second' && BENCH_STATUS="PASS" || BENCH_STATUS="FAIL"
    echo "[solace_bench] baseline NPS    : ${BENCH_NPS}"
    echo "[solace_bench] baseline status : ${BENCH_STATUS}"

    # ── 3. Perft depth-5 ────────────────────────────────────────────────────
    echo "[solace_bench] running perft 5 ..."
    PERFT_OUTPUT="$(printf '%s\nposition startpos\ngo perft 5\nquit\n' "${NET_OPTS}" | timeout 60 "${SOLACE}" 2>&1 | tee -a "${LOG_FILE}")"
    PERFT_NODES="$(echo "${PERFT_OUTPUT}" | grep -i 'Nodes searched' | grep -oE '[0-9]+' | tail -1 || echo 0)"
    if   [[ "${PERFT_NODES}" -eq "${PERFT_EXPECTED}" ]]; then PERFT_STATUS="PASS"
    elif [[ "${PERFT_NODES}" -eq 0 ]];                   then PERFT_STATUS="FAIL"
    else PERFT_STATUS="FAIL (got ${PERFT_NODES}, expected ${PERFT_EXPECTED})"
    fi
    echo "[solace_bench] perft 5 nodes  : ${PERFT_NODES}"
    echo "[solace_bench] perft 5 status : ${PERFT_STATUS}"

    # ── 4. Solace NNUE aggression bench (optional) ───────────────────────────
    if [[ -n "${SOLACE_NET}" ]]; then
        echo "[solace_bench] running Solace NNUE bench ..."
        AGGR_OUTPUT="$(printf '%s\nsetoption name SolaceAggressionMode value NNUE\nsetoption name SolaceAggressionNet value %s\nbench\nquit\n' \
            "${NET_OPTS}" "${SOLACE_NET}" | timeout 180 "${SOLACE}" 2>&1 | tee -a "${LOG_FILE}")"

        AGGR_BENCH_NPS="$(echo "${AGGR_OUTPUT}" | grep -i 'Nodes/second' | grep -oE '[0-9]+' | tail -1 || echo 0)"
        echo "${AGGR_OUTPUT}" | grep -q 'Nodes/second' && AGGR_BENCH_STATUS="PASS" || AGGR_BENCH_STATUS="FAIL"

        AGGR_LINE="$(echo "${AGGR_OUTPUT}" | grep 'info string solace_aggr' | tail -1 || true)"
        if [[ -n "${AGGR_LINE}" ]]; then
            AGGR_TOTAL_MOVES="$( echo "${AGGR_LINE}" | grep -oP 'total_moves \K[0-9]+'    || echo 0)"
            AGGR_SACRIFICES="$(  echo "${AGGR_LINE}" | grep -oP 'sacrifices \K[0-9]+'     || echo 0)"
            AGGR_SAC_PER_1K="$(  echo "${AGGR_LINE}" | grep -oP 'sac_per_1k \K[0-9]+'    || echo 0)"
            AGGR_KING_ATTACKS="$(echo "${AGGR_LINE}" | grep -oP 'king_attacks \K[0-9]+'   || echo 0)"
            AGGR_KING_PER_1K="$( echo "${AGGR_LINE}" | grep -oP 'king_per_1k \K[0-9]+'   || echo 0)"
            AGGR_DRAW_VICINITY="$(echo "${AGGR_LINE}" | grep -oP 'draw_vicinity \K[0-9]+' || echo 0)"
            echo "[solace_bench] aggr NPS          : ${AGGR_BENCH_NPS}"
            echo "[solace_bench] aggr sacrifices   : ${AGGR_SACRIFICES} (${AGGR_SAC_PER_1K}/1k)"
            echo "[solace_bench] aggr king attacks : ${AGGR_KING_ATTACKS} (${AGGR_KING_PER_1K}/1k)"
            echo "[solace_bench] aggr draw vicinity: ${AGGR_DRAW_VICINITY}"
        else
            echo "[solace_bench] WARNING: no solace_aggr line in output (nets may not be loaded)."
        fi
    fi
fi

# ── 5. JSON summary ───────────────────────────────────────────────────────────
cat > "${JSON_FILE}" << EOF
{
  "timestamp":           "${TIMESTAMP}",
  "binary":              "${SOLACE}",
  "id_name":             "${ID_NAME}",
  "id_author":           "${ID_AUTHOR}",
  "uciok":               ${UCI_OK},
  "solace_options_ok":   ${SOLACE_OPTS_OK},
  "big_net":             "${BIG_NET}",
  "small_net":           "${SMALL_NET}",
  "solace_net":          "${SOLACE_NET}",
  "bench_nps":           ${BENCH_NPS},
  "bench_status":        "${BENCH_STATUS}",
  "perft5_nodes":        ${PERFT_NODES},
  "perft5_expected":     ${PERFT_EXPECTED},
  "perft5_status":       "${PERFT_STATUS}",
  "aggr_bench_nps":      ${AGGR_BENCH_NPS},
  "aggr_bench_status":   "${AGGR_BENCH_STATUS}",
  "aggr_total_moves":    ${AGGR_TOTAL_MOVES},
  "aggr_sacrifices":     ${AGGR_SACRIFICES},
  "aggr_sac_per_1k":     ${AGGR_SAC_PER_1K},
  "aggr_king_attacks":   ${AGGR_KING_ATTACKS},
  "aggr_king_per_1k":    ${AGGR_KING_PER_1K},
  "aggr_draw_vicinity":  ${AGGR_DRAW_VICINITY}
}
EOF

echo "[solace_bench] JSON : ${JSON_FILE}"
cat "${JSON_FILE}"
echo "[solace_bench] done."
