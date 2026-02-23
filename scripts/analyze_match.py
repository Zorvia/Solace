#!/usr/bin/env python3
"""
Solace — match result analyzer
================================
Reads a match JSON (from run_match.sh) and optionally the companion PGN file to
compute:
  - Elo difference (Δelo) between Solace and the reference engine
  - 95% confidence interval on Δelo
  - LOS — Likelihood Of Superiority (probability Solace is stronger)
  - Draw rate, win rate
  - Whether the result meets the "beats Stockfish" threshold
  - Per-game sacrifice frequency (from solace_aggr info strings in PGN comments)

Statistical model:
  Uses the standard pentanomial + trinomial Elo estimation from Fishtest.
  For W/D/L without pentanomial data we fall back to the trinomial model:
    score = (W + 0.5*D) / N
    Δelo  = -400 * log10(1/score - 1)
    se    = sqrt(score*(1-score)/N) * 400 / (ln(10) * score*(1-score))
    CI    = Δelo ± 1.96*se
    LOS   = 0.5 * erfc(-( W-L ) / sqrt(2*(W+L)))

Usage:
    python3 analyze_match.py match_results/match_20260223T123456Z.json
    python3 analyze_match.py match_results/match_*.json --threshold 10
    python3 analyze_match.py match_results/match_*.json --json

Exit code 0 if LOS >= 95% AND Δelo >= threshold, else 1 (useful for CI gates).

GPLv3 — part of the Solace project (Stockfish fork).
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional


# ── Elo statistics ────────────────────────────────────────────────────────────

def elo_from_score(score: float) -> float:
    if score <= 0.0 or score >= 1.0:
        return float("nan")
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_se(wins: int, draws: int, losses: int) -> float:
    n = wins + draws + losses
    if n == 0:
        return float("nan")
    score = (wins + 0.5 * draws) / n
    if score <= 0.0 or score >= 1.0:
        return float("nan")
    # Variance of the score under the trinomial model
    mu2 = score * score
    var = (wins * (1.0 - score) ** 2
           + draws * (0.5 - score) ** 2
           + losses * (0.0 - score) ** 2) / n
    # Propagate to Elo via delta method: σ_Elo = |dElo/dScore| * σ_score
    d_elo_d_score = 400.0 / (math.log(10.0) * score * (1.0 - score))
    return d_elo_d_score * math.sqrt(var / n)


def los(wins: int, losses: int) -> float:
    """
    Likelihood of superiority: P(engine A is stronger than B).
    Uses the normal approximation erfc.
    """
    total = wins + losses
    if total == 0:
        return 0.5
    z = (wins - losses) / math.sqrt(total)
    # CDF of standard normal via erfc approximation
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# ── PGN parser: aggregate solace_aggr comments + basic result extraction ──────

RESULT_RE  = re.compile(r'\[Result\s+"([^"]+)"\]')
WHITE_RE   = re.compile(r'\[White\s+"([^"]+)"\]')
AGGR_RE    = re.compile(
    r'info string solace_aggr'
    r'.*?total_moves\s+(\d+)'
    r'.*?sacrifices\s+(\d+)'
    r'.*?king_attacks\s+(\d+)'
)


def parse_pgn(pgn_path: Path, solace_name: str = "Solace"):
    """
    Returns (wins, draws, losses, aggr_stats_list) from Solace's perspective.
    aggr_stats_list: list of per-game dicts with 'total_moves', 'sacrifices', 'king_attacks'.
    """
    wins = draws = losses = 0
    aggr_list = []

    with pgn_path.open(encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    # Split into games by double newline before first [Event
    games = re.split(r'\n(?=\[Event\s)', text)

    for game in games:
        result_m = RESULT_RE.search(game)
        white_m  = WHITE_RE.search(game)
        if not result_m:
            continue
        result = result_m.group(1)
        white  = white_m.group(1) if white_m else ""
        solace_is_white = solace_name.lower() in white.lower()

        if result == "1-0":
            if solace_is_white: wins   += 1
            else:               losses += 1
        elif result == "0-1":
            if solace_is_white: losses += 1
            else:               wins   += 1
        elif result in ("1/2-1/2", "½-½"):
            draws += 1

        aggr_m = AGGR_RE.search(game)
        if aggr_m:
            aggr_list.append({
                "total_moves":  int(aggr_m.group(1)),
                "sacrifices":   int(aggr_m.group(2)),
                "king_attacks": int(aggr_m.group(3)),
            })

    return wins, draws, losses, aggr_list


def aggregate_aggr(aggr_list: list) -> dict:
    if not aggr_list:
        return {}
    total_moves  = sum(g["total_moves"]  for g in aggr_list)
    sacrifices   = sum(g["sacrifices"]   for g in aggr_list)
    king_attacks = sum(g["king_attacks"] for g in aggr_list)
    n = len(aggr_list)
    return {
        "games_with_data": n,
        "total_moves":     total_moves,
        "sacrifices":      sacrifices,
        "sac_per_1k":      round(1000.0 * sacrifices / total_moves, 2) if total_moves else 0.0,
        "king_attacks":    king_attacks,
        "king_per_1k":     round(1000.0 * king_attacks / total_moves, 2) if total_moves else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def analyze(match_json: Path, elo_threshold: float, as_json: bool) -> bool:
    with match_json.open() as fh:
        meta = json.load(fh)

    wins   = int(meta.get("wins",   0))
    draws  = int(meta.get("draws",  0))
    losses = int(meta.get("losses", 0))
    total  = int(meta.get("total_played", wins + draws + losses))

    pgn_path_str = meta.get("pgn_file", "")
    aggr_data: dict = {}

    if pgn_path_str:
        pgn_path = Path(pgn_path_str)
        if pgn_path.exists():
            pgn_w, pgn_d, pgn_l, aggr_list = parse_pgn(pgn_path)
            # Prefer PGN-derived counts if available and non-trivially different
            if pgn_w + pgn_d + pgn_l > 0:
                wins, draws, losses = pgn_w, pgn_d, pgn_l
                total = wins + draws + losses
            aggr_data = aggregate_aggr(aggr_list)

    if total == 0:
        print(f"[analyze] No game results found in {match_json}.", file=sys.stderr)
        return False

    score    = (wins + 0.5 * draws) / total
    elo_diff = elo_from_score(score)
    se       = elo_se(wins, draws, losses)
    ci_lo    = elo_diff - 1.96 * se
    ci_hi    = elo_diff + 1.96 * se
    los_val  = los(wins, losses)
    draw_rate = draws / total
    win_rate  = wins  / total

    beats = (los_val >= 0.95) and (elo_diff >= elo_threshold)

    result = {
        "match_file":    str(match_json),
        "solace_bin":    meta.get("solace_bin", ""),
        "ref_bin":       meta.get("ref_bin",    ""),
        "aggr_mode":     meta.get("aggr_mode",  ""),
        "tc":            meta.get("tc",          ""),
        "wins":          wins,
        "draws":         draws,
        "losses":        losses,
        "total":         total,
        "score":         round(score, 4),
        "elo_diff":      round(elo_diff, 1),
        "elo_se":        round(se, 1),
        "ci_lo":         round(ci_lo, 1),
        "ci_hi":         round(ci_hi, 1),
        "los":           round(los_val, 4),
        "los_pct":       round(los_val * 100.0, 1),
        "draw_rate":     round(draw_rate, 3),
        "win_rate":      round(win_rate, 3),
        "elo_threshold": elo_threshold,
        "beats_ref":     beats,
        "aggr_stats":    aggr_data,
    }

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        w = max(len(str(match_json)), 60)
        sep = "─" * 62
        print(f"\n{sep}")
        print(f"  Solace vs Stockfish — {meta.get('tc','?')} — {meta.get('aggr_mode','?')} mode")
        print(sep)
        print(f"  Games     : {total}  (W={wins}  D={draws}  L={losses})")
        print(f"  Score     : {score:.4f}  ({win_rate*100:.1f}% win  {draw_rate*100:.1f}% draw)")
        print(f"  Δelo      : {elo_diff:+.1f}  ±{se:.1f}  (95% CI: [{ci_lo:+.1f}, {ci_hi:+.1f}])")
        print(f"  LOS       : {los_val*100:.1f}%")
        if aggr_data:
            print(f"  Sac/1k    : {aggr_data.get('sac_per_1k', 0.0):.1f}")
            print(f"  KingAtk/1k: {aggr_data.get('king_per_1k', 0.0):.1f}")
        verdict = "✓ BEATS REFERENCE" if beats else "✗ does not meet threshold"
        print(f"  Threshold : Δelo ≥ {elo_threshold:.0f} AND LOS ≥ 95%  →  {verdict}")
        print(sep)

    return beats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match_files", nargs="+", metavar="match.json",
                    help="One or more match JSON files from run_match.sh")
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="Minimum Δelo to consider Solace a winner (default: 10)")
    ap.add_argument("--json", action="store_true",
                    help="Emit results as JSON instead of formatted text")
    args = ap.parse_args()

    all_pass = True
    for path_str in args.match_files:
        p = Path(path_str)
        if not p.exists():
            print(f"[analyze] File not found: {p}", file=sys.stderr)
            all_pass = False
            continue
        passed = analyze(p, elo_threshold=args.threshold, as_json=args.json)
        if not passed:
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
