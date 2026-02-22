#!/usr/bin/env python3
"""
Solace — aggression metrics log parser
=======================================
Reads engine stdout logs (or stdin) that contain lines of the form:

    info string solace_aggr total_moves N sacrifices N sac_per_1k N \
        king_attacks N king_per_1k N draw_vicinity N aggr_level N

Produces a CSV summary and a JSON statistics object aggregated across all
matching lines found in the input.

Usage:
    python3 parse_aggression_log.py [logfile ...] [--out results.csv] [--json]

If no logfiles are given, reads from stdin.

GPLv3 — part of the Solace project (Stockfish fork).
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List, Optional


AGGR_PATTERN = re.compile(
    r"info string solace_aggr"
    r"\s+total_moves\s+(?P<total_moves>\d+)"
    r"\s+sacrifices\s+(?P<sacrifices>\d+)"
    r"\s+sac_per_1k\s+(?P<sac_per_1k>\d+)"
    r"\s+king_attacks\s+(?P<king_attacks>\d+)"
    r"\s+king_per_1k\s+(?P<king_per_1k>\d+)"
    r"\s+draw_vicinity\s+(?P<draw_vicinity>\d+)"
    r"\s+aggr_level\s+(?P<aggr_level>\d+)"
)


@dataclass
class AggrRecord:
    total_moves: int
    sacrifices: int
    sac_per_1k: int
    king_attacks: int
    king_per_1k: int
    draw_vicinity: int
    aggr_level: int


def parse_stream(lines) -> List[AggrRecord]:
    records: List[AggrRecord] = []
    for line in lines:
        m = AGGR_PATTERN.search(line)
        if m:
            records.append(AggrRecord(**{k: int(v) for k, v in m.groupdict().items()}))
    return records


def aggregate(records: List[AggrRecord]) -> dict:
    if not records:
        return {}

    n = len(records)
    total_moves  = sum(r.total_moves  for r in records)
    sacrifices   = sum(r.sacrifices   for r in records)
    king_attacks = sum(r.king_attacks for r in records)
    draw_vicinity = sum(r.draw_vicinity for r in records)
    aggr_levels  = [r.aggr_level for r in records]

    sac_per_1k  = round(1000.0 * sacrifices  / total_moves, 2) if total_moves else 0.0
    king_per_1k = round(1000.0 * king_attacks / total_moves, 2) if total_moves else 0.0
    draw_rate   = round(draw_vicinity / total_moves, 4) if total_moves else 0.0

    return {
        "samples":              n,
        "total_moves":          total_moves,
        "sacrifices":           sacrifices,
        "sac_per_1k":           sac_per_1k,
        "king_attacks":         king_attacks,
        "king_per_1k":          king_per_1k,
        "draw_vicinity_moves":  draw_vicinity,
        "draw_rate":            draw_rate,
        "aggr_level_min":       min(aggr_levels),
        "aggr_level_max":       max(aggr_levels),
        "aggr_level_mean":      round(sum(aggr_levels) / n, 1),
    }


def write_csv(records: List[AggrRecord], path: str) -> None:
    if not records:
        print("[parse_aggression_log] No records to write.", file=sys.stderr)
        return
    field_names = [f.name for f in fields(AggrRecord)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    print(f"[parse_aggression_log] Wrote {len(records)} rows to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfiles", nargs="*", help="Log files to parse (default: stdin)")
    ap.add_argument("--out",  default=None, metavar="FILE.csv",
                    help="Write per-record CSV to this file")
    ap.add_argument("--json", action="store_true",
                    help="Print aggregated JSON summary to stdout")
    args = ap.parse_args()

    records: List[AggrRecord] = []

    if args.logfiles:
        for path in args.logfiles:
            p = Path(path)
            if not p.exists():
                print(f"[parse_aggression_log] WARNING: {path} not found", file=sys.stderr)
                continue
            with p.open() as fh:
                records.extend(parse_stream(fh))
    else:
        records.extend(parse_stream(sys.stdin))

    if not records:
        print("[parse_aggression_log] No solace_aggr lines found.", file=sys.stderr)
        sys.exit(1)

    print(f"[parse_aggression_log] Parsed {len(records)} solace_aggr records.")

    summary = aggregate(records)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\nAggregated aggression metrics:")
        print(f"  Samples              : {summary['samples']}")
        print(f"  Total moves          : {summary['total_moves']}")
        print(f"  Sacrifices / 1k      : {summary['sac_per_1k']}")
        print(f"  King-attack / 1k     : {summary['king_per_1k']}")
        print(f"  Draw-vicinity rate   : {summary['draw_rate']:.4f}")
        print(f"  AggrLevel range      : {summary['aggr_level_min']}–{summary['aggr_level_max']}"
              f" (mean {summary['aggr_level_mean']})")

    if args.out:
        write_csv(records, args.out)


if __name__ == "__main__":
    main()
