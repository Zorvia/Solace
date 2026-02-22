#!/usr/bin/env python3
"""
Solace — aggressive games fetcher
==================================
Downloads PGN collections filtered for gambits and sacrificial attacking games
from the Lichess open database (https://database.lichess.org/) and writes a
versioned, SHA-256-hashed manifest so training datasets are reproducible.

Network is ONLY used when --download is passed and a file is not already
present locally.  If you pre-download the files manually, the script will
validate them and write the manifest without any network access.

Sources used (all CC0 / public domain):
  - Lichess open database monthly PGN dumps
    https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst
  - Lichess elite database (2400+ rated games)
    https://database.nikonoel.fr/

Usage:
    python3 fetch_aggressive_games.py --month 2024-01 --out-dir data/raw
    python3 fetch_aggressive_games.py --local path/to/games.pgn --out-dir data/raw
    python3 fetch_aggressive_games.py --manifest data/raw/manifest.json

GPLv3 — part of the Solace project (Stockfish fork).
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Aggressive opening ECO prefixes to keep ──────────────────────────────────
# King's Gambit (C30-C39), Evan's Gambit (C51-C52), Sicilian (B20-B99),
# King's Indian (E60-E99), Alekhine (B02-B05), Dragon (B70-B79),
# Benko Gambit (A57-A59), Budapest Gambit (A51-A52),
# Danish Gambit (C21), Scotch Gambit (C44-C45)
AGGRESSIVE_ECO_RANGES = [
    ("B20", "B99"),  # Sicilian
    ("C30", "C39"),  # King's Gambit
    ("C51", "C52"),  # Evans Gambit
    ("C21", "C21"),  # Danish Gambit
    ("C44", "C45"),  # Scotch Gambit
    ("E60", "E99"),  # King's Indian
    ("A51", "A52"),  # Budapest Gambit
    ("A57", "A59"),  # Benko Gambit
    ("B02", "B05"),  # Alekhine
    ("B70", "B79"),  # Dragon
]

# Regex patterns for PGN header tag parsing
TAG_RE   = re.compile(r'^\[(\w+)\s+"([^"]*)"\]')
MOVE_RE  = re.compile(r'^1\.\s')


def eco_is_aggressive(eco: str) -> bool:
    if not eco or len(eco) < 3:
        return False
    letter = eco[0]
    try:
        number = int(eco[1:3])
    except ValueError:
        return False
    for (lo, hi) in AGGRESSIVE_ECO_RANGES:
        if letter == lo[0] and int(lo[1:]) <= number <= int(hi[1:]):
            return True
    return False


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def download_file(url: str, dest: Path, show_progress: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done  = 0
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if show_progress and total:
                        pct = int(100 * done / total)
                        print(f"\r  downloading {dest.name}: {pct}%", end="", flush=True)
        print()
        tmp.rename(dest)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def open_pgn(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def filter_pgn(src: Path, dst: Path,
               min_elo: int,
               max_games: int) -> dict:
    """
    Stream-parses src PGN, writes games matching aggressive ECO + min_elo to dst.
    Returns a stats dict.
    """
    stats = {"read": 0, "kept": 0, "skipped_elo": 0, "skipped_eco": 0}
    game_lines: list[str] = []
    in_moves = False
    headers: dict[str, str] = {}

    def flush_game():
        nonlocal stats
        if not game_lines:
            return
        stats["read"] += 1
        eco      = headers.get("ECO", "")
        result   = headers.get("Result", "*")
        w_elo    = int(headers.get("WhiteElo", "0") or "0")
        b_elo    = int(headers.get("BlackElo", "0") or "0")
        avg_elo  = (w_elo + b_elo) // 2 if w_elo and b_elo else 0

        if result not in ("1-0", "0-1"):
            stats["skipped_eco"] += 1
            return
        if avg_elo > 0 and avg_elo < min_elo:
            stats["skipped_elo"] += 1
            return
        if not eco_is_aggressive(eco):
            stats["skipped_eco"] += 1
            return

        out_fh.write("\n".join(game_lines) + "\n\n")
        stats["kept"] += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open_pgn(src) as in_fh, dst.open("w", encoding="utf-8") as out_fh:
        for raw_line in in_fh:
            if stats["kept"] >= max_games:
                break
            line = raw_line.rstrip()
            m = TAG_RE.match(line)
            if m:
                if in_moves:
                    flush_game()
                    game_lines = []
                    headers    = {}
                    in_moves   = False
                headers[m.group(1)] = m.group(2)
                game_lines.append(line)
            elif line.strip() == "" and game_lines and in_moves:
                game_lines.append(line)
                flush_game()
                game_lines = []
                headers    = {}
                in_moves   = False
            else:
                if MOVE_RE.match(line):
                    in_moves = True
                game_lines.append(line)

        if game_lines and in_moves:
            flush_game()

    return stats


def write_manifest(entry: dict, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if manifest_path.exists():
        with manifest_path.open() as fh:
            existing = json.load(fh)
    key = entry["filtered_sha256"]
    existing = [e for e in existing if e.get("filtered_sha256") != key]
    existing.append(entry)
    with manifest_path.open("w") as fh:
        json.dump(existing, fh, indent=2)
    print(f"[fetch] Manifest updated: {manifest_path}")


def build_lichess_url(month: str) -> str:
    return (
        f"https://database.lichess.org/standard/"
        f"lichess_db_standard_rated_{month}.pgn.zst"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month",    metavar="YYYY-MM",
                    help="Lichess monthly dump to download (e.g. 2024-01)")
    ap.add_argument("--local",    metavar="FILE",
                    help="Use a local PGN / PGN.GZ file instead of downloading")
    ap.add_argument("--out-dir",  default="data/raw",
                    help="Directory to write filtered PGN and manifest (default: data/raw)")
    ap.add_argument("--min-elo",  type=int, default=1800,
                    help="Minimum average player Elo to keep (default: 1800)")
    ap.add_argument("--max-games", type=int, default=100_000,
                    help="Stop after keeping this many games (default: 100000)")
    ap.add_argument("--download", action="store_true",
                    help="Actually fetch from the internet (omit for dry-run/local)")
    ap.add_argument("--manifest", metavar="FILE",
                    help="Print contents of an existing manifest file and exit")
    args = ap.parse_args()

    if args.manifest:
        p = Path(args.manifest)
        if not p.exists():
            print(f"Manifest not found: {p}", file=sys.stderr)
            sys.exit(1)
        with p.open() as fh:
            print(json.dumps(json.load(fh), indent=2))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if args.local:
        src = Path(args.local)
        if not src.exists():
            print(f"[fetch] ERROR: local file not found: {src}", file=sys.stderr)
            sys.exit(1)
        source_label = src.name
    elif args.month:
        url = build_lichess_url(args.month)
        fname = url.split("/")[-1]
        src   = out_dir / fname
        source_label = url
        if not src.exists():
            if not args.download:
                print(f"[fetch] File not present locally. Re-run with --download to fetch:")
                print(f"        {url}")
                print(f"  OR manually download to: {src}")
                sys.exit(0)
            print(f"[fetch] Downloading {url} ...")
            download_file(url, src)
        else:
            print(f"[fetch] Using cached file: {src}")
    else:
        ap.print_help()
        sys.exit(1)

    print(f"[fetch] Source    : {src} ({src.stat().st_size // 1024 // 1024} MB)")
    print(f"[fetch] Min ELO   : {args.min_elo}")
    print(f"[fetch] Max games : {args.max_games}")

    stem = src.stem.replace(".pgn", "").replace(".zst", "")
    dst  = out_dir / f"{stem}_aggressive.pgn"

    print(f"[fetch] Filtering to {dst} ...")
    stats = filter_pgn(src, dst, min_elo=args.min_elo, max_games=args.max_games)

    print(f"[fetch] Games read   : {stats['read']:,}")
    print(f"[fetch] Kept         : {stats['kept']:,}")
    print(f"[fetch] Skip ECO     : {stats['skipped_eco']:,}")
    print(f"[fetch] Skip ELO     : {stats['skipped_elo']:,}")

    if stats["kept"] == 0:
        print("[fetch] WARNING: no games kept — check ECO filter or ELO threshold.")
        sys.exit(1)

    filtered_sha = sha256_file(dst)
    entry = {
        "created_utc":    datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source":         source_label,
        "filtered_file":  str(dst),
        "filtered_sha256": filtered_sha,
        "games_kept":     stats["kept"],
        "min_elo":        args.min_elo,
        "max_games":      args.max_games,
        "eco_ranges":     [f"{lo}-{hi}" for lo, hi in AGGRESSIVE_ECO_RANGES],
    }

    write_manifest(entry, manifest_path)
    print(f"[fetch] SHA-256 ({dst.name}): {filtered_sha}")
    print("[fetch] Done.")


if __name__ == "__main__":
    main()
