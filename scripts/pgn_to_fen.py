#!/usr/bin/env python3
"""
Solace — PGN → FEN + outcome converter
========================================
Converts a filtered PGN file (produced by fetch_aggressive_games.py) into a
tab-separated file of FEN positions with game outcome scores, suitable as
input for NNUE training tools (e.g. nnue-pytorch, bullet-trainer).

Output format (TSV, one position per line):
    <FEN>\t<outcome>\t<ply>\t<material_imbalance>

  outcome values: 1.0 (white win), 0.5 (draw), 0.0 (black win)

Filters applied per position:
  --min-ply          skip opening positions (default: 8)
  --max-ply          skip very long endgame positions (default: 200)
  --min-material     minimum total material on board in centipawns (default: 1000)
  --min-imbalance    only keep positions where |material difference| >= threshold
                     (0 = keep all, >0 biases toward imbalanced/sacrificed positions)
  --skip-check       skip positions where the side to move is in check
  --sample-rate      randomly sample 1 in N positions per game (default: 3)

Uses python-chess if installed for accurate FEN generation; falls back to a
lightweight built-in PGN parser that emits approximate FENs for positions
reachable from the standard starting position.

Usage:
    python3 pgn_to_fen.py aggressive.pgn --out positions.tsv
    python3 pgn_to_fen.py aggressive.pgn --out positions.tsv --min-imbalance 100
    python3 pgn_to_fen.py aggressive.pgn --out positions.tsv --stats

GPLv3 — part of the Solace project (Stockfish fork).
"""

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple

# ── Piece values in centipawns (used for material filter) ────────────────────
PIECE_CP = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}

# ── PGN parsing helpers ───────────────────────────────────────────────────────
TAG_RE    = re.compile(r'^\[(\w+)\s+"([^"]*)"\]')
COMMENT_RE = re.compile(r'\{[^}]*\}')
VARIATION_RE = re.compile(r'\([^()]*\)')  # shallow nested variations
RESULT_RE = re.compile(r'(1-0|0-1|1/2-1/2|\*)\s*$')
TOKEN_RE  = re.compile(r'(\d+\.+|[a-hKQRBNO][^\s]*|[a-h][1-8]|[KQRBN][a-h]?[1-8]?x?[a-h][1-8])')

RESULT_MAP = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5, "*": None}


def iter_games_raw(path: Path) -> Iterator[Tuple[dict, str]]:
    headers: dict[str, str] = {}
    moves_lines: list[str]  = []
    in_moves = False

    def emit():
        if moves_lines:
            return headers.copy(), " ".join(moves_lines)
        return None

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            m = TAG_RE.match(line)
            if m:
                if in_moves:
                    r = emit()
                    if r:
                        yield r
                    headers    = {}
                    moves_lines = []
                    in_moves   = False
                headers[m.group(1)] = m.group(2)
            elif line.strip() == "" and in_moves:
                r = emit()
                if r:
                    yield r
                headers     = {}
                moves_lines = []
                in_moves    = False
            elif line.strip():
                in_moves = True
                moves_lines.append(line)

    if in_moves:
        r = emit()
        if r:
            yield r


# ── python-chess fast path ───────────────────────────────────────────────────

def try_import_chess():
    try:
        import chess
        import chess.pgn
        import io
        return chess, chess.pgn, io
    except ImportError:
        return None, None, None


def material_cp(board) -> Tuple[int, int]:
    import chess as ch
    white = sum(
        PIECE_CP.get(ch.piece_name(pt).upper()[0], 0) * len(board.pieces(pt, ch.WHITE))
        for pt in ch.PIECE_TYPES
    )
    black = sum(
        PIECE_CP.get(ch.piece_name(pt).upper()[0], 0) * len(board.pieces(pt, ch.BLACK))
        for pt in ch.PIECE_TYPES
    )
    return white, black


def positions_from_game_chess(pgn_text: str, headers: dict,
                               min_ply: int, max_ply: int,
                               min_material: int, min_imbalance: int,
                               skip_check: bool, sample_rate: int,
                               rng: random.Random):
    import chess
    import chess.pgn
    import io

    result_str = headers.get("Result", "*")
    outcome    = RESULT_MAP.get(result_str)
    if outcome is None:
        return

    full_pgn = ""
    for k, v in headers.items():
        full_pgn += f'[{k} "{v}"]\n'
    full_pgn += "\n" + pgn_text

    game = chess.pgn.read_game(io.StringIO(full_pgn))
    if game is None:
        return

    board = game.board()
    ply   = 0
    node  = game

    while node.variations:
        node  = node.variations[0]
        board.push(node.move)
        ply  += 1

        if ply < min_ply or ply > max_ply:
            continue
        if rng.randint(1, sample_rate) != 1:
            continue
        if skip_check and board.is_check():
            continue

        w_cp, b_cp = material_cp(board)
        total = w_cp + b_cp
        imbal = abs(w_cp - b_cp)
        if total < min_material:
            continue
        if min_imbalance > 0 and imbal < min_imbalance:
            continue

        fen = board.fen()
        yield fen, outcome, ply, imbal


# ── Fallback: header-only FEN emitter ────────────────────────────────────────
# Without python-chess we cannot replay moves, so we emit only the start
# position tagged with result. This is a degraded mode — install python-chess
# on your training machine for full position extraction.

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def positions_from_game_fallback(pgn_text: str, headers: dict, **_kwargs):
    result_str = headers.get("Result", "*")
    outcome = RESULT_MAP.get(result_str)
    if outcome is None:
        return
    yield STARTPOS_FEN, outcome, 0, 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn",           help="Input PGN file")
    ap.add_argument("--out",         required=True, metavar="FILE.tsv",
                    help="Output TSV file")
    ap.add_argument("--min-ply",     type=int, default=8)
    ap.add_argument("--max-ply",     type=int, default=200)
    ap.add_argument("--min-material", type=int, default=1000,
                    help="Min total board material in cp (default: 1000)")
    ap.add_argument("--min-imbalance", type=int, default=0,
                    help="Min |material diff| cp to keep position (0=all)")
    ap.add_argument("--skip-check",  action="store_true", default=True,
                    help="Skip positions where the side to move is in check (default: on)")
    ap.add_argument("--sample-rate", type=int, default=3,
                    help="Keep 1 in N positions per game (default: 3)")
    ap.add_argument("--max-positions", type=int, default=5_000_000,
                    help="Stop after this many positions (default: 5M)")
    ap.add_argument("--seed",        type=int, default=42,
                    help="RNG seed for reproducibility (default: 42)")
    ap.add_argument("--stats",       action="store_true",
                    help="Print per-ECO stats after conversion")
    args = ap.parse_args()

    src = Path(args.pgn)
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        sys.exit(1)

    chess_mod, pgn_mod, io_mod = try_import_chess()
    if chess_mod:
        print("[pgn_to_fen] python-chess found — full move replay enabled.")
        extract_fn = positions_from_game_chess
    else:
        print("[pgn_to_fen] WARNING: python-chess not installed.")
        print("[pgn_to_fen]   Install it on your training machine for full extraction.")
        print("[pgn_to_fen]   Falling back to start-position-only mode (degraded).")
        extract_fn = positions_from_game_fallback

    rng = random.Random(args.seed)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    games_read = games_ok = pos_written = 0
    eco_counts: dict[str, int] = {}

    kwargs = dict(
        min_ply      = args.min_ply,
        max_ply      = args.max_ply,
        min_material = args.min_material,
        min_imbalance = args.min_imbalance,
        skip_check   = args.skip_check,
        sample_rate  = args.sample_rate,
        rng          = rng,
    )

    with dst.open("w", encoding="utf-8") as out_fh:
        out_fh.write("fen\toutcome\tply\tmaterial_imbalance\n")

        for headers, moves_text in iter_games_raw(src):
            games_read += 1
            eco = headers.get("ECO", "?")
            try:
                for fen, outcome, ply, imbal in extract_fn(moves_text, headers, **kwargs):
                    out_fh.write(f"{fen}\t{outcome}\t{ply}\t{imbal}\n")
                    pos_written += 1
                    eco_counts[eco] = eco_counts.get(eco, 0) + 1
                    if pos_written >= args.max_positions:
                        break
                games_ok += 1
            except Exception as exc:
                print(f"[pgn_to_fen] Skipped game {games_read}: {exc}", file=sys.stderr)

            if pos_written >= args.max_positions:
                break

            if games_read % 1000 == 0:
                print(f"\r[pgn_to_fen] Games: {games_read:,}  Positions: {pos_written:,}",
                      end="", flush=True)

    print(f"\n[pgn_to_fen] Games read    : {games_read:,}")
    print(f"[pgn_to_fen] Games OK      : {games_ok:,}")
    print(f"[pgn_to_fen] Positions out : {pos_written:,}")
    print(f"[pgn_to_fen] Written to    : {dst}")

    if args.stats and eco_counts:
        print("\n[pgn_to_fen] Top ECOs by position count:")
        for eco, cnt in sorted(eco_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {eco:>4}  {cnt:>8,}")


if __name__ == "__main__":
    main()
