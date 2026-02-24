#!/usr/bin/env python3
"""
Solace — SPSA parameter tuner
==============================
Implements Simultaneous Perturbation Stochastic Approximation (SPSA) to
automatically find optimal values for the Solace aggression search parameters.

The tuner drives Solace directly via UCI, discovers which spin options exist,
perturbs them ±c_k, plays mini-matches in both directions, and estimates the
gradient of Elo with respect to each parameter.  This mirrors the method used
by Fishtest for all Stockfish parameter tuning.

Algorithm (Spall 1998):
  θ_{k+1} = θ_k + a_k * g_k
  g_k = [ Elo(θ+c_k*Δ) - Elo(θ-c_k*Δ) ] / (2*c_k*Δ)   (Δ_i ∈ {-1,+1})

Hyperparameters:
  A      = Proportion of total iterations to stabilise before decay (default: 0.1*iters)
  alpha  = Learning-rate decay exponent (default: 0.602)
  gamma  = Perturbation-decay exponent  (default: 0.101)
  a      = Initial step size            (default: auto from target Δ and A)
  c      = Initial perturbation         (default: estimated from parameter ranges)

Usage:
    # Tune all discovered Solace* parameters (requires match runner)
    python3 spsa_tuner.py --engine src/solace --ref stockfish \\
        --big-net nn-big.nnue --small-net nn-small.nnue \\
        --iters 50 --games-per-side 50 --tc 10+0.1 \\
        --out tuning/

    # Tune only specific parameters
    python3 spsa_tuner.py --engine src/solace --ref stockfish \\
        --params SolaceLmrKingProx,SolaceLmrSacrifice \\
        --iters 100 --games-per-side 100 --out tuning/

    # Dry run: show parameters the engine exposes, exit
    python3 spsa_tuner.py --engine src/solace --dry-run

Output:
    tuning/spsa_log.csv         — per-iteration parameter values and Elo estimates
    tuning/spsa_results.txt     — final values in fishtest-paste format
    tuning/tune_results.h       — C++ snippet to paste into tune.cpp read_results()

Requirements: cutechess-cli or fastchess on PATH.

GPLv3 — part of the Solace project (Stockfish fork).
"""

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


# ── UCI helpers ───────────────────────────────────────────────────────────────

def discover_tunable_params(engine_path: str) -> dict[str, dict]:
    """
    Launch engine, send 'uci', collect spin options whose names start with
    'Solace' (or the given prefix). Returns {name: {default, min, max}}.
    """
    proc = subprocess.run(
        [engine_path],
        input="uci\nquit\n",
        capture_output=True, text=True, timeout=10
    )
    params = {}
    for line in proc.stdout.splitlines():
        m = re.match(
            r"^option name (\S+) type spin default (\d+) min (\d+) max (\d+)",
            line
        )
        if m:
            name = m.group(1)
            params[name] = {
                "default": int(m.group(2)),
                "min":     int(m.group(3)),
                "max":     int(m.group(4)),
                "current": int(m.group(2)),
            }
    return params


# ── Mini-match runner ─────────────────────────────────────────────────────────

def find_runner() -> Optional[str]:
    for name in ("fastchess", "cutechess-cli"):
        path = shutil.which(name)
        if path:
            return path
    return None


def run_minigame(engine_path: str, ref_path: str,
                 params: dict[str, int],
                 big_net: str, small_net: str,
                 n_games: int, tc: str,
                 runner: str, seed: int) -> tuple[int, int, int]:
    """
    Play n_games between engine (with params) and ref.
    Returns (wins, draws, losses) from engine's perspective.
    """
    # Build setoption strings for engine
    engine_opts = " ".join(
        f"option.{k}={v}" for k, v in params.items()
    )
    if big_net:
        engine_opts = f"option.EvalFile={big_net} option.EvalFileSmall={small_net} " + engine_opts
    ref_opts = ""
    if big_net:
        ref_opts = f"option.EvalFile={big_net} option.EvalFileSmall={small_net}"
    engine_opts += " option.Hash=16"
    ref_opts    += " option.Hash=16"

    with tempfile.NamedTemporaryFile(suffix=".pgn", delete=False) as pf:
        pgn_path = pf.name

    cmd = [
        runner,
        "-engine", f"name=Solace cmd={engine_path}", *engine_opts.split(),
        "-engine", f"name=Ref    cmd={ref_path}",    *ref_opts.split(),
        "-each",   f"tc={tc}",
        "-rounds", str(n_games),
        "-concurrency", "1",
        "-pgnout", pgn_path,
        "-recover",
        "-repeat",
        "-resign", "movecount=4", "score=600",
        "-draw",   "movenumber=40", "movecount=8", "score=10",
        "-seed",   str(seed),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=n_games * 120)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 0, 0, 0
    finally:
        Path(pgn_path).unlink(missing_ok=True)

    w = d = l = 0
    m = re.search(r"Score of Solace.*?(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", output, re.IGNORECASE)
    if m:
        w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        # fastchess format
        mw = re.search(r"Wins:\s*(\d+)",   output, re.IGNORECASE)
        ml = re.search(r"Losses:\s*(\d+)", output, re.IGNORECASE)
        md = re.search(r"Draws:\s*(\d+)",  output, re.IGNORECASE)
        if mw: w = int(mw.group(1))
        if ml: l = int(ml.group(1))
        if md: d = int(md.group(1))

    return w, d, l


def score_to_elo(w: int, d: int, l: int) -> float:
    n = w + d + l
    if n == 0:
        return 0.0
    s = (w + 0.5 * d) / n
    s = max(1e-6, min(1.0 - 1e-6, s))
    return -400.0 * math.log10(1.0 / s - 1.0)


# ── SPSA core ─────────────────────────────────────────────────────────────────

class SPSATuner:
    def __init__(self,
                 params:       dict[str, dict],
                 iters:        int,
                 games_per:    int,
                 A_ratio:      float = 0.1,
                 alpha:        float = 0.602,
                 gamma:        float = 0.101,
                 target_delta: float = 10.0,
                 rng:          random.Random = None):
        self.params  = params   # {name: {current, min, max, default}}
        self.iters   = iters
        self.games   = games_per
        self.alpha   = alpha
        self.gamma   = gamma
        self.rng     = rng or random.Random(42)

        names = sorted(params.keys())
        self.names = names
        self.theta = {n: float(params[n]["current"]) for n in names}

        A = max(1, int(A_ratio * iters))
        # Auto-scale a so first step ≈ target_delta / |gradient_estimate|
        # Conservative default: a = target_delta * (A+1)^alpha / (p * c)
        p = len(names)
        c = self._c(1)
        self.a = target_delta * (A + 1) ** alpha / (p * c) if p > 0 else 1.0
        self.A = A

    def _a(self, k: int) -> float:
        return self.a / (k + 1 + self.A) ** self.alpha

    def _c(self, k: int) -> float:
        # Perturbation scale: fraction of parameter range
        scales = [self.params[n]["max"] - self.params[n]["min"] for n in self.names]
        avg_scale = sum(scales) / len(scales) if scales else 1.0
        return max(1.0, avg_scale * 0.05 / (k ** self.gamma + 1e-9))

    def _clamp(self, name: str, value: float) -> float:
        lo = float(self.params[name]["min"])
        hi = float(self.params[name]["max"])
        return max(lo, min(hi, value))

    def run(self,
            engine_path: str, ref_path: str,
            big_net: str, small_net: str,
            tc: str, runner: str,
            log_path: Path) -> dict[str, float]:

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file  = log_path.parent / "spsa_log.csv"
        fieldnames = ["iter", "elo_plus", "elo_minus"] + self.names

        with log_file.open("w", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            writer.writeheader()

            for k in range(1, self.iters + 1):
                a_k = self._a(k)
                c_k = max(1.0, self._c(k))

                # Bernoulli ±1 perturbation vector
                delta = {n: self.rng.choice([-1, 1]) for n in self.names}

                # θ+ and θ-
                theta_plus  = {n: self._clamp(n, self.theta[n] + c_k * delta[n])
                               for n in self.names}
                theta_minus = {n: self._clamp(n, self.theta[n] - c_k * delta[n])
                               for n in self.names}

                # Also pass Param mode so bonuses activate
                plus_full  = dict(theta_plus)
                minus_full = dict(theta_minus)
                plus_full["SolaceAggressionMode"]  = "Param"
                plus_full["SolaceAggressionLevel"] = 75
                minus_full["SolaceAggressionMode"]  = "Param"
                minus_full["SolaceAggressionLevel"] = 75

                seed = self.rng.randint(1, 2**31)

                w_p, d_p, l_p = run_minigame(engine_path, ref_path, plus_full,
                                              big_net, small_net, self.games, tc, runner, seed)
                w_m, d_m, l_m = run_minigame(engine_path, ref_path, minus_full,
                                              big_net, small_net, self.games, tc, runner, seed)

                elo_plus  = score_to_elo(w_p, d_p, l_p)
                elo_minus = score_to_elo(w_m, d_m, l_m)
                elo_diff  = elo_plus - elo_minus

                # Gradient estimate and update
                for n in self.names:
                    if abs(delta[n] * c_k) < 1e-9:
                        continue
                    g_k = elo_diff / (2.0 * c_k * delta[n])
                    self.theta[n] = self._clamp(n, self.theta[n] + a_k * g_k)

                row = {"iter": k, "elo_plus": round(elo_plus, 2),
                       "elo_minus": round(elo_minus, 2)}
                row.update({n: round(self.theta[n], 2) for n in self.names})
                writer.writerow(row)
                csvf.flush()

                theta_str = "  ".join(f"{n}={self.theta[n]:.1f}" for n in self.names)
                print(f"[spsa] iter {k:>4}/{self.iters}  "
                      f"Δelo={elo_diff:+.1f}  a_k={a_k:.4f}  c_k={c_k:.1f}  "
                      f"| {theta_str}", flush=True)

        return {n: round(self.theta[n]) for n in self.names}


# ── Output writers ────────────────────────────────────────────────────────────

def write_results(final: dict[str, float],
                  initial: dict[str, dict],
                  out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fishtest-paste format
    txt_path = out_dir / "spsa_results.txt"
    with txt_path.open("w") as fh:
        for name, val in final.items():
            p = initial[name]
            r = p["max"] - p["min"]
            fh.write(f"param: {name}, best: {val:.1f}, "
                     f"start: {p['default']}, "
                     f"min: {p['min']}, max: {p['max']}, "
                     f"c_end: {r/20:.2f}, "
                     f"a_end: 0.0020\n")
    print(f"[spsa] Results: {txt_path}")

    # C++ snippet for tune.cpp read_results()
    cpp_path = out_dir / "tune_results.h"
    with cpp_path.open("w") as fh:
        fh.write("// Paste into Tune::read_results() in src/tune.cpp\n")
        fh.write("// Generated by spsa_tuner.py\n\n")
        for name, val in final.items():
            fh.write(f'  TuneResults["{name}"] = {int(round(val))};\n')
    print(f"[spsa] C++ snippet: {cpp_path}")

    # JSON summary
    json_path = out_dir / "spsa_summary.json"
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": {
            name: {"initial": initial[name]["default"], "final": int(round(val))}
            for name, val in final.items()
        }
    }
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[spsa] Summary JSON: {json_path}")
    print("\n[spsa] Final parameter values:")
    for name, val in final.items():
        delta = int(round(val)) - initial[name]["default"]
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<28} {int(round(val)):>6}  (was {initial[name]['default']}, {sign}{delta})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine",         required=True,  metavar="PATH")
    ap.add_argument("--ref",            default="",     metavar="PATH",
                    help="Reference engine (default: same binary, mode=Off)")
    ap.add_argument("--big-net",        default="",     metavar="FILE.nnue")
    ap.add_argument("--small-net",      default="",     metavar="FILE.nnue")
    ap.add_argument("--iters",          type=int,   default=50)
    ap.add_argument("--games-per-side", type=int,   default=50,
                    help="Games per ±perturbation per iteration (default: 50)")
    ap.add_argument("--tc",             default="10+0.1")
    ap.add_argument("--out",            default="tuning", metavar="DIR")
    ap.add_argument("--params",         default="",
                    help="Comma-separated param names to tune (default: all Solace* spin options)")
    ap.add_argument("--A-ratio",        type=float, default=0.1,
                    help="Fraction of iters to use as SPSA stability constant A (default: 0.1)")
    ap.add_argument("--target-delta",   type=float, default=5.0,
                    help="Target Elo improvement to calibrate initial step size (default: 5.0)")
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--runner",         default="",     metavar="PATH",
                    help="Explicit path to fastchess or cutechess-cli")
    ap.add_argument("--dry-run",        action="store_true",
                    help="Print discoverable parameters and exit without tuning")
    args = ap.parse_args()

    if not Path(args.engine).exists():
        print(f"ERROR: engine not found: {args.engine}", file=sys.stderr); sys.exit(1)

    # Discover tunable parameters
    all_params = discover_tunable_params(args.engine)
    solace_params = {k: v for k, v in all_params.items() if k.startswith("Solace")}

    if args.dry_run:
        print(f"Discoverable Solace parameters in {args.engine}:")
        for name, p in sorted(solace_params.items()):
            print(f"  {name:<30}  default={p['default']:<6}  range=[{p['min']}, {p['max']}]")
        print(f"\nTotal: {len(solace_params)} tunable parameters")
        sys.exit(0)

    if args.params:
        wanted = {n.strip() for n in args.params.split(",")}
        tune_params = {k: v for k, v in solace_params.items() if k in wanted}
        missing = wanted - set(tune_params)
        if missing:
            print(f"WARNING: params not found in engine: {missing}", file=sys.stderr)
    else:
        # Tune only search parameters (not Mode/Level/Net which are control knobs)
        tune_params = {k: v for k, v in solace_params.items()
                       if k not in {"SolaceAggressionMode", "SolaceAggressionLevel",
                                    "SolaceAggressionNet"}}

    if not tune_params:
        print("ERROR: no tunable parameters found. Use --dry-run to inspect.", file=sys.stderr)
        sys.exit(1)

    runner = args.runner or find_runner()
    if not runner:
        print("ERROR: no match runner found. Install fastchess or cutechess-cli, "
              "or pass --runner.", file=sys.stderr)
        sys.exit(1)

    ref = args.ref or args.engine

    print(f"[spsa] Engine     : {args.engine}")
    print(f"[spsa] Reference  : {ref}")
    print(f"[spsa] Runner     : {runner}")
    print(f"[spsa] Parameters : {', '.join(sorted(tune_params))}")
    print(f"[spsa] Iterations : {args.iters}")
    print(f"[spsa] Games/side : {args.games_per_side}")
    print(f"[spsa] TC         : {args.tc}")
    print(f"[spsa] Output dir : {args.out}")
    print()

    rng    = random.Random(args.seed)
    tuner  = SPSATuner(
        params       = tune_params,
        iters        = args.iters,
        games_per    = args.games_per_side,
        A_ratio      = args.A_ratio,
        target_delta = args.target_delta,
        rng          = rng,
    )

    out_dir  = Path(args.out)
    log_path = out_dir / "spsa_log.csv"
    t0       = time.time()

    final = tuner.run(
        engine_path = args.engine,
        ref_path    = ref,
        big_net     = args.big_net,
        small_net   = args.small_net,
        tc          = args.tc,
        runner      = runner,
        log_path    = log_path,
    )

    elapsed = time.time() - t0
    print(f"\n[spsa] Completed {args.iters} iterations in {elapsed:.0f}s "
          f"({elapsed/args.iters:.1f}s/iter)")

    write_results(final, tune_params, out_dir)


if __name__ == "__main__":
    main()
