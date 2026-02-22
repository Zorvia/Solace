#!/usr/bin/env python3
"""
Solace — NNUE training loop (numpy, CPU-only)
==============================================
Trains a HalfKAv2_hm-compatible network from a FEN+outcome TSV file produced
by pgn_to_fen.py.  Outputs .npz checkpoints loadable by nnue_export.py.

Architecture (mirrors Stockfish big net):
  Input  : HalfKAv2_hm sparse features (up to 32 active per side)
  FT     : sparse embedding → L1=1024 per side (float32 during training)
  FC0    : 2*L1 → L2=31  (tanh, then squared-clamped)
  FC1    : L2*2 → L3=32  (clamped-relu)
  FC2    : L3 → 1        (output, centipawns)

Aggression bias: positions with |material_imbalance| >= --aggr-imbalance
receive a higher weight in the loss, teaching the net to be more accurate
in unbalanced/sacrificed positions (the core of Solace's style).

Loss: WDL-based MSE  L = w * (sigmoid(cp/600) - target_wdl)^2

Usage:
    python3 train_nnue.py positions.tsv --epochs 5 --lr 0.001 --out checkpoints/
    python3 train_nnue.py positions.tsv --resume checkpoints/epoch_3.npz --epochs 2

Requires: numpy (no torch, no GPU needed — though GPU speedup requires PyTorch;
          install pytorch and use nnue-pytorch for full-scale training runs)

GPLv3 — part of the Solace project (Stockfish fork).
"""

import argparse
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# ── Architecture constants (must match Stockfish big net) ────────────────────
L1            = 1024    # FT output per side
L2            = 31      # FC0 output
L3            = 32      # FC1 output
PSQT_BUCKETS  = 8
LAYER_STACKS  = 8

# HalfKAv2_hm constants
SQUARE_NB     = 64
PS_NB         = 11 * SQUARE_NB   # 704
FT_INPUT_DIM  = SQUARE_NB * PS_NB // 2   # 22528

KING_BUCKETS = [
    28,29,30,31,31,30,29,28,
    24,25,26,27,27,26,25,24,
    20,21,22,23,23,22,21,20,
    16,17,18,19,19,18,17,16,
    12,13,14,15,15,14,13,12,
     8, 9,10,11,11,10, 9, 8,
     4, 5, 6, 7, 7, 6, 5, 4,
     0, 1, 2, 3, 3, 2, 1, 0,
]
KING_BUCKET_OFFSETS = [b * PS_NB for b in KING_BUCKETS]

# OrientTBL: XOR value to apply to square index based on king's file
# If king is on a-d files (files 0-3), XOR with SQ_H1 (7) = mirror file
# If king is on e-h files (files 4-7), XOR with SQ_A1 (0) = no mirror
ORIENT_TBL = [7 if (sq % 8) < 4 else 0 for sq in range(64)]

# PieceSquareIndex from white's perspective
# ps = piece_type_index * SQUARE_NB  (for the 10 non-king piece types + king)
PS_W_PAWN, PS_B_PAWN   = 0*SQUARE_NB, 1*SQUARE_NB
PS_W_KNIGHT,PS_B_KNIGHT= 2*SQUARE_NB, 3*SQUARE_NB
PS_W_BISHOP,PS_B_BISHOP= 4*SQUARE_NB, 5*SQUARE_NB
PS_W_ROOK,  PS_B_ROOK  = 6*SQUARE_NB, 7*SQUARE_NB
PS_W_QUEEN, PS_B_QUEEN = 8*SQUARE_NB, 9*SQUARE_NB
PS_KING                 = 10*SQUARE_NB

# From perspective of each color:
# white perspective: own=W pieces, enemy=B pieces
# black perspective: own=B pieces, enemy=W pieces (and board is flipped)
PIECE_TO_PS = {
    "WHITE": {
        "P": PS_W_PAWN,   "N": PS_W_KNIGHT, "B": PS_W_BISHOP,
        "R": PS_W_ROOK,   "Q": PS_W_QUEEN,  "K": PS_KING,
        "p": PS_B_PAWN,   "n": PS_B_KNIGHT, "b": PS_B_BISHOP,
        "r": PS_B_ROOK,   "q": PS_B_QUEEN,  "k": PS_KING,
    },
    "BLACK": {
        # From black's perspective: own pieces are black, enemy are white
        # AND the entire board is rotated 180 degrees (sq ^= 56)
        "p": PS_W_PAWN,   "n": PS_W_KNIGHT, "b": PS_W_BISHOP,
        "r": PS_W_ROOK,   "q": PS_W_QUEEN,  "k": PS_KING,
        "P": PS_B_PAWN,   "N": PS_B_KNIGHT, "B": PS_B_BISHOP,
        "R": PS_B_ROOK,   "Q": PS_B_QUEEN,  "K": PS_KING,
    },
}


def sq_from_file_rank(file: int, rank: int) -> int:
    return rank * 8 + file


def fen_to_features(fen: str):
    """
    Returns (white_indices, black_indices): lists of int HalfKAv2_hm feature indices.
    Up to 32 active features per side (one per piece on the board).
    """
    parts = fen.split()
    board_str = parts[0]
    side_to_move = parts[1] if len(parts) > 1 else "w"

    pieces: list[tuple[int, str]] = []  # (square, piece_char)
    rank = 7
    for row in board_str.split("/"):
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                sq = sq_from_file_rank(file, rank)
                pieces.append((sq, ch))
                file += 1
        rank -= 1

    w_king_sq = next((sq for sq, p in pieces if p == "K"), None)
    b_king_sq = next((sq for sq, p in pieces if p == "k"), None)
    if w_king_sq is None or b_king_sq is None:
        return None, None

    def make_index(perspective: str, sq: int, piece_ch: str, ksq: int) -> int:
        flip = 56 if perspective == "BLACK" else 0
        oriented_sq = sq ^ ORIENT_TBL[ksq] ^ flip
        ksq_flipped  = ksq ^ flip
        ps_offset    = PIECE_TO_PS[perspective][piece_ch]
        return oriented_sq + ps_offset + KING_BUCKET_OFFSETS[ksq_flipped]

    w_indices, b_indices = [], []
    for sq, piece_ch in pieces:
        if piece_ch in "Kk":
            continue  # king handled via bucket, not direct feature
        try:
            w_indices.append(make_index("WHITE", sq, piece_ch, w_king_sq))
            b_indices.append(make_index("BLACK", sq, piece_ch, b_king_sq))
        except KeyError:
            pass

    return w_indices, b_indices


# ── Simplified single-bucket network (training uses mean across buckets) ─────
class SolaceNet:
    """
    Simplified training architecture:
      FT  : input_dim → L1  (sparse embedding, one per side)
      FC0 : 2*L1 → L2       (int8-quantizable)
      FC1 : L2*2 → L3
      FC2 : L3 → 1

    We train a single set of FC0/FC1/FC2 weights (bucket selection is
    a position-dependent choice based on remaining pieces; we use bucket 0
    for simplicity during initial training, which is standard for small runs).
    """

    def __init__(self, rng: np.random.Generator):
        scale = 0.01
        self.ft_biases  = np.zeros(L1, dtype=np.float32)
        # FT weights stored as dense (input_dim × L1) — sparse update in forward
        self.ft_weights = (rng.standard_normal((FT_INPUT_DIM, L1)) * scale).astype(np.float32)

        self.fc0_b = np.zeros(L2 + 1, dtype=np.float32)
        self.fc0_w = (rng.standard_normal((L2 + 1, L1 * 2)) * scale).astype(np.float32)
        # fc0_b[L2] = psqt component bias, set to a small positive constant
        self.fc0_b[L2] = 0.0

        self.fc1_b = np.zeros(L3, dtype=np.float32)
        self.fc1_w = (rng.standard_normal((L3, L2 * 2)) * scale).astype(np.float32)

        self.fc2_b = np.zeros(1, dtype=np.float32)
        self.fc2_w = (rng.standard_normal((1, L3)) * scale).astype(np.float32)

    def forward(self, w_indices, b_indices):
        w_acc = self.ft_biases.copy()
        for idx in w_indices:
            if 0 <= idx < FT_INPUT_DIM:
                w_acc += self.ft_weights[idx]

        b_acc = self.ft_biases.copy()
        for idx in b_indices:
            if 0 <= idx < FT_INPUT_DIM:
                b_acc += self.ft_weights[idx]

        # Clamp to [0, 127] simulating ClippedReLU (scaled to float)
        w_clamped = np.clip(w_acc, 0.0, 1.0)
        b_clamped = np.clip(b_acc, 0.0, 1.0)

        ft_out = np.concatenate([w_clamped, b_clamped])  # 2*L1

        # FC0 + SqrClippedReLU || ClippedReLU (concatenated)
        fc0_pre = self.fc0_w @ ft_out + self.fc0_b    # L2+1
        fc0_sqr = np.clip(fc0_pre[:L2], 0.0, 1.0) ** 2  # SqrClipped
        fc0_lin = np.clip(fc0_pre[:L2], 0.0, 1.0)        # Clipped
        psqt    = fc0_pre[L2]

        fc0_out = np.concatenate([fc0_sqr, fc0_lin])  # L2*2

        # FC1 + ClippedReLU
        fc1_pre = self.fc1_w @ fc0_out + self.fc1_b    # L3
        fc1_out = np.clip(fc1_pre, 0.0, 1.0)

        # FC2 → output (centipawns / 600 for WDL sigmoid)
        output_cp = float((self.fc2_w @ fc1_out + self.fc2_b)[0]) + psqt
        return output_cp, w_acc, b_acc, ft_out, fc0_pre, fc0_out, fc1_out

    def to_npz(self) -> dict:
        return {
            "ft_biases":  self.ft_biases,
            "ft_weights": self.ft_weights,
            "fc0_biases": self.fc0_b,
            "fc0_weights": self.fc0_w,
            "fc1_biases": self.fc1_b,
            "fc1_weights": self.fc1_w,
            "fc2_biases": self.fc2_b,
            "fc2_weights": self.fc2_w,
        }

    @classmethod
    def from_npz(cls, d: dict) -> "SolaceNet":
        net = cls.__new__(cls)
        net.ft_biases  = d["ft_biases"]
        net.ft_weights = d["ft_weights"]
        net.fc0_b      = d["fc0_biases"]
        net.fc0_w      = d["fc0_weights"]
        net.fc1_b      = d["fc1_biases"]
        net.fc1_w      = d["fc1_weights"]
        net.fc2_b      = d["fc2_biases"]
        net.fc2_w      = d["fc2_weights"]
        return net


# ── WDL loss (sigmoid-based MSE) ─────────────────────────────────────────────

def wdl_sigmoid(cp: float, scale: float = 600.0) -> float:
    return 1.0 / (1.0 + np.exp(-cp / scale))


def loss_grad(predicted_cp: float, target_wdl: float,
              sample_weight: float = 1.0) -> tuple[float, float]:
    pred_wdl = wdl_sigmoid(predicted_cp)
    diff     = pred_wdl - target_wdl
    loss     = sample_weight * diff * diff
    # dL/d(pred_cp)
    d_wdl    = 2.0 * sample_weight * diff * pred_wdl * (1.0 - pred_wdl) / 600.0
    return loss, d_wdl


# ── Training loop ─────────────────────────────────────────────────────────────

def read_positions(tsv_path: Path, max_rows: int, rng_seed: int):
    rows = []
    with tsv_path.open() as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            fen, outcome, ply, imbal = parts[0], float(parts[1]), int(parts[2]), int(parts[3])
            rows.append((fen, outcome, imbal))
            if len(rows) >= max_rows:
                break
    rng = random.Random(rng_seed)
    rng.shuffle(rows)
    return rows


def train(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng_np = np.random.default_rng(args.seed)

    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        d   = dict(np.load(args.resume))
        net = SolaceNet.from_npz(d)
        start_epoch = int(d.get("epoch", np.array(0))) + 1
    else:
        net         = SolaceNet(rng_np)
        start_epoch = 1

    print(f"[train] Loading positions from {args.tsv} ...")
    positions = read_positions(Path(args.tsv), args.max_positions, args.seed)
    print(f"[train] Loaded {len(positions):,} positions.")

    lr         = args.lr
    aggr_w_bonus = args.aggr_weight_bonus
    aggr_thresh  = args.aggr_imbalance

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0         = time.time()
        total_loss = 0.0
        n_ok       = 0
        n_skip     = 0

        random.Random(args.seed + epoch).shuffle(positions)

        for i, (fen, target, imbal) in enumerate(positions):
            w_idx, b_idx = fen_to_features(fen)
            if w_idx is None:
                n_skip += 1
                continue

            # Aggression loss weighting
            sample_w = 1.0 + aggr_w_bonus * (1.0 if imbal >= aggr_thresh else 0.0)

            try:
                cp_out, w_acc, b_acc, ft_out, fc0_pre, fc0_out, fc1_out = net.forward(w_idx, b_idx)
            except Exception:
                n_skip += 1
                continue

            loss, d_cp = loss_grad(cp_out, target, sample_w)
            total_loss += loss

            # Backprop: FC2
            d_fc2_w = d_cp * fc1_out[np.newaxis, :]
            d_fc1   = d_cp * net.fc2_w[0]

            # FC1 ClippedReLU backward
            mask_fc1 = (fc1_out > 0.0) & (fc1_out < 1.0)
            d_fc1   *= mask_fc1.astype(np.float32)
            d_fc1_w  = d_fc1[:, np.newaxis] * fc0_out[np.newaxis, :]
            d_fc0_out = net.fc1_w.T @ d_fc1

            # FC0 SqrCReLU || CReLU backward
            fc0_lin_raw = fc0_pre[:L2]
            sqr_mask = (fc0_lin_raw > 0.0) & (fc0_lin_raw < 1.0)
            lin_mask = sqr_mask  # same range
            d_fc0_sqr = d_fc0_out[:L2] * 2.0 * np.clip(fc0_lin_raw, 0.0, 1.0) * sqr_mask
            d_fc0_lin = d_fc0_out[L2:] * lin_mask.astype(np.float32)
            d_fc0_pre        = np.zeros(L2 + 1, dtype=np.float32)
            d_fc0_pre[:L2]  += d_fc0_sqr + d_fc0_lin
            d_fc0_pre[L2]    = d_cp   # psqt passthrough
            d_fc0_w  = d_fc0_pre[:, np.newaxis] * ft_out[np.newaxis, :]
            d_ft_out = net.fc0_w.T @ d_fc0_pre

            # FT ClippedReLU backward
            w_mask = (w_acc > 0.0) & (w_acc < 1.0)
            b_mask = (b_acc > 0.0) & (b_acc < 1.0)
            d_w = d_ft_out[:L1] * w_mask.astype(np.float32)
            d_b = d_ft_out[L1:] * b_mask.astype(np.float32)

            # Gradient updates — vanilla SGD
            net.fc2_w  -= lr * d_fc2_w
            net.fc2_b  -= lr * np.array([d_cp], dtype=np.float32)
            net.fc1_w  -= lr * d_fc1_w
            net.fc1_b  -= lr * d_fc1
            net.fc0_w  -= lr * d_fc0_w
            net.fc0_b  -= lr * d_fc0_pre

            for idx in w_idx:
                if 0 <= idx < FT_INPUT_DIM:
                    net.ft_weights[idx] -= lr * d_w
            for idx in b_idx:
                if 0 <= idx < FT_INPUT_DIM:
                    net.ft_weights[idx] -= lr * d_b
            net.ft_biases -= lr * (d_w + d_b)

            n_ok += 1

            if (i + 1) % 5000 == 0:
                avg = total_loss / max(n_ok, 1)
                elapsed = time.time() - t0
                print(f"  epoch {epoch}  pos {i+1:>8,}  avg_loss={avg:.6f}  "
                      f"elapsed={elapsed:.1f}s  skip={n_skip}", flush=True)

        avg_loss = total_loss / max(n_ok, 1)
        elapsed  = time.time() - t0
        print(f"[train] Epoch {epoch} complete: avg_loss={avg_loss:.6f}  "
              f"positions={n_ok:,}  skipped={n_skip}  time={elapsed:.1f}s")

        ckpt_path = out_dir / f"epoch_{epoch:03d}.npz"
        save_data = net.to_npz()
        save_data["epoch"]     = np.array(epoch)
        save_data["avg_loss"]  = np.array(avg_loss)
        save_data["lr"]        = np.array(lr)
        save_data["aggr_level"]= np.array(args.aggr_imbalance)
        np.savez(ckpt_path, **save_data)
        print(f"[train] Checkpoint: {ckpt_path}")

        # Learning rate decay
        lr *= args.lr_decay

    print("[train] Training complete.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv",              help="Input FEN+outcome TSV (from pgn_to_fen.py)")
    ap.add_argument("--epochs",         type=int,   default=3)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--lr-decay",       type=float, default=0.9,
                    help="LR multiplier applied after each epoch (default: 0.9)")
    ap.add_argument("--max-positions",  type=int,   default=1_000_000)
    ap.add_argument("--out",            default="checkpoints",
                    help="Directory for .npz checkpoints (default: checkpoints/)")
    ap.add_argument("--resume",         metavar="CKPT.npz",
                    help="Resume from a checkpoint file")
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--aggr-imbalance", type=int,   default=100,
                    help="Material imbalance cp threshold for aggression weight boost")
    ap.add_argument("--aggr-weight-bonus", type=float, default=1.0,
                    help="Extra loss weight applied to aggressive positions (default: 1.0 = 2x)")
    args = ap.parse_args()
    sys.exit(train(args))


if __name__ == "__main__":
    main()
