#!/usr/bin/env python3
"""
Solace — NNUE checkpoint exporter
===================================
Converts a .npz training checkpoint (produced by train_nnue.py) to a
Stockfish-compatible .nnue binary file.

Binary format (little-endian throughout):
  [File header]
    uint32  Version   = 0x7AF32F20
    uint32  hash      = FT_hash ^ Arch_hash
    uint32  desc_len
    bytes   description

  [Feature Transformer]
    uint32  FT_hash (sub-section hash)
    LEB128  biases     L1 × int16
    LEB128  weights    FT_INPUT_DIM × L1 × int16
    LEB128  psqtWeights FT_INPUT_DIM × PSQT_BUCKETS × int32  (all zeros here)
    LEB128  threatWeights  ThreatInputDim × L1 × int16        (big net only)
    LEB128  threatPsqtWeights  ThreatInputDim × PSQT_BUCKETS × int32

  [LAYER_STACKS × NetworkArchitecture]   (one bucket per stack)
    uint32  Arch_hash
    int32[L2+1]  fc0 biases
    int8[L1*2*(L2+1)]  fc0 weights   (layout: row-major, weight[out][in])
    int32[L3]    fc1 biases
    int8[L2*2*L3] fc1 weights
    int32[1]     fc2 biases
    int8[L3]     fc2 weights

Quantization:
  FT weights/biases: float32 → int16  (scale = 127 * 64 = 8128)
  FC weights:        float32 → int8   (scale = 64)
  FC biases:         float32 → int32  (scale = 127 * 64 * 64 = 520192 for fc0,
                                                127 * 64 = 8128 for fc1/fc2)

Note: The threat feature weights (FullThreats) are not trained here; they are
      zeroed out in the exported file.  The engine will load and use the file
      without threats, which is equivalent to a non-threat-aware evaluation.

Usage:
    python3 nnue_export.py checkpoints/epoch_003.npz --out solace_v0.nnue
    python3 nnue_export.py checkpoints/epoch_003.npz --out solace_v0.nnue --verify-hash

GPLv3 — part of the Solace project (Stockfish fork).
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

# ── Architecture constants ────────────────────────────────────────────────────
L1            = 1024
L2            = 31
L3            = 32
PSQT_BUCKETS  = 8
LAYER_STACKS  = 8
SQUARE_NB     = 64
PS_NB         = 11 * SQUARE_NB
FT_INPUT_DIM  = SQUARE_NB * PS_NB // 2     # 22528

THREAT_INPUT_DIM = 60144  # FullThreats::Dimensions (zeroed out)

# File format version
NNUE_VERSION = 0x7AF32F20

# ── Hash computation (mirrors C++ constexpr logic) ───────────────────────────

def u32(x: int) -> int:
    return x & 0xFFFFFFFF


def arch_get_hash() -> int:
    h = u32(0xEC42E90D)
    h = u32(h ^ (L1 * 2))
    # fc_0: AffineTransformSparseInput<L1*2, L2+1>
    h_fc0 = u32(0xCC03DAE4 + (L2 + 1))
    h_fc0 = u32(h_fc0 ^ (h >> 1) ^ (h << 31))
    h = h_fc0
    # ac_sqr_0: SqrClippedReLU (same hash formula as ClippedReLU)
    h = u32(0x538D24C7 + h)
    # ac_0: ClippedReLU<L2+1>
    h = u32(0x538D24C7 + h)
    # fc_1: AffineTransform<L2*2, L3>
    h_fc1 = u32(0xCC03DAE4 + L3)
    h_fc1 = u32(h_fc1 ^ (h >> 1) ^ (h << 31))
    h = h_fc1
    # ac_1: ClippedReLU<L3>
    h = u32(0x538D24C7 + h)
    # fc_2: AffineTransform<L3, 1>
    h_fc2 = u32(0xCC03DAE4 + 1)
    h_fc2 = u32(h_fc2 ^ (h >> 1) ^ (h << 31))
    h = h_fc2
    return h


def ft_get_hash() -> int:
    # UseThreats = True for big net; uses FullThreats::HashValue
    return u32(0x8f234cb8 ^ (L1 * 2))


def network_hash() -> int:
    return u32(ft_get_hash() ^ arch_get_hash())


# ── LEB128 encoding (used by FT write_leb_128) ───────────────────────────────

def encode_leb128_i16_array(arr: np.ndarray) -> bytes:
    """Encode a flat int16 array using signed LEB128."""
    out = bytearray()
    for v in arr.astype(np.int16).flat:
        v = int(v)
        while True:
            byte = v & 0x7F
            v >>= 7
            if (v == 0 and not (byte & 0x40)) or (v == -1 and (byte & 0x40)):
                out.append(byte)
                break
            out.append(byte | 0x80)
    return bytes(out)


def encode_leb128_i32_array(arr: np.ndarray) -> bytes:
    """Encode a flat int32 array using signed LEB128."""
    out = bytearray()
    for v in arr.astype(np.int32).flat:
        v = int(v)
        while True:
            byte = v & 0x7F
            v >>= 7
            if (v == 0 and not (byte & 0x40)) or (v == -1 and (byte & 0x40)):
                out.append(byte)
                break
            out.append(byte | 0x80)
    return bytes(out)


# ── Quantisation helpers ──────────────────────────────────────────────────────

FT_WEIGHT_SCALE  = 127 * 64           # 8128  → int16
FT_BIAS_SCALE    = 127 * 64           # 8128  → int16
FC_WEIGHT_SCALE  = 64                 # → int8
FC0_BIAS_SCALE   = 127 * 64 * 64     # 520192 → int32  (compensates int8×int8 accumulation)
FC1_BIAS_SCALE   = 127 * 64          # 8128   → int32
FC2_BIAS_SCALE   = 127 * 64          # 8128   → int32


def quantise_ft_biases(biases: np.ndarray) -> np.ndarray:
    return np.clip(np.round(biases * FT_BIAS_SCALE), -32768, 32767).astype(np.int16)


def quantise_ft_weights(weights: np.ndarray) -> np.ndarray:
    return np.clip(np.round(weights * FT_WEIGHT_SCALE), -32768, 32767).astype(np.int16)


def quantise_fc_weights(w: np.ndarray) -> np.ndarray:
    return np.clip(np.round(w * FC_WEIGHT_SCALE), -128, 127).astype(np.int8)


def quantise_fc0_biases(b: np.ndarray) -> np.ndarray:
    return np.round(b * FC0_BIAS_SCALE).astype(np.int32)


def quantise_fc1_biases(b: np.ndarray) -> np.ndarray:
    return np.round(b * FC1_BIAS_SCALE).astype(np.int32)


def quantise_fc2_biases(b: np.ndarray) -> np.ndarray:
    return np.round(b * FC2_BIAS_SCALE).astype(np.int32)


# ── Binary writer ─────────────────────────────────────────────────────────────

def write_u32_le(fh, v: int):
    fh.write(struct.pack("<I", v & 0xFFFFFFFF))


def write_i32_le_array(fh, arr: np.ndarray):
    fh.write(arr.astype("<i4").tobytes())


def write_i8_array(fh, arr: np.ndarray):
    fh.write(arr.astype(np.int8).tobytes())


def export_nnue(npz_path: Path, out_path: Path, description: str, verify_hash: bool):
    print(f"[export] Loading checkpoint: {npz_path}")
    d = dict(np.load(npz_path))

    ft_biases  = d["ft_biases"]    # float32 [L1]
    ft_weights = d["ft_weights"]   # float32 [FT_INPUT_DIM, L1]
    fc0_b      = d["fc0_biases"]   # float32 [L2+1]
    fc0_w      = d["fc0_weights"]  # float32 [L2+1, L1*2]
    fc1_b      = d["fc1_biases"]   # float32 [L3]
    fc1_w      = d["fc1_weights"]  # float32 [L3, L2*2]
    fc2_b      = d["fc2_biases"]   # float32 [1]
    fc2_w      = d["fc2_weights"]  # float32 [1, L3]

    epoch     = int(d.get("epoch",    np.array(0)))
    avg_loss  = float(d.get("avg_loss", np.array(-1.0)))
    print(f"[export] Epoch={epoch}  avg_loss={avg_loss:.6f}")

    # Compute hashes
    ft_hash   = ft_get_hash()
    arch_hash = arch_get_hash()
    net_hash  = network_hash()
    print(f"[export] FT hash   : 0x{ft_hash:08x}")
    print(f"[export] Arch hash : 0x{arch_hash:08x}")
    print(f"[export] Net hash  : 0x{net_hash:08x}")

    # Quantise
    q_ft_bias  = quantise_ft_biases(ft_biases)          # int16 [L1]
    q_ft_wt    = quantise_ft_weights(ft_weights)         # int16 [FT_INPUT_DIM, L1]
    q_fc0_b    = quantise_fc0_biases(fc0_b)              # int32 [L2+1]
    q_fc0_w    = quantise_fc_weights(fc0_w)              # int8  [L2+1, L1*2]
    q_fc1_b    = quantise_fc1_biases(fc1_b)              # int32 [L3]
    q_fc1_w    = quantise_fc_weights(fc1_w)              # int8  [L3, L2*2]
    q_fc2_b    = quantise_fc2_biases(fc2_b)              # int32 [1]
    q_fc2_w    = quantise_fc_weights(fc2_w)              # int8  [1, L3]

    # Threat weights (not trained) — zero
    zero_threat_wt  = np.zeros((THREAT_INPUT_DIM, L1),           dtype=np.int16)
    zero_threat_psqt= np.zeros((THREAT_INPUT_DIM, PSQT_BUCKETS), dtype=np.int32)
    zero_psqt       = np.zeros((FT_INPUT_DIM,     PSQT_BUCKETS), dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        # ── File header ───────────────────────────────────────────────────────
        desc_bytes = description.encode("utf-8")
        write_u32_le(fh, NNUE_VERSION)
        write_u32_le(fh, net_hash)
        write_u32_le(fh, len(desc_bytes))
        fh.write(desc_bytes)

        # ── Feature Transformer ───────────────────────────────────────────────
        write_u32_le(fh, ft_hash)
        # biases (int16, LEB128)
        fh.write(encode_leb128_i16_array(q_ft_bias))
        # threatWeights (int16, LEB128) — zeroed
        fh.write(encode_leb128_i16_array(zero_threat_wt))
        # weights (int16, LEB128) — row-major [FT_INPUT_DIM × L1]
        fh.write(encode_leb128_i16_array(q_ft_wt))
        # combined psqtWeights: threat first, then normal (int32, LEB128)
        combined_psqt = np.concatenate([
            zero_threat_psqt.reshape(-1),
            zero_psqt.reshape(-1)
        ])
        fh.write(encode_leb128_i32_array(combined_psqt))

        # ── Layer stacks (8 identical buckets from this single-bucket model) ─
        for bucket in range(LAYER_STACKS):
            write_u32_le(fh, arch_hash)
            # fc0 biases: int32 LE, length L2+1
            write_i32_le_array(fh, q_fc0_b)
            # fc0 weights: int8, shape [L2+1, L1*2]
            write_i8_array(fh, q_fc0_w)
            # fc1 biases: int32 LE, length L3
            write_i32_le_array(fh, q_fc1_b)
            # fc1 weights: int8, shape [L3, L2*2]
            write_i8_array(fh, q_fc1_w)
            # fc2 biases: int32 LE, length 1
            write_i32_le_array(fh, q_fc2_b)
            # fc2 weights: int8, shape [1, L3]
            write_i8_array(fh, q_fc2_w)

    size_bytes = out_path.stat().st_size
    sha256     = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"[export] Written:  {out_path}  ({size_bytes:,} bytes)")
    print(f"[export] SHA-256:  {sha256}")
    print(f"[export] Filename: nn-{sha256[:12]}.nnue  ← rename to this for EvalFile UCI option")

    if verify_hash:
        expected_name = f"nn-{sha256[:12]}.nnue"
        if out_path.name == expected_name:
            print("[export] Hash-based filename: MATCH ✓")
        else:
            print(f"[export] Hash-based filename: MISMATCH — rename to {expected_name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint",     help=".npz checkpoint from train_nnue.py")
    ap.add_argument("--out",          required=True, metavar="FILE.nnue",
                    help="Output .nnue file path")
    ap.add_argument("--description",  default="Solace aggressive net",
                    help="Description string embedded in .nnue header")
    ap.add_argument("--verify-hash",  action="store_true",
                    help="Print whether output filename matches SHA-256 convention")
    args = ap.parse_args()

    export_nnue(
        npz_path     = Path(args.checkpoint),
        out_path     = Path(args.out),
        description  = args.description,
        verify_hash  = args.verify_hash,
    )


if __name__ == "__main__":
    main()
