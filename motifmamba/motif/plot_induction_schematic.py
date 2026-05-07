#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot an illustrative induction extrapolation comparison figure."
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/workspace/motif/logs/induction_extrapolation",
        help="Output directory for schematic plot.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="induction_schematic_comparison",
        help="Output filename prefix without extension.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lengths = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], dtype=np.int64)

    # Illustrative curves (not measured):
    # 1) L<=256: all models are close to 1.
    # 2) L>=512: transformer/lstm drop sharply and approach 0 with small fluctuations.
    # 3) rwkv > transformer/lstm; mamba > rwkv; motifmamba > mamba.
    curves = {
        "transformer": np.array([0.995, 0.992, 0.986, 0.34, 0.10, 0.040, 0.015, 0.022, 0.011]),
        "lstm": np.array([0.993, 0.989, 0.982, 0.29, 0.085, 0.032, 0.013, 0.019, 0.010]),
        "rwkv": np.array([0.996, 0.994, 0.989, 0.70, 0.42, 0.24, 0.15, 0.11, 0.08]),
        "mamba": np.array([0.997, 0.995, 0.991, 0.76, 0.48, 0.30, 0.20, 0.14, 0.10]),
        "motifmamba": np.array([0.999, 0.998, 0.995, 0.82, 0.56, 0.36, 0.24, 0.17, 0.12]),
    }

    colors = {
        "transformer": "#1f77b4",
        "mamba": "#ff7f0e",
        "lstm": "#9467bd",
        "rwkv": "#8c564b",
        "motifmamba": "#2ca02c",
    }
    markers = {
        "transformer": "o",
        "mamba": "s",
        "lstm": "^",
        "rwkv": "D",
        "motifmamba": "*",
    }

    plt.figure(figsize=(10, 5.8), dpi=180)
    order = ["transformer", "mamba", "lstm", "rwkv", "motifmamba"]
    for name in order:
        lw = 2.8 if name == "motifmamba" else 2.0
        ms = 9 if name == "motifmamba" else 6
        plt.plot(
            lengths,
            curves[name],
            label=name,
            color=colors[name],
            marker=markers[name],
            linewidth=lw,
            markersize=ms,
            alpha=0.95,
        )

    plt.xscale("log", base=2)
    plt.xticks(lengths, [str(x) for x in lengths])
    plt.ylim(0.0, 1.02)
    plt.xlabel("Sequence Length (log2 scale)")
    plt.ylabel("Accuracy (Illustrative)")
    plt.title("Induction Extrapolation (Schematic Comparison)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.35)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out_png = out_dir / f"{args.name}.png"
    out_svg = out_dir / f"{args.name}.svg"
    plt.savefig(out_png)
    plt.savefig(out_svg)

    print(f"[done] png={out_png}")
    print(f"[done] svg={out_svg}")


if __name__ == "__main__":
    main()
