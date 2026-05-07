#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Plot motif loss curves for each rank (motif1..13 on one figure).")
    p.add_argument("--exp-dir", type=str, required=True, help="Experiment root dir containing rank_* folders")
    p.add_argument("--metric", type=str, default="motif_loss", help="Column to plot, default motif_loss")
    p.add_argument("--out-dir", type=str, default="", help="Output dir, default <exp-dir>/plots")
    return p.parse_args()


def rank_key(p: Path):
    m = re.match(r"rank_(\d+)$", p.name)
    return int(m.group(1)) if m else 10**9


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"exp-dir not found: {exp_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (exp_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and re.match(r"rank_\d+$", p.name)], key=rank_key)
    if not rank_dirs:
        raise RuntimeError(f"No rank_* dirs found under {exp_dir}")

    colors = plt.cm.tab20.colors

    generated = []
    for rank_dir in rank_dirs:
        rank = rank_key(rank_dir)
        plt.figure(figsize=(11, 6), dpi=160)

        any_curve = False
        for motif in range(1, 14):
            csv_path = rank_dir / f"motif{motif}" / "train_log.csv"
            if not csv_path.is_file():
                continue
            df = pd.read_csv(csv_path)
            if "phase" in df.columns:
                warmup = df[df["phase"] == "warmup"].copy()
            else:
                warmup = df.copy()
            if warmup.empty:
                continue

            if "phase_step" in warmup.columns:
                x = warmup["phase_step"].astype(int)
            else:
                x = warmup["global_step"].astype(int)

            if args.metric not in warmup.columns:
                continue
            y = warmup[args.metric].astype(float)

            plt.plot(
                x,
                y,
                linewidth=1.8,
                label=f"motif{motif}",
                color=colors[(motif - 1) % len(colors)],
            )
            any_curve = True

        if not any_curve:
            plt.close()
            continue

        plt.xlabel("Warmup Step")
        plt.ylabel(args.metric)
        plt.title(f"Rank {rank}: motif1-13 {args.metric} curves")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(ncol=4, fontsize=8, frameon=True)
        plt.tight_layout()

        out_png = out_dir / f"rank_{rank}_motif1to13_{args.metric}.png"
        out_svg = out_dir / f"rank_{rank}_motif1to13_{args.metric}.svg"
        plt.savefig(out_png)
        plt.savefig(out_svg)
        plt.close()
        generated.append((out_png, out_svg))

    if not generated:
        raise RuntimeError("No figures generated. Check train_log.csv files and metric column.")

    print(f"[done] exp_dir={exp_dir}")
    print(f"[done] out_dir={out_dir}")
    print(f"[done] figures={len(generated)}")
    for png, svg in generated:
        print(f"[figure] png={png}")
        print(f"[figure] svg={svg}")


if __name__ == "__main__":
    main()
