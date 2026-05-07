#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Plot motif loss curves per rank with flexible motif dir names.")
    p.add_argument("--exp-dir", type=str, required=True)
    p.add_argument("--metric", type=str, default="motif_loss")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument(
        "--motif13-rescale",
        type=float,
        default=1.0,
        help="Divide motif13 curve by this factor before plotting (e.g., 300).",
    )
    p.add_argument(
        "--motif13-rescale-auto",
        action="store_true",
        help="Auto-rescale motif13 per rank so its first point matches median first-point of motif1..12.",
    )
    p.add_argument(
        "--max-step",
        type=int,
        default=-1,
        help="If >0, only plot points with step <= max-step.",
    )
    p.add_argument(
        "--fill-to-step",
        type=int,
        default=-1,
        help="If >0, extend curves shorter than this step using recent trend.",
    )
    p.add_argument(
        "--fill-window",
        type=int,
        default=20,
        help="Number of tail points used to estimate trend for extension.",
    )
    p.add_argument(
        "--fill-mode",
        type=str,
        default="log_linear",
        choices=["linear", "log_linear"],
        help="Trend model used for curve extension.",
    )
    p.add_argument(
        "--smooth-method",
        type=str,
        default="none",
        choices=["none", "moving_avg", "median", "ema"],
        help="Smoothing applied to y curve before plotting.",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Window size for moving_avg / median smoothing.",
    )
    p.add_argument(
        "--smooth-ema-span",
        type=int,
        default=12,
        help="EMA span when --smooth-method=ema.",
    )
    p.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable background grid lines.",
    )
    p.add_argument(
        "--axis-endpoints-only",
        action="store_true",
        help="Show only start/end ticks on both axes.",
    )
    p.add_argument(
        "--axis-start-zero",
        action="store_true",
        help="Force x/y axis lower bounds to 0.",
    )
    return p.parse_args()


def rank_key(p: Path):
    m = re.match(r"rank_(\d+)$", p.name)
    return int(m.group(1)) if m else 10**9


def motif_key(p: Path):
    m = re.match(r"motif(\d+)(?:_|$)", p.name)
    return int(m.group(1)) if m else None


def extend_curve(x: np.ndarray, y: np.ndarray, target_step: int, window: int, mode: str):
    if target_step <= 0 or x.size == 0:
        return x, y
    last_x = int(x[-1])
    if last_x >= int(target_step):
        return x, y
    if x.size < 2:
        return x, y

    w = max(2, min(int(window), int(x.size)))
    xw = x[-w:].astype(float)
    yw = y[-w:].astype(float)

    x_extra = np.arange(last_x + 1, int(target_step) + 1, dtype=int)
    if x_extra.size == 0:
        return x, y

    if mode == "log_linear":
        eps = 1e-12
        z = np.log(np.clip(yw, eps, None))
        a, b = np.polyfit(xw, z, deg=1)
        y_extra = np.exp(a * x_extra + b)
    else:
        a, b = np.polyfit(xw, yw, deg=1)
        y_extra = a * x_extra + b
        y_extra = np.clip(y_extra, 0.0, None)

    x_out = np.concatenate([x, x_extra.astype(x.dtype)])
    y_out = np.concatenate([y, y_extra.astype(y.dtype)])
    return x_out, y_out


def smooth_curve(y: np.ndarray, method: str, window: int, ema_span: int):
    if y.size == 0 or method == "none":
        return y
    s = pd.Series(y.astype(float))
    if method == "moving_avg":
        w = max(1, int(window))
        return s.rolling(window=w, min_periods=1, center=True).mean().to_numpy()
    if method == "median":
        w = max(1, int(window))
        return s.rolling(window=w, min_periods=1, center=True).median().to_numpy()
    if method == "ema":
        sp = max(1, int(ema_span))
        return s.ewm(span=sp, adjust=False).mean().to_numpy()
    return y


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.is_dir():
        raise FileNotFoundError(exp_dir)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (exp_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and re.match(r"rank_\d+$", p.name)], key=rank_key)
    if not rank_dirs:
        raise RuntimeError("No rank_* dirs found")

    colors = plt.cm.tab20.colors
    n_fig = 0

    for rank_dir in rank_dirs:
        motif_dirs = [p for p in rank_dir.iterdir() if p.is_dir() and motif_key(p) is not None]
        motif_dirs.sort(key=lambda p: motif_key(p))

        # Preload data for optional auto-rescale
        rank_data = {}
        for d in motif_dirs:
            m = motif_key(d)
            csv_path = d / "train_log.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if args.metric not in df.columns:
                continue
            if "phase" in df.columns:
                df = df[df["phase"] == "warmup"].copy()
            if df.empty:
                continue
            x = (df["phase_step"].astype(int) if "phase_step" in df.columns else df["global_step"].astype(int)).to_numpy()
            y = df[args.metric].astype(float).to_numpy()
            if args.max_step > 0:
                mask = x <= int(args.max_step)
                x = x[mask]
                y = y[mask]
                if len(x) == 0:
                    continue
            if args.fill_to_step > 0:
                x, y = extend_curve(
                    x=x,
                    y=y,
                    target_step=int(args.fill_to_step),
                    window=int(args.fill_window),
                    mode=str(args.fill_mode),
                )
            rank_data[m] = (x, y)

        motif13_factor = float(args.motif13_rescale)
        if args.motif13_rescale_auto:
            base_vals = [float(rank_data[m][1].iloc[0]) for m in rank_data.keys() if 1 <= m <= 12]
            m13_first = float(rank_data[13][1].iloc[0]) if 13 in rank_data else None
            if base_vals and m13_first is not None and m13_first != 0.0:
                target = float(pd.Series(base_vals).median())
                if target > 0:
                    motif13_factor = m13_first / target
            if motif13_factor <= 0:
                motif13_factor = 1.0

        plt.figure(figsize=(11, 6), dpi=160)
        any_curve = False

        for m in sorted(rank_data.keys()):
            x, y = rank_data[m]
            label = f"motif{m}"
            if m == 13 and motif13_factor != 1.0:
                y = y / float(motif13_factor)
                label = f"motif13(/ {motif13_factor:g})"
            y = smooth_curve(
                y=y,
                method=str(args.smooth_method),
                window=int(args.smooth_window),
                ema_span=int(args.smooth_ema_span),
            )

            plt.plot(
                x,
                y,
                linewidth=1.8,
                label=label,
                color=colors[(m - 1) % len(colors)],
            )
            any_curve = True

        if not any_curve:
            plt.close()
            continue

        rk = rank_key(rank_dir)
        plt.xlabel("Warmup Step")
        plt.ylabel(args.metric)
        title = f"Rank {rk}: motif1-13 {args.metric} curves"
        if motif13_factor != 1.0:
            suffix = "auto" if args.motif13_rescale_auto else "manual"
            title += f" (motif13 / {motif13_factor:g}, {suffix})"
        plt.title(title)
        ax = plt.gca()
        x_end_target = None
        if args.fill_to_step > 0:
            x_end_target = float(args.fill_to_step)
        elif args.max_step > 0:
            x_end_target = float(args.max_step)

        if args.axis_start_zero:
            _, xmax = ax.get_xlim()
            _, ymax = ax.get_ylim()
            ax.set_xlim(left=0.0, right=(x_end_target if x_end_target is not None else xmax))
            ax.set_ylim(bottom=0.0, top=ymax)
        elif x_end_target is not None:
            xmin, _ = ax.get_xlim()
            ax.set_xlim(left=xmin, right=x_end_target)
        if not args.no_grid:
            plt.grid(True, linestyle="--", alpha=0.35)
        if args.axis_endpoints_only:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.set_xticks([xmin, xmax])
            ax.set_yticks([ymin, ymax])
        plt.legend(ncol=4, fontsize=8, frameon=True)
        plt.tight_layout()

        out_png = out_dir / f"rank_{rk}_motif1to13_{args.metric}.png"
        out_svg = out_dir / f"rank_{rk}_motif1to13_{args.metric}.svg"
        plt.savefig(out_png)
        plt.savefig(out_svg)
        plt.close()
        print(f"[figure] png={out_png}")
        print(f"[figure] svg={out_svg}")
        n_fig += 1

    if n_fig == 0:
        raise RuntimeError("No figures generated")
    print(f"[done] out_dir={out_dir}, figures={n_fig}")


if __name__ == "__main__":
    main()
