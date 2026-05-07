#!/usr/bin/env python3
import argparse
import csv
import json
import os
from datetime import datetime
from types import SimpleNamespace

import torch

import induction_extrapolation_benchmark as bench
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved induction checkpoints without retraining.")
    p.add_argument(
        "--run-dirs",
        type=str,
        required=True,
        help="Comma-separated run dirs under motif/logs/induction_extrapolation.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--test-lens",
        type=str,
        default="",
        help="Optional comma-separated lengths to override config test lengths.",
    )
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--eval-tokens-budget", type=int, default=131072)
    p.add_argument("--max-eval-batch-size", type=int, default=64)
    p.add_argument("--max-transformer-eval-len", type=int, default=8192)
    p.add_argument("--amp", type=str, default="", choices=["", "none", "bf16", "fp16"])
    p.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional output csv path. Default writes into induction log root with timestamp.",
    )
    return p.parse_args()


def _get_test_lengths(cfg):
    if isinstance(cfg.get("test_lengths"), list) and cfg["test_lengths"]:
        return [int(x) for x in cfg["test_lengths"]]
    if isinstance(cfg.get("test_lens"), str):
        return bench.parse_int_csv(cfg["test_lens"])
    return bench.parse_int_csv(bench.default_test_lengths())


def _get_ckpts(run_dir):
    out = []
    for name in sorted(os.listdir(run_dir)):
        if name.startswith("model_") and name.endswith("_latest.pt"):
            out.append(os.path.join(run_dir, name))
    return out


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        bench.mamba_simple_mod.causal_conv1d_fn = None
        bench.mamba_simplemotif_mod.causal_conv1d_fn = None
        bench.mamba_simple_mod.selective_scan_fn = selective_scan_ref
        bench.mamba_simplemotif_mod.selective_scan_fn = selective_scan_ref

    run_dirs = [x.strip() for x in args.run_dirs.split(",") if x.strip()]
    all_rows = []
    missing_runs = []

    for run_dir in run_dirs:
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.isfile(cfg_path):
            print(f"[skip] missing config: {cfg_path}")
            missing_runs.append((run_dir, "missing_config"))
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        ckpts = _get_ckpts(run_dir)
        if not ckpts:
            print(f"[skip] no checkpoint in {run_dir}")
            missing_runs.append((run_dir, "missing_checkpoint"))
            continue

        test_lengths = (
            bench.parse_int_csv(args.test_lens)
            if args.test_lens.strip()
            else _get_test_lengths(cfg)
        )
        eval_args = SimpleNamespace(
            eval_batches=int(args.eval_batches),
            eval_batch_size=int(args.eval_batch_size),
            eval_tokens_budget=int(args.eval_tokens_budget),
            max_eval_batch_size=int(args.max_eval_batch_size),
            max_transformer_eval_len=int(args.max_transformer_eval_len),
            vocab_size=int(cfg.get("vocab_size", 64)),
            amp=str(args.amp if args.amp else cfg.get("amp", "bf16")),
        )

        for ckpt_path in ckpts:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            arch = ckpt.get("model", os.path.basename(ckpt_path).replace("model_", "").replace("_latest.pt", ""))
            model_cfg = ckpt.get("config", {})
            d_model = int(model_cfg.get("d_model", cfg.get("d_model", 64)))
            n_heads = int(model_cfg.get("n_heads", bench.pick_num_heads(d_model, int(cfg.get("max_heads", 8)))))
            n_layers = int(model_cfg.get("n_layers", cfg.get("n_layers", 2)))
            model = bench.build_model(
                arch=arch,
                vocab_size=int(cfg.get("vocab_size", 64)),
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ffn_mult=float(cfg.get("ffn_mult", 2.0)),
                d_state=int(cfg.get("d_state", 16)),
                d_conv=int(cfg.get("d_conv", 4)),
                expand=int(cfg.get("expand", 2)),
                pq_rank=int(cfg.get("pq_rank", 2)),
                pq_k_init=float(cfg.get("pq_k_init", 1e-4)),
            ).to(device)
            model.load_state_dict(ckpt["state_dict"], strict=True)
            params = int(ckpt.get("params", bench.count_params(model)))
            mcfg = bench.ModelConfig(arch=arch, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
            print("=" * 100)
            print(f"[eval] run_dir={run_dir}")
            print(f"[eval] ckpt={ckpt_path}")
            print(f"[eval] model={arch}, params={params}, amp={eval_args.amp}")
            rows = bench.evaluate_lengths(eval_args, model, arch, params, mcfg, test_lengths, device)
            for r in rows:
                r["run_dir"] = run_dir
                r["checkpoint"] = ckpt_path
            all_rows.extend(rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not all_rows:
        raise RuntimeError("No checkpoints were evaluated.")

    fieldnames = [
        "run_dir",
        "checkpoint",
        "model",
        "params",
        "d_model",
        "n_layers",
        "n_heads",
        "test_len",
        "eval_batches",
        "batch_size",
        "acc",
        "status",
        "seconds",
        "error",
    ]
    if args.out_csv:
        out_csv = args.out_csv
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(
            os.path.dirname(os.path.dirname(run_dirs[0])),
            f"retest_accuracy_by_length_{ts}.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print("=" * 100)
    print(f"[done] rows={len(all_rows)}")
    print(f"[done] csv={out_csv}")
    if missing_runs:
        print(f"[done] missing_runs={missing_runs}")


if __name__ == "__main__":
    main()
