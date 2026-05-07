#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPL on Pile for Mamba + PQ adapter.")
    parser.add_argument("--pretrained-dir", type=str, default="/workspace/mamba-1.4b")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to pq_adapter_best.pt or latest.pt")
    parser.add_argument("--pile-root", type=str, default="/workspace/datasets/pile_standard_pythia")
    parser.add_argument("--pile-shards", type=int, default=1, help="Use first N shards for eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--eval-seqs", type=int, default=128, help="Number of fixed windows to evaluate")
    parser.add_argument("--start-offset", type=int, default=0, help="Token offset in first shard")
    parser.add_argument("--out-csv", type=str, default="")
    return parser.parse_args()


def resolve_dtype(s):
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float32


def list_pile_bins(pile_root, pile_shards):
    pattern = re.compile(r"document-\d{5}-of-\d{5}\.bin$")
    files = sorted([p for p in glob(os.path.join(pile_root, "*.bin")) if pattern.search(os.path.basename(p))])
    if len(files) == 0:
        raise FileNotFoundError(f"No pile bin files found under: {pile_root}")
    if pile_shards > 0:
        files = files[:pile_shards]
    return files


def infer_pq_layout_from_adapter(adapter_state):
    for k, v in adapter_state.items():
        if k.endswith(".P"):
            if v.dim() == 2:
                return int(v.shape[1]), False
            if v.dim() == 3:
                return int(v.shape[2]), True
    raise ValueError("Cannot infer pq_rank/pq_per_dim from adapter state dict (no '.P' tensor found).")


def load_model_with_adapter(pretrained_dir, adapter_path, device, dtype):
    from mamba_ssm.models.config_mamba import MambaConfig
    import mamba_ssm.models.mixer_seq_simple as mixer_seq_simple
    from mamba_ssm.modules.mamba_simplemotif import Mambamotif
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

    adapter_obj = torch.load(adapter_path, map_location="cpu")
    if isinstance(adapter_obj, dict) and "pq_state_dict" in adapter_obj:
        pq_state = adapter_obj["pq_state_dict"]
        adapter_args = adapter_obj.get("args", {})
        pq_k_init = float(adapter_args.get("pq_k_init", 1e-4))
        arg_pq_per_dim = adapter_args.get("pq_per_dim", None)
    elif isinstance(adapter_obj, dict):
        pq_state = adapter_obj
        pq_k_init = 1e-4
        arg_pq_per_dim = None
    else:
        raise ValueError("Adapter file format not recognized.")

    pq_rank, inferred_per_dim = infer_pq_layout_from_adapter(pq_state)
    pq_per_dim = bool(arg_pq_per_dim) if arg_pq_per_dim is not None else inferred_per_dim

    mixer_seq_simple.Mamba = Mambamotif
    cfg = load_config_hf(pretrained_dir)
    config = MambaConfig(**cfg)
    if device.type != "cuda":
        # Triton fused LN is CUDA-only.
        config.fused_add_norm = False
        config.rms_norm = False
    ssm_cfg = dict(config.ssm_cfg or {})
    ssm_cfg["pq_rank"] = int(pq_rank)
    ssm_cfg["pq_per_dim"] = bool(pq_per_dim)
    ssm_cfg["pq_k_init"] = float(pq_k_init)
    ssm_cfg["use_fast_path"] = False
    config.ssm_cfg = ssm_cfg

    model = mixer_seq_simple.MambaLMHeadModel(config, device="cpu", dtype=torch.float32)
    base_state = load_state_dict_hf(pretrained_dir, device="cpu", dtype=None)
    model.load_state_dict(base_state, strict=False)
    model.load_state_dict(pq_state, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, pq_rank, pq_per_dim


def make_windows_from_shards(bin_files, seq_len, eval_seqs, start_offset=0):
    arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in bin_files]
    need = int(eval_seqs)
    windows = []
    stride = seq_len + 1
    shard_i = 0
    offset = int(start_offset)
    while len(windows) < need and shard_i < len(arrays):
        arr = arrays[shard_i]
        n = int(arr.shape[0])
        while offset + stride <= n and len(windows) < need:
            windows.append(np.asarray(arr[offset : offset + stride], dtype=np.int64))
            offset += stride
        shard_i += 1
        offset = 0
    if len(windows) == 0:
        raise RuntimeError("No evaluation windows generated. Check seq_len/start_offset/pile_shards.")
    return windows


@torch.no_grad()
def evaluate_ppl(model, windows, batch_size, device, amp_dtype):
    total_nll = 0.0
    total_tokens = 0
    n_batches = (len(windows) + batch_size - 1) // batch_size
    t0 = time.time()
    for b in range(n_batches):
        chunk = windows[b * batch_size : (b + 1) * batch_size]
        x = torch.tensor(np.stack(chunk, axis=0), dtype=torch.long, device=device)
        inp = x[:, :-1]
        tgt = x[:, 1:]
        if device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=(amp_dtype != torch.float32), dtype=amp_dtype):
                logits = model(inp).logits
        else:
            logits = model(inp).logits
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll.item())
        total_tokens += int(tgt.numel())
        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            cur = total_nll / max(total_tokens, 1)
            print(f"[eval] batch {b+1}/{n_batches}, mean_nll={cur:.6f}, ppl={math_exp(cur):.4f}")
    mean_nll = total_nll / max(total_tokens, 1)
    ppl = math_exp(mean_nll)
    elapsed = time.time() - t0
    return mean_nll, ppl, total_tokens, elapsed


def math_exp(x):
    # Safe enough for NLL ranges we expect.
    return float(np.exp(x))


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    sys.path.insert(0, repo_root)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("[warn] bf16 is not supported on this GPU, fallback to float16.")
        dtype = torch.float16

    bin_files = list_pile_bins(args.pile_root, args.pile_shards)
    print(f"Using {len(bin_files)} pile shards, first shard: {bin_files[0]}")
    print(f"Loading model from {args.pretrained_dir}")
    print(f"Loading adapter from {args.adapter_path}")
    model, pq_rank, pq_per_dim = load_model_with_adapter(args.pretrained_dir, args.adapter_path, device, dtype)
    print(f"Model loaded. pq_rank={pq_rank}, pq_per_dim={pq_per_dim}")

    windows = make_windows_from_shards(
        bin_files=bin_files,
        seq_len=args.seq_len,
        eval_seqs=args.eval_seqs,
        start_offset=args.start_offset,
    )
    print(f"Eval windows: {len(windows)}, seq_len={args.seq_len}, total_pred_tokens={len(windows)*args.seq_len}")

    amp_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
    mean_nll, ppl, total_tokens, elapsed = evaluate_ppl(
        model=model,
        windows=windows,
        batch_size=args.batch_size,
        device=device,
        amp_dtype=amp_dtype,
    )

    print(f"[done] mean_nll={mean_nll:.6f}, ppl={ppl:.6f}, tokens={total_tokens}, elapsed_sec={elapsed:.2f}")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        row = {
            "pretrained_dir": args.pretrained_dir,
            "adapter_path": args.adapter_path,
            "pq_rank": pq_rank,
            "pq_per_dim": pq_per_dim,
            "pile_root": args.pile_root,
            "pile_shards": args.pile_shards,
            "seq_len": args.seq_len,
            "eval_seqs": args.eval_seqs,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "mean_nll": mean_nll,
            "ppl": ppl,
            "tokens": total_tokens,
            "elapsed_sec": elapsed,
        }
        write_header = not os.path.isfile(args.out_csv)
        with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"[done] wrote csv: {args.out_csv}")


if __name__ == "__main__":
    main()
