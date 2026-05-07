#!/usr/bin/env python3
import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MAMBA_ROOT = os.environ.get("MAMBA_ROOT", "/workspace/mamba-main")
import sys
if MAMBA_ROOT not in sys.path:
    sys.path.insert(0, MAMBA_ROOT)

import mamba_ssm.modules.mamba_simple as mamba_simple_mod
import mamba_ssm.modules.mamba_simplemotif as mamba_simplemotif_mod

Mamba = mamba_simple_mod.Mamba
Mambamotif = mamba_simplemotif_mod.Mambamotif


def parse_int_csv(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def default_test_lengths() -> str:
    # 64 .. 1048576
    return ",".join(str(2 ** i) for i in range(6, 21))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_num_heads(d_model: int, max_heads: int, target: int = 4) -> int:
    divisors = [h for h in range(1, max_heads + 1) if d_model % h == 0]
    if not divisors:
        return 1
    return min(divisors, key=lambda h: abs(h - target))


def sinusoidal_pos_encoding(seq_len: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d_model, device=device, dtype=torch.float32).unsqueeze(0)
    angle_rates = 1.0 / torch.pow(10000.0, (2 * torch.floor(i / 2)) / d_model)
    angles = pos * angle_rates
    pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe.to(dtype=dtype)


def make_induction_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if seq_len < 8:
        raise ValueError("seq_len must be >= 8 for induction task")
    if vocab_size < 8:
        raise ValueError("vocab_size must be >= 8 for induction task")

    x = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    target_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    for b in range(batch_size):
        key = int(torch.randint(1, vocab_size, (1,), device=device).item())
        val = int(torch.randint(1, vocab_size, (1,), device=device).item())
        if val == key:
            val = 1 + (val % (vocab_size - 1))
        alt = 1 + (key % (vocab_size - 1))
        row = x[b]
        row[row == key] = alt

        left_lo = 1
        left_hi = max(left_lo, (seq_len // 2) - 2)
        kv_pos = random.randint(left_lo, left_hi)
        q_lo = max(kv_pos + 2, seq_len // 2)
        q_hi = seq_len - 2
        if q_lo > q_hi:
            q_lo = q_hi
        q_pos = random.randint(q_lo, q_hi)

        row[kv_pos] = key
        row[kv_pos + 1] = val
        row[q_pos] = key
        row[q_pos + 1] = val
        target_mask[b, q_pos + 1] = True

    return x, target_mask


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, ffn_mult: float):
        super().__init__()
        ffn_dim = int(round(d_model * float(ffn_mult)))
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=ffn_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        x = self.embed(tokens)
        x = x + sinusoidal_pos_encoding(seqlen, x.shape[-1], x.device, x.dtype).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones((seqlen, seqlen), device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        x = self.norm(x)
        return self.lm_head(x)


class RWKVLikeLM(nn.Module):
    """
    Lightweight RWKV-like recurrent baseline using GRU core.
    This keeps O(L) memory/time characteristics for long-context extrapolation.
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        y, _ = self.rnn(x)
        y = self.norm(y)
        return self.lm_head(y)


class LSTMLM(nn.Module):
    """Plain LSTM baseline for long-context extrapolation comparison."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        y, _ = self.rnn(x)
        y = self.norm(y)
        return self.lm_head(y)


class MambaStackLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        motif: bool = False,
        pq_rank: int = 0,
        pq_k_init: float = 1e-4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if motif:
                block = Mambamotif(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_fast_path=False,
                    layer_idx=i,
                    pq_rank=int(pq_rank),
                    pq_per_dim=False,
                    pq_k_init=float(pq_k_init),
                )
            else:
                block = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_fast_path=False,
                    layer_idx=i,
                )
            self.layers.append(block)
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for ln, block in zip(self.norms, self.layers):
            x = x + block(ln(x))
        x = self.norm_f(x)
        return self.lm_head(x)


def build_model(
    arch: str,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    ffn_mult: float,
    d_state: int,
    d_conv: int,
    expand: int,
    pq_rank: int,
    pq_k_init: float,
) -> nn.Module:
    if arch == "transformer":
        return TransformerLM(vocab_size, d_model, n_layers, n_heads, ffn_mult)
    if arch == "rwkv":
        return RWKVLikeLM(vocab_size, d_model, n_layers)
    if arch == "lstm":
        return LSTMLM(vocab_size, d_model, n_layers)
    if arch == "mamba":
        return MambaStackLM(vocab_size, d_model, n_layers, d_state, d_conv, expand, motif=False)
    if arch == "motifmamba":
        return MambaStackLM(
            vocab_size,
            d_model,
            n_layers,
            d_state,
            d_conv,
            expand,
            motif=True,
            pq_rank=pq_rank,
            pq_k_init=pq_k_init,
        )
    raise ValueError(f"Unknown arch: {arch}")


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return contextlib.nullcontext()
    if amp == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported amp mode: {amp}")


def compute_induction_loss_acc(
    model: nn.Module,
    tokens: torch.Tensor,
    target_mask: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    amp: str,
) -> Tuple[torch.Tensor, float]:
    inp = tokens[:, :-1]
    tgt = tokens[:, 1:]
    m = target_mask[:, 1:]
    denom = m.sum().clamp_min(1)

    with autocast_context(device, amp):
        logits = model(inp)
        per_tok = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            tgt.reshape(-1),
            reduction="none",
        ).view_as(tgt)
        loss = (per_tok * m.float()).sum() / denom

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = (((pred == tgt) & m).float().sum() / denom).item()
    return loss, float(acc)


def is_oom_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "out of memory" in s or "cuda error: out of memory" in s


def auto_batch_size(
    model: nn.Module,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    amp: str,
    lr: float,
    weight_decay: float,
) -> int:
    candidates = [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 2, 1]
    for bs in candidates:
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            model.train()
            opt.zero_grad(set_to_none=True)
            x, m = make_induction_batch(bs, seq_len, vocab_size, device)
            loss, _ = compute_induction_loss_acc(model, x, m, vocab_size, device, amp)
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            return bs
        except RuntimeError as e:
            if is_oom_error(e):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError("Could not find a feasible batch size even at bs=1.")


@dataclass
class ModelConfig:
    arch: str
    d_model: int
    n_heads: int
    n_layers: int


def find_best_d_model_for_arch(
    arch: str,
    target_params: int,
    candidate_dims: List[int],
    args,
) -> Tuple[int, int, int]:
    best = None
    for d_model in candidate_dims:
        n_heads = pick_num_heads(d_model, args.max_heads)
        model = build_model(
            arch=arch,
            vocab_size=args.vocab_size,
            d_model=d_model,
            n_layers=args.n_layers,
            n_heads=n_heads,
            ffn_mult=args.ffn_mult,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            pq_rank=args.pq_rank,
            pq_k_init=args.pq_k_init,
        )
        p = count_params(model)
        diff = abs(p - target_params)
        cand = (diff, p, d_model, n_heads)
        if best is None or cand < best:
            best = cand
    _, p, d_model, n_heads = best
    return d_model, n_heads, p


def build_model_configs(args) -> Dict[str, ModelConfig]:
    archs = [x.strip() for x in args.models.split(",") if x.strip()]
    configs = {}

    base_heads = pick_num_heads(args.d_model, args.max_heads)
    # Use Mamba as target param anchor.
    mamba_probe = build_model(
        arch="mamba",
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=base_heads,
        ffn_mult=args.ffn_mult,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        pq_rank=args.pq_rank,
        pq_k_init=args.pq_k_init,
    )
    mamba_params = count_params(mamba_probe)

    candidate_dims = parse_int_csv(args.match_d_model_candidates)

    for arch in archs:
        if args.auto_match_params and arch in ("transformer", "rwkv", "lstm"):
            d_model, n_heads, _ = find_best_d_model_for_arch(
                arch=arch,
                target_params=mamba_params,
                candidate_dims=candidate_dims,
                args=args,
            )
        else:
            d_model = args.d_model
            n_heads = pick_num_heads(d_model, args.max_heads)
        configs[arch] = ModelConfig(
            arch=arch,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=args.n_layers,
        )
    return configs


def train_one_model(args, model: nn.Module, arch: str, batch_size: int, device: torch.device, train_log_path: str):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_fp16_scaler = device.type == "cuda" and args.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "loss",
                "acc",
                "ema_loss",
                "best_ema_loss",
                "no_improve_steps",
                "stability_span",
                "elapsed_sec",
            ]
        )

    t0 = time.time()
    ema_loss = None
    best_ema_loss = float("inf")
    last_improve_step = 0
    ema_hist = deque(maxlen=max(2, int(args.early_stop_stability_window)))
    stop_reason = "max_steps_reached"
    trained_steps = 0

    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x, m = make_induction_batch(batch_size, args.train_len, args.vocab_size, device)
        try:
            loss, acc = compute_induction_loss_acc(model, x, m, args.vocab_size, device, args.amp)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        except RuntimeError as e:
            if is_oom_error(e):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"[{arch}] OOM during training at step={step}, batch_size={batch_size}. "
                    f"Try smaller --batch-size or use --batch-size 0 (auto)."
                ) from e
            raise

        trained_steps = step
        cur_loss = float(loss.item())
        if ema_loss is None:
            ema_loss = cur_loss
        else:
            ema_loss = float(args.early_stop_ema_beta) * ema_loss + (1.0 - float(args.early_stop_ema_beta)) * cur_loss
        ema_hist.append(ema_loss)

        if ema_loss < (best_ema_loss - float(args.early_stop_min_delta)):
            best_ema_loss = ema_loss
            last_improve_step = step
        no_improve_steps = step - last_improve_step
        stability_span = (
            (max(ema_hist) - min(ema_hist))
            if len(ema_hist) >= max(2, int(args.early_stop_stability_window))
            else float("inf")
        )

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"[train][{arch}] step={step}/{args.train_steps}, loss={cur_loss:.6f}, "
                f"acc={acc:.4f}, ema={ema_loss:.6f}, no_improve={no_improve_steps}, "
                f"stable_span={stability_span if math.isfinite(stability_span) else -1.0:.6f}, elapsed={elapsed:.1f}s"
            )
            with open(train_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        step,
                        cur_loss,
                        float(acc),
                        float(ema_loss),
                        float(best_ema_loss),
                        int(no_improve_steps),
                        float(stability_span if math.isfinite(stability_span) else -1.0),
                        float(elapsed),
                    ]
                )

        # Early stop: plateau + stable EMA
        if (
            args.enable_early_stop
            and step >= int(args.early_stop_min_steps)
            and no_improve_steps >= int(args.early_stop_patience)
            and len(ema_hist) >= max(2, int(args.early_stop_stability_window))
            and stability_span <= float(args.early_stop_stability_tol)
        ):
            stop_reason = (
                f"early_stop(plateau={no_improve_steps},span={stability_span:.6e},"
                f"patience={args.early_stop_patience},tol={args.early_stop_stability_tol})"
            )
            print(f"[train][{arch}] stop: {stop_reason} at step={step}")
            break

    return {
        "trained_steps": int(trained_steps),
        "stop_reason": stop_reason,
        "best_ema_loss": float(best_ema_loss if math.isfinite(best_ema_loss) else float("inf")),
    }


def evaluate_lengths(
    args,
    model: nn.Module,
    arch: str,
    params: int,
    cfg: ModelConfig,
    test_lengths: List[int],
    device: torch.device,
) -> List[Dict]:
    model.eval()
    rows = []
    for L in test_lengths:
        if arch == "transformer" and L > args.max_transformer_eval_len:
            rows.append(
                {
                    "model": arch,
                    "params": params,
                    "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "test_len": L,
                    "eval_batches": 0,
                    "batch_size": 0,
                    "acc": "",
                    "status": f"SKIP_LEN>{args.max_transformer_eval_len}",
                    "seconds": 0.0,
                    "error": "",
                }
            )
            continue

        if args.eval_batch_size > 0:
            bs = args.eval_batch_size
        else:
            bs = max(1, int(args.eval_tokens_budget // max(1, L)))
        bs = min(bs, args.max_eval_batch_size)

        t0 = time.time()
        status = "ok"
        err_msg = ""
        total_acc = 0.0
        total_n = 0
        used_batches = 0
        try:
            for _ in range(args.eval_batches):
                x, m = make_induction_batch(bs, L, args.vocab_size, device)
                with torch.no_grad():
                    _, acc = compute_induction_loss_acc(model, x, m, args.vocab_size, device, args.amp)
                total_acc += acc * bs
                total_n += bs
                used_batches += 1
        except RuntimeError as e:
            if is_oom_error(e):
                status = "OOM"
                err_msg = str(e).replace("\n", " ")[:300]
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

        elapsed = time.time() - t0
        acc_val = ""
        if status == "ok" and total_n > 0:
            acc_val = total_acc / total_n
        rows.append(
            {
                "model": arch,
                "params": params,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "test_len": L,
                "eval_batches": used_batches,
                "batch_size": bs,
                "acc": acc_val,
                "status": status,
                "seconds": elapsed,
                "error": err_msg,
            }
        )
        acc_print = f"{acc_val:.6f}" if isinstance(acc_val, float) else "NA"
        print(f"[eval][{arch}] L={L}, acc={acc_print}, status={status}, bs={bs}, t={elapsed:.1f}s")
    return rows


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_accuracy_table(path: str, rows: List[Dict], test_lengths: List[int]):
    models = sorted(set(r["model"] for r in rows))
    by_model = {m: {} for m in models}
    params_by_model = {}
    for r in rows:
        by_model[r["model"]][int(r["test_len"])] = r
        params_by_model[r["model"]] = int(r["params"])

    headers = ["model", "params"] + [f"L{L}" for L in test_lengths]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for m in models:
            row = [m, params_by_model.get(m, "")]
            for L in test_lengths:
                r = by_model[m].get(L)
                if r is None:
                    row.append("")
                elif r["status"] != "ok":
                    row.append(r["status"])
                else:
                    row.append(f"{float(r['acc']):.6f}")
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train short (L=256), test long induction extrapolation benchmark for transformer/rwkv/lstm/mamba/motifmamba."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="/workspace/motif/logs/induction_extrapolation")

    parser.add_argument("--models", type=str, default="transformer,mamba,motifmamba")
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--train-len", type=int, default=256)
    parser.add_argument("--test-lens", type=str, default=default_test_lengths())

    parser.add_argument("--train-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=0, help="<=0 means auto-find by memory")
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=0, help="<=0 means auto from eval_tokens_budget")
    parser.add_argument("--eval-tokens-budget", type=int, default=131072, help="Used only when eval-batch-size<=0")
    parser.add_argument("--max-eval-batch-size", type=int, default=64)
    parser.add_argument("--max-transformer-eval-len", type=int, default=8192)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    parser.add_argument("--enable-early-stop", action="store_true", help="Enable early stop when training is stable")
    parser.add_argument("--disable-early-stop", action="store_true", help="Disable early stop")
    # Looser defaults: easier to early-stop once training plateaus.
    parser.add_argument("--early-stop-min-steps", type=int, default=2000)
    parser.add_argument("--early-stop-patience", type=int, default=1200)
    parser.add_argument("--early-stop-min-delta", type=float, default=5e-4)
    parser.add_argument("--early-stop-ema-beta", type=float, default=0.95)
    parser.add_argument("--early-stop-stability-window", type=int, default=400)
    parser.add_argument("--early-stop-stability-tol", type=float, default=2e-3)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=float, default=2.0)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)

    parser.add_argument("--pq-rank", type=int, default=2)
    parser.add_argument("--pq-k-init", type=float, default=1e-4)

    parser.add_argument("--auto-match-params", action="store_true", help="Auto-match transformer/rwkv/lstm params to mamba")
    parser.add_argument("--match-d-model-candidates", type=str, default="24,32,40,48,56,64,72,80,96,112,128,144,160")
    args = parser.parse_args()
    # default: enabled unless explicitly disabled
    if args.disable_early_stop:
        args.enable_early_stop = False
    elif not args.enable_early_stop:
        args.enable_early_stop = True
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        # Force CPU-safe path (causal_conv1d CUDA ext does not support CPU tensors).
        mamba_simple_mod.causal_conv1d_fn = None
        mamba_simplemotif_mod.causal_conv1d_fn = None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"induction_short{args.train_len}_steps{args.train_steps}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    test_lengths = parse_int_csv(args.test_lens)
    configs = build_model_configs(args)
    archs = [x.strip() for x in args.models.split(",") if x.strip()]
    pq_grad_mode_env = os.environ.get("MOTIFMAMBA_PQ_GRAD_MODE", "exact")
    if "motifmamba" in archs:
        print(f"[motifmamba] MOTIFMAMBA_PQ_GRAD_MODE={pq_grad_mode_env}")
    if device.type != "cuda":
        incompatible = [a for a in archs if a in ("mamba", "motifmamba")]
        if incompatible:
            raise RuntimeError(
                "Current selective_scan build is CUDA-only. "
                f"Please run on GPU for models: {','.join(incompatible)} (e.g. --device cuda:0)."
            )

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "resolved_device": str(device),
                "pq_grad_mode_env": pq_grad_mode_env,
                "test_lengths": test_lengths,
                "model_configs": {k: vars(v) for k, v in configs.items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_rows = []
    all_eval_rows = []

    for arch in archs:
        cfg = configs[arch]
        print("=" * 100)
        print(f"[model] {arch} | d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")
        model = build_model(
            arch=arch,
            vocab_size=args.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_mult=args.ffn_mult,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            pq_rank=args.pq_rank,
            pq_k_init=args.pq_k_init,
        ).to(device)
        params = count_params(model)
        print(f"[model] {arch} params={params}")

        if args.batch_size > 0:
            train_bs = int(args.batch_size)
        else:
            print(f"[model] auto finding train batch size for {arch} ...")
            train_bs = auto_batch_size(
                model=model,
                seq_len=args.train_len,
                vocab_size=args.vocab_size,
                device=device,
                amp=args.amp,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        print(f"[model] {arch} train_batch_size={train_bs}")

        train_log_path = os.path.join(out_dir, f"train_{arch}.csv")
        train_info = train_one_model(args, model, arch, train_bs, device, train_log_path)

        ckpt_path = os.path.join(out_dir, f"model_{arch}_latest.pt")
        torch.save(
            {
                "model": arch,
                "config": vars(cfg),
                "params": params,
                "train_info": train_info,
                "state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        print(f"[save] {ckpt_path}")

        eval_rows = evaluate_lengths(args, model, arch, params, cfg, test_lengths, device)
        all_eval_rows.extend(eval_rows)
        summary_rows.append(
            {
                "model": arch,
                "params": params,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "train_batch_size": train_bs,
                "trained_steps": int(train_info["trained_steps"]),
                "stop_reason": train_info["stop_reason"],
                "best_ema_loss": float(train_info["best_ema_loss"]),
                "checkpoint": ckpt_path,
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = os.path.join(out_dir, "model_summary.csv")
    write_csv(
        summary_path,
        summary_rows,
        [
            "model",
            "params",
            "d_model",
            "n_layers",
            "n_heads",
            "train_batch_size",
            "trained_steps",
            "stop_reason",
            "best_ema_loss",
            "checkpoint",
        ],
    )
    eval_path = os.path.join(out_dir, "accuracy_by_length.csv")
    write_csv(
        eval_path,
        all_eval_rows,
        [
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
        ],
    )
    table_path = os.path.join(out_dir, "accuracy_table.csv")
    write_accuracy_table(table_path, all_eval_rows, test_lengths)

    print("=" * 100)
    print(f"[done] out_dir={out_dir}")
    print(f"[done] summary={summary_path}")
    print(f"[done] by_length={eval_path}")
    print(f"[done] table={table_path}")


if __name__ == "__main__":
    main()
