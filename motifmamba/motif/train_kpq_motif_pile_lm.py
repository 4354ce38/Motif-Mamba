#!/usr/bin/env python3
import argparse
import csv
import json
import os
import math
import random
import re
import sys
import time
from glob import glob

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, ShardingStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Mamba checkpoint, warm up motif on k/P/Q, then train LM on Pile with only k/P/Q trainable."
    )
    parser.add_argument("--pretrained-dir", type=str, default="/workspace/mamba-1.4b")
    parser.add_argument("--pile-root", type=str, default="/workspace/datasets/pile_standard_pythia")
    parser.add_argument("--out-dir", type=str, default="/workspace/logs/pq_motif_pile/mamba1.4b_joint")
    parser.add_argument(
        "--init-adapter",
        type=str,
        default="",
        help="Optional path to a saved pq adapter checkpoint (pq_adapter_latest.pt) used to initialize k/P/Q before training.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dist-strategy",
        type=str,
        default="ddp",
        choices=["ddp", "fsdp"],
        help="Distributed strategy when WORLD_SIZE>1.",
    )
    parser.add_argument(
        "--fsdp-sharding-strategy",
        type=str,
        default="shard_grad_op",
        choices=["shard_grad_op", "full_shard"],
        help="FSDP sharding mode (shard_grad_op ~= ZeRO-2, full_shard ~= ZeRO-3).",
    )
    parser.add_argument(
        "--fsdp-offload-params",
        action="store_true",
        help="Enable FSDP CPU param offload (slower, lower GPU memory).",
    )

    parser.add_argument("--pq-rank", type=int, default=2)
    parser.add_argument("--pq-per-dim", dest="pq_per_dim", action="store_true", help="Use per-channel P/Q (shape [d_inner, d_state, r])")
    parser.add_argument("--no-pq-per-dim", dest="pq_per_dim", action="store_false", help="Use shared P/Q across channels")
    parser.set_defaults(pq_per_dim=True)
    parser.add_argument("--pq-k-init", type=float, default=1e-4)
    parser.add_argument("--train-pq-k", dest="train_pq_k", action="store_true", help="Train pq_k together with P/Q")
    parser.add_argument("--no-train-pq-k", dest="train_pq_k", action="store_false", help="Freeze pq_k and train only P/Q")
    parser.set_defaults(train_pq_k=True)
    parser.add_argument(
        "--train-only-layer",
        type=int,
        default=-1,
        help="If >=0, train k/P/Q only for this backbone layer index, freeze k/P/Q in other layers.",
    )
    parser.add_argument(
        "--train-base-model",
        dest="train_base_model",
        action="store_true",
        help="Train base Mamba parameters together with k/P/Q in joint stage",
    )
    parser.add_argument(
        "--no-train-base-model",
        dest="train_base_model",
        action="store_false",
        help="Freeze base Mamba parameters and train only k/P/Q",
    )
    parser.set_defaults(train_base_model=False)
    parser.add_argument(
        "--fulltrain-master-fp32",
        dest="fulltrain_master_fp32",
        action="store_true",
        help="When training base model, keep trainable weights in fp32 for stability",
    )
    parser.add_argument(
        "--no-fulltrain-master-fp32",
        dest="fulltrain_master_fp32",
        action="store_false",
        help="Disable fp32 master weights in full-model training",
    )
    parser.set_defaults(fulltrain_master_fp32=True)

    parser.add_argument("--motif-class", type=str, default="2", help="2 or motif_MOP etc.")
    parser.add_argument(
        "--motif-target-values",
        type=str,
        default="",
        help="Optional 13 comma-separated motif target values; overrides --motif-class/--motif-target when set.",
    )
    parser.add_argument("--motif-target", type=float, default=None)
    parser.add_argument("--motif-amplitude", type=float, default=1e5)
    parser.add_argument("--motif-bias", type=float, default=5e-5)
    parser.add_argument(
        "--motif-loss-form",
        type=str,
        default="mse",
        choices=["mse", "log_mse"],
        help="Loss form for active motif terms. log_mse helps tiny-prob motifs (e.g., motif13).",
    )
    parser.add_argument(
        "--motif-log-eps",
        type=float,
        default=1e-12,
        help="Epsilon used in log_mse: log(obs+eps)-log(target+eps).",
    )
    parser.add_argument(
        "--motif-loss-weights",
        type=str,
        default="",
        help="Optional 13 comma-separated nonnegative weights for motif losses.",
    )
    parser.add_argument("--motif-coef", type=float, default=0.1, help="Used in LM joint phase")
    parser.add_argument(
        "--motif-per-dim-mode",
        type=str,
        default="sampled_channel",
        choices=["sampled_channel", "mean_matrix"],
        help="How to compute motif loss for per-channel P/Q",
    )
    parser.add_argument(
        "--motif-loss-only-layer",
        type=int,
        default=-1,
        help="If >=0, compute motif loss only on this backbone layer index; ignore other layers.",
    )
    parser.add_argument(
        "--motif-channel-samples",
        type=int,
        default=16,
        help="Number of channels sampled per block when motif-per-dim-mode=sampled_channel (0 means all)",
    )

    parser.add_argument("--warmup-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-coef", type=float, default=1e6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Stop when motif <= init * ratio")
    parser.add_argument("--warmup-max-steps", type=int, default=20000)
    parser.add_argument("--warmup-log-every", type=int, default=20)
    parser.add_argument(
        "--warmup-no-early-stop",
        action="store_true",
        help="Do not stop warmup early when ratio target is reached; always run full warmup-max-steps.",
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="Run motif warmup stage only, then exit without joint LM training.",
    )
    parser.add_argument("--warmup-strict", dest="warmup_strict", action="store_true", help="Fail if ratio target is not reached")
    parser.add_argument("--no-warmup-strict", dest="warmup_strict", action="store_false", help="Allow joint phase even if warmup target is not reached")
    parser.set_defaults(warmup_strict=True)

    parser.add_argument("--train-lr", type=float, default=1e-4)
    parser.add_argument("--train-weight-decay", type=float, default=0.0)
    parser.add_argument("--train-steps", type=int, default=0, help="If <= 0, auto-compute from --train-epochs over all shards")
    parser.add_argument("--train-epochs", type=float, default=1.0, help="Used when --train-steps <= 0")
    parser.add_argument("--max-train-tokens", type=int, default=2_000_000_000, help="Stop when tokens_seen reaches this budget (0 disables)")
    parser.add_argument("--min-train-tokens", type=int, default=200_000_000, help="Do not early-stop before this many tokens")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Number of micro-steps per optimizer step")
    parser.add_argument("--eval-every-steps", type=int, default=100, help="Evaluate proxy PPL every N optimizer steps")
    parser.add_argument("--eval-tokens", type=int, default=2_000_000, help="Tokens used per proxy eval")
    parser.add_argument("--early-stop-patience", type=int, default=8, help="Stop after this many non-improving evals")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3, help="Minimum PPL drop to count as improvement")
    parser.add_argument("--disable-early-stop", action="store_true", help="Disable early stopping and only use max-train-tokens/steps")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--nan-backoff-retries",
        type=int,
        default=6,
        help="When lm/motif loss becomes non-finite in joint stage, retry this many times after shrinking k/P/Q.",
    )
    parser.add_argument(
        "--nan-backoff-scale-k",
        type=float,
        default=0.5,
        help="Multiply pq_k by this factor on each non-finite retry.",
    )
    parser.add_argument(
        "--nan-backoff-scale-pq",
        type=float,
        default=1.0,
        help="Optionally multiply P/Q by this factor on each non-finite retry (1.0 keeps P/Q unchanged).",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument(
        "--save-full-model",
        dest="save_full_model",
        action="store_true",
        help="Save full model checkpoint (base + k/P/Q) in addition to adapter",
    )
    parser.add_argument(
        "--no-save-full-model",
        dest="save_full_model",
        action="store_false",
        help="Do not save full model checkpoint",
    )
    parser.set_defaults(save_full_model=False)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed(requested_device: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = torch.device(requested_device)
    return device, rank, local_rank, world_size


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    if isinstance(model, (DDP, FSDP)):
        return model.module
    return model


def reshape_scalar_params_for_fsdp(module) -> int:
    # FSDP does not support 0-dim (scalar) parameters; convert them to shape [1].
    changed = 0
    for submodule in module.modules():
        for pname, p in list(submodule._parameters.items()):
            if p is None:
                continue
            if isinstance(p, torch.nn.Parameter) and p.ndim == 0:
                new_p = torch.nn.Parameter(p.detach().reshape(1), requires_grad=p.requires_grad)
                submodule._parameters[pname] = new_p
                changed += 1
    return changed


def summarize_param_dtypes(module):
    return sorted({str(p.dtype) for p in module.parameters()})


def ddp_reduce_mean_scalar(value, device: torch.device) -> float:
    if torch.is_tensor(value):
        t = value.detach().to(device=device, dtype=torch.float64)
    else:
        t = torch.tensor(float(value), device=device, dtype=torch.float64)
    if is_dist_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= float(dist.get_world_size())
    return float(t.item())


def ddp_reduce_mean_tensor(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.detach().clone()
    if is_dist_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        out /= float(dist.get_world_size())
    return out


class PileSequentialBatcher:
    def __init__(self, pile_root: str, stream_rank: int = 0, stream_world_size: int = 1):
        pattern = re.compile(r"document-\d{5}-of-\d{5}\.bin$")
        self.files = sorted([p for p in glob(os.path.join(pile_root, "*.bin")) if pattern.search(os.path.basename(p))])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No Pile .bin files found under: {pile_root}")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.files]
        self.lengths = [int(a.shape[0]) for a in self.arrays]
        self.shard_idx = 0
        self.offset = 0
        self.epoch = 0
        self.stream_rank = max(0, int(stream_rank))
        self.stream_world_size = max(1, int(stream_world_size))
        self._stream_need = None

    def _advance_shard(self):
        self.shard_idx += 1
        self.offset = 0
        if self.shard_idx >= len(self.arrays):
            self.shard_idx = 0
            self.epoch += 1

    def _init_stream_offset(self, need: int):
        if self._stream_need == need:
            return
        self._stream_need = need
        self.offset = need * self.stream_rank
        while True:
            n = self.lengths[self.shard_idx]
            if self.offset + need <= n:
                break
            self.offset -= n
            self._advance_shard()

    @property
    def total_tokens(self) -> int:
        return int(sum(self.lengths))

    def next_batch(self, batch_size: int, seq_len: int) -> torch.Tensor:
        need = seq_len + 1
        self._init_stream_offset(need)
        stride = need * self.stream_world_size
        out = torch.empty((batch_size, need), dtype=torch.long)
        for i in range(batch_size):
            while True:
                arr = self.arrays[self.shard_idx]
                n = self.lengths[self.shard_idx]
                if self.offset + need <= n:
                    chunk = np.asarray(arr[self.offset : self.offset + need], dtype=np.int64)
                    out[i].copy_(torch.from_numpy(chunk))
                    self.offset += stride
                    while self.offset + need > n:
                        self.offset -= n
                        self._advance_shard()
                    break
                self._advance_shard()
        return out


class PileEvalBatcher:
    """
    A fixed-position stream for stable proxy PPL evaluation.
    """

    def __init__(self, pile_root: str):
        pattern = re.compile(r"document-\d{5}-of-\d{5}\.bin$")
        files = sorted([p for p in glob(os.path.join(pile_root, "*.bin")) if pattern.search(os.path.basename(p))])
        if len(files) == 0:
            raise FileNotFoundError(f"No Pile .bin files found under: {pile_root}")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in files]
        self.lengths = [int(a.shape[0]) for a in self.arrays]
        self.file_idx = 0
        self.offset = 0

    def reset(self):
        self.file_idx = 0
        self.offset = 0

    def next_batch(self, batch_size: int, seq_len: int) -> torch.Tensor:
        need = seq_len + 1
        out = torch.empty((batch_size, need), dtype=torch.long)
        for i in range(batch_size):
            while True:
                arr = self.arrays[self.file_idx]
                n = self.lengths[self.file_idx]
                if self.offset + need <= n:
                    chunk = np.asarray(arr[self.offset : self.offset + need], dtype=np.int64)
                    out[i].copy_(torch.from_numpy(chunk))
                    self.offset += need
                    break
                self.file_idx += 1
                self.offset = 0
                if self.file_idx >= len(self.arrays):
                    self.file_idx = 0
        return out


class motifRegular:
    def __init__(self, fre, device="cuda", cc=16, loss_form: str = "mse", log_eps: float = 1e-12, loss_weights=None):
        self.device = device
        self.L = torch.ones([1, cc], device=self.device)
        self.P = torch.zeros([cc, cc], device=self.device)
        self.obs = torch.zeros([14], requires_grad=False, device=self.device)
        self.fre = torch.zeros([13], device=self.device)
        self.loss_form = str(loss_form)
        self.log_eps = float(log_eps)
        if loss_weights is None:
            self.loss_weights = torch.ones([13], device=self.device)
        else:
            if len(loss_weights) != 13:
                raise ValueError("loss_weights must have length 13")
            self.loss_weights = torch.tensor(loss_weights, device=self.device, dtype=torch.float32)
        self.sum = int(cc * (cc - 1) * (cc - 2) / 6)
        for i in range(13):
            self.fre[i] = fre[i]
        for i in range(cc):
            for j in range(cc):
                if i != j:
                    self.P[i][j] = 1

    def cal(self, a, amplitude, bias):
        a2 = a * a
        w = torch.sigmoid(amplitude * (a2 - bias))
        w = w * self.P
        pmw = self.P - w
        w0 = pmw * pmw.T
        w1 = w * pmw.T
        w2 = pmw * w.T
        w3 = w * w.T
        q = torch.zeros([14], device=self.device)
        q[1] = 0.5 * self.L @ (w1 * (w1 @ w0)) @ self.L.T
        q[2] = 0.5 * self.L @ (w0 * (w1 @ w2)) @ self.L.T
        q[3] = self.L @ (w1 * (w0 @ w2)) @ self.L.T
        q[4] = self.L @ (w1 * (w1 @ w2)) @ self.L.T
        q[5] = self.L @ (w3 * (w1 @ w0)) @ self.L.T
        q[6] = self.L @ (w3 * (w2 @ w0)) @ self.L.T
        q[7] = 0.5 * self.L @ (w3 * (w1 @ w2)) @ self.L.T
        q[8] = 0.5 * self.L @ (w3 * (w2 @ w1)) @ self.L.T
        q[9] = 0.5 * self.L @ (w3 * (w3 @ w0)) @ self.L.T
        q[10] = (1.0 / 3.0) * self.L @ (w1 * (w2 @ w2)) @ self.L.T
        q[11] = self.L @ (w3 * (w2 @ w2)) @ self.L.T
        q[12] = self.L @ (w3 * (w3 @ w2)) @ self.L.T
        q[13] = (1.0 / 6.0) * self.L @ (w3 * (w3 @ w3)) @ self.L.T
        r = torch.zeros([1], device=self.device)
        for i in range(13):
            if self.fre[i] > 0:
                obs_i = q[i + 1] / self.sum
                tgt_i = self.fre[i]
                if self.loss_form == "log_mse":
                    eps = float(self.log_eps)
                    diff = torch.log(torch.clamp(obs_i, min=eps)) - torch.log(torch.clamp(tgt_i, min=eps))
                else:
                    diff = obs_i - tgt_i
                r[0] += self.loss_weights[i] * (diff ** 2)
        with torch.no_grad():
            for i in range(13):
                self.obs[i + 1] = q[i + 1] / self.sum
        return r[0], self.obs


def parse_motif_class(motif_class: str):
    s = str(motif_class).strip()
    presets = {
        "MOP": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.130366, 0.249035, -1],
        "MOPE": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.130366, 0.349035, -1],
        "MOP_E": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.130366, 0.349035, -1],  # alias
        "FRP": [0.091003, 0.287659, 0.178217, 0.107868, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "FRPE": [0.09003, 0.387659, 0.178217, 0.107868, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "FRP_E": [0.09003, 0.387659, 0.178217, 0.107868, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # alias
        "MAVGF": [0.0455015, 0.1438295, 0.0891085, 0.053934, -1, -1, -1, -1, -1, -1, 0.065183, 0.1245175, 0.0],
        "2AVG12": [-1, 0.125, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.125, -1],
    }
    su = s.upper()
    if su.startswith("MOTIF_"):
        su = su[len("MOTIF_"):]
    if su in presets:
        return {"kind": "preset", "name": su, "target_values": presets[su]}
    if s.endswith("E") or s.endswith("e"):
        idx = int(s[:-1])
        if idx < 1 or idx > 13:
            raise ValueError("--motif-class with E suffix must be in 1E..13E")
        return {"kind": "single", "idx": idx, "target": 0.4}
    idx = int(s)
    if idx < 1 or idx > 13:
        raise ValueError("--motif-class must be 1..13 or motif preset")
    return {"kind": "single", "idx": idx, "target": 0.25}


def parse_motif_target_values(spec: str):
    s = str(spec).strip()
    if s == "":
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 13:
        raise ValueError("--motif-target-values must contain exactly 13 comma-separated numbers")
    vals = [float(x) for x in parts]
    return vals


def parse_motif_loss_weights(spec: str):
    s = str(spec).strip()
    if s == "":
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 13:
        raise ValueError("--motif-loss-weights must contain exactly 13 comma-separated numbers")
    vals = [float(x) for x in parts]
    for i, v in enumerate(vals, start=1):
        if v < 0:
            raise ValueError(f"--motif-loss-weights[{i}] must be >= 0")
    return vals


def build_model_with_pq(pretrained_dir, device, model_dtype, pq_rank, pq_per_dim, pq_k_init):
    from mamba_ssm.models.config_mamba import MambaConfig
    import mamba_ssm.models.mixer_seq_simple as mixer_seq_simple
    from mamba_ssm.modules.mamba_simplemotif import Mambamotif
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

    mixer_seq_simple.Mamba = Mambamotif
    cfg = load_config_hf(pretrained_dir)
    config = MambaConfig(**cfg)
    if device.type != "cuda":
        config.fused_add_norm = False
        config.rms_norm = False
    ssm_cfg = dict(config.ssm_cfg or {})
    ssm_cfg["pq_rank"] = int(pq_rank)
    ssm_cfg["pq_per_dim"] = bool(pq_per_dim)
    ssm_cfg["pq_k_init"] = float(pq_k_init)
    ssm_cfg["use_fast_path"] = False
    config.ssm_cfg = ssm_cfg

    model = mixer_seq_simple.MambaLMHeadModel(config, device=device, dtype=model_dtype)
    base_state = load_state_dict_hf(pretrained_dir, device="cpu", dtype=None)
    incompatible = model.load_state_dict(base_state, strict=False)
    return model, incompatible


def configure_trainable_params(model, train_pq_k: bool, train_base_model: bool, train_only_layer: int = -1):
    kpq = []
    train_named = []
    for name, p in model.named_parameters():
        is_P = name.endswith(".P")
        is_Q = name.endswith(".Q")
        is_k = name.endswith(".pq_k")
        is_kpq = is_P or is_Q or is_k

        if is_kpq:
            req = (not is_k) or train_pq_k
            if int(train_only_layer) >= 0:
                layer_tag = f".layers.{int(train_only_layer)}."
                req = req and (layer_tag in name)
        else:
            req = train_base_model
        p.requires_grad_(req)

        if is_kpq and req:
            kpq.append((name, p))
        if req:
            train_named.append((name, p))

    if len(kpq) == 0:
        raise RuntimeError("No trainable k/P/Q params found. Check pq_rank and train_pq_k.")
    if len(train_named) == 0:
        raise RuntimeError("No trainable params found. Check train flags.")
    return kpq, train_named


def apply_kpq_backoff(model, scale_k: float, scale_pq: float):
    with torch.no_grad():
        for module in model.modules():
            pq_k = getattr(module, "pq_k", None)
            P = getattr(module, "P", None)
            Q = getattr(module, "Q", None)
            if pq_k is not None and scale_k != 1.0:
                pq_k.mul_(float(scale_k))
            if P is not None and scale_pq != 1.0:
                P.mul_(float(scale_pq))
            if Q is not None and scale_pq != 1.0:
                Q.mul_(float(scale_pq))


def compute_model_motif_loss(
    model,
    motif_regularizer,
    amplitude,
    bias,
    per_dim_mode="sampled_channel",
    channel_samples=16,
    only_layer=-1,
):
    losses = []
    obs = []
    layer_filter = int(only_layer)
    for module_name, module in model.named_modules():
        if layer_filter >= 0:
            tag = f".layers.{layer_filter}.mixer"
            if tag not in module_name:
                continue
        P = getattr(module, "P", None)
        Q = getattr(module, "Q", None)
        if P is None or Q is None:
            continue
        pq_k = getattr(module, "pq_k", None)
        P_eff = P if pq_k is None else (pq_k * P)
        pq_square = torch.matmul(P_eff.float(), Q.float())
        if pq_square.dim() == 3:
            if per_dim_mode == "mean_matrix":
                m_loss, m_obs = motif_regularizer.cal(pq_square.mean(dim=0), amplitude=amplitude, bias=bias)
                losses.append(m_loss)
                obs.append(m_obs)
            else:
                d_inner = int(pq_square.shape[0])
                n_sample = d_inner if channel_samples <= 0 else min(int(channel_samples), d_inner)
                if n_sample == d_inner:
                    channel_idx = torch.arange(d_inner, device=pq_square.device)
                else:
                    channel_idx = torch.randperm(d_inner, device=pq_square.device)[:n_sample]
                blk_losses = []
                blk_obs = []
                for c in channel_idx:
                    c_loss, c_obs = motif_regularizer.cal(
                        pq_square[int(c.item())], amplitude=amplitude, bias=bias
                    )
                    blk_losses.append(c_loss)
                    blk_obs.append(c_obs)
                losses.append(torch.stack(blk_losses).mean())
                obs.append(torch.stack(blk_obs).mean(dim=0))
        else:
            m_loss, m_obs = motif_regularizer.cal(pq_square, amplitude=amplitude, bias=bias)
            losses.append(m_loss)
            obs.append(m_obs)
    if len(losses) == 0:
        raise RuntimeError("No motif-compatible blocks (with P/Q) found.")
    return torch.stack(losses).mean(), torch.stack(obs).mean(dim=0)


def compute_lm_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def evaluate_proxy_ppl(model, eval_batcher, device, amp_dtype, batch_size, seq_len, eval_tokens):
    eval_batcher.reset()
    tokens_per_batch = int(batch_size * seq_len)
    n_batches = max(1, int(np.ceil(float(eval_tokens) / float(tokens_per_batch))))
    loss_sum = 0.0
    count = 0
    was_training = model.training
    model.eval()
    for _ in range(n_batches):
        batch = eval_batcher.next_batch(batch_size, seq_len).to(device, non_blocking=True)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        if device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=(amp_dtype != torch.float32), dtype=amp_dtype):
                logits = model(inp).logits
                loss = compute_lm_loss(logits, tgt)
        else:
            logits = model(inp).logits
            loss = compute_lm_loss(logits, tgt)
        loss_sum += float(loss.item())
        count += 1
    if was_training:
        model.train()
    mean_loss = loss_sum / max(1, count)
    return float(math.exp(min(50.0, mean_loss))), mean_loss


def format_seconds(total_sec: float) -> str:
    total_sec = max(0, int(total_sec))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def save_adapter(path, model, extra=None):
    state = {}
    for name, p in model.named_parameters():
        if name.endswith(".P") or name.endswith(".Q") or name.endswith(".pq_k"):
            state[name] = p.detach().cpu()
    payload = {"pq_state_dict": state}
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def save_full_checkpoint(path, model, optimizer=None, scaler=None, extra=None):
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def save_adapter_distributed(path, model, extra=None, use_fsdp=False, fsdp_full_shard=False):
    if use_fsdp and fsdp_full_shard:
        with FSDP.summon_full_params(model, recurse=True, writeback=False):
            save_adapter(path, unwrap_model(model), extra=extra)
        return
    save_adapter(path, unwrap_model(model), extra=extra)


def load_adapter_init(model, init_adapter_path: str, is_main: bool = True):
    if not init_adapter_path:
        return
    if not os.path.isfile(init_adapter_path):
        raise FileNotFoundError(f"--init-adapter not found: {init_adapter_path}")

    payload = torch.load(init_adapter_path, map_location="cpu")
    if isinstance(payload, dict) and "pq_state_dict" in payload:
        state = payload["pq_state_dict"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise ValueError(f"--init-adapter must be a dict-like checkpoint: {init_adapter_path}")

    if not isinstance(state, dict):
        raise ValueError(f"Invalid adapter payload in {init_adapter_path}: missing state_dict")

    tensor_state = {k: v for k, v in state.items() if torch.is_tensor(v)}
    if len(tensor_state) == 0:
        raise ValueError(f"No tensor state entries found in {init_adapter_path}")

    model_keys = set(model.state_dict().keys())
    loadable = {k: v for k, v in tensor_state.items() if k in model_keys}
    skipped = [k for k in tensor_state.keys() if k not in model_keys]
    if len(loadable) == 0:
        raise RuntimeError(
            f"No adapter keys matched current model when loading {init_adapter_path}. "
            f"Check pq_rank / pq_per_dim / model size."
        )

    incompatible = model.load_state_dict(loadable, strict=False)
    if is_main:
        print(
            f"[init-adapter] loaded_tensors={len(loadable)}, skipped_unknown={len(skipped)}, "
            f"missing_keys_reported={len(incompatible.missing_keys)}"
        )
        if len(skipped) > 0:
            preview = ", ".join(skipped[:5])
            print(f"[init-adapter] first_skipped_keys={preview}")


def main():
    args = parse_args()
    this_file = os.path.abspath(__file__)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(this_file), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    device, rank, local_rank, world_size = init_distributed(args.device)
    is_main = rank == 0
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        csv_path = os.path.join(args.out_dir, "train_log.csv")
        adapter_latest = os.path.join(args.out_dir, "pq_adapter_latest.pt")
        adapter_best = os.path.join(args.out_dir, "pq_adapter_best.pt")
        full_latest = os.path.join(args.out_dir, "full_model_latest.pt")
        full_best = os.path.join(args.out_dir, "full_model_best.pt")

        set_seed(args.seed + rank)
        model_dtype = resolve_dtype(args.dtype)
        if model_dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            if is_main:
                print("[warn] bf16 not supported; fallback to float16")
            model_dtype = torch.float16

        if is_main:
            print(f"Device: {device}, model_dtype: {model_dtype}, world_size={world_size}, rank={rank}")
            print("Loading model...")
        model, incompatible = build_model_with_pq(
            pretrained_dir=args.pretrained_dir,
            device=device,
            model_dtype=model_dtype,
            pq_rank=args.pq_rank,
            pq_per_dim=args.pq_per_dim,
            pq_k_init=args.pq_k_init,
        )
        if is_main:
            print(
                f"Loaded. missing_keys={len(incompatible.missing_keys)}, unexpected_keys={len(incompatible.unexpected_keys)}"
            )
            print(
                f"PQ config: rank={args.pq_rank}, per_dim={args.pq_per_dim}, "
                f"train_pq_k={args.train_pq_k}, train_base_model={args.train_base_model}, "
                f"train_only_layer={args.train_only_layer}"
            )
        if args.train_base_model and args.fulltrain_master_fp32:
            model.to(device=device, dtype=torch.float32)
            if is_main:
                print("[stability] train_base_model enabled: using fp32 master weights for trainable params")
        if args.init_adapter:
            load_adapter_init(model, args.init_adapter, is_main=is_main)
        if is_main:
            print(
                f"Motif per-dim mode: {args.motif_per_dim_mode}, "
                f"channel_samples={args.motif_channel_samples}"
            )

        kpq, train_named = configure_trainable_params(
            model,
            train_pq_k=args.train_pq_k,
            train_base_model=args.train_base_model,
            train_only_layer=args.train_only_layer,
        )
        kpq_params = [p for _, p in kpq]
        train_params = [p for _, p in train_named]
        if is_main:
            print(f"Trainable k/P/Q params: {sum(p.numel() for _, p in kpq)}")
            print(f"Trainable total params: {sum(p.numel() for _, p in train_named)}")

        use_fsdp = (world_size > 1 and args.dist_strategy == "fsdp")
        fsdp_full_shard = False
        if world_size > 1:
            if use_fsdp:
                scalar_fixed = reshape_scalar_params_for_fsdp(model)
                if is_main and scalar_fixed > 0:
                    print(f"[dist] fsdp scalar-compat: reshaped {scalar_fixed} scalar params to 1D")
                fsdp_param_dtype = torch.float32 if (args.train_base_model and args.fulltrain_master_fp32) else model_dtype
                model = model.to(device=device, dtype=fsdp_param_dtype)
                if is_main:
                    print(f"[dist] fsdp param dtype cast -> {fsdp_param_dtype}")
                    print(f"[dist] fsdp param dtypes after cast: {summarize_param_dtypes(model)}")
                fsdp_sharding = (
                    ShardingStrategy.FULL_SHARD
                    if args.fsdp_sharding_strategy == "full_shard"
                    else ShardingStrategy.SHARD_GRAD_OP
                )
                fsdp_full_shard = fsdp_sharding == ShardingStrategy.FULL_SHARD
                fsdp_kwargs = {
                    "sharding_strategy": fsdp_sharding,
                    "cpu_offload": CPUOffload(offload_params=bool(args.fsdp_offload_params)),
                    "use_orig_params": True,
                }
                if device.type == "cuda":
                    fsdp_kwargs["device_id"] = local_rank
                model = FSDP(model, **fsdp_kwargs)
                # Read trainable params from wrapped model for optimizer compatibility.
                kpq = []
                train_named = []
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if name.endswith(".P") or name.endswith(".Q") or name.endswith(".pq_k"):
                        kpq.append((name, p))
                    train_named.append((name, p))
                kpq_params = [p for _, p in kpq]
                train_params = [p for _, p in train_named]
                if is_main:
                    print(
                        f"[dist] strategy=fsdp, sharding={args.fsdp_sharding_strategy}, "
                        f"cpu_offload={args.fsdp_offload_params}"
                    )
                    if args.save_full_model:
                        print("[warn] --save-full-model is disabled under FSDP in this script.")
                        args.save_full_model = False
            else:
                if device.type == "cuda":
                    model = DDP(
                        model,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        broadcast_buffers=False,
                        find_unused_parameters=False,
                    )
                else:
                    model = DDP(
                        model,
                        device_ids=None,
                        output_device=None,
                        broadcast_buffers=False,
                        find_unused_parameters=False,
                    )
                if is_main:
                    print("[dist] strategy=ddp")
        model_for_save = unwrap_model(model)

        target_vals = parse_motif_target_values(args.motif_target_values)
        if target_vals is not None:
            if args.motif_target is not None and is_main:
                print("[warn] --motif-target ignored when --motif-target-values is set")
            target_freq = torch.tensor(target_vals, device=device, dtype=torch.float32)
            motif_desc = "custom_13_values"
        else:
            motif_cfg = parse_motif_class(args.motif_class)
            if motif_cfg["kind"] == "single":
                target_val = motif_cfg["target"] if args.motif_target is None else float(args.motif_target)
                target_freq = -torch.ones(13, device=device)
                target_freq[motif_cfg["idx"] - 1] = target_val
                motif_desc = f"class={args.motif_class}, target={target_val}"
            else:
                if args.motif_target is not None and is_main:
                    print("[warn] --motif-target ignored in preset mode")
                target_freq = torch.tensor(motif_cfg["target_values"], device=device, dtype=torch.float32)
                motif_desc = f"preset={motif_cfg['name']}"
        motif_loss_weights = parse_motif_loss_weights(args.motif_loss_weights)
        motif_regularizer = motifRegular(
            target_freq,
            device=device,
            cc=16,
            loss_form=args.motif_loss_form,
            log_eps=args.motif_log_eps,
            loss_weights=motif_loss_weights,
        )
        if is_main:
            print(f"Motif setting: {motif_desc}")
            if motif_loss_weights is not None:
                print(f"Motif loss weights: {motif_loss_weights}")

        # Warmup stage: motif only
        warmup_opt = AdamW(kpq_params, lr=args.warmup_lr, weight_decay=args.warmup_weight_decay)
        with torch.no_grad():
            init_motif, _ = compute_model_motif_loss(
                model,
                motif_regularizer,
                args.motif_amplitude,
                args.motif_bias,
                per_dim_mode=args.motif_per_dim_mode,
                channel_samples=args.motif_channel_samples,
                only_layer=args.motif_loss_only_layer,
            )
        init_motif_val = ddp_reduce_mean_scalar(init_motif, device)
        target_motif = init_motif_val * float(args.warmup_ratio)
        if is_main:
            print(
                f"[warmup] init_motif={init_motif_val:.6e}, target={target_motif:.6e}, max_steps={args.warmup_max_steps}"
            )

        header = [
            "global_step",
            "phase",
            "phase_step",
            "lm_loss",
            "ppl",
            "eval_lm_loss",
            "eval_ppl",
            "best_eval_ppl",
            "motif_loss",
            "total_loss",
            "lr",
            "tokens_seen",
            "elapsed_sec",
        ] + [f"motif_{i}" for i in range(1, 14)]
        if is_main:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()

        start_t = time.time()
        warmup_t0 = time.time()
        global_step = 0
        warmup_reached = False
        best_warmup_motif = float("inf")
        last_warmup_motif = float("nan")
        last_warmup_step = 0
        best_ppl = float("inf")
        best_eval_ppl = float("inf")
        bad_eval_count = 0
        tokens_seen = 0
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and model_dtype == torch.float16))

        for wstep in range(1, args.warmup_max_steps + 1):
            warmup_opt.zero_grad(set_to_none=True)
            motif_loss = None
            motif_obs = None
            loss = None
            for retry_idx in range(max(0, int(args.nan_backoff_retries)) + 1):
                motif_loss, motif_obs = compute_model_motif_loss(
                    model,
                    motif_regularizer,
                    args.motif_amplitude,
                    args.motif_bias,
                    per_dim_mode=args.motif_per_dim_mode,
                    channel_samples=args.motif_channel_samples,
                    only_layer=args.motif_loss_only_layer,
                )
                loss = float(args.warmup_coef) * motif_loss
                if torch.isfinite(loss):
                    break
                if retry_idx >= int(args.nan_backoff_retries):
                    raise RuntimeError(
                        f"Warmup loss became non-finite at step {wstep} after {retry_idx} retries. "
                        f"Try lower --warmup-coef / --warmup-lr / --motif-amplitude."
                    )
                apply_kpq_backoff(
                    model,
                    scale_k=float(args.nan_backoff_scale_k),
                    scale_pq=float(args.nan_backoff_scale_pq),
                )
                if is_main:
                    print(
                        f"[nan-backoff][warmup] step={wstep}, retry={retry_idx + 1}, "
                        f"scale_k={args.nan_backoff_scale_k}, scale_pq={args.nan_backoff_scale_pq}"
                    )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(kpq_params, float(args.grad_clip))
            warmup_opt.step()

            global_step += 1
            motif_loss_v = ddp_reduce_mean_scalar(motif_loss, device)
            best_warmup_motif = min(best_warmup_motif, motif_loss_v)
            last_warmup_motif = motif_loss_v
            last_warmup_step = wstep
            total_loss_v = ddp_reduce_mean_scalar(loss, device)
            motif_obs_mean = ddp_reduce_mean_tensor(motif_obs.float())

            if is_main:
                row = {
                    "global_step": global_step,
                    "phase": "warmup",
                    "phase_step": wstep,
                    "lm_loss": 0.0,
                    "ppl": 0.0,
                    "eval_lm_loss": 0.0,
                    "eval_ppl": 0.0,
                    "best_eval_ppl": 0.0,
                    "motif_loss": motif_loss_v,
                    "total_loss": total_loss_v,
                    "lr": float(warmup_opt.param_groups[0]["lr"]),
                    "tokens_seen": tokens_seen,
                    "elapsed_sec": float(time.time() - start_t),
                }
                for i in range(1, 14):
                    row[f"motif_{i}"] = float(motif_obs_mean[i].item())
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writerow(row)

                if wstep % args.warmup_log_every == 0 or wstep == 1:
                    pct = 100.0 * float(wstep) / float(max(1, args.warmup_max_steps))
                    elapsed = time.time() - warmup_t0
                    sec_per_step = elapsed / float(max(1, wstep))
                    eta_sec = sec_per_step * float(max(0, args.warmup_max_steps - wstep))
                    print(
                        f"[warmup] step={wstep}/{args.warmup_max_steps} ({pct:.1f}%), "
                        f"eta={format_seconds(eta_sec)}, motif={motif_loss_v:.6e}, target={target_motif:.6e}"
                    )

            if (not args.warmup_no_early_stop) and motif_loss_v <= target_motif:
                warmup_reached = True
                if is_main:
                    print(f"[warmup] target reached at step={wstep}, motif={motif_loss_v:.6e}")
                break

        if args.warmup_no_early_stop and (best_warmup_motif <= target_motif):
            warmup_reached = True

        if is_main:
            warmup_summary = {
                "init_motif": float(init_motif_val),
                "target_motif": float(target_motif),
                "warmup_ratio": float(args.warmup_ratio),
                "reached": bool(warmup_reached),
                "best_motif": float(best_warmup_motif),
                "last_motif": float(last_warmup_motif),
                "steps_run": int(last_warmup_step),
                "max_steps": int(args.warmup_max_steps),
                "pq_rank": int(args.pq_rank),
                "motif_desc": motif_desc,
            }
            warmup_summary_path = os.path.join(args.out_dir, "warmup_summary.json")
            with open(warmup_summary_path, "w", encoding="utf-8") as f:
                json.dump(warmup_summary, f, ensure_ascii=False, indent=2)
            print(f"[warmup] summary={warmup_summary_path}")

        if (not warmup_reached) and args.warmup_strict:
            raise RuntimeError("Warmup target not reached and --warmup-strict is set.")

        if args.warmup_only:
            if is_main:
                final_extra = {
                    "global_step": global_step,
                    "phase_step": int(last_warmup_step),
                    "best_ppl": 0.0,
                    "best_eval_ppl": 0.0,
                    "train_steps": 0,
                    "stop_reason": "warmup_only",
                    "world_size": world_size,
                    "args": vars(args),
                }
                save_adapter_distributed(
                    adapter_latest,
                    model,
                    extra=final_extra,
                    use_fsdp=use_fsdp,
                    fsdp_full_shard=fsdp_full_shard,
                )
                print("[done] warmup-only finished")
                print(f"[done] csv={csv_path}")
                print(f"[done] adapter_latest={adapter_latest}")
            return

        # Joint stage: Pile LM + motif
        if is_main:
            print("[joint] starting Pile LM training...")
        batcher = PileSequentialBatcher(args.pile_root, stream_rank=rank, stream_world_size=world_size)
        eval_batcher = PileEvalBatcher(args.pile_root) if (is_main or use_fsdp) else None
        train_opt = AdamW(train_params, lr=args.train_lr, weight_decay=args.train_weight_decay)
        amp_dtype = model_dtype if model_dtype in (torch.float16, torch.bfloat16) else torch.float32
        model.train()

        if args.grad_accum_steps < 1:
            raise ValueError("--grad-accum-steps must be >= 1")
        tokens_per_step = int(args.batch_size * (args.seq_len + 1) * args.grad_accum_steps * world_size)
        if args.train_steps > 0:
            train_steps = int(args.train_steps)
        else:
            steps_per_epoch = batcher.total_tokens // tokens_per_step
            if steps_per_epoch <= 0:
                raise RuntimeError(
                    f"Pile tokens ({batcher.total_tokens}) too small for batch_size={args.batch_size}, seq_len={args.seq_len}"
                )
            train_steps = max(1, int(np.ceil(float(args.train_epochs) * float(steps_per_epoch))))
            if is_main:
                print(
                    f"[joint] auto train_steps={train_steps} from total_tokens={batcher.total_tokens}, "
                    f"tokens_per_step={tokens_per_step}, train_epochs={args.train_epochs}"
                )
        if is_main:
            print(
                f"[joint] max_train_tokens={args.max_train_tokens}, min_train_tokens={args.min_train_tokens}, "
                f"eval_every_steps={args.eval_every_steps}, eval_tokens={args.eval_tokens}, "
                f"early_stop={'off' if args.disable_early_stop else 'on'}"
            )
        tokens_per_opt_step = int(args.batch_size * args.seq_len * args.grad_accum_steps * world_size)
        if args.max_train_tokens > 0:
            max_steps_by_tokens = int(np.ceil(float(args.max_train_tokens) / float(max(1, tokens_per_opt_step))))
            progress_total_steps = max(1, min(train_steps, max_steps_by_tokens))
        else:
            progress_total_steps = max(1, train_steps)
        if is_main:
            print(
                f"[joint] progress_total_steps={progress_total_steps}, "
                f"tokens_per_opt_step={tokens_per_opt_step}"
            )
        stop_reason = "train_steps_reached"
        step_time_ema = None

        for step in range(1, train_steps + 1):
            step_t0 = time.time()
            train_opt.zero_grad(set_to_none=True)
            lm_loss_sum = 0.0
            for micro_idx in range(args.grad_accum_steps):
                batch = batcher.next_batch(args.batch_size, args.seq_len).to(device, non_blocking=True)
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                lm_loss = None
                for retry_idx in range(max(0, int(args.nan_backoff_retries)) + 1):
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda", enabled=(amp_dtype != torch.float32), dtype=amp_dtype):
                            logits = model(inp).logits
                            lm_loss = compute_lm_loss(logits, tgt)
                    else:
                        logits = model(inp).logits
                        lm_loss = compute_lm_loss(logits, tgt)
                    if torch.isfinite(lm_loss):
                        break
                    if retry_idx >= int(args.nan_backoff_retries):
                        raise RuntimeError(
                            f"LM loss non-finite at joint step {step}, micro {micro_idx + 1}/{args.grad_accum_steps} "
                            f"after {retry_idx} retries. Try lower --warmup-coef / --train-lr or smaller --pq-k-init."
                        )
                    apply_kpq_backoff(
                        model,
                        scale_k=float(args.nan_backoff_scale_k),
                        scale_pq=float(args.nan_backoff_scale_pq),
                    )
                    if is_main:
                        print(
                            f"[nan-backoff] lm step={step} micro={micro_idx + 1}, retry={retry_idx + 1}, "
                            f"scale_k={args.nan_backoff_scale_k}, scale_pq={args.nan_backoff_scale_pq}"
                        )
                lm_loss_sum += float(lm_loss.item())
                lm_loss_scaled = lm_loss / float(args.grad_accum_steps)
                if scaler.is_enabled():
                    scaler.scale(lm_loss_scaled).backward()
                else:
                    lm_loss_scaled.backward()
                tokens_seen += int(args.batch_size * args.seq_len * world_size)

            motif_loss = None
            motif_obs = None
            for retry_idx in range(max(0, int(args.nan_backoff_retries)) + 1):
                motif_loss, motif_obs = compute_model_motif_loss(
                    model,
                    motif_regularizer,
                    args.motif_amplitude,
                    args.motif_bias,
                    per_dim_mode=args.motif_per_dim_mode,
                    channel_samples=args.motif_channel_samples,
                    only_layer=args.motif_loss_only_layer,
                )
                if torch.isfinite(motif_loss):
                    break
                if retry_idx >= int(args.nan_backoff_retries):
                    raise RuntimeError(
                        f"Motif loss non-finite at joint step {step} after {retry_idx} retries."
                    )
                apply_kpq_backoff(
                    model,
                    scale_k=float(args.nan_backoff_scale_k),
                    scale_pq=float(args.nan_backoff_scale_pq),
                )
                if is_main:
                    print(
                        f"[nan-backoff] motif step={step}, retry={retry_idx + 1}, "
                        f"scale_k={args.nan_backoff_scale_k}, scale_pq={args.nan_backoff_scale_pq}"
                    )
            motif_term = float(args.motif_coef) * motif_loss
            total_loss = (lm_loss_sum / float(args.grad_accum_steps)) + float(motif_term.item())
            if scaler.is_enabled():
                scaler.scale(motif_term).backward()
            else:
                motif_term.backward()
            if not math.isfinite(total_loss):
                raise RuntimeError(
                    f"Joint loss became non-finite at step {step}; "
                    f"lm_loss={lm_loss_sum / float(args.grad_accum_steps):.6e}, "
                    f"motif_loss={float(motif_loss.item()):.6e}, "
                    f"motif_term={float(motif_term.item()):.6e}"
                )

            if scaler.is_enabled():
                if args.grad_clip > 0:
                    scaler.unscale_(train_opt)
                    torch.nn.utils.clip_grad_norm_(train_params, float(args.grad_clip))
                scaler.step(train_opt)
                scaler.update()
            else:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(train_params, float(args.grad_clip))
                train_opt.step()
            step_dt = time.time() - step_t0
            if step_time_ema is None:
                step_time_ema = step_dt
            else:
                step_time_ema = 0.9 * step_time_ema + 0.1 * step_dt

            lm_loss_v = ddp_reduce_mean_scalar(lm_loss_sum / float(args.grad_accum_steps), device)
            motif_loss_v = ddp_reduce_mean_scalar(motif_loss, device)
            total_loss_v = ddp_reduce_mean_scalar(total_loss, device)
            motif_obs_mean = ddp_reduce_mean_tensor(motif_obs.float())

            ppl = float(math.exp(min(50.0, lm_loss_v)))
            is_best = ppl < best_ppl
            if is_best:
                best_ppl = ppl
            global_step += 1

            eval_lm_loss = 0.0
            eval_ppl = 0.0
            is_best_eval = False
            if args.eval_every_steps > 0 and (step % args.eval_every_steps == 0 or step == 1):
                if use_fsdp:
                    eval_ppl, eval_lm_loss = evaluate_proxy_ppl(
                        model=model,
                        eval_batcher=eval_batcher,
                        device=device,
                        amp_dtype=amp_dtype,
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        eval_tokens=args.eval_tokens,
                    )
                    eval_lm_loss = ddp_reduce_mean_scalar(eval_lm_loss, device)
                    eval_ppl = ddp_reduce_mean_scalar(eval_ppl, device)
                    if is_main:
                        if eval_ppl < (best_eval_ppl - float(args.early_stop_min_delta)):
                            best_eval_ppl = eval_ppl
                            bad_eval_count = 0
                            is_best_eval = True
                        else:
                            bad_eval_count += 1
                elif is_main:
                    eval_ppl, eval_lm_loss = evaluate_proxy_ppl(
                        model=model,
                        eval_batcher=eval_batcher,
                        device=device,
                        amp_dtype=amp_dtype,
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        eval_tokens=args.eval_tokens,
                    )
                    if eval_ppl < (best_eval_ppl - float(args.early_stop_min_delta)):
                        best_eval_ppl = eval_ppl
                        bad_eval_count = 0
                        is_best_eval = True
                    else:
                        bad_eval_count += 1
                stats = torch.tensor(
                    [
                        float(eval_lm_loss),
                        float(eval_ppl),
                        float(best_eval_ppl if math.isfinite(best_eval_ppl) else 0.0),
                        float(bad_eval_count),
                        float(1 if is_best_eval else 0),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                if is_dist_initialized():
                    dist.broadcast(stats, src=0)
                eval_lm_loss = float(stats[0].item())
                eval_ppl = float(stats[1].item())
                best_eval_ppl = float(stats[2].item())
                bad_eval_count = int(stats[3].item())
                is_best_eval = bool(int(stats[4].item()))

            if is_main:
                row = {
                    "global_step": global_step,
                    "phase": "joint",
                    "phase_step": step,
                    "lm_loss": lm_loss_v,
                    "ppl": ppl,
                    "eval_lm_loss": float(eval_lm_loss),
                    "eval_ppl": float(eval_ppl),
                    "best_eval_ppl": float(best_eval_ppl if math.isfinite(best_eval_ppl) else 0.0),
                    "motif_loss": motif_loss_v,
                    "total_loss": total_loss_v,
                    "lr": float(train_opt.param_groups[0]["lr"]),
                    "tokens_seen": tokens_seen,
                    "elapsed_sec": float(time.time() - start_t),
                }
                for i in range(1, 14):
                    row[f"motif_{i}"] = float(motif_obs_mean[i].item())

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writerow(row)

                if step % args.log_every == 0 or step == 1:
                    progress_now = min(step, progress_total_steps)
                    pct = 100.0 * float(progress_now) / float(progress_total_steps)
                    eta_sec = (progress_total_steps - progress_now) * (step_time_ema if step_time_ema is not None else 0.0)
                    token_msg = (
                        f"{tokens_seen}/{args.max_train_tokens}"
                        if args.max_train_tokens > 0
                        else f"{tokens_seen}"
                    )
                    print(
                        f"[joint] step={step}/{train_steps} (~{pct:.1f}%), eta={format_seconds(eta_sec)}, "
                        f"tokens={token_msg}, lm_loss={lm_loss_v:.6f}, ppl={ppl:.4f}, "
                        f"motif={motif_loss_v:.6e}, total={total_loss_v:.6f}, epoch_pass={batcher.epoch}"
                    )
                    if eval_ppl > 0:
                        print(
                            f"[joint][eval] eval_lm_loss={eval_lm_loss:.6f}, eval_ppl={eval_ppl:.4f}, "
                            f"best_eval_ppl={best_eval_ppl:.4f}, bad_eval_count={bad_eval_count}"
                        )

            if is_main and (step % args.save_every == 0 or step == train_steps):
                extra = {
                    "global_step": global_step,
                    "phase_step": step,
                    "best_ppl": best_ppl,
                    "best_eval_ppl": best_eval_ppl,
                    "train_steps": train_steps,
                    "world_size": world_size,
                    "args": vars(args),
                }
                save_adapter_distributed(
                    adapter_latest,
                    model,
                    extra=extra,
                    use_fsdp=use_fsdp,
                    fsdp_full_shard=fsdp_full_shard,
                )
                if args.save_full_model:
                    save_full_checkpoint(full_latest, model_for_save, optimizer=train_opt, scaler=scaler, extra=extra)
                if is_best_eval or (not math.isfinite(best_eval_ppl) and is_best):
                    save_adapter_distributed(
                        adapter_best,
                        model,
                        extra=extra,
                        use_fsdp=use_fsdp,
                        fsdp_full_shard=fsdp_full_shard,
                    )
                    if args.save_full_model:
                        save_full_checkpoint(full_best, model_for_save, optimizer=train_opt, scaler=scaler, extra=extra)

            should_stop = False
            if args.max_train_tokens > 0 and tokens_seen >= int(args.max_train_tokens):
                stop_reason = f"max_train_tokens_reached({args.max_train_tokens})"
                should_stop = True
            if (
                (not args.disable_early_stop)
                and args.eval_every_steps > 0
                and tokens_seen >= int(args.min_train_tokens)
                and bad_eval_count >= int(args.early_stop_patience)
            ):
                stop_reason = (
                    f"early_stop(patience={args.early_stop_patience}, "
                    f"min_tokens={args.min_train_tokens}, bad_eval_count={bad_eval_count})"
                )
                should_stop = True

            if is_dist_initialized():
                stop_flag = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int32)
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                should_stop = bool(int(stop_flag.item()))
            if should_stop:
                if is_main:
                    print(f"[joint] stop: {stop_reason}")
                break

        # Final forced save to avoid missing checkpoints when early stop happens before save_every.
        if is_main:
            final_extra = {
                "global_step": global_step,
                "phase_step": min(step if "step" in locals() else 0, train_steps if "train_steps" in locals() else 0),
                "best_ppl": best_ppl,
                "best_eval_ppl": best_eval_ppl,
                "train_steps": train_steps if "train_steps" in locals() else 0,
                "stop_reason": stop_reason,
                "world_size": world_size,
                "args": vars(args),
            }
            save_adapter_distributed(
                adapter_latest,
                model,
                extra=final_extra,
                use_fsdp=use_fsdp,
                fsdp_full_shard=fsdp_full_shard,
            )
            if args.save_full_model:
                save_full_checkpoint(full_latest, model_for_save, optimizer=train_opt, scaler=scaler, extra=final_extra)

            print("[done] training finished")
            print(f"[done] stop_reason={stop_reason}")
            print(f"[done] csv={csv_path}")
            print(f"[done] adapter_latest={adapter_latest}")
            print(f"[done] adapter_best={adapter_best}")
            if args.save_full_model:
                print(f"[done] full_latest={full_latest}")
                print(f"[done] full_best={full_best}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
