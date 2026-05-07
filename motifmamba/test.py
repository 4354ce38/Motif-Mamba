#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F
from einops import rearrange

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mamba_ssm.modules.mamba_simplemotif as mmod
from mamba_ssm.modules.mamba_simplemotif import Mambamotif
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def summarize_tensor(name, t, n=8):
    t_det = t.detach()
    t_cpu = t_det.float().cpu()
    flat = t_cpu.reshape(-1)
    if flat.numel() == 0:
        print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, EMPTY")
        return
    sample = ", ".join(f"{v:.6g}" for v in flat[:n].tolist())
    print(
        f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
        f"min={flat.min().item():.6g}, max={flat.max().item():.6g}, "
        f"mean={flat.mean().item():.6g}, sample=[{sample}]"
    )


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Device: {device}")
    print("Build a tiny Mambamotif and print internal variables (A, dt, B, C, P, Q, k...).")

    # Minimal motif-enabled Mamba
    model = Mambamotif(
        d_model=16,
        d_state=8,
        d_conv=4,
        expand=2,
        pq_rank=2,
        pq_per_dim=False,
        pq_k_init=1e-4,
        device=device,
        dtype=dtype,
    ).to(device)
    model.eval()

    # Minimal input: (batch, seqlen, d_model)
    hidden_states = torch.randn(1, 4, 16, device=device, dtype=dtype)
    summarize_tensor("hidden_states", hidden_states)

    with torch.no_grad():
        batch, seqlen, _ = hidden_states.shape

        # Reproduce the same intermediate variables as in Mambamotif.forward
        xz = rearrange(
            model.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if model.in_proj.bias is not None:
            xz = xz + rearrange(model.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)
        if mmod.causal_conv1d_fn is None or device.type != "cuda":
            x = model.act(model.conv1d(x)[..., :seqlen])
        else:
            x = mmod.causal_conv1d_fn(
                x=x,
                weight=rearrange(model.conv1d.weight, "d 1 w -> d w"),
                bias=model.conv1d.bias,
                activation=model.activation,
            )

        x_dbl = model.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt_in, B, C = torch.split(x_dbl, [model.dt_rank, model.d_state, model.d_state], dim=-1)

        dt = model.dt_proj.weight @ dt_in.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) n -> b n l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=seqlen).contiguous()

        A = -torch.exp(model.A_log.float())
        D = model.D.float()
        delta_bias = model.dt_proj.bias.float()
        dt_eff = F.softplus(dt + delta_bias[None, :, None])

        P = model.P
        Q = model.Q
        k = model.pq_k
        P_eff = P * k

        print("\n===== Core Variables =====")
        summarize_tensor("A", A)
        summarize_tensor("dt(raw)", dt)
        summarize_tensor("dt(effective=softplus(dt+bias))", dt_eff)
        summarize_tensor("B", B)
        summarize_tensor("C", C)
        summarize_tensor("D", D)
        summarize_tensor("z", z)
        summarize_tensor("P", P)
        summarize_tensor("Q", Q)
        summarize_tensor("k", k)
        summarize_tensor("P_eff=k*P", P_eff)
        summarize_tensor("PQ=P_eff@Q", P_eff @ Q)

        # Print concrete slices for readability
        print("\n===== Selected Slices =====")
        print("A[0, :8] =", A[0, :8].detach().cpu())
        print("dt_eff[0, 0, :] =", dt_eff[0, 0, :].detach().cpu())
        print("B[0, :, 0] =", B[0, :, 0].detach().cpu())
        print("C[0, :, 0] =", C[0, :, 0].detach().cpu())
        print("(P_eff@Q)[0, :] =", (P_eff @ Q)[0, :].detach().cpu())

        if device.type != "cuda":
            print("\n[Info] CUDA not available.")
            print("[Info] Printed all internal variables. Skip selective_scan_fn/model.forward test")
            print("       because current motif full-matrix path requires CUDA tensors.")
            return

        print("\n===== CUDA Forward Check =====")
        y_scan, last_state = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=True,
            P=P_eff,
            Q=Q,
        )
        y = rearrange(y_scan, "b d l -> b l d")
        out_manual = model.out_proj(y)
        out_model = model(hidden_states)

        summarize_tensor("y_scan", y_scan)
        summarize_tensor("last_state", last_state)
        summarize_tensor("out_manual", out_manual)
        summarize_tensor("out_model", out_model)

        max_diff = (out_manual - out_model).abs().max().item()
        print(f"max_abs_diff(out_manual, out_model) = {max_diff:.6g}")


if __name__ == "__main__":
    main()
