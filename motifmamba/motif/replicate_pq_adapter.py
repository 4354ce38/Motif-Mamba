#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Replicate one layer/channel PQ to all layers/channels in a pq adapter.")
    p.add_argument("--in-adapter", required=True, help="Input adapter path (pq_adapter_latest.pt)")
    p.add_argument("--out-adapter", required=True, help="Output adapter path")
    p.add_argument("--source-layer", type=int, default=0, help="Source backbone layer index")
    p.add_argument(
        "--source-channel",
        type=int,
        default=-1,
        help="If P/Q are 3D (per-channel), choose source channel index; <0 means use channel-mean.",
    )
    p.add_argument(
        "--no-copy-pq-k",
        action="store_true",
        help="Do not overwrite pq_k values when replicating.",
    )
    return p.parse_args()


def find_layer_keys(state):
    layers = defaultdict(dict)
    pat = re.compile(r"^(.*\.layers\.(\d+)\.mixer)\.(P|Q|pq_k)$")
    for k, v in state.items():
        m = pat.match(k)
        if not m:
            continue
        base, layer_str, suffix = m.group(1), m.group(2), m.group(3)
        layer = int(layer_str)
        layers[layer][suffix] = (k, v)
    return layers


def make_source_tensor(tensor, source_channel):
    if tensor.ndim == 2:
        return tensor.detach().clone()
    if tensor.ndim == 3:
        if source_channel >= 0:
            if source_channel >= tensor.shape[0]:
                raise ValueError(
                    f"source_channel={source_channel} out of range for shape[0]={tensor.shape[0]}"
                )
            base2d = tensor[source_channel].detach().clone()
        else:
            base2d = tensor.detach().mean(dim=0)
        return base2d
    raise ValueError(f"Unsupported tensor ndim={tensor.ndim}, shape={tuple(tensor.shape)}")


def broadcast_like(base2d, like_tensor):
    if like_tensor.ndim == 2:
        if tuple(base2d.shape) != tuple(like_tensor.shape):
            raise ValueError(
                f"Shape mismatch for 2D target: source={tuple(base2d.shape)} target={tuple(like_tensor.shape)}"
            )
        return base2d.detach().clone().to(dtype=like_tensor.dtype)
    if like_tensor.ndim == 3:
        d = like_tensor.shape[0]
        if tuple(base2d.shape) != tuple(like_tensor.shape[1:]):
            raise ValueError(
                f"Shape mismatch for 3D target: source={tuple(base2d.shape)} target_tail={tuple(like_tensor.shape[1:])}"
            )
        out = base2d.detach().clone().unsqueeze(0).repeat(d, 1, 1)
        return out.to(dtype=like_tensor.dtype)
    raise ValueError(f"Unsupported target ndim={like_tensor.ndim}, shape={tuple(like_tensor.shape)}")


def main():
    args = parse_args()

    if not os.path.isfile(args.in_adapter):
        raise FileNotFoundError(args.in_adapter)

    payload = torch.load(args.in_adapter, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Adapter payload must be a dict")

    state = payload.get("pq_state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("Adapter pq_state_dict must be a dict")

    layers = find_layer_keys(state)
    if len(layers) == 0:
        raise RuntimeError("No layer PQ keys found in adapter")
    if args.source_layer not in layers:
        raise ValueError(f"source_layer={args.source_layer} not found. found_layers={sorted(layers.keys())}")

    src = layers[args.source_layer]
    if "P" not in src or "Q" not in src:
        raise RuntimeError(f"source layer {args.source_layer} missing P/Q")

    src_P = src["P"][1]
    src_Q = src["Q"][1]
    baseP = make_source_tensor(src_P, args.source_channel)
    baseQ = make_source_tensor(src_Q, args.source_channel)
    src_k = src.get("pq_k", (None, None))[1]

    out_state = {}
    for k, v in state.items():
        out_state[k] = v.detach().clone() if torch.is_tensor(v) else v

    n_layer = 0
    n_chan = 0
    for layer, bundle in layers.items():
        if "P" in bundle:
            kP, oldP = bundle["P"]
            newP = broadcast_like(baseP, oldP)
            if oldP.ndim == 3:
                n_chan += int(oldP.shape[0])
            out_state[kP] = newP
        if "Q" in bundle:
            kQ, oldQ = bundle["Q"]
            newQ = broadcast_like(baseQ, oldQ)
            out_state[kQ] = newQ
        if (not args.no_copy_pq_k) and ("pq_k" in bundle) and (src_k is not None):
            kk, oldk = bundle["pq_k"]
            out_state[kk] = src_k.detach().clone().to(dtype=oldk.dtype)
        n_layer += 1

    out_payload = dict(payload)
    if "pq_state_dict" in payload:
        out_payload["pq_state_dict"] = out_state
    else:
        out_payload = out_state

    os.makedirs(os.path.dirname(os.path.abspath(args.out_adapter)), exist_ok=True)
    torch.save(out_payload, args.out_adapter)

    print(f"[done] in={args.in_adapter}")
    print(f"[done] out={args.out_adapter}")
    print(f"[done] source_layer={args.source_layer}, source_channel={args.source_channel}")
    print(f"[done] layers_replicated={n_layer}, channel_slots_touched={n_chan}")


if __name__ == "__main__":
    main()
