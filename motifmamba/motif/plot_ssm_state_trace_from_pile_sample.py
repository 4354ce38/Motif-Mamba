#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

# Avoid matplotlib cache permission issues on shared environments.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot input tokens and internal SSM state traces over time for one Pile sample."
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default="/workspace/motif/logs/pile_mamba130m_motif2avg12_rank2_fulltrain_1M_seq2048_20260415_095438",
        help="Directory containing full_model_latest.pt / pq_adapter_latest.pt",
    )
    p.add_argument("--pretrained-dir", type=str, default="/workspace/mamba-130m")
    p.add_argument("--full-ckpt", type=str, default="")
    p.add_argument("--adapter-path", type=str, default="")
    p.add_argument("--pile-root", type=str, default="/workspace/datasets/pile_standard_pythia_10M")
    p.add_argument("--pile-shard-index", type=int, default=0)
    p.add_argument("--sample-index", type=int, default=0, help="Window index in selected shard")
    p.add_argument(
        "--sample-mode",
        type=str,
        default="auto_sentence",
        choices=["fixed", "auto_sentence"],
        help="fixed: use --sample-index directly; auto_sentence: search for a more sentence-like sample.",
    )
    p.add_argument("--auto-search-windows", type=int, default=2000, help="How many windows to scan in auto mode.")
    p.add_argument("--auto-search-step", type=int, default=1, help="Sample-index stride during auto search.")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--plot-steps", type=int, default=128, help="Only visualize the first N time steps (capped at 128).")
    p.add_argument("--row2-layers", type=int, default=6, help="How many representative layers to keep in row 2.")
    p.add_argument("--row2-normalize", dest="row2_normalize", action="store_true", help="Normalize row2 per selected layer (z-score).")
    p.add_argument("--no-row2-normalize", dest="row2_normalize", action="store_false", help="Disable row2 per-layer normalization.")
    p.set_defaults(row2_normalize=True)
    p.add_argument("--tokenizer-path", type=str, default="EleutherAI/gpt-neox-20b")
    p.add_argument(
        "--selected-layer",
        type=int,
        default=-1,
        help="Layer index for detailed state trajectory; supports negative index (-1 = last layer).",
    )
    p.add_argument("--units-to-plot", type=int, default=1, help="Ignored now: third row always plots one unit.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--out-prefix", type=str, default="pile_sample_ssm_trace")
    return p.parse_args()


def resolve_dtype(s: str) -> torch.dtype:
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float32


def list_pile_bins(pile_root: str):
    pattern = re.compile(r"document-\d{5}-of-\d{5}\.bin$")
    files = sorted([p for p in glob(os.path.join(pile_root, "*.bin")) if pattern.search(os.path.basename(p))])
    if len(files) == 0:
        raise FileNotFoundError(f"No Pile .bin files found under: {pile_root}")
    return files


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def _infer_pq_layout(state: Dict[str, torch.Tensor]) -> Tuple[int, bool]:
    for k, v in state.items():
        if not k.endswith(".Q"):
            continue
        if v.ndim == 3:
            return int(v.shape[1]), True
        if v.ndim == 2:
            return int(v.shape[0]), False
    for k, v in state.items():
        if not k.endswith(".P"):
            continue
        if v.ndim == 3:
            return int(v.shape[2]), True
        if v.ndim == 2:
            return int(v.shape[1]), False
    raise RuntimeError("Unable to infer pq_rank/pq_per_dim from state_dict.")


def _extract_model_state(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload)}")


def _load_full_checkpoint(path: str):
    payload = torch.load(path, map_location="cpu")
    state = _strip_module_prefix(_extract_model_state(payload))
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    pq_rank, pq_per_dim = _infer_pq_layout(state)
    pq_k_init = float(args.get("pq_k_init", 1e-4))
    if args.get("pq_per_dim", None) is not None:
        pq_per_dim = bool(args["pq_per_dim"])
    return state, args, pq_rank, pq_per_dim, pq_k_init


def _load_adapter(path: str):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "pq_state_dict" in payload:
        state = payload["pq_state_dict"]
        args = payload.get("args", {})
        pq_k_init = float(args.get("pq_k_init", 1e-4))
        pq_per_dim_arg = args.get("pq_per_dim", None)
    elif isinstance(payload, dict):
        state = payload
        pq_k_init = 1e-4
        pq_per_dim_arg = None
    else:
        raise RuntimeError(f"Unsupported adapter payload type: {type(payload)}")
    state = _strip_module_prefix(state)
    pq_rank, inferred_pq_per_dim = _infer_pq_layout(state)
    pq_per_dim = bool(pq_per_dim_arg) if pq_per_dim_arg is not None else inferred_pq_per_dim
    return state, pq_rank, pq_per_dim, pq_k_init


def build_model(
    pretrained_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    full_state: Optional[Dict[str, torch.Tensor]],
    adapter_state: Optional[Dict[str, torch.Tensor]],
    pq_rank: int,
    pq_per_dim: bool,
    pq_k_init: float,
):
    from mamba_ssm.models.config_mamba import MambaConfig
    import mamba_ssm.models.mixer_seq_simple as mixer_seq_simple
    from mamba_ssm.modules.mamba_simplemotif import Mambamotif
    import mamba_ssm.modules.mamba_simplemotif as mamba_simplemotif_mod
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

    if device.type != "cuda":
        # The optional causal-conv1d CUDA extension is not available on CPU.
        mamba_simplemotif_mod.causal_conv1d_fn = None
        mamba_simplemotif_mod.causal_conv1d_update = None
        # Full-matrix PQ scan on CPU must use reference implementation.
        mamba_simplemotif_mod.selective_scan_fn = selective_scan_ref

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

    model = mixer_seq_simple.MambaLMHeadModel(config, device="cpu", dtype=torch.float32)
    base_state = load_state_dict_hf(pretrained_dir, device="cpu", dtype=None)
    model.load_state_dict(base_state, strict=False)
    if full_state is not None:
        model.load_state_dict(full_state, strict=False)
    if adapter_state is not None:
        model.load_state_dict(adapter_state, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def pick_sample_tokens(pile_root: str, shard_index: int, sample_index: int, seq_len: int):
    files = list_pile_bins(pile_root)
    if shard_index < 0 or shard_index >= len(files):
        raise ValueError(f"pile-shard-index out of range: {shard_index}, num_files={len(files)}")
    arr = np.memmap(files[shard_index], dtype=np.uint16, mode="r")
    need = seq_len + 1
    start = int(sample_index) * need
    if start + need > int(arr.shape[0]):
        raise ValueError(
            f"Not enough tokens in shard for sample-index={sample_index}, seq-len={seq_len}. "
            f"max_start={int(arr.shape[0]) - need}"
        )
    window = np.asarray(arr[start : start + need], dtype=np.int64)
    input_ids = torch.from_numpy(window[:-1]).long().unsqueeze(0)
    return input_ids, files[shard_index], start


def _format_token_piece(piece: str, max_len: int = 200) -> str:
    txt = piece.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if txt == "":
        txt = "<empty>"
    if len(txt) > max_len:
        txt = txt[: max_len - 3] + "..."
    return txt


def load_tokenizer(tokenizer_path: str, pretrained_dir: str):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None, "none(transformers_missing)"

    candidates = []
    if tokenizer_path:
        candidates.append(tokenizer_path)
    if pretrained_dir and pretrained_dir not in candidates:
        candidates.append(pretrained_dir)

    tok = None
    used = ""
    for c in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(c, local_files_only=True, trust_remote_code=True)
            used = f"local:{c}"
            break
        except Exception:
            continue
    return tok, (used if tok is not None else "none(not_found_local_cache)")


def decode_token_pieces(token_ids: np.ndarray, tok) -> List[str]:
    if tok is None:
        return [f"<id:{int(t)}>" for t in token_ids]

    pieces: List[str] = []
    for tid in token_ids.tolist():
        try:
            p = tok.decode([int(tid)], clean_up_tokenization_spaces=False)
        except Exception:
            p = f"<id:{int(tid)}>"
        pieces.append(_format_token_piece(p))
    return pieces


def decode_full_text(token_ids: np.ndarray, tok) -> str:
    if tok is None:
        return " ".join([f"<id:{int(t)}>" for t in token_ids])
    try:
        txt = tok.decode(token_ids.tolist(), clean_up_tokenization_spaces=True)
    except Exception:
        txt = " ".join([f"<id:{int(t)}>" for t in token_ids])
    txt = txt.replace("\r", " ").replace("\n", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def sentence_like_score(text: str) -> float:
    t = text.strip()
    if len(t) < 20:
        return -1e9
    n = max(len(t), 1)
    letters = sum(ch.isalpha() for ch in t)
    spaces = sum(ch.isspace() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    punct = sum(ch in ".,;:!?" for ch in t)
    weird = sum(ch in "<>{}[]|\\@" for ch in t)
    score = 0.0
    score += (letters + 0.5 * spaces) / n
    score += 0.25 * min(1.0, punct / max(1, n // 20))
    score += 0.05 * min(1.0, digits / max(1, n // 30))
    if re.search(r"[.!?]\s", t):
        score += 0.2
    if re.match(r"^[A-Z\"'(\[]", t):
        score += 0.08
    score -= 0.3 * (weird / n)
    return score


def select_sentence_like_sample(
    pile_root: str,
    shard_index: int,
    seq_len: int,
    tok,
    start_sample_index: int,
    search_windows: int,
    search_step: int,
):
    files = list_pile_bins(pile_root)
    if shard_index < 0 or shard_index >= len(files):
        raise ValueError(f"pile-shard-index out of range: {shard_index}, num_files={len(files)}")
    arr = np.memmap(files[shard_index], dtype=np.uint16, mode="r")
    need = seq_len + 1
    max_windows = max(0, (int(arr.shape[0]) - need) // need + 1)
    if max_windows <= 0:
        raise RuntimeError("No valid windows found in selected shard.")

    best = None
    best_score = -1e18
    start = max(0, int(start_sample_index))
    limit = min(max_windows, start + max(1, int(search_windows)) * max(1, int(search_step)))
    step = max(1, int(search_step))

    for si in range(start, limit, step):
        st = si * need
        window = np.asarray(arr[st : st + need], dtype=np.int64)
        token_ids = window[:-1]
        preview_len = min(64, token_ids.shape[0])
        txt = decode_full_text(token_ids[:preview_len], tok)
        s = sentence_like_score(txt)
        if s > best_score:
            best_score = s
            best = (si, token_ids, txt)

    if best is None:
        raise RuntimeError("Failed to select a sentence-like sample.")

    si, token_ids, txt = best
    input_ids = torch.from_numpy(token_ids).long().unsqueeze(0)
    return input_ids, files[shard_index], si * need, si, txt, best_score


@torch.no_grad()
def collect_state_traces(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    selected_layer: int,
):
    from mamba_ssm.utils.generation import InferenceParams

    tokens = input_ids.to(device=device, dtype=torch.long)
    T = int(tokens.shape[1])
    infer = InferenceParams(max_seqlen=T, max_batch_size=1)

    layer_keys = None
    selected_key = None
    selected_states = []
    layer_rms = []

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    amp_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32

    for t in range(T):
        x_t = tokens[:, t : t + 1]
        if use_amp:
            with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                _ = model(x_t, inference_params=infer, num_last_tokens=1).logits
        else:
            _ = model(x_t, inference_params=infer, num_last_tokens=1).logits
        infer.seqlen_offset += 1

        if layer_keys is None:
            layer_keys = sorted(infer.key_value_memory_dict.keys())
            if len(layer_keys) == 0:
                raise RuntimeError("No layer states found in inference cache.")
            sel = selected_layer
            if sel < 0:
                sel = len(layer_keys) + sel
            if sel < 0 or sel >= len(layer_keys):
                raise ValueError(f"selected-layer={selected_layer} out of range for {len(layer_keys)} layers")
            selected_key = layer_keys[sel]

        rms_row = []
        for k in layer_keys:
            cache = infer.key_value_memory_dict[k]
            if not (isinstance(cache, tuple) and len(cache) >= 2):
                raise RuntimeError(f"Unexpected cache layout at layer {k}: {type(cache)}")
            ssm_state = cache[1]  # (B, d_inner, d_state)
            rms = torch.sqrt(torch.mean(ssm_state[0].float() ** 2)).item()
            rms_row.append(rms)
        layer_rms.append(rms_row)

        sel_state = infer.key_value_memory_dict[selected_key][1][0].detach().float().cpu().reshape(-1)
        selected_states.append(sel_state)

    layer_rms = np.asarray(layer_rms, dtype=np.float32)  # (T, n_layer)
    selected_states = torch.stack(selected_states, dim=0).numpy()  # (T, units)
    return layer_keys, selected_key, layer_rms, selected_states


def plot_traces(
    token_ids: np.ndarray,
    token_texts: List[str],
    layer_rms: np.ndarray,
    selected_states: np.ndarray,
    selected_key: int,
    units_to_plot: int,
    plot_steps: int,
    row2_layers: int,
    row2_normalize: bool,
    sample_preview_text: str,
    out_png: str,
):
    T_all = token_ids.shape[0]
    T = min(int(plot_steps), int(T_all), 128)
    n_layer = layer_rms.shape[1]
    t = np.arange(1, T + 1)
    row2_m = max(1, min(int(row2_layers), int(n_layer)))
    row2_idx = np.unique(np.linspace(0, n_layer - 1, num=row2_m, dtype=int))
    row2_rms = layer_rms[:T][:, row2_idx]  # (T, m), raw
    row2_mu = row2_rms.mean(axis=0, keepdims=True)
    row2_std = row2_rms.std(axis=0, keepdims=True)
    row2_std = np.where(row2_std < 1e-8, 1.0, row2_std)
    row2_rms_norm = (row2_rms - row2_mu) / row2_std
    row2_show = row2_rms_norm if row2_normalize else row2_rms

    variances = selected_states[:T].var(axis=0)
    # Keep only one most responsive unit in the third row.
    k = 1
    top_idx = np.argsort(-variances)[:k]
    top_traces = selected_states[:T, top_idx]  # (T, k)

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.2, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

    ax0.set_xlim(0.5, T + 0.5)
    ax0.set_ylim(0.0, 1.0)
    ax0.set_yticks([])
    ax0.grid(axis="x", alpha=0.25)
    for i in range(T):
        tok_piece = token_texts[i]
        ax0.text(
            i + 1,
            0.5,
            f"t={i+1}\n{tok_piece}",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "#f7f7f7", "ec": "#bbbbbb", "alpha": 0.95},
        )
    if T == 1:
        xticks = [1]
    else:
        xticks = [1, T]
    ax0.set_xticks(xticks)
    ax0.set_title(f"Input Tokens Aligned by Time (t=1..{T})")

    im = ax1.imshow(
        row2_show.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[0.5, T + 0.5, 0, len(row2_idx) - 1],
        cmap=("RdBu_r" if row2_normalize else "viridis"),
    )
    if row2_normalize:
        ax1.set_title(f"SSM State RMS z-score (Row2 Kept {len(row2_idx)} Layers, t=1..{T})")
    else:
        ax1.set_title(f"SSM State RMS (Row2 Kept {len(row2_idx)} Layers, t=1..{T})")
    ax1.set_ylabel("Layer index (selected)")
    ax1.set_xticks(xticks)
    ax1.set_yticks(np.arange(len(row2_idx)))
    ax1.set_yticklabels([str(int(x)) for x in row2_idx])
    fig.colorbar(im, ax=ax1, label=("z-score RMS(ssm_state)" if row2_normalize else "RMS(ssm_state)"))

    ax2.plot(t, top_traces[:, 0], linewidth=1.2, color="#d62728", label=f"u{int(top_idx[0])}")
    ax2.set_title(f"Selected Layer={selected_key}: Single Most-Variant SSM Unit (t=1..{T})")
    ax2.set_xlabel("Time step t")
    ax2.set_ylabel("State value")
    ax2.grid(True, alpha=0.25)
    ax2.set_xticks(xticks)
    ax2.legend(loc="upper right", fontsize=9, frameon=False)

    preview = sample_preview_text
    if len(preview) > 180:
        preview = preview[:177] + "..."
    fig.suptitle(f"Sample Preview: {preview}", fontsize=12, y=1.02)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return top_idx, row2_idx, row2_rms, row2_rms_norm


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else run_dir
    os.makedirs(out_dir, exist_ok=True)

    full_ckpt = os.path.abspath(args.full_ckpt) if args.full_ckpt else os.path.join(run_dir, "full_model_latest.pt")
    adapter_path = os.path.abspath(args.adapter_path) if args.adapter_path else os.path.join(run_dir, "pq_adapter_latest.pt")

    pretrained_dir = os.path.abspath(args.pretrained_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("[warn] bf16 not supported on this GPU, fallback to float16.")
        dtype = torch.float16
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print("[warn] fp16/bf16 on CPU is slow or unsupported in some ops. Falling back to float32.")
        dtype = torch.float32

    full_state = None
    adapter_state = None
    ckpt_args = {}
    pq_rank = None
    pq_per_dim = None
    pq_k_init = 1e-4

    if os.path.isfile(full_ckpt):
        full_state, ckpt_args, pq_rank, pq_per_dim, pq_k_init = _load_full_checkpoint(full_ckpt)
        print(f"[load] full checkpoint: {full_ckpt}")
    else:
        print(f"[warn] full checkpoint not found: {full_ckpt}")

    if os.path.isfile(adapter_path):
        adapter_state, ar, ap, ak = _load_adapter(adapter_path)
        print(f"[load] adapter: {adapter_path}")
        if pq_rank is None:
            pq_rank = ar
        if pq_per_dim is None:
            pq_per_dim = ap
        pq_k_init = ak
    else:
        print(f"[warn] adapter not found: {adapter_path}")

    if pq_rank is None:
        raise RuntimeError("Cannot infer pq_rank. Need full checkpoint or adapter.")
    if pq_per_dim is None:
        pq_per_dim = False

    ckpt_pretrained = ckpt_args.get("pretrained_dir", "") if isinstance(ckpt_args, dict) else ""
    if ckpt_pretrained and os.path.isdir(ckpt_pretrained):
        pretrained_dir = ckpt_pretrained
    print(f"[load] pretrained_dir={pretrained_dir}, pq_rank={pq_rank}, pq_per_dim={pq_per_dim}, pq_k_init={pq_k_init}")

    tok, tok_source = load_tokenizer(args.tokenizer_path, pretrained_dir)
    print(f"[tokenizer] source={tok_source}")

    if args.sample_mode == "auto_sentence":
        input_ids, shard_path, start_offset, chosen_sample_idx, sample_preview_text, sample_score = select_sentence_like_sample(
            pile_root=args.pile_root,
            shard_index=args.pile_shard_index,
            seq_len=args.seq_len,
            tok=tok,
            start_sample_index=args.sample_index,
            search_windows=args.auto_search_windows,
            search_step=args.auto_search_step,
        )
        print(
            f"[sample] mode=auto_sentence chosen_sample_index={chosen_sample_idx} "
            f"score={sample_score:.4f}"
        )
    else:
        input_ids, shard_path, start_offset = pick_sample_tokens(
            pile_root=args.pile_root,
            shard_index=args.pile_shard_index,
            sample_index=args.sample_index,
            seq_len=args.seq_len,
        )
        chosen_sample_idx = int(args.sample_index)
        sample_preview_text = decode_full_text(input_ids[0].cpu().numpy()[:64], tok)
        print(f"[sample] mode=fixed sample_index={chosen_sample_idx}")

    print(
        f"[sample] task=pile_lm shard={os.path.basename(shard_path)} "
        f"start_offset={start_offset} seq_len={args.seq_len}"
    )

    model = build_model(
        pretrained_dir=pretrained_dir,
        device=device,
        dtype=dtype,
        full_state=full_state,
        adapter_state=adapter_state,
        pq_rank=pq_rank,
        pq_per_dim=pq_per_dim,
        pq_k_init=pq_k_init,
    )

    layer_keys, selected_key, layer_rms, selected_states = collect_state_traces(
        model=model,
        input_ids=input_ids,
        device=device,
        dtype=dtype,
        selected_layer=args.selected_layer,
    )
    print(f"[trace] n_layers={len(layer_keys)}, selected_layer_key={selected_key}, timesteps={input_ids.shape[1]}")

    out_png = os.path.join(out_dir, f"{args.out_prefix}.png")
    out_npz = os.path.join(out_dir, f"{args.out_prefix}.npz")
    out_tokens_csv = os.path.join(out_dir, f"{args.out_prefix}_tokens.csv")
    out_sample_txt = os.path.join(out_dir, f"{args.out_prefix}_sample_text.txt")
    out_meta_txt = os.path.join(out_dir, f"{args.out_prefix}_meta.txt")
    token_ids_np = input_ids[0].cpu().numpy()
    token_texts = decode_token_pieces(token_ids=token_ids_np, tok=tok)

    steps = min(int(args.plot_steps), int(token_ids_np.shape[0]), 128)
    print(f"[plot] only first {steps} steps: t=1..{steps} (cap=128)")
    for i in range(steps):
        print(f"[token] t={i+1:02d} id={int(token_ids_np[i])} piece={token_texts[i]}")
    print(f"[sample_text] {sample_preview_text}")

    top_idx, row2_idx, row2_rms, row2_rms_norm = plot_traces(
        token_ids=token_ids_np,
        token_texts=token_texts,
        layer_rms=layer_rms,
        selected_states=selected_states,
        selected_key=selected_key,
        units_to_plot=args.units_to_plot,
        plot_steps=steps,
        row2_layers=args.row2_layers,
        row2_normalize=args.row2_normalize,
        sample_preview_text=sample_preview_text,
        out_png=out_png,
    )
    np.savez_compressed(
        out_npz,
        input_ids=token_ids_np,
        token_texts=np.asarray(token_texts),
        layer_rms=layer_rms,
        selected_states=selected_states,
        layer_keys=np.asarray(layer_keys, dtype=np.int32),
        selected_layer_key=np.asarray([selected_key], dtype=np.int32),
        plot_steps=np.asarray([steps], dtype=np.int32),
        top_unit_indices=np.asarray(top_idx, dtype=np.int64),
        row2_layer_indices=np.asarray(row2_idx, dtype=np.int32),
        row2_layer_rms=np.asarray(row2_rms, dtype=np.float32),
        row2_layer_rms_norm=np.asarray(row2_rms_norm, dtype=np.float32),
    )
    with open(out_tokens_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "token_id", "token_text"])
        for i in range(steps):
            writer.writerow([i + 1, int(token_ids_np[i]), token_texts[i]])
    with open(out_sample_txt, "w", encoding="utf-8") as f:
        f.write(sample_preview_text + "\n")
    with open(out_meta_txt, "w", encoding="utf-8") as f:
        f.write(f"sample_mode={args.sample_mode}\n")
        f.write(f"chosen_sample_index={chosen_sample_idx}\n")
        f.write(f"pile_shard={os.path.basename(shard_path)}\n")
        f.write(f"start_offset={start_offset}\n")
        f.write(f"seq_len={args.seq_len}\n")
        f.write(f"plot_steps={steps}\n")
        f.write(f"row2_layers_kept={len(row2_idx)}\n")
        f.write(f"row2_normalize={bool(args.row2_normalize)}\n")
        f.write("row2_layer_indices=" + ",".join([str(int(x)) for x in row2_idx]) + "\n")
        f.write(f"tokenizer_source={tok_source}\n")

    print(f"[done] figure={out_png}")
    print(f"[done] traces={out_npz}")
    print(f"[done] tokens_csv={out_tokens_csv}")
    print(f"[done] sample_text={out_sample_txt}")
    print(f"[done] meta={out_meta_txt}")


if __name__ == "__main__":
    main()
