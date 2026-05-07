#!/usr/bin/env python3
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = "/workspace"
PY = "/home/user/miniconda3/envs/motifmamba/bin/python"
TRAIN = f"{ROOT}/mamba-main/motif/train_kpq_motif_pile_lm.py"
BASE_OUT = Path(f"{ROOT}/motif/logs/motif13_hparam_search")
BASE_OUT.mkdir(parents=True, exist_ok=True)

base = [
    PY, "-u", TRAIN,
    "--pretrained-dir", f"{ROOT}/mamba-130m",
    "--pile-root", f"{ROOT}/datasets/pile_standard_pythia",
    "--device", "cuda:0",
    "--dtype", "float32",
    "--seed", "42",
    "--motif-class", "13",
    "--pq-rank", "1",
    "--no-pq-per-dim",
    "--train-only-layer", "0",
    "--motif-loss-only-layer", "0",
    "--warmup-only",
    "--no-warmup-strict",
    "--warmup-max-steps", "120",
    "--warmup-ratio", "0.1",
    "--max-train-tokens", "0",
    "--min-train-tokens", "0",
    "--batch-size", "1",
    "--seq-len", "2048",
]

pq_k_inits = [1e-2, 1e-1, 1.0]
warmup_lrs = [3e-3, 1e-2, 3e-2]
amplitudes = [1e3, 3e2]
biases = [0.0, -1e-6]

results = []
idx = 0
for pk, lr, amp, bs in itertools.product(pq_k_inits, warmup_lrs, amplitudes, biases):
    idx += 1
    run_dir = BASE_OUT / f"trial_{idx:04d}_pk{pk:g}_lr{lr:g}_a{amp:g}_b{bs:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = base + [
        "--out-dir", str(run_dir),
        "--pq-k-init", str(pk),
        "--warmup-lr", str(lr),
        "--warmup-coef", "1e6",
        "--motif-amplitude", str(amp),
        "--motif-bias", str(bs),
        "--motif-per-dim-mode", "sampled_channel",
        "--motif-channel-samples", "16",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}/mamba-main:" + env.get("PYTHONPATH", "")
    env["MPLCONFIGDIR"] = "/tmp/mpl"

    log_path = run_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    row = {
        "trial": idx,
        "pq_k_init": pk,
        "warmup_lr": lr,
        "amplitude": amp,
        "bias": bs,
        "ok": p.returncode == 0,
        "run_dir": str(run_dir),
    }

    summ = run_dir / "warmup_summary.json"
    if summ.exists():
        d = json.loads(summ.read_text(encoding="utf-8"))
        init = float(d.get("init_motif", float("nan")))
        best = float(d.get("best_motif", float("nan")))
        row.update({
            "init": init,
            "best": best,
            "ratio": (best / init) if init != 0 else float("nan"),
            "reached": bool(d.get("reached", False)),
            "steps": int(d.get("steps_run", 0)),
        })
    results.append(row)

    print(f"trial={idx} ok={row.get('ok')} ratio={row.get('ratio')} pk={pk} lr={lr} amp={amp} bias={bs}")

# sort best by ratio
valid = [r for r in results if isinstance(r.get("ratio"), float)]
valid.sort(key=lambda x: x.get("ratio", 1e9))
out_json = BASE_OUT / "search_results.json"
out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[done] results={out_json}")
print("[top10]")
for r in valid[:10]:
    print(r)
