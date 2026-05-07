# MotifMamba Code Export

This folder is a consolidated export of MotifMamba-related code from the local `mamba-main` tree, prepared for GitHub upload.

## Included
- `motif/`
: MotifMamba training/eval scripts (LM training, evaluation, plotting, batch launch scripts).
- `mamba_ssm/modules/mamba_simplemotif.py`
: MotifMamba module implementation.
- `mamba_ssm/ops/selective_scan_interface.py`
: selective scan interface with motif/PQ paths.
- `csrc/selective_scan/`
: CUDA/C++ selective scan extension sources used by MotifMamba.
- `mamba_ssm/models/{config_mamba.py,mixer_seq_simple.py}` and `mamba_ssm/utils/hf.py`
: minimal model/config loading dependencies used by motif scripts.
- `setup.py`, `pyproject.toml`, `LICENSE`.

## Notes
- This export preserves original scripts as much as possible, including local path defaults in some `.sh` files.
- Before running on a new machine, update env vars in launcher scripts (e.g. `ROOT`, `PY`, `PRETRAINED_DIR`, `PILE_ROOT`, `OUT_DIR`).
- Large datasets/checkpoints are intentionally not included.

## Source Snapshot
Date:
- 2026-05-07
