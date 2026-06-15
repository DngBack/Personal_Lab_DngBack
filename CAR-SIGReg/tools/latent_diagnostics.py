"""
Latent Diagnostics for CAR-SIGReg.

Loads a trained LeWM checkpoint, samples a validation batch, and computes:
  - Covariance eigenvalue spectrum
  - Effective rank (entropy-based)
  - Action sensitivity (finite-difference on act_emb)
  - Controllability alignment (fraction of action energy in top-k PCA dims)

Results are saved to a JSON file for paper tables.

Usage
-----
    # Set up environment first:
    export PYTHONPATH=CAR-SIGReg:ca-lewm/third_party/le-wm
    source ca-lewm/third_party/le-wm/.venv/bin/activate
    export STABLEWM_HOME=~/.stable-wm   # or ~/.stable_worldmodel

    python CAR-SIGReg/tools/latent_diagnostics.py \\
        --checkpoint tworoom/car_select_10ep/lewm_object.ckpt \\
        --dataset tworoom \\
        --num-batches 20 \\
        --top-k 16 \\
        --out car_sigreg_diagnostics.json

    # Compare multiple checkpoints:
    python CAR-SIGReg/tools/latent_diagnostics.py \\
        --checkpoint tworoom/lewm_baseline/lewm_object.ckpt \\
        --label lewm_baseline \\
        --out all_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Pure metric functions (no model-loading dependencies)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_covariance(emb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute sample covariance of emb.

    Args:
        emb: (B, T, D) or (N, D)

    Returns:
        C: (D, D) float32 covariance matrix
    """
    z = emb.reshape(-1, emb.size(-1)).float()
    z = z - z.mean(dim=0, keepdim=True)
    n = max(z.size(0) - 1, 1)
    return z.T @ z / n


@torch.no_grad()
def effective_rank(emb: torch.Tensor, eps: float = 1e-8) -> tuple[float, torch.Tensor]:
    """
    Entropy-based effective rank of the latent covariance.

    Args:
        emb: (B, T, D)

    Returns:
        (erank scalar, eigvals tensor descending)
    """
    C = compute_covariance(emb, eps=eps)
    eigvals = torch.linalg.eigvalsh(C).clamp_min(0).flip(0)  # descending
    p = eigvals / (eigvals.sum() + eps)
    erank = torch.exp(-(p * (p + eps).log()).sum())
    return erank.item(), eigvals.cpu()


@torch.no_grad()
def action_sensitivity(
    model,
    ctx_emb: torch.Tensor,
    ctx_act: torch.Tensor,
    eps: float = 0.05,
) -> torch.Tensor:
    """
    Estimate latent sensitivity to action via finite difference on act_emb.

    Args:
        model:   object with a .predict(emb, act_emb) method.
        ctx_emb: (B, T_ctx, D)
        ctx_act: (B, T_ctx, A)
        eps:     finite-difference step size.

    Returns:
        dz: (B, T_ctx, D) — per-sample, per-step sensitivity vector.
    """
    noise = torch.randn_like(ctx_act)
    z_plus  = model.predict(ctx_emb, ctx_act + eps * noise)
    z_minus = model.predict(ctx_emb, ctx_act - eps * noise)
    return (z_plus - z_minus) / (2.0 * eps)


@torch.no_grad()
def controllability_alignment(
    emb: torch.Tensor,
    dz: torch.Tensor,
    top_k: int = 16,
    eps: float = 1e-8,
) -> float:
    """
    Fraction of action-sensitivity energy captured by the top-k PCA directions.

    A value close to 1 means the action-sensitive directions coincide with the
    high-variance subspace.  Low values indicate mismatch (the failure mode
    that CAR-SIGReg is designed to fix).

    Args:
        emb:   (B, T, D) — used to compute PCA basis.
        dz:    (B, T, D) — sensitivity vectors from action_sensitivity().
        top_k: number of PCA eigenvectors to use as the "active" subspace.

    Returns:
        alignment scalar in [0, 1].
    """
    C = compute_covariance(emb, eps=eps)
    _, U = torch.linalg.eigh(C)
    U = U.flip(1)  # descending eigenvalue order → columns = top eigenvectors

    U_A = U[:, :top_k].to(dz.device)
    dz_flat = dz.reshape(-1, dz.size(-1)).float()

    active_energy = (dz_flat @ U_A).square().sum(dim=-1).mean()
    total_energy  = dz_flat.square().sum(dim=-1).mean()

    return (active_energy / (total_energy + eps)).item()


@torch.no_grad()
def per_direction_ctrl_score(
    emb: torch.Tensor,
    dz: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each PCA eigenvector, compute its variance score and ctrl score.

    Useful for plotting the joint distribution that CAR-SIGReg optimises.

    Args:
        emb: (B, T, D)
        dz:  (B, T, D)

    Returns:
        (var_score, ctrl_score) — both (D,) normalised to sum=1.
    """
    C = compute_covariance(emb, eps=eps)
    eigvals, U = torch.linalg.eigh(C)
    eigvals = eigvals.flip(0).clamp_min(0)
    U = U.flip(1)

    dz_flat = dz.reshape(-1, dz.size(-1)).float()
    dz_u = dz_flat @ U.to(dz.device)
    ctrl_raw = dz_u.square().mean(dim=0).cpu()

    var_score  = eigvals / (eigvals.sum() + eps)
    ctrl_score = ctrl_raw / (ctrl_raw.sum() + eps)

    return var_score, ctrl_score


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def resolve_checkpoint_path(policy: str) -> Path:
    """
    Resolve a checkpoint path the same way eval.py does:
      - If absolute and exists → use directly.
      - Otherwise prefix with $STABLEWM_HOME/checkpoints/.
    Appends _object.ckpt suffix if the path has no extension.
    """
    p = Path(policy)
    if p.is_absolute() and p.exists():
        return p

    home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    base = home / "checkpoints" / policy

    if base.exists():
        return base

    with_suffix = base.with_name(base.name + "_object.ckpt")
    if with_suffix.exists():
        return with_suffix

    for suffix in ("_object.ckpt", "_weights.ckpt", ".ckpt"):
        candidate = base.with_suffix(suffix) if base.suffix == "" else base.parent / (base.stem + suffix)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve checkpoint for policy='{policy}'. "
        f"Tried {base} and variants. Set STABLEWM_HOME correctly."
    )


def load_model_from_checkpoint(ckpt_path: Path, device: str = "cpu"):
    """
    Load a JEPA world model from a LeWM _object.ckpt file.

    These checkpoints store the raw JEPA object (not a Lightning state dict).
    Falls back to torch.load with weights_only=False.
    """
    try:
        model = torch.load(ckpt_path, map_location=device, weights_only=False)
        if hasattr(model, "eval"):
            model.eval()
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint {ckpt_path}: {e}\n"
            "Make sure PYTHONPATH includes ca-lewm/third_party/le-wm/"
        ) from e


def load_dataset_batches(
    dataset_name: str,
    num_steps: int = 4,
    batch_size: int = 64,
    num_batches: int = 20,
    device: str = "cpu",
    seed: int = 42,
):
    """
    Load a few batches from a LeWM dataset for diagnostic purposes.

    Returns list of batch dicts, each with keys: pixels, action, proprio.
    """
    try:
        import stable_worldmodel as swm
        import stable_pretraining as spt
    except ImportError as e:
        raise ImportError(
            "stable_worldmodel / stable_pretraining not found. "
            "Run with PYTHONPATH=ca-lewm/third_party/le-wm/"
        ) from e

    dataset = swm.data.load_dataset(dataset_name, transform=None, num_steps=num_steps)
    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=gen,
        num_workers=0,
    )

    batches = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        batches.append({k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)})
    return batches


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def run_diagnostics(
    model,
    batches: list[dict],
    top_k: int = 16,
    action_eps: float = 0.05,
    history_size: int = 3,
    device: str = "cpu",
) -> dict:
    """
    Run all diagnostic metrics across a list of batches and return aggregated results.

    Returns dict suitable for JSON serialization.
    """
    model = model.to(device).eval()

    all_emb = []
    all_dz  = []

    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        with torch.no_grad():
            output = model.encode(batch)

        emb     = output["emb"].float()      # (B, T, D)
        act_emb = output["act_emb"].float()

        ctx_emb = emb[:, :history_size]
        ctx_act = act_emb[:, :history_size]

        dz = action_sensitivity(model, ctx_emb, ctx_act, eps=action_eps)  # (B, T_ctx, D)

        all_emb.append(emb.cpu())
        all_dz.append(dz.cpu())

    emb_cat = torch.cat(all_emb, dim=0)   # (N*B, T, D)
    dz_cat  = torch.cat(all_dz,  dim=0)

    erank, eigvals = effective_rank(emb_cat)
    ctrl_align_16  = controllability_alignment(emb_cat, dz_cat, top_k=top_k)
    ctrl_align_32  = controllability_alignment(emb_cat, dz_cat, top_k=min(32, emb_cat.size(-1)))

    var_score, ctrl_score = per_direction_ctrl_score(emb_cat, dz_cat)

    # Top-16 alignment of var vs ctrl (cosine sim as a proxy for correlation)
    top16_var  = set(var_score.argsort(descending=True)[:16].tolist())
    top16_ctrl = set(ctrl_score.argsort(descending=True)[:16].tolist())
    subspace_overlap = len(top16_var & top16_ctrl) / 16.0

    return {
        "effective_rank": round(erank, 4),
        "ctrl_align_top16": round(ctrl_align_16, 4),
        "ctrl_align_top32": round(ctrl_align_32, 4),
        "subspace_overlap_top16": round(subspace_overlap, 4),
        "eigvals_top32": eigvals[:32].tolist(),
        "var_score_top32": var_score[:32].tolist(),
        "ctrl_score_top32": ctrl_score[:32].tolist(),
        "embed_dim": int(emb_cat.size(-1)),
        "num_samples": int(emb_cat.shape[0] * emb_cat.shape[1]),
    }


def save_metrics(path: str, metrics: dict) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved diagnostics → {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Latent diagnostics: eff-rank, ctrl-align, action sensitivity."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Policy name (relative to $STABLEWM_HOME/checkpoints/) "
            "or absolute path to a _object.ckpt file."
        ),
    )
    p.add_argument(
        "--dataset",
        default="tworoom",
        help="Dataset name as understood by swm.data.load_dataset (e.g. tworoom, pusht_expert_train).",
    )
    p.add_argument(
        "--num-batches",
        type=int,
        default=20,
        help="Number of batches to sample for diagnostics.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--history-size",
        type=int,
        default=3,
        help="ctx_len (history_size) matching the training config.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Number of top PCA directions for ctrl_align metric.",
    )
    p.add_argument(
        "--action-eps",
        type=float,
        default=0.05,
        help="Finite-difference step on act_emb.",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Optional label to tag results (e.g. 'lewm_baseline', 'car_sigreg').",
    )
    p.add_argument(
        "--out",
        default="diagnostics.json",
        help="Output JSON file path.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[diagnostics] checkpoint : {args.checkpoint}")
    print(f"[diagnostics] dataset    : {args.dataset}")
    print(f"[diagnostics] device     : {args.device}")

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    print(f"[diagnostics] resolved   : {ckpt_path}")

    model = load_model_from_checkpoint(ckpt_path, device=args.device)

    print(f"[diagnostics] loading {args.num_batches} batches from {args.dataset}...")
    batches = load_dataset_batches(
        dataset_name=args.dataset,
        num_steps=args.history_size + 1,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=args.device,
    )

    print("[diagnostics] computing metrics...")
    metrics = run_diagnostics(
        model=model,
        batches=batches,
        top_k=args.top_k,
        action_eps=args.action_eps,
        history_size=args.history_size,
        device=args.device,
    )

    if args.label:
        metrics["label"] = args.label
    metrics["checkpoint"] = str(ckpt_path)
    metrics["dataset"] = args.dataset

    save_metrics(args.out, metrics)

    print("\n--- Summary ---")
    print(f"  effective_rank      : {metrics['effective_rank']}")
    print(f"  ctrl_align_top16    : {metrics['ctrl_align_top16']}")
    print(f"  ctrl_align_top32    : {metrics['ctrl_align_top32']}")
    print(f"  subspace_overlap_16 : {metrics['subspace_overlap_top16']}")


if __name__ == "__main__":
    main()
