#!/usr/bin/env python3
"""Stage 2 prep: cache LeWM fast latents from an HDF5 dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "caches" / "latents"


def img_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tworoom")
    parser.add_argument("--checkpoint", default="quentinll/lewm-tworooms")
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--stablewm-home",
        default=os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable-wm")),
    )
    args = parser.parse_args()

    os.environ["STABLEWM_HOME"] = args.stablewm_home
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(
        args.dataset,
        keys_to_cache=["action", "proprio"],
        cache_dir=args.stablewm_home,
    )
    col_name = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    ep_ids = np.unique(dataset.get_col_data(col_name))[: args.max_episodes]

    model = swm.wm.utils.load_pretrained(args.checkpoint)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True

    transform = img_transform(args.img_size)
    meta = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "num_episodes": int(len(ep_ids)),
        "img_size": args.img_size,
        "latent_dim": None,
    }

    for ep_id in ep_ids:
        mask = dataset.get_col_data(col_name) == ep_id
        indices = np.nonzero(mask)[0]
        rows = dataset.get_row_data(indices)
        pixels = rows["pixels"]
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.numpy()

        latents: list[np.ndarray] = []
        for start in range(0, len(pixels), args.batch_size):
            batch = pixels[start : start + args.batch_size]
            tensor = torch.stack([transform(frame) for frame in batch]).cuda()
            with torch.no_grad():
                out = model.encode({"pixels": tensor.unsqueeze(1)})
            emb = out["emb"].squeeze(1).cpu().numpy()
            latents.append(emb)
            if meta["latent_dim"] is None:
                meta["latent_dim"] = int(emb.shape[-1])

        ep_latents = np.concatenate(latents, axis=0)
        np.savez_compressed(
            out_dir / f"ep_{int(ep_id):06d}.npz",
            ep_id=int(ep_id),
            step_idx=rows["step_idx"],
            proprio=rows["proprio"],
            action=rows["action"],
            latents=ep_latents,
        )

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"cached {len(ep_ids)} episodes -> {out_dir}")


if __name__ == "__main__":
    main()
