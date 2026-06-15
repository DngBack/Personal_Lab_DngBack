"""
CAR-SIGReg training script.

This is a patched version of le-wm/train.py that replaces the original
SIGReg with CARSIGReg (Controllability-Aware Rank-Adaptive SIGReg).

Run from the repo root (Personal_Lab_DngBack/) with:

    export PYTHONPATH=CAR-SIGReg:ca-lewm/third_party/le-wm
    source ca-lewm/third_party/le-wm/.venv/bin/activate
    python CAR-SIGReg/train.py data=tworoom wandb.enabled=false

Hydra will discover config from CAR-SIGReg/config/train/ (relative to
this file's location via config_path="./config/train").
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

# Avoid CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (torch 2.12+cu130 vs system cuDNN).
torch.backends.cudnn.enabled = False

from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from car_sigreg import CARSIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


# Keys logged every step (both loss and diagnostic scalars).
_LOG_KEYS = (
    "loss",
    "pred_loss",
    "sigreg_loss",
    "inactive_loss",
    "ctrl_loss",
    "eff_rank",
    "active_rank",
    "ctrl_align",
    "inactive_energy",
)


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute CAR-SIGReg losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]       # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # Prediction loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    # Debug: print emb shape on first step to verify embed_dim
    if self.global_step == 0 and stage == "train":
        print(f"[CAR-SIGReg] emb shape: {tuple(emb.shape)}", flush=True)

    # CAR-SIGReg losses
    reg_out = self.sigreg(
        emb,
        predictor=self.model.predict,
        ctx_emb=ctx_emb,
        ctx_act=ctx_act,
        use_ctrl_loss=cfg.loss.sigreg.use_ctrl_loss,
    )
    output.update(reg_out)

    # Weighted total loss
    output["loss"] = (
        output["pred_loss"]
        + cfg.loss.sigreg.weight * output["sigreg_loss"]
        + cfg.loss.sigreg.inactive_weight * output["inactive_loss"]
        + cfg.loss.sigreg.ctrl_weight * output["ctrl_loss"]
    )

    # Gate ctrl_loss after warmup when not using use_ctrl_loss flag
    # (controllability already influences basis selection every update_basis_every steps)

    log_dict = {
        f"{stage}/{k}": output[k].detach()
        for k in _LOG_KEYS
        if k in output and torch.is_tensor(output[k])
    }
    self.log_dict(log_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    # CARSIGReg V1 has no learnable parameters; optimizer only covers model.
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=CARSIGReg(**OmegaConf.to_container(cfg.loss.sigreg.kwargs, resolve=True)),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()


if __name__ == "__main__":
    run()
