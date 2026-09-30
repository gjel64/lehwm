import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg, kl_loss
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback
from hjepa import freeze_level0, _flat


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def lehjepa_forward(self, batch, stage, cfg):

    ctx_len = cfg.history_size
    n = cfg.n # number of steps predicted by predictor1
    model = self.model
    freeze_level0(model)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)


    ######## Level 0 : LeWM ########

    with torch.no_grad():
        output = model.encode(batch)
        z0 = output["emb"]            # (B, T, D0)
        act_emb = output["act_emb"]   # (B, T, D0)
        B, T, _ = z0.shape
        assert T == ctx_len + n, f"need {ctx_len + n} frames, got {T}"

        # Rollout AR of P0 over n steps, sliding window of ctx_len frames.
        roll = z0[:, :ctx_len]
        for k in range(n):
            pred = model.predict(roll[:, -ctx_len:], act_emb[:, k:k + ctx_len])
            roll = torch.cat([roll, pred[:, -1:]], dim=1)
        z0_roll = roll[:, -1:]        # (B, 1, D0) : predictions of T-1 frames, given the first ctx_len frames
 
    t, tn = ctx_len - 1, ctx_len - 1 + n


    ######## Level 1 : LeHWM ########

    z1 = _flat(model.elevator, z0)                     # (B, T, D1)
    z1_t, z1_tn = z1[:, t:t + 1], z1[:, tn:tn + 1]     # (B, 1, D1)
 
    m, mu, logvar = model.posterior(z1_t, z1_tn)       # (B, 1, m_dim)
    pred1 = model.predictor1(z1_t, model.m_encoder(m)) # (B, 1, D1)
    pred1 = _flat(model.pred_proj1, pred1)
 
    # Consistency 
    cons_tgt = _flat(model.elevator, z0_roll).detach()
 
    # Skill 
    pi_in = torch.cat([z0[:, t], m[:, 0]], dim=-1)   # (B, D0 + m_dim)
    a_pred = model.pi(pi_in)                                  # (B, n * action_dim)
    a_true = batch["action"][:, t:tn].reshape(B, -1).float()  # (B, n * action_dim)


    ######## Losses ########

    step = getattr(self, "global_step", 0)
    w_kl = cfg.loss.kl.weight * min(1.0, step / max(1, cfg.loss.kl.warmup_steps))
 
    output["pred1_loss"] = (pred1 - z1_tn).pow(2).mean()
    output["sigreg1_loss"] = self.sigreg(z1.transpose(0, 1)) # SIGReg needs (T, B, D)
    output["kl_loss"] = kl_loss(mu, logvar, free_bits=cfg.loss.kl.free_bits)
    output["cons_loss"] = (pred1 - cons_tgt).pow(2).mean()
    output["skill_loss"] = (a_pred - a_true).pow(2).mean()
 
    output["loss"] = (
        output["pred1_loss"]
        + cfg.loss.sigreg.weight * output["sigreg1_loss"]
        + w_kl * output["kl_loss"]
        + cfg.loss.cons.weight * output["cons_loss"]
        + cfg.loss.skill.weight * output["skill_loss"]
    )
 
    logs = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(logs, on_step=True, sync_dist=True)
    self.log(f"{stage}/w_kl", float(w_kl), on_step=True)
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
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    # loads LeWM weights.pt from HF repo and freeze level 0
    lewm_sd = torch.load(Path(cfg.lewm_weights).expanduser(), map_location="cpu", weights_only=False)
    world_model.load_lewm(lewm_sd)
    freeze_level0(world_model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lehjepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
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
    return


if __name__ == "__main__":
    run()
