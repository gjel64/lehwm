from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

def plot_losses(run_dir):
    """Trace chaque loss (train par pas, val par epoch) depuis metrics.csv -> loss.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(Path(run_dir) / "metrics.csv")
    # "fit/<name>loss" (par pas) et "validate/<name>loss_epoch" (par epoch) -> {name: {stage: colonne}}
    curves = {}
    for c in df.columns:
        if "/" in c and c.removesuffix("_epoch").endswith("loss"):
            stage, name = c.removesuffix("_epoch").split("/", 1)
            curves.setdefault(name, {})[stage] = c
    if not curves:
        return
    fig, axes = plt.subplots(1, len(curves), figsize=(3.6 * len(curves), 3.2), squeeze=False)
    for ax, (name, stages) in zip(axes[0], sorted(curves.items())):
        for stage, col in sorted(stages.items()):  # fit/train avant validate/val
            d = df[["step", col]].dropna()
            if col.endswith("_epoch"):
                ax.plot(d["step"], d[col], "o-", ms=4, lw=2, color="#eb6834", label=stage)
            else:
                ax.plot(d["step"], d[col], lw=0.5, alpha=0.3, color="#2a78d6")
                ax.plot(d["step"], d[col].ewm(alpha=0.05).mean(), lw=2, color="#2a78d6", label=stage)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel("step")
        ax.grid(color="#e4e3df", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "loss.png", dpi=120)
    plt.close(fig)


class RunDirCallback(Callback):
    """A chaque fin d'epoch : sauve le modèle dans run_dir/weights.pt et retrace loss.png."""

    def __init__(self, run_dir, cfg):
        super().__init__()
        self.run_dir = Path(run_dir).resolve()
        self.cfg = cfg

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        from stable_worldmodel.wm.utils import save_pretrained
        # chemin absolu : save_pretrained l'utilise tel quel au lieu de $STABLEWM_HOME
        save_pretrained(pl_module.model, run_name=str(self.run_dir), config=self.cfg, filename="weights.pt")
        for logger in trainer.loggers:
            logger.save()
        plot_losses(self.run_dir)
