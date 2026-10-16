"""Convertit un LeWM du papier H-JEPA (cube_lewm_object.ckpt) en state_dict pour notre niveau 0.

Le checkpoint H-JEPA est un objet picklé (models.hjepa.HJEPA à 1 niveau) : on le charge avec le
code de third_party/H-JEPA, on renomme les clés vers la disposition de jepa.JEPA, puis on vérifie
que notre modèle donne exactement les mêmes embeddings / prédictions que le leur.

  python convert_lewm.py ~/hjepa_home/ckpts/cube/cube_lewm/seed42/cube_lewm_object.ckpt
  -> ~/hjepa_home/ckpts/cube/cube_lewm/seed42/lewm_state_dict.pt   (cfg.lewm_weights de train.py)
"""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
HJEPA_CODE = ROOT / "third_party/H-JEPA/h_jepa"

# préfixes H-JEPA (models.jepa.JEPA) -> nos modules (jepa.JEPA)
RENAME = {
    "encoder.encoder.": "encoder.",
    "encoder.projector.": "projector.",
    "predictor.predictor.": "predictor.",
    "predictor.projector.": "pred_proj.",
    "action_embed.": "action_encoder.",
}


def load_hjepa_level1(path):
    sys.path.insert(0, str(HJEPA_CODE))  # pour dépickler models.*, loss
    obj = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    assert obj.num_levels == 1, f"attendu un LeWM à 1 niveau, reçu {obj.num_levels} niveaux"
    return obj.jepas[0].eval()


def convert(state_dict):
    out = {}
    for k, v in state_dict.items():
        # file d'actions latentes (niveaux > 1 seulement) ; mask_token : jamais utilisé sans masquage
        if k in ("latent_action_queue", "encoder.encoder.embeddings.mask_token"):
            continue
        new = next((dst + k[len(src):] for src, dst in RENAME.items() if k.startswith(src)), None)
        assert new is not None, f"clé non gérée : {k}"
        out[new] = v
    return out


def build_ours():
    import hydra
    import stable_pretraining  # noqa: F401  enregistre ${eval:...} sans vérifier -> avant hjepa
    import hjepa  # noqa: F401

    cfg = OmegaConf.load(ROOT / "config/train/lewm.yaml")
    cfg.model = OmegaConf.load(ROOT / "config/train/model/lewm.yaml")
    cfg.action_dim = 25  # frameskip 5 x action 5D
    return hydra.utils.instantiate(cfg.model)


@torch.no_grad()
def check(theirs, ours):
    g = torch.Generator().manual_seed(0)
    pixels = torch.randn(2, 3, 3, 224, 224, generator=g)
    action = torch.randn(2, 3, 25, generator=g)
    a = theirs.encode({"pixels": pixels.clone(), "action": action.clone()})
    b = ours.encode({"pixels": pixels.clone(), "action": action.clone()})
    errs = {
        "emb": (a["embed_0"] - b["emb"]).abs().max().item(),
        "act_emb": (a["action_0"] - b["act_emb"]).abs().max().item(),
        "pred": (theirs.predict(a["embed_0"], a["action_0"]) - ours.predict(b["emb"], b["act_emb"])).abs().max().item(),
    }
    print("écart max :", {k: f"{v:.2e}" for k, v in errs.items()})
    assert all(v < 1e-4 for v in errs.values()), "le modèle converti ne reproduit pas le LeWM H-JEPA"


def main():
    src = Path(sys.argv[1]).expanduser()
    # notre modèle d'abord : une fois h_jepa/ dans sys.path, ses utils.py / data.py / loss.py
    # masquent des modules importés par stable_pretraining
    ours = build_ours().eval()
    theirs = load_hjepa_level1(src)
    sd = convert(theirs.state_dict())

    ours.load_lewm(sd)  # vérifie que seuls les modules du niveau 1 manquent
    check(theirs, ours)

    dst = src.parent / "lewm_state_dict.pt"
    torch.save(sd, dst)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
