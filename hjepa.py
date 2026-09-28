
import torch
from torch import nn
from omegaconf import OmegaConf
 
from jepa import JEPA
from module import kl_loss
 
# À appeler AVANT de charger la config Hydra (sinon ${eval:...} ne résout pas)
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)
 
 
LEVEL0 = ("encoder", "projector", "predictor", "pred_proj", "action_encoder")
LEVEL1 = ("elevator", "posterior", "m_encoder", "predictor1", "pred_proj1", "pi")
 
 
class HJEPA(JEPA):
    """LeWM (level 0, frozen) + level 1's modules.
    jepa.JEPA n'accepte pas les nouveaux modules en kwargs -> sous-classe."""
 
    def __init__(self, *, elevator, posterior, m_encoder, predictor1, pred_proj1, pi, **lewm_kwargs):
        super().__init__(**lewm_kwargs)
        self.elevator = elevator
        self.posterior = posterior
        self.m_encoder = m_encoder
        self.predictor1 = predictor1
        self.pred_proj1 = pred_proj1
        self.pi = pi
 
    def load_lewm(self, state_dict):
        """Charge les poids LeWM ; seuls les modules du niveau 1 doivent manquer."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        assert not unexpected, f"clés inattendues : {unexpected[:5]}"
        bad = [k for k in missing if k.split(".")[0] not in LEVEL1]
        assert not bad, f"poids LeWM manquants : {bad[:5]}"
 
 
def freeze_level0(model):
    """À refaire à chaque step : model.train() remet les BN gelées en mode train."""
    for name in LEVEL0:
        mod = getattr(model, name)
        mod.eval()
        mod.requires_grad_(False)
 
 
def _flat(mod, x):
    """Applique un MLP à BatchNorm sur (B, T, D) en passant par (B*T, D)."""
    B, T, _ = x.shape
    return mod(x.reshape(B * T, -1)).reshape(B, T, -1)
 