import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# -----------------------------------------------------------------------------
# Utility: KL‑divergence and CE‑like evidential loss (unchanged)
# -----------------------------------------------------------------------------

def _kl_div(alpha: torch.Tensor, c: int) -> torch.Tensor:
    beta = torch.ones((1, c), device=alpha.device)
    S_alpha = torch.sum(alpha, 1, keepdim=True)
    S_beta = torch.sum(beta, 1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), 1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), 1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    return torch.sum((alpha - beta) * (dg1 - dg0), 1, keepdim=True) + lnB + lnB_uni


def _ce_loss(y: torch.Tensor, alpha: torch.Tensor, c: int, step: int, anneal: int) -> torch.Tensor:
    S = torch.sum(alpha, 1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(y, num_classes=c).float()
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), 1, keepdim=True)
    coef = min(1.0, step / anneal)
    alp = E * (1 - label) + 1
    return A + coef * _kl_div(alp, c)

# -----------------------------------------------------------------------------
# Core modules
# -----------------------------------------------------------------------------

class Classifier(nn.Module):
    """Small MLP that converts *feature_out* → evidential 2‑class output."""

    def __init__(self, feature_out: int):
        super().__init__()
        hid = feature_out // 2
        self.net = nn.Sequential(
            nn.Linear(feature_out, feature_out),
            nn.ReLU(),
            nn.Linear(feature_out, hid),
            nn.ReLU(),
        )
        self.fc_evd = nn.Linear(hid, 2)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.softplus(self.fc_evd(x))


class fushion_decision(nn.Module):
    """Evidential fusion + uncertainty for an arbitrary number of views.

    Parameters
    ----------
    views : int  –  expected number of input streams.
    feature_out : int – embedding dimension of each stream.
    lambda_epochs : int – KL annealing schedule.
    """

    def __init__(self, views: int, feature_out: int, lambda_epochs: int = 50):
        super().__init__()
        self.views = views
        self.lambda_epochs = lambda_epochs
        self.classifiers = nn.ModuleList([Classifier(feature_out) for _ in range(views)])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        feats: List[torch.Tensor],   # list length = n_views
        labels: torch.Tensor,
        global_step: int,
    ):
        evidence: Dict[int, torch.Tensor] = {}
        loss = 0.0
        alpha: Dict[int, torch.Tensor] = {}

        for v, x in enumerate(feats):
            #print(x)
            evidence[v] = self.classifiers[v](x)
            alpha[v] = evidence[v] + 1.0
            loss += _ce_loss(labels, alpha[v], c=2, step=global_step, anneal=self.lambda_epochs)
            
        # (B, views, 1) uncertainty coefficient
        uncertain = self._ds_uncertain(alpha)
        return evidence, uncertain, torch.mean(loss)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ds_uncertain(self, alpha: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Compute per‑view uncertainty U = 2 / sum(alpha)."""
        u_list = []
        for v in range(len(alpha)):
            S_v = torch.sum(alpha[v], 1, keepdim=True)
            u_list.append(2.0 / S_v)
        return torch.stack(u_list, 1)  # (B, views, 1)
