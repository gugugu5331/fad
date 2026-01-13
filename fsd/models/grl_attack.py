from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .grl import grad_reverse
from .fakespeech_detection import FakeSpeechDetection


class AttackDiscriminator(nn.Module):
    def __init__(self, in_dim: int, num_attacks: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_attacks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class GRLOutput:
    logits: torch.Tensor
    emb: torch.Tensor
    evn_single: torch.Tensor
    uncertain: torch.Tensor
    loss_aux: torch.Tensor
    stream_embs: torch.Tensor
    attack_logits: torch.Tensor  # (B, n_streams, num_attacks)


class FakeSpeechDetectionAttackGRL(nn.Module):
    """FakeSpeechDetection + GRL attack-type adversary.

    目标：学习对攻击类型 (Axx) 不敏感的伪造语音表征，便于后续对 spoof 样本做聚类。
    """

    def __init__(
        self,
        *,
        detectors: Optional[Sequence[nn.Module]] = None,
        feature_out: int = 256,
        fixed_size: Optional[int] = None,
        device: Optional[str] = None,
        num_attacks: int,
        grl_lambda: float = 1.0,
        attack_hidden: int = 256,
        attack_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = FakeSpeechDetection(
            detectors=detectors,
            feature_out=feature_out,
            fixed_size=fixed_size,
            device=device,
        )
        self.num_attacks = int(num_attacks)
        self.grl_lambda = float(grl_lambda)
        n_streams = len(self.backbone.detectors)
        if n_streams <= 0:
            raise ValueError("FakeSpeechDetectionAttackGRL requires at least one detector stream")
        self.num_streams = int(n_streams)
        self.attack_discs = nn.ModuleList(
            [
                AttackDiscriminator(
                    in_dim=feature_out,
                    num_attacks=self.num_attacks,
                    hidden=int(attack_hidden),
                    dropout=float(attack_dropout),
                )
                for _ in range(n_streams)
            ]
        )

    def forward(
        self,
        wave: torch.Tensor,
        lfcc: torch.Tensor,
        dct2d: torch.Tensor,
        label: torch.Tensor,
        attack: torch.Tensor,
        *,
        global_step: int = 0,
        grl_lambda: Optional[float] = None,
    ) -> GRLOutput:
        logits, emb, evn_single, uncertain, loss_aux, stream_embs = self.backbone(
            [wave, lfcc, dct2d], label, global_step, return_streams=True
        )
        lambd = self.grl_lambda if grl_lambda is None else float(grl_lambda)
        attack_logits = torch.stack(
            [disc(grad_reverse(stream_embs[:, i, :], lambd)) for i, disc in enumerate(self.attack_discs)],
            dim=1,
        )
        return GRLOutput(
            logits=logits,
            emb=emb,
            evn_single=evn_single,
            uncertain=uncertain,
            loss_aux=loss_aux,
            stream_embs=stream_embs,
            attack_logits=attack_logits,
        )
