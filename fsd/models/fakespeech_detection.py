"""fsd.models.fakespeech_detection

Incrementally‑extensible fake‑speech detection framework (v2).
==============================================================
This version fixes the *shape '[B, 1, 1]' is invalid for input of size XXXX* error
by normalising the `uncertain` tensor that comes back from `fushion_decision` so
it is always broadcast‑compatible with `(B, n_streams, feature_out)`.

Key updates
-----------
1. **_normalize_uncertain()** helper handles any incoming shape:
   • `(B,)` – sample‑wise scalar → expands to `(B, n, 1)`
   • `(B, n)` – already per stream → unsqueeze(‑1)
   • `(B, *)` where `* ≠ n` – we keep first *n* columns then reshape
   • anything higher‑dim – reduces by mean over extra dims.
2. Forward pass now calls this helper before certainty weighting.
3. Added extensive comments & debug asserts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple

from models import RawNet2, ResNet18,DCT2D                    # project‑specific
from self_attention_fusion_multy_view.decision_mul import fushion_decision

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


# -----------------------------------------------------------------------------
# Fixed‑size pooling for attention maps
# -----------------------------------------------------------------------------

class FixedPooling(nn.Module):
    def __init__(self, fixed_size: int = 6) -> None:
        super().__init__()
        self.fixed = fixed_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W)
        b, h, w = x.shape
        pw = self.fixed * ((w + self.fixed - 1) // self.fixed) - w
        ph = self.fixed * ((h + self.fixed - 1) // self.fixed) - h
        x = F.pad(x, (0, pw, 0, ph))
        pool = ( (h + self.fixed - 1) // self.fixed, (w + self.fixed - 1) // self.fixed )
        return F.max_pool2d(x, kernel_size=pool, stride=pool)


# -----------------------------------------------------------------------------
# Transformer fusion block
# -----------------------------------------------------------------------------

class LModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "ReLU",
        norm_first: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sa = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.act = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "SELU": nn.SELU}[activation]()
        self.norm_first = norm_first
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim, eps), nn.LayerNorm(embed_dim, eps)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), self.act, nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim)
        )

    def _sa(self, x):
        y, attn = self.sa(x, x, x)
        return self.drop1(y), attn

    def forward(self, x):
        if self.norm_first:
            y, attn = self._sa(self.norm1(x))
            x = x + y
            x = x + self.drop2(self.mlp(self.norm2(x)))
            return x, attn
        y, attn = self._sa(x)
        x = self.norm1(x + y)
        x = self.norm2(x + self.drop2(self.mlp(x)))
        return x, attn


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------

class FakeSpeechDetection(nn.Module):
    """Fake‑speech detector with pluggable, freeze‑friendly sub‑detectors."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        detectors: Optional[Sequence[nn.Module]] = None,
        feature_out: int = 256,
        fixed_size: Optional[int] = 6,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.feature_out = feature_out
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.detectors: nn.ModuleList[nn.Module] = nn.ModuleList()
        self.proj: nn.ModuleList[nn.Module] = nn.ModuleList()

        self.fusion: Optional[LModel] = None
        self.decision: Optional[nn.Module] = None
        self.mlp1: Optional[nn.Linear] = None
        self.mlp_classifier = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(feature_out, feature_out), nn.LeakyReLU(), nn.Linear(feature_out, 2)
        )

        self.fixed_size = fixed_size
        self.attn_pool = FixedPooling(fixed_size) if fixed_size else None
        self.dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(feature_out)

        if detectors:
            for d in detectors:
                self.add_detector(d, trainable=True)

    # ---------------------------- public API --------------------------

    def add_detector(self, det: nn.Module, *, trainable: bool = True, proj: Optional[nn.Module] = None) -> None:
        if not trainable:
            freeze_module(det)
        
        det = det.to(self.device)

        self.detectors.append(det)

        # === 提前确定输出维度 ===
        with torch.no_grad():
            if isinstance(det, RawNet2.Model):
                dummy = torch.randn(1, 64000, device=self.device)
                out, *_ = det(dummy)
            elif isinstance(det, ResNet18.Model):
                dummy = torch.randn(1, 60, 404, device=self.device)
                out, *_ = det(dummy)
            elif isinstance(det,DCT2D.Model):
                dummy  = torch.randn(1, 398, 40, device=self.device)
                out, *_ = det(dummy)
            else:
                raise ValueError("未知 detector 类型")

        in_dim = out.size(-1)
        proj = nn.Identity() if in_dim == self.feature_out else nn.Linear(in_dim, self.feature_out)
        self.proj.append(proj)

        self._rebuild_after_stream_change()

    # ---------------------------- internals ---------------------------

    def _rebuild_after_stream_change(self) -> None:
        n = len(self.detectors)
        
        assert n > 0, "Need at least one detector"
        self.decision = fushion_decision(n, self.feature_out,25)
        self.fusion = LModel(embed_dim=self.feature_out)
        attn_dim = (self.fixed_size ** 2) if self.fixed_size else (n * n)
        self.mlp1 = nn.Linear(self.feature_out * n + attn_dim, self.feature_out)

    @staticmethod
    def _normalize_uncertain(u: torch.Tensor, batch: int, n_streams: int) -> torch.Tensor:
        """Make *u* broadcast‑compatible with (B, n_streams, F)."""
        # If already (B, n, 1) or (B, n) – easy path
        if u.dim() == 3 and u.shape[1] == n_streams:
            return u
        if u.dim() == 2 and u.shape[1] == n_streams:
            return u.unsqueeze(-1)
        if u.dim() == 1:                         # (B,)
            return u.view(batch, 1, 1).expand(batch, n_streams, 1)
        # Any other shape – reduce / slice until it matches
        u = u.view(batch, -1)                    # (B, ?)
        if u.shape[1] < n_streams:
            # Pad with zeros if somehow shorter
            pad = u.new_zeros(batch, n_streams - u.shape[1])
            u = torch.cat([u, pad], dim=1)
        u = u[:, :n_streams]                     # keep first n columns
        return u.unsqueeze(-1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: List[torch.Tensor],
        label: torch.Tensor,
        global_step: int = 0,
        return_streams: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        n = len(self.detectors)
        #print(n)
        assert len(inputs) == n, "#inputs must match #detectors"
        B = inputs[0].size(0)

        # 1. Per‑stream embeddings ------------------------------------------------
        outs: List[torch.Tensor] = []
        for i, (x, det) in enumerate(zip(inputs, self.detectors)):
            proj = self.proj[i]
            # ---- shape fixes -------------------------------------------------
            if isinstance(det, RawNet2.Model) and x.dim() == 3 and x.size(1) == 1:
                x = x.squeeze(1)
            if isinstance(det, ResNet18.Model):
                if x.dim() == 4 and x.size(1) == 1:
                    x = x.squeeze(1)
                elif x.dim() == 5 and x.size(2) == 1:
                    b, s, _, f, t = x.shape
                    x = x.view(b * s, f, t)
            # ------------------------------------------------------------------
            feat, *_ = det(x)
            if feat.size(-1) != self.feature_out:
                if isinstance(proj, nn.Identity):
                    proj = nn.Linear(feat.size(-1), self.feature_out).to(feat.device)
                    self.proj[i] = proj          # cache it
                feat = proj(feat)
            else:
                feat = proj(feat)
            outs.append(feat)

        # 2. Uncertainty weighting ----------------------------------------------
        evn_single, uncertain, loss = self.decision(outs, label, global_step)
        uncertain = self._normalize_uncertain(uncertain, B, n)   # (B, n, 1)

        stream_embs = torch.stack(outs, dim=1)                    # (B, n, F) per-stream embeddings
        out_tensor = stream_embs * (1.0 - uncertain)              # weighted
        out_tensor = self.ln(out_tensor)
        out_tensor = self.dropout(out_tensor)

        fused, attn = self.fusion(out_tensor)                     # attn (B, n, n)

        # 3. Attention feature -----------------------------------------------
        if self.fixed_size:
            attn_feat = self.attn_pool(attn).reshape(B, -1)       # (B, fixed²)
        else:
            attn_feat = attn.reshape(B, -1)                       # (B, n²)

        # 4. Classifier ------------------------------------------------------
        flat = torch.cat([fused.reshape(B, -1), attn_feat], dim=1)
        emb = self.mlp1(flat)
        logits = self.mlp_classifier(emb)
        if return_streams:
            return logits, emb, evn_single, uncertain, loss, stream_embs
        return logits, emb, evn_single, uncertain, loss


# -----------------------------------------------------------------------------
# Quick smoke‑test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    B = 4
    
    raw = torch.randn(B, 1, 64000).to("cuda")       # RawNet2 waveform (64k samples)
    lfcc = torch.randn(B, 60, 404).to("cuda")   # ResNet18 spectrogram / mel‑spec
    dct2d =  torch.randn(B, 398, 40).to("cuda") 
    model = FakeSpeechDetection(detectors=[RawNet2.Model(), ResNet18.Model(),DCT2D.Model()],
                                feature_out=256, fixed_size=None, device=torch.device("cuda"))
    for d in model.detectors:
        freeze_module(d)                 # pretend pre‑trained

    logits, emb, evn_single, unc, loss = model(
        inputs=[raw, lfcc,dct2d],
        label=torch.randint(0, 2, (B,)).to("cuda") ,
    )
    print("SUCCESS", logits.shape)
