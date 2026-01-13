from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fsd.data.ssl import ASVspoof19TrainDataset
from fsd.train.base import build_arg_parser
from fsd.utils.checkpoint import load_stream_ckpt
from fsd.utils.runtime import configure_runtime
from models import DCT2D, RawNet2, ResNet18


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


class StreamPretrainModel(nn.Module):
    def __init__(self, *, detector: nn.Module, proj: nn.Module, feature_out: int):
        super().__init__()
        self.detector = detector
        self.proj = proj
        self.head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(feature_out, feature_out),
            nn.LeakyReLU(),
            nn.Linear(feature_out, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, *_ = self.detector(x)
        feat = self.proj(feat)
        logits = self.head(feat)
        return logits, feat


def _infer_proj(detector: nn.Module, sample: torch.Tensor, *, device: torch.device, feature_out: int) -> nn.Module:
    detector = detector.to(device)
    with torch.no_grad():
        feat, *_ = detector(sample.to(device))
    in_dim = int(feat.size(-1))
    if in_dim == int(feature_out):
        return nn.Identity()
    return nn.Linear(in_dim, int(feature_out))


def parse_args():
    parser = build_arg_parser("Pretrain a single stream encoder (raw / lfcc / dct2d).")
    parser.add_argument("--stream", choices=["raw", "lfcc", "dct2d"], required=True)
    parser.add_argument("--feature_out", type=int, default=256)
    parser.add_argument("--init_stream_ckpt", type=Path, help="Optional: resume stream pretrain from a previous ckpt.")

    parser.set_defaults(cache_features=False)
    parser.set_defaults(feature_device="cpu")
    parser.set_defaults(use_dp=False)
    parser.set_defaults(ckpt_dir=Path("./ckpt/pretrain_stream/"))
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = configure_runtime(
        gpu_id=getattr(args, "gpu_id", None),
        cpu_affinity=getattr(args, "cpu_affinity", None),
        omp_threads=getattr(args, "omp_threads", None),
        mkl_threads=getattr(args, "mkl_threads", None),
        torch_threads=getattr(args, "torch_threads", None),
        torch_interop_threads=getattr(args, "torch_interop_threads", None),
    )
    if runtime.cpu_affinity is not None:
        print(f"[info] cpu_affinity={','.join(map(str, runtime.cpu_affinity))}")

    set_seed(args.seed)
    device = torch.device(args.device)

    train_dst = ASVspoof19TrainDataset(
        args.train_protocol,
        args.train_wave_dir,
        pad_to=args.pad_to,
        args=args,
        algo=args.algo,
        device=args.feature_device,
        disk_cache_dir=args.train_cache_dir,
    )
    dev_dst = ASVspoof19TrainDataset(
        args.dev_protocol,
        args.dev_wave_dir,
        pad_to=args.pad_to,
        args=args,
        algo=args.algo,
        device=args.feature_device,
        disk_cache_dir=args.dev_cache_dir,
    )

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_dst,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(int(args.num_workers), 0),
        pin_memory=pin_mem,
    )
    dev_loader = DataLoader(
        dev_dst,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(int(args.num_workers), 0),
        pin_memory=pin_mem,
    )

    if args.stream == "raw":
        detector = RawNet2.Model()
        sample_x, *_ = train_dst[0]
        sample_x = sample_x.unsqueeze(0)  # (1, T)
    elif args.stream == "lfcc":
        detector = ResNet18.Model()
        _, sample_x, *_ = train_dst[0]
        sample_x = sample_x.unsqueeze(0)  # (1, 60, T)
    elif args.stream == "dct2d":
        detector = DCT2D.Model()
        *_, sample_x, _ = train_dst[0]
        sample_x = sample_x.unsqueeze(0)  # (1, frames, 40)
    else:
        raise ValueError(f"Unknown stream: {args.stream}")

    proj = _infer_proj(detector, sample_x, device=device, feature_out=args.feature_out).to(device)
    model = StreamPretrainModel(detector=detector.to(device), proj=proj, feature_out=args.feature_out).to(device)

    if args.init_stream_ckpt is not None and Path(args.init_stream_ckpt).exists():
        det_state, proj_state = load_stream_ckpt(args.init_stream_ckpt)
        missing, unexpected = model.detector.load_state_dict(det_state, strict=False)
        if missing or unexpected:
            print(f"[warn] detector ckpt load: missing={len(missing)} unexpected={len(unexpected)}")
        if proj_state is not None:
            missing, unexpected = model.proj.load_state_dict(proj_state, strict=False)
            if missing or unexpected:
                print(f"[warn] proj ckpt load: missing={len(missing)} unexpected={len(unexpected)}")

    cls_weight = torch.tensor([0.3, 0.7], device=device, dtype=torch.float32)
    ce = nn.CrossEntropyLoss(weight=cls_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.ckpt_dir) / args.stream
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dev = 0.0
    best_path: Optional[Path] = None

    for epoch in range(1, int(args.epochs) + 1):
        # ---------------- train ----------------
        model.train()
        tr_loss = tr_acc = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Train[{args.stream}] Ep{epoch:03d}", unit="batch", leave=True)
        for wave, lfcc, dct2d, y in pbar:
            if args.stream == "raw":
                x = wave
            elif args.stream == "lfcc":
                x = lfcc
            else:
                x = dct2d

            non_blocking = device.type == "cuda"
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)

            logits, _ = model(x)
            loss = ce(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = int(y.size(0))
            tr_loss += loss.item() * bs
            tr_acc += accuracy(logits, y) * bs
            n += bs
            pbar.set_postfix({"loss": f"{tr_loss / n:.4f}", "acc": f"{100 * tr_acc / n:.2f}%"})

        # ---------------- dev ----------------
        model.eval()
        dv_loss = dv_acc = 0.0
        n = 0
        with torch.no_grad():
            for wave, lfcc, dct2d, y in tqdm(dev_loader, desc=f"Dev[{args.stream}] Ep{epoch:03d}", unit="batch", leave=True):
                if args.stream == "raw":
                    x = wave
                elif args.stream == "lfcc":
                    x = lfcc
                else:
                    x = dct2d

                non_blocking = device.type == "cuda"
                x = x.to(device, non_blocking=non_blocking)
                y = y.to(device, non_blocking=non_blocking)

                logits, _ = model(x)
                loss = ce(logits, y)
                bs = int(y.size(0))
                dv_loss += loss.item() * bs
                dv_acc += accuracy(logits, y) * bs
                n += bs

        dv_loss = dv_loss / max(n, 1)
        dv_acc = dv_acc / max(n, 1)
        print(f"[Epoch {epoch:03d}] dev loss {dv_loss:.4f} | acc {100 * dv_acc:.2f}%")

        if dv_acc > best_dev:
            best_dev = dv_acc
            best_path = out_dir / f"best_{epoch:03d}_{dv_acc:.4f}.pth"
            torch.save(
                {
                    "stream": args.stream,
                    "feature_out": int(args.feature_out),
                    "detector": model.detector.state_dict(),
                    "proj": model.proj.state_dict() if not isinstance(model.proj, nn.Identity) else None,
                    "epoch": int(epoch),
                    "dev_acc": float(dv_acc),
                },
                best_path,
            )
            print(f"  ↳ saved best stream ckpt: {best_path}")

    if best_path is not None:
        print(f"[done] best dev acc {best_dev:.4f} @ {best_path}")


if __name__ == "__main__":
    main()

