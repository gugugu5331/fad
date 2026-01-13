from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[2]))

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fsd.data.cached_protocol import ASVspoof19CachedWithAttack
from fsd.models.grl_attack import FakeSpeechDetectionAttackGRL
from fsd.train.base import Dc_loss
from fsd.utils.checkpoint import load_model_state, load_stream_ckpt
from models import DCT2D, RawNet2, ResNet18
from fsd.utils.runtime import configure_runtime
from fsd.utils.spoof_clustering import kmeans, save_cluster_assignments


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def _grl_lambda_schedule(p: float) -> float:
    # DANN schedule: 2/(1+exp(-10p)) - 1
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def run_epoch(
    model: FakeSpeechDetectionAttackGRL,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grl_lambda: float = 1.0,
    attack_loss_weight: float = 0.1,
    alpha: float = 1.0,
    phi: float = 1.0,
    global_step_base: int = 0,
    total_steps: int = 1,
    use_grl_schedule: bool = False,
    frozen_detectors_eval: bool = True,
) -> Tuple[float, float, float]:
    train = optimizer is not None
    model.train(train)
    if train and frozen_detectors_eval:
        for det in model.backbone.detectors:
            if not any(p.requires_grad for p in det.parameters()):
                det.eval()

    cls_weight = torch.tensor([0.3, 0.7], device=device, dtype=torch.float32)
    ce_cls = nn.CrossEntropyLoss(weight=cls_weight)
    ce_adv = nn.CrossEntropyLoss()
    dc_loss_fn = Dc_loss(device)
    tot_loss = tot_acc = 0.0
    tot_adv_correct = 0.0
    tot_adv_n = 0.0
    n = 0

    pbar = tqdm(loader, desc="Train" if train else "Dev", unit="batch", leave=True)
    for step, (wave, lfcc, dct2d, y, attack, utt) in enumerate(pbar):
        step_idx = global_step_base + step
        p = step_idx / max(total_steps - 1, 1)
        curr_grl = grl_lambda * (_grl_lambda_schedule(p) if use_grl_schedule else 1.0)

        non_blocking = device.type == "cuda"
        wave = wave.to(device, non_blocking=non_blocking)
        lfcc = lfcc.to(device, non_blocking=non_blocking)
        dct2d = dct2d.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        attack = attack.to(device, non_blocking=non_blocking)

        out = model(wave, lfcc, dct2d, y, attack, global_step=step_idx, grl_lambda=curr_grl)
        total_loss = ce_cls(out.logits, y)
        certainloss = out.loss_aux.mean()
        dcloss = dc_loss_fn(out.evn_single)
        cls_loss = total_loss + float(alpha) * certainloss + float(phi) * dcloss

        mask = attack >= 0  # bonafide -> -1
        if mask.any():
            # attack_logits: (B, n_streams, num_attacks) -> flatten to (S*n_streams, num_attacks)
            logits_sp = out.attack_logits[mask]
            n_streams = int(logits_sp.shape[1])
            logits_flat = logits_sp.reshape(-1, logits_sp.shape[-1])
            targets = attack[mask].repeat_interleave(n_streams)
            adv_loss = ce_adv(logits_flat, targets)
            adv_pred = logits_flat.argmax(dim=1)
            adv_correct = (adv_pred == targets).float().sum().item()
            adv_n = float(targets.numel())
        else:
            adv_loss = out.attack_logits.sum() * 0.0
            adv_correct = 0.0
            adv_n = 0.0

        loss = cls_loss + float(attack_loss_weight) * adv_loss

        bs = int(y.size(0))
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        tot_loss += loss.item() * bs
        tot_acc += _accuracy(out.logits, y) * bs
        tot_adv_correct += adv_correct
        tot_adv_n += adv_n
        n += bs

        adv_acc_running = (tot_adv_correct / tot_adv_n) if tot_adv_n > 0 else float("nan")
        pbar.set_postfix(
            {
                "loss": f"{tot_loss / max(n, 1):.4f}",
                "acc": f"{100 * tot_acc / max(n, 1):.2f}%",
                "adv_acc": f"{100 * adv_acc_running:.2f}%" if not math.isnan(adv_acc_running) else "-",
                "adv_n": int(tot_adv_n),
                "att_w": f"{attack_loss_weight:.2f}",
                "grl": f"{curr_grl:.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}" if train else "-",
            }
        )

    adv_acc = (tot_adv_correct / tot_adv_n) if tot_adv_n > 0 else float("nan")
    return tot_loss / max(n, 1), tot_acc / max(n, 1), adv_acc


@torch.inference_mode()
def extract_spoof_embeddings(
    model: FakeSpeechDetectionAttackGRL,
    loader: DataLoader,
    *,
    device: torch.device,
    max_samples: int = 0,
) -> Tuple[torch.Tensor, list[str]]:
    model.eval()
    embs = []
    utts: list[str] = []

    n_kept = 0
    for wave, lfcc, dct2d, y, attack, batch_utts in tqdm(loader, desc="Extract embeddings", unit="batch", leave=True):
        non_blocking = device.type == "cuda"
        wave = wave.to(device, non_blocking=non_blocking)
        lfcc = lfcc.to(device, non_blocking=non_blocking)
        dct2d = dct2d.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        attack = attack.to(device, non_blocking=non_blocking)

        out = model(wave, lfcc, dct2d, y, attack, global_step=0, grl_lambda=0.0)
        mask = y == 1
        if not mask.any():
            continue

        batch_emb = out.emb[mask].detach().cpu()
        batch_utts = [u for u, m in zip(batch_utts, mask.detach().cpu().tolist()) if m]
        if max_samples > 0 and (n_kept + len(batch_utts)) > max_samples:
            keep = max_samples - n_kept
            batch_emb = batch_emb[:keep]
            batch_utts = batch_utts[:keep]

        embs.append(batch_emb)
        utts.extend(batch_utts)
        n_kept += len(batch_utts)
        if max_samples > 0 and n_kept >= max_samples:
            break

    if not embs:
        raise RuntimeError("No spoof samples found for clustering")
    return torch.cat(embs, dim=0), utts


def parse_args():
    ap = argparse.ArgumentParser("Train GRL attack-invariant model and cluster spoof embeddings.")
    ap.add_argument("--train_protocol", type=Path, default=Path("/ssd/lxx/code/database/ASVspoof_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"))
    ap.add_argument("--dev_protocol", type=Path, default=Path("/ssd/lxx/code/database/ASVspoof_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"))
    ap.add_argument("--eval_protocol", type=Path, default=Path("/ssd/lxx/code/database/ASVspoof_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"))
    ap.add_argument("--train_cache_dir", type=Path, default=Path("feature_cache/train"))
    ap.add_argument("--dev_cache_dir", type=Path, default=Path("feature_cache/dev"))
    ap.add_argument("--eval_cache_dir", type=Path, default=Path("feature_cache/eval"))
    ap.add_argument(
        "--include_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将 ASVspoof2019 eval (A07-A19) 拼接进训练集，覆盖 A01-A19。",
    )

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu_id", type=int, help="Use only this physical GPU id (sets CUDA_VISIBLE_DEVICES).")
    ap.add_argument("--cpu_affinity", help="Pin process to CPU cores, e.g. '0-15' or '0-7,16-23'.")
    ap.add_argument("--omp_threads", type=int, help="Set OMP_NUM_THREADS.")
    ap.add_argument("--mkl_threads", type=int, help="Set MKL_NUM_THREADS.")
    ap.add_argument("--torch_threads", type=int, help="torch.set_num_threads().")
    ap.add_argument("--torch_interop_threads", type=int, help="torch.set_num_interop_threads().")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--init_ckpt", type=Path, help="Load full model weights before training (for fine-tuning).")
    ap.add_argument("--init_raw_ckpt", type=Path, help="Load per-stream (raw) pretrain checkpoint.")
    ap.add_argument("--init_lfcc_ckpt", type=Path, help="Load per-stream (lfcc) pretrain checkpoint.")
    ap.add_argument("--init_dct2d_ckpt", type=Path, help="Load per-stream (dct2d) pretrain checkpoint.")
    ap.add_argument("--detector_trainable", choices=["all", "none", "last"], default="all",
                    help="Which detector parameters to train: all / none (freeze) / last (partial unfreeze).")
    ap.add_argument("--detector_lr", type=float,
                    help="Optional LR for detector parameters (separate param group).")

    ap.add_argument("--grl_lambda", type=float, default=1.0)
    ap.add_argument("--grl_schedule", action="store_true", help="Use DANN schedule to ramp GRL lambda.")
    ap.add_argument("--attack_loss_weight", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.2)
    # ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--phi", type=float, default=0.2)

    ap.add_argument("--ckpt_dir", type=Path, default=Path("./ckpt/grl_attack_invariant/"))

    ap.add_argument("--cluster_k", type=int, default=0, help="If >0, run k-means on spoof embeddings after training.")
    ap.add_argument("--cluster_split", choices=["train", "dev", "eval"], default="dev", help="Which split to cluster.")
    ap.add_argument("--cluster_iters", type=int, default=25)
    ap.add_argument("--cluster_max_samples", type=int, default=0, help="If >0, limit spoof samples used for clustering.")
    ap.add_argument("--cluster_out", type=Path, default=Path("./clusters/spoof_clusters.txt"))
    return ap.parse_args()


def _set_detectors_trainable(model: FakeSpeechDetectionAttackGRL, mode: str) -> None:
    mode = str(mode).lower().strip()
    if mode not in {"all", "none", "last"}:
        raise ValueError(f"Unknown detector_trainable mode: {mode}")

    for det in model.backbone.detectors:
        for p in det.parameters():
            p.requires_grad = (mode == "all")

    if mode != "last":
        return

    for det in model.backbone.detectors:
        for p in det.parameters():
            p.requires_grad = False

        if isinstance(det, RawNet2.Model):
            patterns = (
                "block4",
                "block5",
                "bn_before_gru",
                "gru",
                "fc1_gru",
                "fc2_gru",
                "fc_attention4",
                "fc_attention5",
            )
        elif isinstance(det, ResNet18.Model):
            patterns = ("resnet18_model.layer4", "resnet18_model.fc")
        elif isinstance(det, DCT2D.Model):
            patterns = ("rnn", "attention_layer", "mlp", "last_layer")
        else:
            patterns = ()

        matched = 0
        for name, p in det.named_parameters():
            if any(pat in name for pat in patterns):
                p.requires_grad = True
                matched += 1
        if matched == 0 and patterns:
            print(f"[warn] detector_trainable=last: no params matched for {type(det).__name__}")


def main():
    args = parse_args()

    runtime = configure_runtime(
        gpu_id=args.gpu_id,
        cpu_affinity=args.cpu_affinity,
        omp_threads=args.omp_threads,
        mkl_threads=args.mkl_threads,
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
    )
    if runtime.cpu_affinity is not None:
        print(f"[info] cpu_affinity={','.join(map(str, runtime.cpu_affinity))}")
    if args.gpu_id is not None and str(args.device).startswith("cuda"):
        args.device = "cuda:0" if ":" in str(args.device) else "cuda"

    set_seed(args.seed)

    device = torch.device(args.device)
    pin_memory = device.type == "cuda"

    # 固定攻击类型映射：A01~A19（train/dev 只包含 A01~A06，eval 包含 A07~A19）
    attack_to_idx = {f"A{i:02d}": i - 1 for i in range(1, 20)}
    num_attacks = len(attack_to_idx)

    train_ds = ASVspoof19CachedWithAttack(args.train_protocol, args.train_cache_dir, attack_to_idx=attack_to_idx)
    dev_ds = ASVspoof19CachedWithAttack(args.dev_protocol, args.dev_cache_dir, attack_to_idx=attack_to_idx)
    eval_ds = None
    if args.include_eval:
        eval_ds = ASVspoof19CachedWithAttack(args.eval_protocol, args.eval_cache_dir, attack_to_idx=attack_to_idx)

    if eval_ds is not None:
        from torch.utils.data import ConcatDataset

        train_ds_for_loader = ConcatDataset([train_ds, eval_ds])
        print(
            f"[info] attacks={num_attacks}  train={len(train_ds)} + eval={len(eval_ds)} -> {len(train_ds_for_loader)}  dev={len(dev_ds)}"
        )
    else:
        train_ds_for_loader = train_ds
        print(f"[info] attacks={num_attacks}  train={len(train_ds)}  dev={len(dev_ds)}")

    train_loader = DataLoader(
        train_ds_for_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = FakeSpeechDetectionAttackGRL(
        detectors=[RawNet2.Model(), ResNet18.Model(), DCT2D.Model()],
        feature_out=256,
        fixed_size=None,
        device=args.device,
        num_attacks=num_attacks,
        grl_lambda=args.grl_lambda,
    ).to(device)

    if args.init_ckpt is not None:
        missing, unexpected = load_model_state(model, args.init_ckpt, strict=False)
        print(f"[info] loaded init_ckpt={args.init_ckpt} (missing={len(missing)} unexpected={len(unexpected)})")

    stream_map = {"raw": 0, "lfcc": 1, "dct2d": 2}
    for stream, idx in stream_map.items():
        ckpt_path = getattr(args, f"init_{stream}_ckpt", None)
        if not ckpt_path:
            continue
        det_state, proj_state = load_stream_ckpt(ckpt_path)
        missing, unexpected = model.backbone.detectors[idx].load_state_dict(det_state, strict=False)
        if missing or unexpected:
            print(f"[warn] init_{stream}_ckpt detector load: missing={len(missing)} unexpected={len(unexpected)}")
        if proj_state is not None:
            missing, unexpected = model.backbone.proj[idx].load_state_dict(proj_state, strict=False)
            if missing or unexpected:
                print(f"[warn] init_{stream}_ckpt proj load: missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[info] loaded init_{stream}_ckpt={ckpt_path}")

    _set_detectors_trainable(model, args.detector_trainable)

    if args.detector_lr is not None:
        det_params = [p for p in model.backbone.detectors.parameters() if p.requires_grad]
        det_ids = {id(p) for p in det_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in det_ids]
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": float(args.lr)})
        if det_params:
            param_groups.append({"params": det_params, "lr": float(args.detector_lr)})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(len(train_loader), 1)

    best_dev_acc = -1.0
    best_ckpt: Optional[Path] = None
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        global_step_base = (epoch - 1) * max(len(train_loader), 1)
        tr_loss, tr_acc, _ = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            grl_lambda=args.grl_lambda,
            attack_loss_weight=args.attack_loss_weight,
            alpha=args.alpha,
            phi=args.phi,
            global_step_base=global_step_base,
            total_steps=total_steps,
            use_grl_schedule=args.grl_schedule,
        )
        dv_loss, dv_acc, _ = run_epoch(
            model,
            dev_loader,
            device=device,
            optimizer=None,
            grl_lambda=args.grl_lambda,
            attack_loss_weight=args.attack_loss_weight,
            alpha=args.alpha,
            phi=args.phi,
            global_step_base=global_step_base,
            total_steps=total_steps,
            use_grl_schedule=args.grl_schedule,
        )

        print(
            f"[Epoch {epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
            f"dev loss {dv_loss:.4f} acc {dv_acc*100:.2f}%"
        )
        if dv_acc > best_dev_acc:
            best_dev_acc = dv_acc
            ckpt_path = args.ckpt_dir / f"best_{epoch:03d}_{dv_acc:.4f}.pth"
            torch.save(model.state_dict(), ckpt_path)
            best_ckpt = ckpt_path
            print("[info] saved best checkpoint:", ckpt_path)

    if args.cluster_k > 0:
        if best_ckpt is not None and best_ckpt.exists():
            try:
                state = torch.load(best_ckpt, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(state)
            print("[info] loaded best checkpoint for clustering:", best_ckpt)

        if args.cluster_split == "train":
            cluster_ds = train_ds
        elif args.cluster_split == "eval":
            if eval_ds is None:
                raise ValueError("cluster_split=eval 但当前 --no-include_eval；请加上 --include_eval 并提供 eval cache。")
            cluster_ds = eval_ds
        else:
            cluster_ds = dev_ds
        cluster_loader = DataLoader(
            cluster_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        emb, utts = extract_spoof_embeddings(model, cluster_loader, device=device, max_samples=args.cluster_max_samples)
        labels, _ = kmeans(emb, args.cluster_k, num_iters=args.cluster_iters, seed=args.seed)
        save_cluster_assignments(args.cluster_out, utts, labels)
        print(f"[info] saved spoof clusters: {args.cluster_out}")


if __name__ == "__main__":
    main()
