"""fsd.train.base – Training / validation / evaluation for FakeSpeechDetection.

This script wires together:
  • FakeSpeechDetection (incremental, in fakespeech_detection.py)
  • FakeSpeechDataset       – training / dev loader  (fsd_dataset.py)
  • ASVspoofEvalDataset     – evaluation loader      (asvspoof_eval_dataset.py)

Run example
-----------
python main.py \
  --train_protocol  protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --train_wave_dir  data/LA/flac \
  --dev_protocol    protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --dev_wave_dir    data/LA/flac \
  --eval_list       protocols/ASVspoof2021.LA.cm.eval.trl.txt \
  --eval_wave_dir   data/LA_eval/flac \
  --epochs 50 --batch_size 32 --lr 1e-4
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[2]))

import argparse, os, json, math
from pathlib import Path
from typing import Optional

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from fsd.models.fakespeech_detection import FakeSpeechDetection, freeze_module
from fsd.data.ssl import ASVspoof19TrainDataset, ASVspoof21EvalDataset
from fsd.utils.checkpoint import load_model_state, load_stream_ckpt
from fsd.utils.runtime import configure_runtime
from models import RawNet2, ResNet18, DCT2D

from tqdm import tqdm  # ★ 只需这一行

DEFAULT_CACHE_ROOT = Path("./feature_cache")


def _optional_path(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() == "none":
        return None
    return Path(value)


class _DPWrapper(nn.Module):
    """Wrap FakeSpeechDetection to make DataParallel scatter friendly."""
    def __init__(self, core: FakeSpeechDetection):
        super().__init__()
        self.core = core

    def forward(self, *args, **kwargs):
        global_step = kwargs.pop("global_step", 0)
        if kwargs:
            raise TypeError(f"Unexpected keyword args: {sorted(kwargs.keys())}")

        if len(args) == 2:
            inputs, label = args
            return self.core(inputs, label, global_step)

        if len(args) == 3 and isinstance(args[0], (list, tuple)):
            inputs, label, global_step = args
            return self.core(inputs, label, global_step)

        if len(args) == 4:
            wave, lfcc, dct2d, label = args
            return self.core([wave, lfcc, dct2d], label, global_step)

        if len(args) == 5:
            wave, lfcc, dct2d, label, global_step = args
            return self.core([wave, lfcc, dct2d], label, global_step)

        raise TypeError(
            "_DPWrapper.forward expected either "
            "(wave, lfcc, dct2d, label[, global_step]) or "
            "([wave, lfcc, dct2d], label[, global_step])"
        )


def _unwrap_core(model: nn.Module) -> FakeSpeechDetection:
    if isinstance(model, nn.DataParallel):
        model = model.module
    if isinstance(model, _DPWrapper):
        return model.core
    if hasattr(model, "core") and isinstance(getattr(model, "core"), FakeSpeechDetection):
        return model.core
    raise TypeError(f"Could not unwrap FakeSpeechDetection from {type(model)}")


def _set_module_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(enabled)


def _set_detectors_trainable(core: FakeSpeechDetection, mode: str) -> None:
    mode = str(mode).lower().strip()
    if mode not in {"all", "none", "last"}:
        raise ValueError(f"Unknown detector_trainable mode: {mode}")

    if mode == "all":
        for det in core.detectors:
            _set_module_requires_grad(det, True)
        return

    if mode == "none":
        for det in core.detectors:
            _set_module_requires_grad(det, False)
        return

    # mode == "last": freeze then unfreeze last few layers per detector type.
    for det in core.detectors:
        _set_module_requires_grad(det, False)

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

        if not patterns:
            continue

        matched = 0
        for name, p in det.named_parameters():
            if any(pat in name for pat in patterns):
                p.requires_grad = True
                matched += 1
        if matched == 0:
            print(f"[warn] detector_trainable=last: no params matched for {type(det).__name__}")
# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor):
    return (logits.argmax(1) == y).float().mean().item()


class Dc_loss(nn.Module):
    def __init__(self, device: torch.device | str):
        super().__init__()
        self.device = device

    def forward(self, evidences):
        num_views = len(evidences)
        if num_views <= 1:
            return torch.tensor(0.0, device=self.device)

        batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
        p = torch.zeros((num_views, batch_size, num_classes), device=self.device)
        u = torch.zeros((num_views, batch_size), device=self.device)
        for v in range(num_views):
            alpha = evidences[v] + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            p[v] = alpha / S
            u[v] = torch.squeeze(num_classes / S)
        dc_sum = 0
        for i in range(num_views):
            pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
            cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
            dc = pd * cc
            dc_sum = dc_sum + torch.sum(dc, dim=0)
        return torch.mean(dc_sum)


def run_epoch(model: nn.Module, loader: DataLoader, *, device: str,
              optimizer: Optional[torch.optim.Optimizer] = None,
              scheduler: Optional[OneCycleLR] = None,epoch = 0,
              alpha: float = 1.0,
              phi: float = 1.0,
              frozen_detectors_eval: bool = True):
    train = optimizer is not None
    model.train() if train else model.eval()
    if train and frozen_detectors_eval:
        core = _unwrap_core(model)
        for det in core.detectors:
            if not any(p.requires_grad for p in det.parameters()):
                det.eval()
    weight = torch.tensor([0.3, 0.7], device=device, dtype=torch.float32)
    total_loss_fn = nn.CrossEntropyLoss(weight=weight)
    dc_loss_fn = Dc_loss(device)
    tot_loss = tot_acc = tot_acc_1 = tot_acc_2 = tot_acc_3 =n = 0

    pbar = tqdm(loader,
            desc=f"{'Train' if train else 'dev '} Ep{epoch:03d}",
            unit="batch",
            leave=True)
    #logits, emb, evn_single, uncertain, loss
    for wave, lfcc, dct2d,y in pbar:
        wave, lfcc,dct2d, y = wave.to(device), lfcc.to(device),dct2d.to(device), y.to(device)
        logits, emb, evn_single, uncertain, loss_aux = model(wave, lfcc, dct2d, y, global_step = epoch)
        total_loss = total_loss_fn(logits, y)
        certainloss = loss_aux.mean()
        dcloss = dc_loss_fn(evn_single)
        loss = total_loss + float(alpha) * certainloss + float(phi) * dcloss
        bs = y.size(0)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
        # -------- 统计 & 进度条动态后缀 --------
        tot_loss += loss.item() * bs
        tot_acc += accuracy(logits, y) * bs
        tot_acc_1  += accuracy(evn_single[0], y) * bs
        tot_acc_2  += accuracy(evn_single[1], y) * bs
        tot_acc_3  += accuracy(evn_single[2], y) * bs
        n        += bs
        pbar.set_postfix({
            "loss": f"{tot_loss / n:.4f}",
            "acc":  f"{100 * tot_acc / n:.2f}%",
            "acc_1":  f"{100 * tot_acc_1 / n:.2f}%",
            "acc_2":  f"{100 * tot_acc_2 / n:.2f}%",
            "acc_3":  f"{100 * tot_acc_3 / n:.2f}%",
            "lr":   f"{optimizer.param_groups[0]['lr']:.2e}" if train else "-"
        })
    return tot_loss / n, tot_acc / n

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace):
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
    if getattr(args, "gpu_id", None) is not None:
        if str(args.device).startswith("cuda"):
            args.device = "cuda:0" if ":" in str(args.device) else "cuda"
        if hasattr(args, "feature_device") and str(args.feature_device).startswith("cuda"):
            args.feature_device = "cuda"

    set_seed(args.seed)
    device = torch.device(args.device)


    # -------- datasets / loaders --------
    feature_device_cuda = str(args.feature_device).startswith("cuda")
    train_workers = args.num_workers
    if feature_device_cuda and train_workers:
        print(f"[info] feature_device is CUDA; 为避免 fork + CUDA 报错，将 num_workers 从 {train_workers} 改为 0")
        train_workers = 0

    train_dst = ASVspoof19TrainDataset(
        args.train_protocol,
        args.train_wave_dir,
        pad_to=args.pad_to,
        args=args,
        algo=args.algo,
        device=args.feature_device,
        disk_cache_dir=args.train_cache_dir,
    )
    dev_dst   = ASVspoof19TrainDataset(
        args.dev_protocol,
        args.dev_wave_dir,
        pad_to=args.pad_to,
        args=args,
        algo=args.algo,
        device=args.feature_device,
        disk_cache_dir=args.dev_cache_dir,
    )
    
    if args.cache_features:
        cache_dev = getattr(args, "cache_features_device", "cpu")
        cache_workers = getattr(args, "cache_features_workers", 0)
        print(f"[info] caching training features to memory on {cache_dev}…")
        train_dst.cache_all_features(cache_device=cache_dev, cache_workers=cache_workers)
        print(f"[info] caching dev features to memory on {cache_dev}…")
        dev_dst.cache_all_features(cache_device=cache_dev, cache_workers=cache_workers)
        train_workers = 0  # cached in-memory -> avoid extra worker/IPC overhead
        cache_on_cuda = str(cache_dev).startswith("cuda")
    else:
        cache_on_cuda = False

    data_on_cuda = cache_on_cuda if args.cache_features else feature_device_cuda
    pin_mem = (device.type == "cuda") and (not data_on_cuda)  # 若数据已在 CUDA 上，不能再 pin
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, shuffle=True,
                              num_workers=train_workers, pin_memory=pin_mem)
    dev_loader   = DataLoader(dev_dst,   batch_size=args.batch_size, shuffle=False,
                              num_workers=train_workers, pin_memory=pin_mem)

    # -------- model --------
    core_model = FakeSpeechDetection(
        detectors=[RawNet2.Model(), ResNet18.Model(), DCT2D.Model()],
        feature_out=256,
        fixed_size=None,
        device=args.device,
    ).to(device)

    wrapped_model = _DPWrapper(core_model)
    if args.use_dp and torch.cuda.device_count() > 1:
        model = nn.DataParallel(wrapped_model)
        print(f"[info] Using DataParallel on {torch.cuda.device_count()} GPUs")
    else:
        model = wrapped_model

    # -------- checkpoint init (optional) --------
    if getattr(args, "init_ckpt", None):
        missing, unexpected = load_model_state(model, args.init_ckpt, strict=False)
        print(f"[info] loaded init_ckpt={args.init_ckpt} (missing={len(missing)} unexpected={len(unexpected)})")

    core = _unwrap_core(model)
    stream_map = {
        "raw": 0,
        "lfcc": 1,
        "dct2d": 2,
    }
    for stream, idx in stream_map.items():
        ckpt_path = getattr(args, f"init_{stream}_ckpt", None)
        if not ckpt_path:
            continue
        det_state, proj_state = load_stream_ckpt(ckpt_path)
        missing, unexpected = core.detectors[idx].load_state_dict(det_state, strict=False)
        if missing or unexpected:
            print(f"[warn] init_{stream}_ckpt detector load: missing={len(missing)} unexpected={len(unexpected)}")
        if proj_state is not None:
            missing, unexpected = core.proj[idx].load_state_dict(proj_state, strict=False)
            if missing or unexpected:
                print(f"[warn] init_{stream}_ckpt proj load: missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[info] loaded init_{stream}_ckpt={ckpt_path}")

    # -------- freeze / finetune controls --------
    _set_detectors_trainable(core, getattr(args, "detector_trainable", "all"))

    # -------- optimizer & scheduler --------
    detector_lr = getattr(args, "detector_lr", None)
    if detector_lr is not None:
        det_params = [p for p in core.detectors.parameters() if p.requires_grad]
        det_ids = {id(p) for p in det_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in det_ids]
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": float(args.lr)})
        if det_params:
            param_groups.append({"params": det_params, "lr": float(detector_lr)})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    #max_lr = 3e-3          # 举例

    # 2) One-CycleLR 设置
    scheduler = OneCycleLR(
        optimizer,
        max_lr       = [pg.get("lr", args.lr) for pg in optimizer.param_groups],
        total_steps  = args.epochs * steps_per_epoch,
        pct_start    = 0.05,      # 5 % warm-up
        div_factor   = 25,        # 初始 LR = max_lr/25
        final_div_factor = 50     # 结束 LR = max_lr/50
    )

    # -------- training loop --------
    best_dev_acc = 0.0
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device=device,
                                    optimizer=optimizer, scheduler=scheduler,epoch=epoch,
                                    alpha=args.alpha, phi=args.phi )
        dv_loss, dv_acc = run_epoch(model, dev_loader, device=device,epoch=epoch,
                                    alpha=args.alpha, phi=args.phi)
        print(f"[Epoch {epoch:03d}] train loss {tr_loss:.4f} | acc {tr_acc*100:.2f}%  "
              f"dev loss {dv_loss:.4f} | acc {dv_acc*100:.2f}%")
        if dv_acc > best_dev_acc:
            best_dev_acc = dv_acc
            ckpt_path = args.ckpt_dir / f"best_{epoch:03d}_{dv_acc:.4f}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print("  ↳ saved best checkpoint", ckpt_path)

    # -------- evaluation (optional) --------
    if args.eval_list:
        if args.ckpt is not None and Path(args.ckpt).exists():
            #print(torch.load(args.ckpt,map_location=device))
            model.load_state_dict(torch.load(args.ckpt,map_location=device))
            print('Model loaded : {}'.format(args.ckpt))

        eval_ids = []
        with open(args.eval_list) as fh:
            for ln in fh:
                parts = ln.strip().split()
                if not parts:
                    continue
                eval_ids.append(parts[1] if len(parts) > 1 else parts[0])
        eval_dst = ASVspoof21EvalDataset(
            eval_ids,
            args.eval_wave_dir,
            pad_to=args.pad_to,
            device=args.feature_device,
            disk_cache_dir=args.eval_cache_dir,
        )
        eval_workers = 0 if feature_device_cuda else args.num_workers
        eval_loader = DataLoader(eval_dst, batch_size=args.batch_size,
                                 shuffle=False, num_workers=eval_workers, pin_memory=not feature_device_cuda)
        out_file = args.output_dir / "best_040_0.9961_without_softmax.txt"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        model.eval()
        with torch.no_grad(), open(out_file, "w") as fh:
            for wave, lfcc,dct2d, utt in eval_loader:
                wave, lfcc,dct2d = wave.to(device), lfcc.to(device),dct2d.to(device)
                dummy_y = torch.zeros(len(wave), dtype=torch.long, device=device)
                logits, *_ = model(wave, lfcc, dct2d, dummy_y, global_step=0)
                prob = logits[:, 1]  
                score = ( -prob).cpu().numpy()  
                for u, s in zip(utt, score):
                    fh.write(f"{u} {s:.6f}\n")
        print("Evaluation scores saved to", out_file)


def build_arg_parser(description: str | None = None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--train_protocol", type=Path,default="/ssd/lxx/code/database/ASVspoof_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    ap.add_argument("--train_wave_dir", type=Path,default="/ssd/lxx/LA/ASVspoof2019_LA_train/flac")
    ap.add_argument("--dev_protocol", type=Path,default="/ssd/lxx/code/database/ASVspoof_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    ap.add_argument("--dev_wave_dir", type=Path,default= "/ssd/lxx/LA/ASVspoof2019_LA_dev/flac")
    ap.add_argument("--eval_list", type=Path,default="/ssd/lxx/code/database/ASVspoof_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    ap.add_argument("--eval_wave_dir", type=Path,default="/ssd/lxx/LA/ASVspoof2019_LA_eval/flac")
    ap.add_argument("--pad_to", type=int, default=64600)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.2)
    # ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--phi", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu_id", type=int, help="Use only this physical GPU id (sets CUDA_VISIBLE_DEVICES).")
    ap.add_argument("--cpu_affinity", help="Pin process to CPU cores, e.g. '0-15' or '0-7,16-23'.")
    ap.add_argument("--omp_threads", type=int, help="Set OMP_NUM_THREADS.")
    ap.add_argument("--mkl_threads", type=int, help="Set MKL_NUM_THREADS.")
    ap.add_argument("--torch_threads", type=int, help="torch.set_num_threads().")
    ap.add_argument("--torch_interop_threads", type=int, help="torch.set_num_interop_threads().")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--init_ckpt", type=Path, help="Load full model weights before training (for fine-tuning).")
    ap.add_argument("--init_raw_ckpt", type=Path, help="Load per-stream (raw) pretrain checkpoint.")
    ap.add_argument("--init_lfcc_ckpt", type=Path, help="Load per-stream (lfcc) pretrain checkpoint.")
    ap.add_argument("--init_dct2d_ckpt", type=Path, help="Load per-stream (dct2d) pretrain checkpoint.")
    ap.add_argument("--detector_trainable", choices=["all", "none", "last"], default="all",
                    help="Which detector parameters to train: all / none (freeze) / last (partial unfreeze).")
    ap.add_argument("--detector_lr", type=float,
                    help="Optional LR for detector parameters (separate param group).")
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--ckpt_dir", type=Path, default=Path("./ckpt/with_noise_4_resnet_rawnet_dct2d_1E-4_withklloss_50epoch_respectly/"))
    ap.add_argument("--output_dir", type=Path, default=Path("./scores/LA21/with_noise_4_resnet_rawnet_dct2d_1E-4_withklloss_50epoch_respectly/"))
    ap.add_argument("--feature_device", default="cuda", help="Device used for on-the-fly feature computation (e.g., cuda or cpu)")
    ap.add_argument("--use_dp", action="store_true", help="Use DataParallel for multi-GPU training.")
    ap.add_argument("--cache_features", action=argparse.BooleanOptionalAction, default=True,
                    help="Precompute features once on the feature device and cache them in memory for training/dev.")
    ap.add_argument(
        "--cache_features_device",
        default="cpu",
        help="Device to store in-memory cached features (recommended: cpu).",
    )
    ap.add_argument(
        "--cache_features_workers",
        type=int,
        default=0,
        help="Workers used when preloading features into memory (0 avoids /dev/shm issues).",
    )

    cache_root = DEFAULT_CACHE_ROOT
    ap.add_argument("--train_cache_dir", type=_optional_path, default=cache_root / "train",
                    help="Directory with cached train features (.pt). Use 'none' to disable disk caching for training data.")
    ap.add_argument("--dev_cache_dir", type=_optional_path, default=cache_root / "dev",
                    help="Directory with cached dev features (.pt). Use 'none' to disable disk caching for dev data.")
    ap.add_argument("--eval_cache_dir", type=_optional_path, default=cache_root / "eval",
                    help="Directory with cached eval features (.pt). Use 'none' to disable disk caching for eval data.")

    ##===================================================Rawboost data augmentation ======================================================================#
    ap.add_argument('--algo', type=int, default=4,
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

    # LnL_convolutive_noise parameters
    ap.add_argument('--nBands', type=int, default=5,
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    ap.add_argument('--minF', type=int, default=20,
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    ap.add_argument('--maxF', type=int, default=8000,
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    ap.add_argument('--minBW', type=int, default=100,
                    help='minimum width [Hz] of filter.[default=100] ')
    ap.add_argument('--maxBW', type=int, default=1000,
                    help='maximum width [Hz] of filter.[default=1000] ')
    ap.add_argument('--minCoeff', type=int, default=10,
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    ap.add_argument('--maxCoeff', type=int, default=100,
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    ap.add_argument('--minG', type=int, default=0,
                    help='minimum gain factor of linear component.[default=0]')
    ap.add_argument('--maxG', type=int, default=0,
                    help='maximum gain factor of linear component.[default=0]')
    ap.add_argument('--minBiasLinNonLin', type=int, default=5,
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    ap.add_argument('--maxBiasLinNonLin', type=int, default=20,
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    ap.add_argument('--N_f', type=int, default=5,
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    ap.add_argument('--P', type=int, default=10,
                    help='Maximum number of uniformly distributed samples in [%%].[defaul=10]')
    ap.add_argument('--g_sd', type=int, default=2,
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    ap.add_argument('--SNRmin', type=int, default=10,
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    ap.add_argument('--SNRmax', type=int, default=40,
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')

    ap.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    return ap


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)
