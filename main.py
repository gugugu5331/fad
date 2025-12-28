"""main.py – Full training / validation / evaluation script for FakeSpeechDetection.

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
import argparse, os, json, math
from pathlib import Path
from typing import Optional

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from model import FakeSpeechDetection, freeze_module
from data_utils_SSL import ASVspoof19TrainDataset
from data_utils_SSL import ASVspoof21EvalDataset
from models import RawNet2, ResNet18,DCT2D

from tqdm import tqdm  # ★ 只需这一行
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


def run_epoch(model: nn.Module, loader: DataLoader, *, device: str,
              optimizer: Optional[torch.optim.Optimizer] = None,
              scheduler: Optional[OneCycleLR] = None,epoch = 0):
    train = optimizer is not None
    model.train() if train else model.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss = tot_acc_1 = tot_acc_2 = tot_acc_3 =n = 0

    pbar = tqdm(loader,
            desc=f"{'Train' if train else 'dev '} Ep{epoch:03d}",
            unit="batch",
            leave=True)
    #logits, emb, evn_single, uncertain, loss
    for wave, lfcc, dct2d,y in pbar:
        wave, lfcc,dct2d, y = wave.to(device), lfcc.to(device),dct2d.to(device), y.to(device)
        logits, emb, evn_single, uncertain, loss_aux = model([wave, lfcc,dct2d], y, global_step = epoch)
        #loss = ce(logits, y) + loss_aux.mean()
        loss =  loss_aux.mean()
        bs = y.size(0)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
        # -------- 统计 & 进度条动态后缀 --------
        tot_loss += loss.item() * bs
        tot_acc_1  += accuracy(evn_single[0], y) * bs
        tot_acc_2  += accuracy(evn_single[1], y) * bs
        tot_acc_3  += accuracy(evn_single[2], y) * bs
        n        += bs
        pbar.set_postfix({
            "loss": f"{tot_loss / n:.4f}",
            "acc_1":  f"{100 * tot_acc_1 / n:.2f}%",
            "acc_2":  f"{100 * tot_acc_2 / n:.2f}%",
            "acc_3":  f"{100 * tot_acc_3 / n:.2f}%",
            "lr":   f"{optimizer.param_groups[0]['lr']:.2e}" if train else "-"
        })
    return tot_loss / n, (tot_acc_1 + tot_acc_2 + tot_acc_3) / 3 / n

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device(args.device)

   
    # -------- datasets / loaders --------
    train_dst = ASVspoof19TrainDataset(args.train_protocol, args.train_wave_dir,
                                  pad_to=args.pad_to, args=args,algo=args.algo,device=args.device)
    dev_dst   = ASVspoof19TrainDataset(args.dev_protocol,   args.dev_wave_dir,
                                  pad_to=args.pad_to,args=args,algo=args.algo,device=args.device)
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, shuffle=True,
                              num_workers=12, pin_memory=True)
    dev_loader   = DataLoader(dev_dst,   batch_size=args.batch_size, shuffle=False,
                              num_workers=12, pin_memory=True)

    # -------- model --------
    model = FakeSpeechDetection(detectors=[RawNet2.Model(), ResNet18.Model(),DCT2D.Model()],
                                feature_out=256, fixed_size=None, device=args.device)
    #model.add_detector(DCT2D.Model(),trainable = True)
    model = nn.DataParallel(model).to(device)

    # -------- optimizer & scheduler --------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    #max_lr = 3e-3          # 举例

    # 2) One-CycleLR 设置
    scheduler = OneCycleLR(
        optimizer,
        max_lr       = args.lr,
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
                                    optimizer=optimizer, scheduler=scheduler,epoch=epoch )
        dv_loss, dv_acc = run_epoch(model, dev_loader, device=device,epoch=epoch)
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

        eval_ids = [ln.strip().split()[0] for ln in open(args.eval_list)]
        eval_dst = ASVspoof21EvalDataset(eval_ids, args.eval_wave_dir,
                                       pad_to=args.pad_to, device=args.device)
        eval_loader = DataLoader(eval_dst, batch_size=args.batch_size,
                                 shuffle=False, num_workers=8)
        out_file = args.output_dir / "best_040_0.9961_without_softmax.txt"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        model.eval()
        with torch.no_grad(), open(out_file, "w") as fh:
            for wave, lfcc,dct2d, utt in eval_loader:
                wave, lfcc,dct2d = wave.to(device), lfcc.to(device),dct2d.to(device)
                logits, *_ = model([wave, lfcc,dct2d], torch.zeros(len(wave), dtype=torch.long, device=device), 0)
                prob = logits[:, 1]  
                score = ( -prob).cpu().numpy()  
                for u, s in zip(utt, score):
                    fh.write(f"{u} {s:.6f}\n")
        print("Evaluation scores saved to", out_file)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--ckpt_dir", type=Path, default=Path("./ckpt/with_noise_4_resnet_rawnet_dct2d_1E-4_withklloss_50epoch_respectly/"))
    ap.add_argument("--output_dir", type=Path, default=Path("./scores/LA21/with_noise_4_resnet_rawnet_dct2d_1E-4_withklloss_50epoch_respectly/"))
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
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    ap.add_argument('--g_sd', type=int, default=2,
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    ap.add_argument('--SNRmin', type=int, default=10,
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    ap.add_argument('--SNRmax', type=int, default=40,
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')

    ap.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    args = ap.parse_args()
    main(args)
