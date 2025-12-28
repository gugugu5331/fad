
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import random
from typing import Optional
import torchaudio
from pathlib import Path
from typing import Callable, Optional, Tuple, List

from feature import dct2d
from tqdm import tqdm
from spafe.utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
from spafe.utils.exceptions import ParameterError, ErrorMsgs
from spafe.fbanks.mel_fbanks import mel_filter_banks
#from spafe.utils.cepstral import cms, cmvn, lifter_ceps
from scipy.fftpack import dct, dctn

import math
import torch.nn.functional as F
from torch_dct import dct as _dct
# ------------------------------------------------------------------
# util: pad / hz↔mel ------------------------------------------------
# ------------------------------------------------------------------
def pad(x: torch.Tensor | np.ndarray, max_len: int = 64600):
    """沿时间维 repeat or truncate 到固定长度 (默认为 4 s)。"""
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    L = x.size(0)
    if L >= max_len:
        return x[:max_len]
    n_rep = max_len // L + 1
    return x.repeat(n_rep)[:max_len]

def _hz2mel(hz): return 2595.0 * torch.log10(1 + hz / 700.)
def _mel2hz(mel): return 700.0 * (10**(mel / 2595.) - 1)

# ------------------------------------------------------------------
# 1️⃣  compute_feature  (外部名字保持不变) ---------------------------
# ------------------------------------------------------------------
def compute_feature(raw_data, sr, device: str | torch.device = "cpu"):
    """
    Pad + Mel-2D-DCT 特征 (torch 实现).
    return: Tensor(float32)  shape = (n_frames, 40)
    """
    if not isinstance(raw_data, torch.Tensor):
        raw_data = torch.from_numpy(raw_data)
    x = pad(raw_data)                     # (T,)
    return mel_2D(x, fs=sr, device=device)  # 调用同文件下的新 mel_2D

def dct_t(x: torch.Tensor, dim: int = -1, norm: str | None = "ortho"):
    """torch-dct 的 dim-aware 包装：在任意维上做 DCT-II。"""
    if dim < 0:                 # 允许负索引
        dim += x.ndim
    if dim == x.ndim - 1:       # 已经是最后一维，直接算
        return _dct(x, norm=norm)
    # 否则把目标维换到最后，再换回来
    y = x.transpose(dim, -1)
    y = _dct(y, norm=norm)
    return y.transpose(dim, -1)

# ------------------------------------------------------------------
# 2️⃣  mel_2D  (外部名字保持不变) ----------------------------------
# ------------------------------------------------------------------
def mel_2D(
    sig: torch.Tensor,
    fs: int = 16000,
    num_ceps: int = 40,
    win_len: float = 0.032,
    win_hop: float = 0.016,
    nfilts: int = 128,
    nfft: int = 1024,
    dct_type: int = 2,
    device: str | torch.device = "cpu",
    **kwargs,        # 其余参数保持接口但不再使用
):
    """
    PyTorch-only 版本：
      • framing + Hamming
      • power spec → Mel filterbank (128)
      • log10 → 2-D DCT-II → 取前 40 维
    输出 (frames, 40)  float32
    """
    x = sig.to(device).float()

    # ---------- framing ------------------------------------------
    win_len_s = int(round(fs * win_len))
    hop_s     = int(round(fs * win_hop))
    pad_len   = (math.ceil((x.numel() - win_len_s) / hop_s) * hop_s
                 + win_len_s - x.numel())
    x = F.pad(x, (0, pad_len))

    frames = x.unfold(0, win_len_s, hop_s)          # (N, win)
    window = torch.hamming_window(win_len_s, periodic=False, device=device)
    frames = frames * window

    # ---------- power spectrum -----------------------------------
    spec = torch.fft.rfft(frames, nfft)             # (N, nfft//2+1)
    pow_spec = spec.real.pow(2) + spec.imag.pow(2)

    # ---------- Mel filterbank -----------------------------------
    f_min, f_max = 0.0, fs / 2
    mel_pts = torch.linspace(_hz2mel(torch.tensor(f_min)),
                             _hz2mel(torch.tensor(f_max)),
                             nfilts + 2, device=device)
    bin_pts = torch.floor((nfft + 1) * _mel2hz(mel_pts) / fs).long()

    fb = torch.zeros(nfilts, nfft // 2 + 1, device=device)
    for m in range(1, nfilts + 1):
        f_l, f_c, f_r = bin_pts[m-1:m+2]
        fb[m-1, f_l:f_c] = (torch.arange(f_l, f_c, device=device) - f_l) / (f_c - f_l)
        fb[m-1, f_c:f_r] = (f_r - torch.arange(f_c, f_r, device=device)) / (f_r - f_c)

    mel_spec = pow_spec @ fb.T                          # (N, 128)
    log_mel = torch.log10(mel_spec.clamp_min(1e-10))

    # ---------- 2-D DCT-II ---------------------------------------
    dct_f  = dct_t(log_mel, norm='ortho', dim=1)
    dct_2d = dct_t(dct_f,  norm='ortho', dim=0)
    return dct_2d[:, :num_ceps].contiguous().float()



def genSpoof_list2019(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_eval):
        for line in l_meta:
            key = line.strip()
            # _, key, _, _, _ = line.strip().split()
            file_list.append(key)
        return file_list

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 0 if label == 'bonafide' else 1
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            #_, key, _, _, _ = line.strip().split()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 0 if label == 'bonafide' else 1
        return d_meta,file_list




def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

WaveAugment = Callable[[torch.Tensor], torch.Tensor]



# -----------------------------------------------------------------------------
# RawBoost wrapper
# -----------------------------------------------------------------------------
class RawBoostAugment:
    """Thin wrapper around official RawBoost algos.

    Parameters follow original repo naming; only commonly‑used ones exposed.
    algo: 1=LnL, 2=ISD, 3=SSI, 4=(1+2+3), 5=(1+2), 6=(1+3), 7=(2+3), 8=parallel(1,2)
    """
    def __init__(self, *, algo:int=3, sample_rate:int=16000, **kwargs):
        if rb is None:
            raise ImportError("rawboost package not found; `pip install rawboost`.")
        self.algo = algo
        self.sr   = sample_rate
        self.kw   = kwargs
        # Instantiate individual processors
        self.proc1 = rb.LnL_convolutive_noise(sr=sample_rate, **kwargs)
        self.proc2 = rb.ISD_additive_noise(sr=sample_rate, **kwargs)
        self.proc3 = rb.SSI_additive_noise(sr=sample_rate, **kwargs)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        w = wav.clone().numpy()
        if self.algo in {1,4,5,6,8}:
            w = self.proc1.process(w)
        if self.algo in {2,4,5,7,8}:
            w = self.proc2.process(w)
        if self.algo in {3,4,6,7}:
            w = self.proc3.process(w)
        return torch.from_numpy(w.astype(np.float32))
    
def _pad_wave(w: torch.Tensor, pad_to: int) -> torch.Tensor:
    """Pad or truncate 1‑D waveform to *pad_to* samples."""
    L = w.size(0)
    if L == pad_to:
        return w
    if L > pad_to:
        return w[:pad_to]
    # repeat then cut
    n_rep = pad_to // L + 1
    return w.repeat(n_rep)[:pad_to]


class ASVspoof19TrainDataset(Dataset):
    """Dataset for FakeSpeechDetection.

    Parameters
    ----------
    proto_txt : str | Path
        Protocol file with format "<utt_id> <label>" per line (label in
        {bonafide, spoof} or {0,1}).
    root_wave : str | Path
        Directory where waveform files live.  Accepts `.wav`, `.flac`, or `.npy`
        (saved float32 wave).
    root_spec : str | Path | None
        If provided, load `*.npy` spectrogram from this dir; otherwise compute
        LFCC (+Δ +ΔΔ) on the fly.
    pad_to : int
        Fixed sample length after padding/clipping.
    sample_rate : int
        Target sample rate for LFCC calculation.
    augment_fn : callable | None
        Optional waveform augmentation (RawBoost, noise, etc).  Signature
        `augment_fn(wave: Tensor) -> Tensor`.
    device : torch device string – set to "cuda" to push LFCC transform.
    """

    def __init__(
        self,
        proto_txt: str | Path,
        root_wave: str | Path,
        *,
        root_spec: str | Path | None = None,
        
        pad_to: int = 64600,
        sample_rate: int = 16000,
        augment_fn: Optional[WaveAugment] = None,
        args,
        algo,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.algo = algo
        self.entries: List[Tuple[str, int]] = []  # (utt_id, label)
        with open(proto_txt) as fh:
            for ln in fh:
                parts = ln.strip().split()
                utt, lab_token = parts[1], parts[-1].lower()
                if lab_token.lower() in {"bonafide", "0"}:
                    lab_idx = 0
                elif lab_token.lower() in {"spoof", "1"}:
                    lab_idx = 1
                else:
                    raise ValueError(f"Unknown label {lab_token}")
                self.entries.append((utt, lab_idx))

        self.root_wave = Path(root_wave)
        self.root_spec = Path(root_spec) if root_spec else None
        self.pad_to = pad_to
        self.augment_fn = process_Rawboost_feature
        self.feature_device = device or "cpu"
        self.cache_data: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None

        # torchaudio LFCC extractor
        self.lfcc_tf = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            speckwargs={"n_fft": 1024, "win_length": 320, "hop_length": 160},
            n_filter=60, n_lfcc=20,
        )
        if self.feature_device and torch.cuda.is_available() and str(self.feature_device).startswith("cuda"):
            self.lfcc_tf = self.lfcc_tf.to(self.feature_device)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _load_wave(self, utt: str) -> torch.Tensor:
        """Load wav/flac/npy as 1‑D float Tensor."""
        # priority: .npy > .wav > .flac
        for ext in (".npy", ".wav", ".flac"):
            f = self.root_wave / f"{utt}{ext}"
            if f.exists():
                if ext == ".npy":
                    w = np.load(f).astype(np.float32)
                else:
                    w, sr = torchaudio.load(f)
                    w = w.mean(0).numpy()  # mono
                return torch.from_numpy(w)
        raise FileNotFoundError(f"Wave for {utt} not found under {self.root_wave}")

    def _make_spec(self, wave: torch.Tensor) -> torch.Tensor:
        """Extract LFCC+Δ+ΔΔ → (60, T) 2‑D tensor.
        Returned tensor layout works with ResNet branch:
          (B, 60, T)  → model 逻辑会自动 unsqueeze 给 (B,1,60,T).
        """
      #  print(wave.shape)
        wave = wave.to(self.feature_device)
        lfcc = self.lfcc_tf(wave)               # (20, T)
        d1   = torch.diff(lfcc, n=1, dim=1)     # (20, T-1)
        d2   = torch.diff(d1,  n=1, dim=1)      # (20, T-2)
        d1   = torch.nn.functional.pad(d1, (1, 0))  # 左补 1 帧
        d2   = torch.nn.functional.pad(d2, (2, 0))  # 左补 2 帧
        spec = torch.cat([lfcc, d1, d2], dim=0)      # (60, T)
        return spec

    def _compute_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        utt, label = self.entries[idx]
        wave = self._load_wave(utt)
        wave = _pad_wave(wave, self.pad_to)
        if self.augment_fn:
            wave = self.augment_fn(wave, 16000, self.args, self.algo)

        if self.root_spec:
            spec_path = self.root_spec / f"{utt}.npy"
            lfcc = torch.from_numpy(np.load(spec_path)).float()  # assume (C,F,T)
        else:
            lfcc = self._make_spec(wave)

        dct2d = compute_feature(wave, 16000, device=self.feature_device)
        return wave, lfcc, dct2d, torch.tensor(label, dtype=torch.long)

    def cache_all_features(self):
        """Compute all features once (on the configured device) and keep them in memory."""
        cached: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for idx in range(len(self.entries)):
            with torch.no_grad():
                wave, lfcc, dct2d, label = self._compute_item(idx)
            cached.append((wave.to(self.feature_device), lfcc.to(self.feature_device), dct2d.to(self.feature_device), label.to(self.feature_device)))
        self.cache_data = cached

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        """Return dataset size based on cached features when present."""
        if self.cache_data is not None:
            return len(self.cache_data)
        return len(self.entries)

    def __getitem__(self, idx: int):
        if self.cache_data is not None:
            return self.cache_data[idx]
        return self._compute_item(idx)


class ASVspoof21EvalDataset(Dataset):
    def __init__(
        self,
        list_IDs: List[str],         # utterance keys without extension
        base_dir: str | Path,        # root folder containing <utt_id>.flac
        pad_to: int = 64600,
        sample_rate: int = 16000,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.pad_to = pad_to
        self.feature_device = device or "cpu"

        # LFCC extractor (moved to cuda if wanted)
        self.lfcc_tf = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            speckwargs={"n_fft": 1024, "win_length": 320, "hop_length": 160},
            n_filter=60, n_lfcc=20,
        )
        if self.feature_device and torch.cuda.is_available() and str(self.feature_device).startswith("cuda"):
            self.lfcc_tf = self.lfcc_tf.to(self.feature_device)

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.list_IDs)
    
    def _make_spec(self, wave: torch.Tensor) -> torch.Tensor:
        """Extract LFCC+Δ+ΔΔ → (60, T) 2‑D tensor.
        Returned tensor layout works with ResNet branch:
          (B, 60, T)  → model 逻辑会自动 unsqueeze 给 (B,1,60,T).
        """
        wave = wave.to(self.feature_device)
        lfcc = self.lfcc_tf(wave)               # (20, T)
        d1   = torch.diff(lfcc, n=1, dim=1)     # (20, T-1)
        d2   = torch.diff(d1,  n=1, dim=1)      # (20, T-2)
        d1   = torch.nn.functional.pad(d1, (1, 0))  # 左补 1 帧
        d2   = torch.nn.functional.pad(d2, (2, 0))  # 左补 2 帧
        spec = torch.cat([lfcc, d1, d2], dim=0)      # (60, T)
        return spec

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        utt = self.list_IDs[idx]
        wav_path = self.base_dir / f"{utt}.flac"
        wave, _ = librosa.load(wav_path, sr=16000)
        wave = _pad(wave, self.pad_to)
        wave_t = torch.from_numpy(wave).float()

        lfcc = self._make_spec(wave_t)
        dct2d = compute_feature(wave_t,16000, device=self.feature_device).float()
        return wave_t, lfcc,dct2d, utt
    
class Dataset_ASVspoof2019_train_NPY(Dataset):
    def __init__(self, root_feat: str, protocol_list: str, pad_to: Optional[int] = None):
        self.root_feat = root_feat
        self.pad_to = pad_to  # 若不为 None，将在时间维度上 pad 或截断到 pad_to
       # print(protocol_list)
        self.samples = []
        label_map = {'bonafide': 0, 'spoof': 1}
        with open(protocol_list, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                utt_id = parts[1]
                label_str = parts[-1]
                label = label_map[label_str]
                npy_path = os.path.join(root_feat, utt_id + ".npy")
                if not os.path.exists(npy_path):
                    raise FileNotFoundError(f"{npy_path} not found")
                self.samples.append((npy_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        feat = np.load(npy_path)               # ndarray (T', D), dtype=float32 或 float16
        feat = torch.from_numpy(feat)          # Tensor (T', D)

        # 如果需要一个固定的时间长度 pad_to，就在这里 pad/truncate
        if self.pad_to is not None:
            T, D = feat.shape
            if T < self.pad_to:
                pad = torch.zeros(self.pad_to - T, D, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=0)
            elif T > self.pad_to:
                feat = feat[: self.pad_to]

        # 此处返回 (feat, label)
        return feat, label

class Dataset_ASVspoof2019_train_NPZ(Dataset):
    """
    从一个 all_feats.npz 中按 key 加载特征，返回 (feat_tensor, label)。

    参数：
      npz_path:  单个 .npz 文件的全路径，例如 "/.../LA_train/all_feats.npz"
      protocol_list: protocol txt 文件路径，比如 "ASVspoof2019.LA.cm.train.trn.txt"
      pad_to:   如果要对时间维度 pad/truncate 到固定长度 pad_to，则传一个 int；否则留 None。
    """
    def __init__(self, npz_path: str, protocol_list: str, pad_to: Optional[int] = None):
        self.npz_path = npz_path
        self.pad_to = pad_to

        # 1. 打开 .npz（使用 mmap_mode="r" 可以延迟加载而不一次性读全内存）
        self.data = np.load(self.npz_path, mmap_mode="r")

        # 2. 解析 protocol_list，获取所有 (key, label)
        #    你的 protocol 行格式比如 ['LA_0079', 'LA_T_1138215', '-', '-', 'bonafide']
        #    用第二列为 key，最后一列为标签
        label_map = {'bonafide': 1, 'spoof': 0}
        self.samples = []
        with open(protocol_list, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                utt_id = parts[1]          # 第二列做 key
                label_str = parts[-1]      # 最后一列为标签
                label = label_map[label_str]
                key = utt_id               # npz 里的 key（确保你的 npz 用的就是这个名字）
                if key not in self.data:
                    raise KeyError(f"{key} not found in {npz_path}")
                self.samples.append((key, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, label = self.samples[idx]
        arr = self.data[key]                 # np.ndarray (T', D)
        feat = torch.from_numpy(arr)         # Tensor (T', D)

        # pad/truncate 到固定长度
        if self.pad_to is not None:
            T, D = feat.shape
            if T < self.pad_to:
                pad = torch.zeros(self.pad_to - T, D, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=0)
            elif T > self.pad_to:
                feat = feat[: self.pad_to]

        return feat, label
            
def _pad(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    """Repeat/truncate to *max_len* samples (64 600 ≈ 4 s @ 16 kHz)."""
    if x.shape[0] >= max_len:
        return x[:max_len]
    n_rep = max_len // x.shape[0] + 1
    return np.tile(x, n_rep)[:max_len]


# class ASVspoof21EvalDataset(Dataset):
#     def __init__(
#         self,
#         list_IDs: List[str],         # utterance keys without extension
#         base_dir: str | Path,        # root folder containing <utt_id>.flac
#         pad_to: int = 64600,
#         sample_rate: int = 16000,
#         device: str | torch.device | None = None,
#     ) -> None:
#         super().__init__()
#         self.list_IDs = list_IDs
#         self.base_dir = Path(base_dir) 
#         self.pad_to = pad_to

#         # LFCC extractor (moved to cuda if wanted)
#         self.lfcc_tf = torchaudio.transforms.LFCC(
#             sample_rate=sample_rate,
#             speckwargs={"n_fft": 1024, "win_length": 320, "hop_length": 160},
#             n_filter=60, n_lfcc=20,
#         )
#         if device and torch.cuda.is_available():
#             self.lfcc_tf = self.lfcc_tf

#     # --------------------------------------------------------------
#     def __len__(self):
#         return len(self.list_IDs)

#     def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
#         utt = self.list_IDs[idx]
#         wav_path = self.base_dir / f"{utt}.flac"
#         wave, _ = librosa.load(wav_path, sr=16000)
#         wave = _pad(wave, self.pad_to)
#         wave_t = torch.from_numpy(wave).float()

#         # # spec: LFCC + Δ + ΔΔ
#         # lfcc = self.lfcc_tf(wave_t)                 # (20,F,T)
#         # d1   = torch.diff(lfcc, n=1, dim=2)        # (20,F,T-1)
#         # d2   = torch.diff(d1,  n=1, dim=2)         # (20,F,T-2)
#         # d1   = torch.nn.functional.pad(d1, (1,0))  # left‑pad
#         # d2   = torch.nn.functional.pad(d2, (2,0))
#         spec = torch.cat([lfcc, d1, d2], dim=0)    # (60,F,T)

#         return wave_t, spec, utt



class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
               '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + utt_id, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id



        #--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return torch.from_numpy(feature.astype(np.float32))
