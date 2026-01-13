"""Protocol + cached-feature dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ProtocolEntry:
    speaker: str
    utt: str
    label: int  # 0=bonafide, 1=spoof


@dataclass(frozen=True)
class ProtocolEntryAttack:
    utt: str
    label: int  # 0=bonafide, 1=spoof
    attack: Optional[str]  # e.g. "A01"; None for bonafide


def _parse_label(token: str) -> int:
    t = token.strip().lower()
    if t in {"bonafide", "0"}:
        return 0
    if t in {"spoof", "1"}:
        return 1
    raise ValueError(f"Unknown label token: {token}")


def _attack_sort_key(token: str):
    m = re.match(r"^A(\d+)$", token)
    if m:
        return (0, int(m.group(1)))
    return (1, token)


class ASVspoof19CachedWithSpeaker(Dataset):
    """Load cached tensors (.pt) + speaker id from ASVspoof2019 protocol."""

    def __init__(self, protocol_path: str | Path, cache_dir: str | Path) -> None:
        super().__init__()
        self.protocol_path = Path(protocol_path)
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"cache_dir not found: {self.cache_dir}")

        entries: List[ProtocolEntry] = []
        speakers: List[str] = []
        with open(self.protocol_path) as fh:
            for ln in fh:
                parts = ln.strip().split()
                if not parts:
                    continue
                if len(parts) < 3:
                    raise ValueError(f"Unexpected protocol line: {ln.strip()}")
                speaker = parts[0]
                utt = parts[1]
                label = _parse_label(parts[-1])
                entries.append(ProtocolEntry(speaker=speaker, utt=utt, label=label))
                speakers.append(speaker)

        uniq_speakers = sorted(set(speakers))
        self.speaker_to_idx: Dict[str, int] = {s: i for i, s in enumerate(uniq_speakers)}
        self.entries = entries

    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_idx)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        cache_file = self.cache_dir / f"{e.utt}.pt"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Cached feature not found: {cache_file}. "
                f"请先运行 cache_features.py 生成 {self.cache_dir} 下的 .pt 文件。"
            )

        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        wave = data["wave"].float()
        lfcc = data["lfcc"].float()
        dct2d = data["dct2d"].float()

        label = torch.tensor(e.label, dtype=torch.long)
        speaker = torch.tensor(self.speaker_to_idx[e.speaker], dtype=torch.long)
        return wave, lfcc, dct2d, label, speaker, e.utt


class ASVspoof19CachedWithAttack(Dataset):
    """Load cached tensors (.pt) + attack id (Axx) from ASVspoof2019 protocol.

    仅对 spoof 样本提供 attack 标签；bonafide 的 attack 设为 -1（训练时应 mask 掉）。
    """

    def __init__(
        self,
        protocol_path: str | Path,
        cache_dir: str | Path,
        *,
        attack_to_idx: Optional[Dict[str, int]] = None,
        allow_new_attacks: bool = False,
    ) -> None:
        super().__init__()
        self.protocol_path = Path(protocol_path)
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"cache_dir not found: {self.cache_dir}")

        raw_entries: List[Tuple[str, int, Optional[str]]] = []
        attacks_found: set[str] = set()
        with open(self.protocol_path) as fh:
            for ln in fh:
                parts = ln.strip().split()
                if not parts:
                    continue
                if len(parts) < 3:
                    raise ValueError(f"Unexpected protocol line: {ln.strip()}")

                utt = parts[1]
                label = _parse_label(parts[-1])
                attack_token = parts[-2] if len(parts) >= 2 else "-"

                if label == 1:
                    if attack_token == "-" or attack_token == "":
                        raise ValueError(f"Missing attack id for spoof line: {ln.strip()}")
                    attacks_found.add(attack_token)
                    raw_entries.append((utt, label, attack_token))
                else:
                    raw_entries.append((utt, label, None))

        if attack_to_idx is None:
            uniq = sorted(attacks_found, key=_attack_sort_key)
            self.attack_to_idx = {a: i for i, a in enumerate(uniq)}
        else:
            self.attack_to_idx = dict(attack_to_idx)
            missing = attacks_found - set(self.attack_to_idx.keys())
            if missing:
                if allow_new_attacks:
                    next_idx = max(self.attack_to_idx.values(), default=-1) + 1
                    for a in sorted(missing, key=_attack_sort_key):
                        self.attack_to_idx[a] = next_idx
                        next_idx += 1
                else:
                    raise ValueError(f"Protocol contains unseen attacks: {sorted(missing)}")

        self.entries: List[ProtocolEntryAttack] = []
        for utt, label, attack in raw_entries:
            self.entries.append(ProtocolEntryAttack(utt=utt, label=label, attack=attack))

    @property
    def num_attacks(self) -> int:
        return len(self.attack_to_idx)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        cache_file = self.cache_dir / f"{e.utt}.pt"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Cached feature not found: {cache_file}. "
                f"请先运行 cache_features.py 生成 {self.cache_dir} 下的 .pt 文件。"
            )

        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        wave = data["wave"].float()
        lfcc = data["lfcc"].float()
        dct2d = data["dct2d"].float()

        label = torch.tensor(e.label, dtype=torch.long)
        if e.attack is None:
            attack = torch.tensor(-1, dtype=torch.long)
        else:
            attack = torch.tensor(self.attack_to_idx[e.attack], dtype=torch.long)
        return wave, lfcc, dct2d, label, attack, e.utt
