from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fsd.data.ssl import ASVspoof19TrainDataset, ASVspoof21EvalDataset
from fsd.train.base import build_arg_parser


def _load_eval_ids(eval_list: Path) -> List[str]:
    ids: List[str] = []
    with open(eval_list) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            ids.append(parts[1] if len(parts) > 1 else parts[0])
    return ids


def _filter_train_entries(entries: Sequence[Tuple[str, int]], cache_dir: Path, overwrite: bool):
    if overwrite:
        return list(entries)
    existing = {p.stem for p in cache_dir.glob("*.pt")}
    if not existing:
        return list(entries)
    filtered = [entry for entry in entries if entry[0] not in existing]
    skipped = len(entries) - len(filtered)
    if skipped:
        print(f"[info] train split: skipping {skipped} already-cached utterances")
    return filtered


def _filter_eval_ids(ids: Iterable[str], cache_dir: Path, overwrite: bool):
    if overwrite:
        return list(ids)
    filtered = []
    skipped = 0
    for utt in ids:
        if (cache_dir / f"{utt}.pt").exists():
            skipped += 1
            continue
        filtered.append(utt)
    if skipped:
        print(f"[info] eval split: skipping {skipped} already-cached utterances")
    return filtered


def _cache_dataset(dataset, cache_dir: Path, desc: str, num_workers: int, overwrite: bool):
    cache_dir.mkdir(parents=True, exist_ok=True)
    worker_count = max(num_workers, 0)

    original_device = getattr(dataset, "feature_device", "cpu")
    dataset.feature_device = "cpu"
    if hasattr(dataset, "lfcc_tf"):
        dataset.lfcc_tf = dataset.lfcc_tf.cpu()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=worker_count,
        collate_fn=lambda x: x[0],
        pin_memory=False,
    )

    try:
        for sample in tqdm(loader, desc=f"Caching {desc}", total=len(dataset)):
            if len(sample) == 5:
                wave, lfcc, dct2d, label, utt = sample
            elif len(sample) == 4:
                wave, lfcc, dct2d, utt = sample
                label = None
            else:
                raise RuntimeError(f"Unexpected sample structure for {desc}: {len(sample)} elements")

            out_file = cache_dir / f"{utt}.pt"
            if out_file.exists() and not overwrite:
                continue

            payload = {
                "wave": wave.detach().cpu(),
                "lfcc": lfcc.detach().cpu(),
                "dct2d": dct2d.detach().cpu(),
            }
            if label is not None:
                payload["label"] = int(label.item())
            torch.save(payload, out_file)
    finally:
        dataset.feature_device = original_device
        if hasattr(dataset, "lfcc_tf") and original_device is not None:
            dataset.lfcc_tf = dataset.lfcc_tf.to(original_device)


def parse_args():
    parser = build_arg_parser("Cache LFCC/RawNet features to disk for later reuse.")
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "eval"], default=["train", "dev", "eval"],
                        help="Which dataset splits to cache.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute and overwrite cache files even if they already exist.")
    parser.add_argument("--cache_workers", type=int,
                        help="Override num_workers when computing features for caching.")
    return parser.parse_args()


def main():
    args = parse_args()
    splits = set(args.splits)
    worker_count = args.cache_workers if args.cache_workers is not None else args.num_workers
    worker_count = worker_count or 0

    if "train" in splits:
        if args.train_cache_dir is None:
            print("[warn] train split requested but --train_cache_dir is disabled; skipping.")
        else:
            cache_dir = Path(args.train_cache_dir)
            train_dst = ASVspoof19TrainDataset(
                args.train_protocol,
                args.train_wave_dir,
                pad_to=args.pad_to,
                args=args,
                algo=args.algo,
                device=args.feature_device,
                return_utt=True,
            )
            train_dst.entries = _filter_train_entries(train_dst.entries, cache_dir, args.overwrite)
            if len(train_dst) == 0:
                print("[info] train split: nothing to cache.")
            else:
                _cache_dataset(train_dst, cache_dir, "train", worker_count, args.overwrite)

    if "dev" in splits:
        if args.dev_cache_dir is None:
            print("[warn] dev split requested but --dev_cache_dir is disabled; skipping.")
        else:
            cache_dir = Path(args.dev_cache_dir)
            dev_dst = ASVspoof19TrainDataset(
                args.dev_protocol,
                args.dev_wave_dir,
                pad_to=args.pad_to,
                args=args,
                algo=args.algo,
                device=args.feature_device,
                return_utt=True,
            )
            dev_dst.entries = _filter_train_entries(dev_dst.entries, cache_dir, args.overwrite)
            if len(dev_dst) == 0:
                print("[info] dev split: nothing to cache.")
            else:
                _cache_dataset(dev_dst, cache_dir, "dev", worker_count, args.overwrite)

    if "eval" in splits:
        if args.eval_cache_dir is None:
            print("[warn] eval split requested but --eval_cache_dir is disabled; skipping.")
        else:
            cache_dir = Path(args.eval_cache_dir)
            eval_ids = _load_eval_ids(args.eval_list)
            eval_ids = _filter_eval_ids(eval_ids, cache_dir, args.overwrite)
            if not eval_ids:
                print("[info] eval split: nothing to cache.")
            else:
                eval_dst = ASVspoof21EvalDataset(
                    eval_ids,
                    args.eval_wave_dir,
                    pad_to=args.pad_to,
                    device=args.feature_device,
                )
                _cache_dataset(eval_dst, cache_dir, "eval", worker_count, args.overwrite)


if __name__ == "__main__":
    main()
