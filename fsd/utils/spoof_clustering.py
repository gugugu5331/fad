"""K-means utilities for spoof embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch


def kmeans(
    x: torch.Tensor,
    k: int,
    *,
    num_iters: int = 25,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple k-means on CPU/GPU with torch only.

    Returns:
      labels: (N,)
      centroids: (K, D)
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D (N,D), got {tuple(x.shape)}")
    n, d = x.shape
    if n == 0:
        raise ValueError("x is empty")
    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, N], got k={k}, N={n}")

    x = x.float()
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen, device=x.device)
    centroids = x[perm[:k]].clone()

    for _ in range(int(num_iters)):
        # squared euclidean distance: ||x-c||^2 = ||x||^2 - 2x·c + ||c||^2
        x2 = (x * x).sum(dim=1, keepdim=True)  # (N,1)
        c2 = (centroids * centroids).sum(dim=1).unsqueeze(0)  # (1,K)
        dist = x2 - 2.0 * (x @ centroids.t()) + c2  # (N,K)
        labels = dist.argmin(dim=1)  # (N,)

        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = x[mask].mean(dim=0)
            else:
                ridx = int(torch.randint(0, n, (1,), generator=gen, device=x.device).item())
                new_centroids[j] = x[ridx]
        centroids = new_centroids

    return labels, centroids


def save_cluster_assignments(out_path: str | Path, utts: Iterable[str], labels: torch.Tensor) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_list = labels.detach().cpu().tolist()
    utt_list: List[str] = list(utts)
    if len(utt_list) != len(labels_list):
        raise ValueError(f"len(utts)={len(utt_list)} != len(labels)={len(labels_list)}")
    with open(out_path, "w") as fh:
        for u, c in zip(utt_list, labels_list):
            fh.write(f"{u} {int(c)}\n")
