"""Backward-compat entrypoint (moved to `fsd.train.cache_features`)."""

from __future__ import annotations

from fsd.train.cache_features import main


if __name__ == "__main__":
    main()

