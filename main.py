"""Backward-compat entrypoint (moved to `fsd.train.base`)."""

from __future__ import annotations

from fsd.train.base import build_arg_parser, main


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)

