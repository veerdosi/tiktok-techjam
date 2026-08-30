#!/usr/bin/env python3
"""challenge benchmark wired to the submission implementation.

The untouched challenge file and its SHA-256 are preserved under ``original/``.
"""

from original import torch_transformer_benchmark as challenge
from submission import UserOptimizedTransformer


challenge.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(challenge.main())

