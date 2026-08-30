#!/usr/bin/env python3
"""Reproduce the untouched Shape-14 challenge allocation failure by dtype."""

from __future__ import annotations

import json

import torch

from implementations import DispatchTransformer
from original.torch_transformer_benchmark import generate_random_case
from shapes import OFFICIAL_SHAPES


class ForwardWitness(DispatchTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.forward_called = False

    def forward(self, x, valid_token_mask=None):
        self.forward_called = True
        return super().forward(x, valid_token_mask)


def probe(dtype):
    config = OFFICIAL_SHAPES[13]
    candidate = ForwardWitness(config).to(device="mps", dtype=dtype).eval()
    required_bytes = config.batch_size * config.seq_len * config.d_model * dtype.itemsize
    try:
        x, mask = generate_random_case(
            config, torch.device("mps"), dtype, 1234, 0.0, 1.0
        )
        with torch.inference_mode():
            candidate(x, mask)
        status = "unexpectedly_completed"
        error = None
    except RuntimeError as exception:
        status = "failed_in_generate_random_case"
        error = str(exception)
    return {
        "dtype": str(dtype),
        "required_input_bytes": required_bytes,
        "required_input_gib": required_bytes / 2**30,
        "status": status,
        "error": error,
        "participant_forward_called": candidate.forward_called,
    }


def main() -> int:
    print(json.dumps([probe(torch.float32), probe(torch.float16)], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
