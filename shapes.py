"""Published TikTok TechJam 2026 Track 3 benchmark shapes."""

from __future__ import annotations

from dataclasses import asdict

from original.torch_transformer_benchmark import TransformerConfig


OFFICIAL_SHAPES = (
    TransformerConfig(64, 128, 128, 4, 128, 4, True),
    TransformerConfig(1, 128, 128, 4, 128, 4, True),
    TransformerConfig(4, 128, 128, 4, 128, 4, True),
    TransformerConfig(16, 128, 128, 4, 128, 4, True),
    TransformerConfig(128, 128, 128, 4, 128, 4, True),
    TransformerConfig(10_000, 128, 128, 4, 128, 4, True),
    TransformerConfig(64, 128, 32, 4, 32, 4, True),
    TransformerConfig(64, 128, 1024, 4, 1024, 4, True),
    TransformerConfig(64, 128, 128, 1, 128, 4, True),
    TransformerConfig(64, 128, 128, 2, 128, 4, True),
    TransformerConfig(64, 128, 128, 16, 128, 4, True),
    TransformerConfig(64, 32, 128, 4, 128, 4, True),
    TransformerConfig(64, 1024, 128, 4, 128, 4, True),
    TransformerConfig(32, 100_000, 1024, 16, 1024, 2, True),
)


def shape_dict(index: int) -> dict[str, int | bool]:
    """Return a JSON-friendly official shape by one-based index."""
    return asdict(OFFICIAL_SHAPES[index - 1])

