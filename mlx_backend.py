"""MLX implementation used for the Apple-M2 extreme-sequence regime."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import torch


def _array(tensor: torch.Tensor, dtype) -> mx.array:
    value = mx.array(np.asarray(tensor.detach().cpu().float().numpy()))
    return value.astype(dtype)


def _linear(x, weight, bias):
    return mx.addmm(bias, x, weight.T)


def _layer_norm(x, weight, bias, eps: float = 1e-5):
    return mx.fast.layer_norm(x, weight, bias, eps=eps)


def _gelu_exact(x):
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


class MLXTransformer:
    """Inference-only Transformer with challenge-compatible PyTorch weights."""

    def __init__(
        self, torch_model, dtype=mx.float16, fuse_qkv: bool = False,
        pad_attention_to_64: bool = False, mixed_linear_fp16: bool = False,
        force_fused_attention: bool = False,
    ):
        self.config = torch_model.config
        self.dtype = dtype
        self.compute_dtype = mx.float16 if mixed_linear_fp16 else dtype
        self.mixed_linear_fp16 = mixed_linear_fp16
        self.force_fused_attention = force_fused_attention
        self.pad_attention_to_64 = pad_attention_to_64
        self.layers = []
        for source in torch_model.layers:
            layer = {
                    "norm1_w": _array(source.norm1.weight, dtype),
                    "norm1_b": _array(source.norm1.bias, dtype),
                    "o_w": _array(source.attention.out_proj.weight, self.compute_dtype),
                    "o_b": _array(source.attention.out_proj.bias, self.compute_dtype),
                    "norm2_w": _array(source.norm2.weight, dtype),
                    "norm2_b": _array(source.norm2.bias, dtype),
                    "ff1_w": _array(source.ffn_in.weight, self.compute_dtype),
                    "ff1_b": _array(source.ffn_in.bias, self.compute_dtype),
                    "ff2_w": _array(source.ffn_out.weight, self.compute_dtype),
                    "ff2_b": _array(source.ffn_out.bias, self.compute_dtype),
                }
            if fuse_qkv:
                layer["qkv_w"] = mx.concatenate(
                    [_array(source.attention.q_proj.weight, self.compute_dtype),
                     _array(source.attention.k_proj.weight, self.compute_dtype),
                     _array(source.attention.v_proj.weight, self.compute_dtype)], axis=0
                )
                layer["qkv_b"] = mx.concatenate(
                    [_array(source.attention.q_proj.bias, self.compute_dtype),
                     _array(source.attention.k_proj.bias, self.compute_dtype),
                     _array(source.attention.v_proj.bias, self.compute_dtype)], axis=0
                )
            else:
                layer.update(
                    q_w=_array(source.attention.q_proj.weight, dtype),
                    q_b=_array(source.attention.q_proj.bias, dtype),
                    k_w=_array(source.attention.k_proj.weight, dtype),
                    k_b=_array(source.attention.k_proj.bias, dtype),
                    v_w=_array(source.attention.v_proj.weight, dtype),
                    v_b=_array(source.attention.v_proj.bias, dtype),
                )
            self.layers.append(layer)
        self.final_w = _array(torch_model.final_norm.weight, dtype)
        self.final_b = _array(torch_model.final_norm.bias, dtype)
        mx.eval(self.layers, self.final_w, self.final_b)

    def _project(self, x, weight, bias):
        if self.mixed_linear_fp16:
            return _linear(x.astype(mx.float16), weight, bias)
        return _linear(x, weight, bias)

    def __call__(self, x):
        heads = self.config.num_heads
        head_dim = self.config.d_model // heads
        scale = head_dim**-0.5
        for layer in self.layers:
            normed = _layer_norm(x, layer["norm1_w"], layer["norm1_b"])
            if "qkv_w" in layer:
                q, k, v = mx.split(
                    self._project(normed, layer["qkv_w"], layer["qkv_b"]), 3, axis=-1
                )
            else:
                q = _linear(normed, layer["q_w"], layer["q_b"])
                k = _linear(normed, layer["k_w"], layer["k_b"])
                v = _linear(normed, layer["v_w"], layer["v_b"])
            batch, seq_len, _ = q.shape
            q = q.reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
            effective_head_dim = head_dim
            if self.pad_attention_to_64 and head_dim < 64:
                padding = [(0, 0), (0, 0), (0, 0), (0, 64 - head_dim)]
                q, k, v = mx.pad(q, padding), mx.pad(k, padding), mx.pad(v, padding)
                effective_head_dim = 64
            attention = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask="causal",
                force_fused=(
                    effective_head_dim in {64, 72, 80, 96, 128, 192, 256}
                    and (
                        self.dtype == mx.float16
                        or self.pad_attention_to_64
                        or self.force_fused_attention
                    )
                ),
            )
            if effective_head_dim != head_dim:
                attention = attention[..., :head_dim]
            attention = attention.transpose(0, 2, 1, 3).reshape(
                batch, seq_len, self.config.d_model
            )
            x = x + self._project(attention, layer["o_w"], layer["o_b"]).astype(self.dtype)
            normed = _layer_norm(x, layer["norm2_w"], layer["norm2_b"])
            hidden = _gelu_exact(self._project(normed, layer["ff1_w"], layer["ff1_b"]))
            x = x + self._project(hidden, layer["ff2_w"], layer["ff2_b"]).astype(self.dtype)
        return _layer_norm(x, self.final_w, self.final_b)
