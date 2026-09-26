from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch

from .versions import is_torch_version


def is_flex_attention_available() -> bool:
    if not is_torch_version(">=", "2.5.0"):
        return False
    return hasattr(torch.nn.attention, "flex_attention")


class FlexAttentionContextParallel:
    def __init__(self, cp_size: int = 1, cp_rank: int = 0):
        self.cp_size = max(1, cp_size)
        self.cp_rank = cp_rank

    def partition_sequence(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.cp_size <= 1:
            return tensor
        seq_len = tensor.size(dim)
        chunk_size = seq_len // self.cp_size
        start = self.cp_rank * chunk_size
        end = (self.cp_rank + 1) * chunk_size if self.cp_rank != self.cp_size - 1 else seq_len
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(start, end)
        return tensor[tuple(slices)]

    def create_block_mask(
        self,
        score_mod: Callable[..., Any],
        B: int,
        H: int,
        Q_LEN: int,
        KV_LEN: int,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> Any:
        if not is_flex_attention_available():
            raise RuntimeError("FlexAttention requires PyTorch >= 2.5.0")
        from torch.nn.attention.flex_attention import create_block_mask

        return create_block_mask(score_mod, B, H, Q_LEN, KV_LEN, device=device, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Optional[Callable[..., Any]] = None,
        block_mask: Optional[Any] = None,
        scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not is_flex_attention_available():
            raise RuntimeError("FlexAttention requires PyTorch >= 2.5.0")
        from torch.nn.attention.flex_attention import flex_attention

        if self.cp_size > 1 and block_mask is None and score_mod is not None:
            B, H, Q_LEN, _ = query.shape
            _, _, KV_LEN, _ = key.shape
            block_mask = self.create_block_mask(score_mod, B, H, Q_LEN, KV_LEN, device=query.device)

        return flex_attention(query, key, value, score_mod=score_mod, block_mask=block_mask, scale=scale, **kwargs)
