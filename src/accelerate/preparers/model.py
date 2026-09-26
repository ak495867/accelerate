from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import BasePreparer

if TYPE_CHECKING:
    from accelerate import Accelerator


class ModelPreparer(BasePreparer):
    def can_prepare(self, obj: Any) -> bool:
        return isinstance(obj, torch.nn.Module)

    def prepare(
        self,
        obj: torch.nn.Module,
        device_placement: Any = None,
        evaluation_mode: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        return self.accelerator.prepare_model(
            obj,
            device_placement=device_placement,
            evaluation_mode=evaluation_mode,
            **kwargs,
        )
