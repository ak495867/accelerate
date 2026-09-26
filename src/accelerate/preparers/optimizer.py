from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import BasePreparer

if TYPE_CHECKING:
    from accelerate import Accelerator


class OptimizerPreparer(BasePreparer):
    def can_prepare(self, obj: Any) -> bool:
        return isinstance(obj, torch.optim.Optimizer)

    def prepare(self, obj: torch.optim.Optimizer, device_placement: Any = None, **kwargs) -> Any:
        return self.accelerator.prepare_optimizer(obj, device_placement=device_placement, **kwargs)
