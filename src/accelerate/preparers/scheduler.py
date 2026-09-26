from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch.optim.lr_scheduler import LRScheduler

from .base import BasePreparer

if TYPE_CHECKING:
    from accelerate import Accelerator


class SchedulerPreparer(BasePreparer):
    def can_prepare(self, obj: Any) -> bool:
        return isinstance(obj, LRScheduler)

    def prepare(self, obj: LRScheduler, **kwargs) -> Any:
        return self.accelerator.prepare_scheduler(obj, **kwargs)
