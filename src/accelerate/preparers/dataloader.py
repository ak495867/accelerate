from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.utils.data

from .base import BasePreparer

if TYPE_CHECKING:
    from accelerate import Accelerator


class DataLoaderPreparer(BasePreparer):
    def can_prepare(self, obj: Any) -> bool:
        return isinstance(obj, torch.utils.data.DataLoader)

    def prepare(
        self,
        obj: torch.utils.data.DataLoader,
        device_placement: Any = None,
        slice_fn_for_dispatch: Any = None,
        **kwargs,
    ) -> Any:
        return self.accelerator.prepare_data_loader(
            obj,
            device_placement=device_placement,
            slice_fn_for_dispatch=slice_fn_for_dispatch,
            **kwargs,
        )
