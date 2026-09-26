from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator


class BasePreparer:
    def __init__(self, accelerator: Accelerator):
        self._accelerator_ref = weakref.ref(accelerator)

    @property
    def accelerator(self) -> Accelerator | None:
        return self._accelerator_ref()

    def can_prepare(self, obj: Any) -> bool:
        raise NotImplementedError

    def prepare(self, obj: Any, **kwargs) -> Any:
        raise NotImplementedError
