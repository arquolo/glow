from __future__ import annotations  # until 3.10

__all__ = ['Compose', 'Transform']

from typing import Any

import numpy as np


class Transform:
    _map: dict[str, Any] = {}

    def __init_subclass__(cls) -> None:
        cls._map = {'image': cls.apply, 'mask': cls.apply_to_mask}

    def prepare(self, rg: np.random.Generator, **data) -> dict[str, Any]:
        return {}

    def apply(self, image: np.ndarray, **extra) -> np.ndarray:
        return image

    def apply_to_mask(self, mask: np.ndarray, **extra) -> np.ndarray:
        return mask

    def __call__(self, rg: np.random.Generator, **data) -> dict:
        extra = self.prepare(rg, **data)
        return {
            k: self._map[k](self, v, **extra) if v is not None else None
            for k, v in data.items()
        }


class Compose(Transform):
    def __init__(self, *transforms: tuple[float, Transform]):
        self.transforms = transforms

    def __call__(self, rg: np.random.Generator, **data) -> dict:
        for prob, fn in self.transforms:
            if rg.uniform() <= prob:
                data = fn(rg, **data)
        return data
