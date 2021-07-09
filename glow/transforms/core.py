from __future__ import annotations

__all__ = [
    'Compose', 'DualStageTransform', 'ImageTransform', 'MaskTransform',
    'Transform'
]

from typing import Any, Protocol, final

import numpy as np


class Transform(Protocol):
    def __call__(self, rng: np.random.Generator, /, **data) -> dict:
        raise NotImplementedError


class _SingleTransform(Transform):
    _key: str

    @final
    def __call__(self, rng: np.random.Generator, /, **data) -> dict:
        return {
            **data,
            self._key: getattr(self, self._key)(data[self._key], rng),
        }


class ImageTransform(_SingleTransform):
    _key = 'image'

    def image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class MaskTransform(_SingleTransform):
    _key = 'mask'

    def mask(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class DualStageTransform(Transform):
    _keys: frozenset[str] = frozenset({'image', 'mask'})

    def prepare(self, rng: np.random.Generator, /, **data) -> dict[str, Any]:
        return {}

    def image(self, image: np.ndarray, **params) -> np.ndarray:
        return image

    def mask(self, mask: np.ndarray, **params) -> np.ndarray:
        return mask

    @final
    def __call__(self, rng: np.random.Generator, /, **data) -> dict:
        if unknown_keys := {*data} - self._keys:
            raise ValueError(f'Got unknown keys in data: {unknown_keys}')
        params = self.prepare(rng, **data)
        return {
            key: getattr(self, key)(value, **params)
            for key, value in data.items() if value is not None
        }


@final
class Compose(Transform):
    probs: tuple[float, ...]
    funcs: tuple[Transform, ...]

    def __init__(self, *transforms: tuple[float, Transform] | Transform):
        # If probability is not set, it's 1
        self.probs, self.funcs = zip(
            *(item if isinstance(item, tuple) else (1, item)
              for item in transforms))

    def __call__(self, rng: np.random.Generator, /, **data) -> dict:
        choices = rng.binomial(1, self.probs).astype(bool)
        for func in np.array(self.funcs)[choices].tolist():
            data = func(rng, **data)
        return data
