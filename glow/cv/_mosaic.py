from __future__ import annotations

__all__ = ['Mosaic']

import dataclasses
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import TypeVar

import cv2
import numpy as np
from tqdm.auto import tqdm

from .. import chunked, mapped
from ..io import TiledImage

Coord = tuple[int, int]
_T = TypeVar('_T')


def _probs_to_hsv(prob: np.ndarray) -> np.ndarray:
    h, w, c = prob.shape
    vmax = 1 if prob.dtype == 'u1' else 255
    hsv = cv2.merge([
        (prob.argmax(-1).astype('f4') * (127 / c)).astype('u1'),
        np.full((h, w), 255, dtype='u1'),
        (prob.max(-1) * vmax).astype('u1'),
    ])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _get_weight(step: int, overlap: int) -> np.ndarray:
    assert overlap
    pad = np.arange(0.5, 0.5 + overlap) / overlap
    return np.r_[pad, np.ones(step - overlap), pad[::-1]].astype('f4')


@dataclass
class Mosaic:
    """
    Helper to split image to tiles and process them.

    Parameters:
    - step - Step between consecutive tiles
    - overlap - Count of pixels that will be shared among overlapping tiles
    """

    step: int
    overlap: int

    def __post_init__(self):
        assert 0 <= self.overlap <= self.step
        assert self.overlap % 2 == 0  # That may be optional

    def as_tiles(self,
                 data: np.ndarray | TiledImage,
                 scale: int = 1,
                 num_workers: int = 1) -> _Tiler:
        """Read tiles from data using scale as stride"""
        shape = tuple(s // scale for s in data.shape[:2])
        ishape = tuple(
            len(range(0, s + self.overlap, self.step)) for s in shape)
        cells = np.ones(ishape, dtype=np.bool_)

        return _Tiler(self.step, self.overlap, shape, cells, data, scale,
                      num_workers)

    def get_weight(self) -> np.ndarray:
        return _get_weight(self.step, self.overlap)


@dataclass
class _Sized(Mosaic):
    shape: tuple[int, ...]
    cells: np.ndarray

    @property
    def ishape(self) -> tuple[int, ...]:
        return self.cells.shape

    def _enumerate(self, it: Iterable[_T]) -> Iterable[tuple[Coord, _T]]:
        return zip(np.argwhere(self.cells).tolist(), it)

    def __len__(self) -> int:
        return int(self.cells.sum())


@dataclass
class _Tiler(_Sized):
    data: np.ndarray | TiledImage
    scale: int
    num_workers: int

    def select(self, mask: np.ndarray, scale: int) -> _Tiler:
        """Drop tiles where `mask` is 0"""
        assert mask.ndim == 2
        mask = mask.astype('u1')

        ih, iw = self.ishape
        step = (self.step * self.scale) // scale
        pad = (self.overlap * self.scale // 2) // scale

        mh, mw = (ih * step), (iw * step)
        if mask.shape[:2] != (mh, mw):
            mask_pad = [(0, s1 - s0) for s0, s1 in zip(mask.shape, (mh, mw))]
            mask = np.pad(mask, mask_pad)[:mh, :mw]

        if self.overlap:
            kernel = np.ones((3, 3), dtype='u1')
            mask = cv2.dilate(mask, kernel, iterations=pad)

        if pad:
            mask = np.pad(mask[:-pad, :-pad], [[pad, 0], [pad, 0]])

        cells = mask.reshape(ih, step, iw, step).any((1, 3))
        return dataclasses.replace(self, cells=cells)

    def _get_tile(self, iy: int, ix: int) -> np.ndarray:
        """Read non-overlapping tile of source image"""
        scale = self.scale
        (y0, y1), (x0, x1) = ((i * self.step - self.overlap,
                               (i + 1) * self.step) for i in (iy, ix))
        if iy and self.cells[iy - 1, ix]:
            y0 += self.overlap
        if ix and self.cells[iy, ix - 1]:
            x0 += self.overlap
        return self.data[y0 * scale:y1 * scale:scale,  # type: ignore
                         x0 * scale:x1 * scale:scale]

    def _rejoin_tiles(
            self, image_parts: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        """Joins non-overlapping parts to tiles"""
        assert self.overlap
        cells = np.pad(self.cells, [(0, 1), (0, 1)])
        row: defaultdict[int, np.ndarray] = defaultdict()

        for (iy, ix), part in self._enumerate(image_parts):
            # Lazy init, first part is always whole
            if row.default_factory is None:
                row.default_factory = partial(np.zeros, part.shape, part.dtype)

            if (tile := row.pop(ix, None)) is not None:
                tile[-part.shape[0]:, -part.shape[1]:] = part
            else:
                tile = part

            yield tile

            if cells[iy, ix + 1]:
                row[ix + 1][:, :self.overlap] = tile[:, -self.overlap:]
            if cells[iy + 1, ix]:
                row[ix][:self.overlap, :] = tile[-self.overlap:, :]

    def _raw_iter(self) -> Iterator[np.ndarray]:
        ys, xs = np.where(self.cells)
        parts = mapped(
            self._get_tile, ys, xs, num_workers=self.num_workers, latency=0)
        return self._rejoin_tiles(parts) if self.overlap else iter(parts)

    def _offset(self) -> Sequence[Coord]:
        offsets = np.argwhere(self.cells) * self.step - self.overlap
        return (offsets * self.scale).tolist()

    def __iter__(self) -> Iterator[tuple[Coord, np.ndarray]]:
        """
        Yield complete tiles built from source image.
        Each tile will have size `(step + overlap)`
        """
        return zip(self._offset(), self._raw_iter())

    def apply(self,
              func: Callable[[Iterable[np.ndarray]], list[np.ndarray]],
              scale: int,
              batch_size: int = 1,
              num_workers: int = 1,
              weighting: bool = True) -> _Merger:
        """
        Applies `func` to tiles in batched way.
        Returns processed tiles with masked edge artifacts together with
        *true* tile offsets.

        `func` should be thread-safe, accept sequence of HWC-formatted
        ndarrays, and return sequence of HWC-formatted ndarrays with
        float dtype.

        Parameters:
        - scale - Scale of output.
        - batch_size - Batch size to use for grouping tiles for func.
        - weighting - Whether to apply weight for each tile,
          or don't when func already applies it.
        """
        ratio = scale / self.scale
        step, overlap, *shape = (
            int(s * ratio) for s in (self.step, self.overlap) + self.shape)

        weight = _get_weight(step, overlap) if overlap and weighting else None

        image_parts = self._raw_iter()
        chunks = chunked(image_parts, batch_size)
        batches = mapped(func, chunks, num_workers=num_workers, latency=0)
        results = chain.from_iterable(batches)

        return _Merger(step, overlap, tuple(shape), self.cells, scale,
                       zip(self._offset(), results), weight)


@dataclass
class _Merger(_Sized):
    scale: int
    source: Iterable[tuple[Coord, np.ndarray]]
    weight: np.ndarray | None

    _cells: np.ndarray = field(init=False)
    _row: dict[int, np.ndarray] = field(default_factory=dict)
    _joint: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self._cells = np.pad(self.cells, [(0, 1), (0, 1)])

    def _update(self, iy: int, ix: int, y: int, x: int,
                tile: np.ndarray) -> tuple[Coord, np.ndarray]:
        """Blends edges of overlapping tiles and returns non-overlapping
        parts of it"""
        if self.weight is not None:
            assert tile.dtype.kind == 'f'
            tile *= self.weight[:, None, None]
            tile *= self.weight[None, :, None]

        if iy and self._cells[iy - 1, ix]:  # have North
            north = self._row.pop(ix)
            tile[:self.overlap, self.step - north.shape[1]:self.step] += north
        else:
            tile = tile[self.overlap:]  # strip North
            y += self.overlap * self.scale

        if ix and self._cells[iy, ix - 1]:  # have West
            west = self._joint.pop()
            if self._cells[iy + 1, [ix - 1, ix]].all():
                tile[-west.shape[0]:, :self.overlap] += west
            else:  # strip South-West
                tile[-west.shape[0] -
                     self.overlap:-self.overlap, :self.overlap] += west
        else:
            tile = tile[:, self.overlap:]  # strip West
            x += self.overlap * self.scale

        tile, east = np.split(tile, [-self.overlap], axis=1)
        if self._cells[iy, ix + 1]:  # have East
            if not (iy and self._cells[iy - 1, [ix, ix + 1]].all()):
                east = east[-self.step:]  # strip North-East
            if not self._cells[iy + 1, [ix, ix + 1]].all():
                east = east[:-self.overlap]  # strip South-East
            self._joint.append(east)

        tile, south = np.split(tile, [-self.overlap])
        if self._cells[iy + 1, ix]:  # have South
            if not (ix and self._cells[[iy, iy + 1], ix - 1].all()):
                # strip South-West
                south = south[:, -(self.step - self.overlap):]
            self._row[ix] = south

        return (y, x), tile

    def _crop(
        self, source: Iterable[tuple[Coord, np.ndarray]]
    ) -> Iterator[tuple[Coord, np.ndarray]]:
        shape = self.shape
        scale = self.scale
        for (y, x), out in source:
            yield (y, x), out[:shape[0] - y // scale, :shape[1] - x // scale]

    def __iter__(self):
        isource = self.source
        if self.overlap:
            isource = (self._update(*iyx, *yx, out)
                       for iyx, (yx, out) in self._enumerate(isource))
        return self._crop(isource)

    def with_view(
            self, view: np.ndarray,
            v_scale: int) -> Iterator[tuple[Coord, np.ndarray, np.ndarray]]:
        assert v_scale >= self.scale
        ratio = v_scale // self.scale

        for (y, x), out in self:
            tw, th = out.shape[:2]
            v = view[y // v_scale:, x // v_scale:][:tw // ratio, :th // ratio]
            yield (y, x), out, v

    def fuse(self, pool: int = 1, progress: bool = False) -> np.ndarray:
        """
        Merges tiles to image, which has Hue proportional to ArgMax(-1),
        and Value proporional to Max(-1).

        Created only for debugging.
        """
        scale = self.scale * pool
        result = np.zeros((*(s // pool for s in self.shape), 3), dtype='u1')

        for (y, x), tile in (tqdm(self, leave=False) if progress else self):
            h, w = (s // pool for s in tile.shape[:2])
            im = tile[::pool, ::pool][:h, :w]
            result[y // scale:, x // scale:][:h, :w] = _probs_to_hsv(im)

        return result
