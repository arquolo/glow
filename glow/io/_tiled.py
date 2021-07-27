from __future__ import annotations

__all__ = ['TiledImage', 'read_tiled']

import ctypes
import os
import sys
import weakref
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, NamedTuple

import cv2
import numpy as np
from lxml.builder import ElementMaker
from lxml.etree import tostring

from .. import call_once, memoize

_TIFF: Any = None
_OSD: Any = None
_TYPE_REGISTRY: dict[str, type[TiledImage]] = {}


def _patch_path(prefix):
    if sys.platform != 'win32':
        return nullcontext()
    return os.add_dll_directory(prefix)


@call_once
def _setup_libs():
    global _TIFF, _OSD
    prefix = Path(__file__).parent / 'libs'
    pattern = ((prefix / '{}-{}.dll').as_posix()
               if sys.platform == 'win32' else '{}.so.{}')
    names = map(pattern.format, ('libtiff', 'libopenslide'), (5, 0))

    with _patch_path(prefix):
        _TIFF, _OSD = map(ctypes.CDLL, names)

    struct_t = ctypes.POINTER(ctypes.c_ubyte)
    (_TIFF.TIFFOpenW
     if sys.platform == 'win32' else _TIFF.TIFFOpen).restype = struct_t

    # _TIFF.TIFFSetErrorHandler(None)
    _OSD.openslide_open.restype = struct_t
    _OSD.openslide_get_error.restype = ctypes.c_char_p
    _OSD.openslide_get_property_value.restype = ctypes.c_char_p
    _OSD.openslide_get_property_names.restype = ctypes.POINTER(ctypes.c_char_p)


# ---------------------------- DeepZoom Image API ----------------------------


class _Dzi(NamedTuple):
    slide: TiledImage
    offset: tuple[int, ...]
    size: tuple[int, ...]
    tile: int = 256

    @property
    def shape(self) -> tuple[int, ...]:
        return self.slide.shape

    def xml(self, fmt: str = 'jpeg') -> str:
        e = ElementMaker()
        return tostring(
            e.Image(
                e.Size(Height=str(self.size[0]), Width=str(self.size[1])),
                TileSize=str(self.tile),
                Overlap='0',
                Format=fmt,
                xmlns='http://schemas.microsoft.com/deepzoom/2008'))

    def at(self, level: int, iy_ix: tuple[int, ...]) -> np.ndarray:
        scale = 2 ** (max(self.slide.shape[:2]).bit_length() - level)
        tile_0 = self.tile * scale
        return self.slide.at(
            tuple(p + ip * tile_0 for p, ip in zip(self.offset, iy_ix)),
            self.tile, scale)


# --------------------------------- decoders ---------------------------------


class _Decoder:
    def __init__(self):
        self._lock = RLock()

    def _get_spec(self, level: int) -> dict:
        raise NotImplementedError

    def _get_patch(self, box, **spec) -> np.ndarray:
        raise NotImplementedError

    @property
    def spacing(self) -> float | None:
        raise NotImplementedError

    @contextmanager
    def _directory(self, level: int):
        with self._lock:
            yield


class TiledImage(_Decoder):
    def __init_subclass__(cls: type[TiledImage], extensions: str) -> None:
        _TYPE_REGISTRY.update({f'.{ext}': cls for ext in extensions.split()})

    def __init__(self, path: Path, num_levels: int = 0) -> None:
        if self.__class__ is TiledImage:
            raise RuntimeError('TiledImage is not for direct construction. '
                               'Use read_tiled() factory function')
        super().__init__()
        self.path = path
        self._num_levels = num_levels
        self._spec = dict(self._init_spec(num_levels))

    def __reduce__(self) -> str | tuple:
        return TiledImage, (self.path, )  # self._slices)

    def _init_spec(self, num_levels: int):
        assert num_levels > 0
        zero, *specs = (self._get_spec(level) for level in range(num_levels))
        h, w, c = zero['shape']
        for spec in [zero, *specs]:
            if spec:
                spec['step'] = step = round(h / spec['shape'][0])
                spec['shape'] = *(s // step for s in (h, w)), c
                yield step, spec

    @property
    def shape(self) -> tuple[int, ...]:
        return self._spec[1]['shape']

    @property
    def scales(self) -> tuple[int, ...]:
        return tuple(self._spec.keys())

    def __repr__(self) -> str:
        return (f'{type(self).__name__}'
                f"('{self.path}', shape={self.shape}, scales={self.scales})")

    def __getitem__(self, slices: tuple[slice, slice] | slice) -> np.ndarray:
        """Retrieves tile"""
        if isinstance(slices, slice):
            slices = (slices, slice(None, None, None))
        if any(s.step == 0 for s in slices):
            raise ValueError('slice step cannot be zero')

        step0, step1 = tuple(s.step or 1 for s in slices)
        if step0 != step1:
            raise ValueError('unequal slices steps are not supported')
        if step0 < 0:
            raise ValueError('slice step should be positive')

        step, spec = max((s, spec) for s, spec in self._spec.items()
                         if s == 1 or s <= step0)
        box = [((s.start or 0) // step,
                (lim if s.stop is None else s.stop // step))
               for s, lim in zip(slices, spec['shape'])]

        with self._directory(spec['level']):
            image = self._get_patch(box, **spec)

        ratio = step / step0
        if ratio == 1.:
            return image
        return cv2.resize(
            image,
            tuple(round(ratio * s) for s in image.shape[1::-1]),
            interpolation=cv2.INTER_AREA)

    def thumbnail(self, scale: int = None) -> tuple[np.ndarray, int]:
        if scale is None:
            scale = self.scales[-1]
        return self[::scale, ::scale], scale

    def at(self,
           z0_yx_offset: tuple[int, ...],
           dsize: int,
           scale: float = 1) -> np.ndarray:
        """Read square region starting with offset"""
        step = max((s for s in self._spec if s <= scale), default=1)
        box = [(int(c) // step, int(c + dsize * scale) // step)
               for c in z0_yx_offset]

        spec = self._spec[step]
        with self._directory(spec['level']):
            image = self._get_patch(box, **spec)

        if scale == 1:
            return image
        return cv2.resize(image, (dsize, dsize), interpolation=cv2.INTER_AREA)

    def dzi(self, tile: int = 256) -> _Dzi:
        raise NotImplementedError


# ----------------------------- OpenSlide decoder -----------------------------


class _OpenslideImage(
        TiledImage,
        extensions='bif mrxs ndpi scn svs svsslide tif tiff vms vmu'):
    def __init__(self, path: Path) -> None:
        _setup_libs()
        self._ptr = _OSD.openslide_open(path.as_posix().encode())
        if not self._ptr:
            raise ValueError(f'File {path} cannot be opened')

        if err := _OSD.openslide_get_error(self._ptr):
            raise ValueError(err)
        weakref.finalize(self, _OSD.openslide_close, self._ptr)

        bg_hex = self._tag(b'openslide.background-color') or b'FFFFFF'
        self.bg_color = np.array([int(bg_hex, 16)], dtype='>u4').view('u1')[1:]

        num_levels = _OSD.openslide_get_level_count(self._ptr)
        super().__init__(path, num_levels)

    def _tag(self, name: bytes) -> bytes:
        return _OSD.openslide_get_property_value(self._ptr, name)

    def _get_spec(self, level: int) -> dict:
        y, x = ctypes.c_int64(), ctypes.c_int64()
        _OSD.openslide_get_level_dimensions(self._ptr, level,
                                            *map(ctypes.byref, (x, y)))
        return {'level': level, 'shape': (y.value, x.value, 3)}

    def _get_patch(self, box, step=0, level=0, **spec):
        (y_min2, y_max2), (x_min2, x_max2) = box

        box = np.array(box)
        valid_box = box.T.clip([0, 0], spec['shape'][:2]).T

        (y_min, y_max), (x_min, x_max) = valid_box
        if y_min == y_max or x_min == x_max:  # Patch is outside slide
            return np.broadcast_to(self.bg_color,
                                   (y_max2 - y_min2, x_max2 - x_min2, 3))

        bgra = np.empty((y_max - y_min, x_max - x_min, 4), dtype='u1')
        bgra_ptr = ctypes.c_void_p(bgra.ctypes.data)
        _OSD.openslide_read_region(self._ptr, bgra_ptr, int(x_min * step),
                                   int(y_min * step), level,
                                   int(x_max - x_min), int(y_max - y_min))

        bgra = cv2.cvtColor(bgra, cv2.COLOR_mRGBA2RGBA)  # premultiplied alpha
        rgb = bgra[..., 2::-1].copy()
        rgb[bgra[..., 3] == 0] = self.bg_color  # alpha blending

        offsets = np.abs(valid_box - box).ravel().tolist()
        if not any(offsets):
            return rgb
        bg_color = self.bg_color.tolist()
        return cv2.copyMakeBorder(
            rgb, *offsets, borderType=cv2.BORDER_CONSTANT, value=bg_color)

    @property
    def metadata(self) -> dict[str, str]:
        m: dict[str, str] = {}
        if names := _OSD.openslide_get_property_names(self._ptr):
            i = 0
            while name := names[i]:
                m[name.decode()] = self._tag(name).decode()
                i += 1
        return m

    @property
    def spacing(self) -> float | None:
        mpp = (self._tag(f'openslide.mpp-{ax}'.encode()) for ax in 'yx')
        if s := [float(m) for m in mpp if m]:
            return float(np.mean(s))
        return None

    def dzi(self, tile: int = 256) -> _Dzi:
        offset = tuple(
            int(self._tag(f'openslide.bounds-{ax}'.encode()) or 0)
            for ax in 'yx')
        size = tuple(
            int(self._tag(f'openslide.bounds-{ax}'.encode()) or lim)
            for ax, lim in zip(('height', 'width'), self.shape[:2]))
        return _Dzi(self, offset, size, tile)


# ------------------------------- Tiff decoder -------------------------------


class Color(Enum):
    MINISBLACK = 1
    RGB = 2
    YCBCR = 6


class Codec(Enum):
    RAW = 1
    CCITT = 2
    CCITTFAX3 = 3
    CCITTFAX4 = 4
    LZW = 5
    JPEG_OLD = 6
    JPEG = 7
    ADOBE_DEFLATE = 8
    RAW_16 = 32771
    PACKBITS = 32773
    THUNDERSCAN = 32809
    DEFLATE = 32946
    DCS = 32947
    JPEG2000_YUV = 33003
    JPEG2000_RGB = 33005
    JBIG = 34661
    SGILOG = 34676
    SGILOG24 = 34677
    JPEG2000 = 34712
    LZMA = 34925
    ZSTD = 50000
    WEBP = 50001


class TiffTag:
    SIZE = (257, 256)
    BITDEPTH = 258
    COMPRESSION = 259
    COLORSPACE = 262
    DESCRIPTION = 270
    SAMPLES_PER_PIXEL = 277
    PLANAR = 284
    TILE = (323, 322)
    RESOLUTION = (283, 282)
    JPEG_TABLES = 347
    TILE_BYTE_COUNTS = 325
    SAMPLE = 339
    BACKGROUND_COLOR = 434


# FIXME: Get around slides from choked SVS encoder
class _TiffImage(TiledImage, extensions='tif tiff'):
    def __init__(self, path: Path) -> None:
        _setup_libs()
        # TODO: use memmap instead of libtiff
        spath = path.as_posix()
        self._ptr = (
            _TIFF.TIFFOpenW(spath, b'rm') if sys.platform == 'win32' else
            _TIFF.TIFFOpen(spath.encode(), b'rm'))
        if not self._ptr:
            raise ValueError(f'File {path} cannot be opened')

        weakref.finalize(self, _TIFF.TIFFClose, self._ptr)

        num_levels = _TIFF.TIFFNumberOfDirectories(self._ptr)
        super().__init__(path, num_levels)

    def _tag(self, type_, tag: int):
        value = type_()
        _TIFF.TIFFGetField(self._ptr, ctypes.c_uint32(tag),
                           ctypes.byref(value))
        return value.value

    @contextmanager
    def _directory(self, level: int):
        with super()._directory(level):
            _TIFF.TIFFSetDirectory(self._ptr, level)
            try:
                yield
            finally:
                _TIFF.TIFFFreeDirectory(self._ptr)

    def _get_spec(self, level) -> dict:
        _TIFF.TIFFSetDirectory(self._ptr, level)

        if not _TIFF.TIFFIsTiled(self._ptr):
            return {}

        if self._tag(ctypes.c_uint16, TiffTag.PLANAR) != 1:
            raise TypeError(f'Level {level} is not contiguous!')

        desc = self._tag(ctypes.c_char_p, TiffTag.DESCRIPTION)
        desc = (desc or b'').decode().replace('\r\n', '|').split('|')

        color = Color(self._tag(ctypes.c_uint16, TiffTag.COLORSPACE))
        spp = (
            self._tag(ctypes.c_uint16, TiffTag.SAMPLES_PER_PIXEL)
            if color in [Color.MINISBLACK, Color.RGB] else 4)

        bg_hex = b'FFFFFF'
        bg_color_ptr = ctypes.c_char_p()
        if _TIFF.TIFFGetField(self._ptr, TiffTag.BACKGROUND_COLOR,
                              ctypes.byref(bg_color_ptr)):
            bg_hex = ctypes.string_at(bg_color_ptr, 3)
        # print(bg_hex)

        spec = {
            'level':
                level,
            'shape': (*(self._tag(ctypes.c_uint32, tag)
                        for tag in TiffTag.SIZE), spp),
            'tile': (*(self._tag(ctypes.c_uint32, tag)
                       for tag in TiffTag.TILE), spp),
            'color':
                color,
            'background':
                np.array([int(bg_hex, 16)], dtype='>u4').view('u1')[1:],
            'description':
                desc,
            'compression':
                Codec(self._tag(ctypes.c_uint16, TiffTag.COMPRESSION)),
        }
        if spec['compression'] is Codec.JPEG:
            count = ctypes.c_int()
            jpt_ptr = ctypes.c_char_p()
            if _TIFF.TIFFGetField(self._ptr, TiffTag.JPEG_TABLES,
                                  ctypes.byref(count),
                                  ctypes.byref(jpt_ptr)) and count.value > 4:
                spec['jpt'] = ctypes.string_at(jpt_ptr, count.value)

        tbc_ptr = ctypes.POINTER(ctypes.c_uint64)()
        if _TIFF.TIFFGetField(self._ptr, TiffTag.TILE_BYTE_COUNTS,
                              ctypes.byref(tbc_ptr)):
            num_tiles = _TIFF.TIFFNumberOfTiles(self._ptr)
            tbc = np.ctypeslib.as_array(tbc_ptr, (num_tiles, ))
            spec['tile_sizes'] = tbc.copy()

        return spec

    def _get_tile(self, y, x, **spec) -> np.ndarray:
        offset = _TIFF.TIFFComputeTile(self._ptr, x, y, 0, 0)
        nbytes = int(spec['tile_sizes'][offset])

        # print(self.path.name, spec['level'], (y, x),
        #       (y // spec['tile'][0], x // spec['tile'][1]), offset)

        if not nbytes:  # If nothing to read, don't read
            # TODO: here should be reading from more-detailed level
            # * If tile is empty on level N,
            # * then all tiles on levels >N are invalid, even when not empty
            return np.broadcast_to(spec['background'], spec['tile'])

        # pview = self._get_patch(self._spec[spec['step'] - 1])

        if spec['compression'] not in [Codec.JPEG2000_RGB, Codec.JPEG2000_YUV]:
            image = np.empty(spec['tile'], dtype='u1')
            isok = _TIFF.TIFFReadTile(self._ptr,
                                      ctypes.c_void_p(image.ctypes.data), x, y,
                                      0, 0)
            assert isok != -1
            return image

        data = ctypes.create_string_buffer(nbytes)
        _TIFF.TIFFReadRawTile(self._ptr, offset, data, len(data))

        import imagecodecs
        if jpt := spec.get('jpt'):
            return imagecodecs.jpeg_decode(
                data, tables=jpt, colorspace=spec['color'].value)
        return imagecodecs.imread(data)

    def _get_patch(self, box, **spec) -> np.ndarray:
        *tile, spp = spec['tile']

        dy, dx = (low for low, _ in box)
        out = np.ascontiguousarray(
            np.broadcast_to(spec['background'],
                            [(high - low) for low, high in box] + [spp]))

        hw = spec['shape'][:2]
        bmin, bmax = np.transpose(box).clip(0, hw)  # type: ignore
        axes = *map(slice, bmin // tile * tile, bmax, tile),
        grid = np.mgrid[axes].transpose(1, 2, 0)  # [H, W, 2]
        if not grid.size:
            return out

        # * block, a bit slower
        # patches = np.block([[[self._get_tile(*iyx, **spec)] for iyx in iyxs]
        #                     for iyxs in grid.tolist()])
        # dst = *map(slice, bmin - (dy, dx), bmax - (dy, dx)),
        # src = *map(slice, bmin % tile, patches.shape[:2] - (-bmax) % tile),
        # out[dst] = patches[src]
        # return out

        # * iterate
        grid = grid.reshape(-1, 2)
        for (iy, ix), (ty_min, tx_min), (ty_max, tx_max) in zip(
                grid.tolist(),
                grid.clip(bmin).tolist(),
                np.clip(grid + tile, 0, bmax).tolist()):
            patch = self._get_tile(iy, ix, **spec)
            out[ty_min - dy:ty_max - dy,
                tx_min - dx:tx_max - dx] = patch[ty_min - iy:ty_max - iy,
                                                 tx_min - ix:tx_max - ix]
        return out

    @property
    def spacing(self) -> float | None:
        s = [(10_000 / m) for t in TiffTag.RESOLUTION
             if (m := self._tag(ctypes.c_float, t))]
        if s:
            return float(np.mean(s))
        for token in self._spec[1]['description']:
            if 'MPP' in token:
                return float(token.split('=')[-1].strip())
        return None

    def dzi(self, tile: int = 256) -> _Dzi:
        return _Dzi(self, (0, 0), self.shape[:2], tile=tile)


# ----------------------------- generic factory -----------------------------


@memoize(
    10_485_760,
    policy='lru',
    key_fn=lambda name: Path(name).resolve().as_posix())
def read_tiled(anypath: Path | str) -> TiledImage:
    """Reads multi-scale images.

    Usage:
    ```
    from glow.io import read_tiled

    slide = read_tiled('test.svs')
    shape: tuple[int, ...] = slide.shape
    scales: tuple[int, ...] = slide.scales

    # Get numpy.ndarray
    image = slide[:2048, :2048]
    ```
    """
    if (path := Path(anypath)).exists():
        try:
            return _TYPE_REGISTRY[path.suffix](path)
        except KeyError:
            raise ValueError(f'Unknown file format {path}') from None
    raise FileNotFoundError(path)
