__all__ = ['TiledImage']

import contextlib
import ctypes
import os
import sys
import weakref
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Tuple, Type, Union
from unittest import mock

import numpy as np

from .. import call_once, memoize

_TIFF: Any = None
_OSD: Any = None


def _patch_path(prefix):
    if sys.platform != 'win32':
        return contextlib.nullcontext()
    if sys.version_info >= (3, 8):
        return os.add_dll_directory(prefix)
    return mock.patch.dict(os.environ,
                           {'PATH': f'{prefix};{os.environ["PATH"]}'})


@call_once
def _setup_libs():
    global _TIFF, _OSD
    prefix = Path(__file__).parent / 'libs'
    pattern = ((prefix / '{}-{}.dll').as_posix()
               if sys.platform == 'win32' else '{}.so.{}')
    names = map(pattern.format, ('libtiff', 'libopenslide'), (5, 0))

    with _patch_path(prefix):
        _TIFF, _OSD = map(ctypes.CDLL, names)

    (_TIFF.TIFFOpenW if sys.platform == 'win32' else
     _TIFF.TIFFOpen).restype = ctypes.POINTER(ctypes.c_ubyte)

    _TIFF.TIFFSetErrorHandler(None)
    _OSD.openslide_open.restype = ctypes.POINTER(ctypes.c_ubyte)
    _OSD.openslide_get_error.restype = ctypes.c_char_p
    _OSD.openslide_get_property_value.restype = ctypes.c_char_p


class Color(Enum):
    MINISBLACK = 1
    RGB = 2
    YCBCR = 6


class Codec(Enum):
    NONE = 1
    CCITTRLE = 2
    CCITTFAX3 = CCITT_T4 = 3
    CCITTFAX4 = CCITT_T6 = 4
    LZW = 5
    OJPEG = 6
    JPEG = 7
    ADOBE_DEFLATE = 8
    NEXT = 32766
    CCITTRLEW = 32771
    PACKBITS = 32773
    THUNDERSCAN = 32809
    IT8CTPAD = 32895
    IT8LW = 32896
    IT8MP = 32897
    IT8BL = 32898
    PIXARFILM = 32908
    PIXARLOG = 32909
    DEFLATE = 32946
    DCS = 32947
    JPEG2000_YUV = 33003
    JPEG2000_RGB = 33005
    JBIG = 34661
    SGILOG = 34676
    SGILOG24 = 34677
    JP2000 = 34712


class _Meta(type):
    @memoize(
        10_485_760,
        policy='lru',
        key_fn=lambda _, name: Path(name).resolve().as_posix())
    def __call__(cls, name):
        return super().__call__(name)


class TiledImage(metaclass=_Meta):
    name: str
    _num_levels = 0
    _suffixes = set()  # type: ignore
    _type_for: Dict[str, Type['TiledImage']] = {}

    def __init_subclass__(cls: Type['TiledImage'], extensions: str) -> None:
        for ext in extensions.split():
            cls._type_for[f'.{ext}'] = cls

    def __new__(cls, name: str) -> 'TiledImage':
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(name)
        if cls is not TiledImage:
            return super().__new__(cls)
        try:
            return super().__new__(cls._type_for[path.suffix])
        except KeyError:
            raise ValueError(f'Unknown file format {path}') from None

    def __init__(self, name: str) -> None:
        self.name = Path(name).as_posix()
        self._lock = RLock()
        self._spec = dict(self._init_spec(self._num_levels))

    def _init_spec(self, num_levels: int):
        assert num_levels
        specs = [self._get_spec(level) for level in range(num_levels)]
        shape = specs[0]['shape']
        for spec in specs:
            if 'tile' in spec:
                spec['step'] = step = shape[0] // spec['shape'][0]
                yield step, spec

    def _get_spec(self, level) -> dict:
        raise NotImplementedError

    def _get_patch(self, box, **spec) -> np.ndarray:
        raise NotImplementedError

    @property
    def shape(self):
        return self._spec[1]['shape']

    @property
    def scales(self):
        return [*self._spec.keys()]

    @property
    def spacing(self) -> List[float]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f'{type(self).__name__}'
                f"('{self.name}', shape={self.shape}, scales={self.scales})")

    @contextlib.contextmanager
    def _directory(self, level):
        with self._lock:
            yield

    def __getitem__(self,
                    slices: Union[Tuple[slice, slice], slice]) -> np.ndarray:
        if isinstance(slices, slice):
            slices = (slices, slice(None, None, slices.step))

        step = slices[0].step
        step = 1 if step is None else step
        spec = self._spec[step]

        box = [((0 if s.start is None else s.start // step),
                (lim if s.stop is None else s.stop // step))
               for s, lim in zip(slices, spec['shape'])]

        with self._directory(spec['level']):
            return self._get_patch(box, **spec)


class _OpenslideImage(
        TiledImage,
        extensions='bif mrxs ndpi scn svs svsslide tif tiff vms vmu'):
    def __init__(self, name: str) -> None:
        _setup_libs()
        self._ptr = _OSD.openslide_open(name.encode())
        err = _OSD.openslide_get_error(self._ptr)
        if err:
            raise ValueError(err)
        weakref.finalize(self, _OSD.openslide_close, self._ptr)

        bg_color_hex = _OSD.openslide_get_property_value(
            self._ptr, 'openslide.background-color')
        self._bg_color = (
            np.full(3, 255, dtype='u1')
            if bg_color_hex is None else bg_color_hex)

        self._num_levels = _OSD.openslide_get_level_count(self._ptr)
        super().__init__(name)

    def _get_spec(self, level):
        y, x = ctypes.c_int64(), ctypes.c_int64()
        _OSD.openslide_get_level_dimensions(self._ptr, level,
                                            *map(ctypes.byref, (x, y)))
        return {'level': level, 'shape': [y.value, x.value], 'tile': True}

    def _get_patch(self, box, step=0, level=0, **spec):
        (y_min, y_max), (x_min, x_max) = box

        data = np.empty((y_max - y_min, x_max - x_min, 4), dtype='u1')
        _OSD.openslide_read_region(
            self._ptr, data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            x_min * step, y_min * step, level, x_max - x_min, y_max - y_min)

        rgb = data[..., 2::-1]
        opacity = data[..., 3:]
        return np.where(
            opacity, (255 * rgb.astype('u2') / opacity.clip(1)).astype('u1'),
            self._bg_color)

    @property
    def spacing(self) -> List[float]:
        mpp = (
            _OSD.openslide_get_property_value(self._ptr, f'openslide.mpp-{ax}')
            for ax in 'yx')
        return [float(m) if m else None for m in mpp]


class _TiffImage(TiledImage, extensions='svs tif tiff'):
    def __init__(self, name: str) -> None:
        _setup_libs()
        self._ptr = (
            _TIFF.TIFFOpenW(name, b'rm') if sys.platform == 'win32' else
            _TIFF.TIFFOpen(name.encode(), b'rm'))
        assert self._ptr
        weakref.finalize(self, _TIFF.TIFFClose, self._ptr)

        self._num_levels = _TIFF.TIFFNumberOfDirectories(self._ptr)
        super().__init__(name)

    def tag(self, type_, tag: int, default=None):
        value = type_()
        res = _TIFF.TIFFGetField(self._ptr, ctypes.c_uint32(tag),
                                 ctypes.byref(value))
        if not res and default is not None:
            return default
        return value.value

    @contextlib.contextmanager
    def _directory(self, level):
        with super()._directory(level):
            _TIFF.TIFFSetDirectory(self._ptr, level)
            yield
            _TIFF.TIFFFreeDirectory(self._ptr)

    def _get_spec(self, level) -> dict:
        _TIFF.TIFFSetDirectory(self._ptr, level)

        planar_config = self.tag(ctypes.c_uint16, 284)
        if planar_config != 1:
            raise TypeError(f'Level {level} is not contiguous!')

        spec = {
            'level': level,
            'shape': [self.tag(ctypes.c_uint32, tag) for tag in (257, 256)],
            'bits_per_sample': self.tag(ctypes.c_uint16, 258),
            'sample_format': self.tag(ctypes.c_uint16, 339, default=1),
        }

        photometric = Color(self.tag(ctypes.c_uint16, 262))
        spec['samples_per_pixel'] = (
            self.tag(ctypes.c_uint16, 277)
            if photometric in (Color.MINISBLACK, Color.RGB) else 4)

        if _TIFF.TIFFIsTiled(self._ptr):
            spec['tile'] = [
                self.tag(ctypes.c_uint32, tag) for tag in (323, 322)
            ]

        spec['compression'] = compression = Codec(
            self.tag(ctypes.c_uint16, 259))
        if compression is Codec.JPEG:
            count = ctypes.c_int()
            jpeg_tables = ctypes.c_char_p()
            if _TIFF.TIFFGetField(
                    self._ptr, 347, ctypes.byref(count),
                    ctypes.byref(jpeg_tables)) and count.value > 4:
                jpt = ctypes.cast(
                    jpeg_tables,
                    ctypes.POINTER(
                        ctypes.c_uint8 * (count.value - 2))).contents
                spec['jpt'] = bytes(jpt)

        return spec

    def _get_tile(self, y, x, **spec) -> np.ndarray:
        if spec['compression'] in (Codec.JPEG2000_RGB, Codec.JPEG2000_YUV):
            return self._get_tile_jpeg(y, x, **spec)

        # ! buggy, incorrect jpeg tables/colormaps
        # if spec['compression'] is Codec.JPEG:
        #     return self._get_tile_jpeg(y, x, **spec)

        return self._get_tile_native(y, x, **spec)

    def _get_tile_native(self, y, x, *, tile, samples_per_pixel,
                         **spec) -> np.ndarray:
        data = np.empty(tile + [samples_per_pixel], dtype='u1')
        isok = _TIFF.TIFFReadTile(
            self._ptr, data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), x,
            y, 0, 0)
        assert isok
        return data

    def _get_tile_jpeg(self, y, x, *, jpt=None, **_) -> np.ndarray:
        offset = _TIFF.TIFFComputeTile(self._ptr, x, y, 0, 0)

        tile_byte_counts = self.tag(ctypes.c_void_p, 325)
        nbytes = ctypes.cast(tile_byte_counts,
                             ctypes.POINTER(ctypes.c_ulonglong))[offset]

        if jpt is not None:
            nbytes += len(jpt)
            data = (ctypes.c_uint8 * nbytes)()
            data_ptr = ctypes.cast(
                ctypes.addressof(data) + len(jpt) - 2,
                ctypes.POINTER(ctypes.c_uint8))
            _TIFF.TIFFReadRawTile(self._ptr, offset, data_ptr, nbytes)
            data[:len(jpt)] = jpt
        else:
            data = (ctypes.c_uint8 * nbytes)()
            _TIFF.TIFFReadRawTile(self._ptr, offset, ctypes.byref(data),
                                  nbytes)

        import imagecodecs
        return imagecodecs.imread(np.asarray(data))

    def _get_patch(self, box, **spec) -> np.ndarray:
        shape = spec['shape']
        tile = spec['tile']
        samples_per_pixel = spec['samples_per_pixel']

        out = np.zeros(
            [(high - low) for low, high in box] + [samples_per_pixel],
            dtype='u1')

        bmin, bmax = np.transpose(box).clip(0, shape)
        dy, dx = bmin
        axes = map(slice, bmin // tile * tile, bmax, tile)
        grid = np.mgrid[tuple(axes)].reshape(2, -1).T

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
    def spacing(self) -> List[float]:
        mpp = (self.tag(ctypes.c_float, x) for x in (283, 282))
        return [(10000 / m) if m else None for m in mpp]
