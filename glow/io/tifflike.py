__all__ = ('TiledImage', )

import os
import contextlib
import ctypes
import enum
import itertools
import threading
import weakref
from pathlib import Path
from unittest import mock

import imagecodecs
import numpy as np

from .. import memoize

if os.name == 'nt':
    _prefix = Path(__file__).parent / 'libs'
    with mock.patch.dict(os.environ,
                         {'PATH': f'{_prefix};{os.environ["PATH"]}'}):
        tiff, openslide = (
            ctypes.CDLL((_prefix / name).as_posix())
            for name in ['libtiff-5.dll', 'libopenslide-0.dll'])
else:
    tiff, openslide = map(ctypes.CDLL, ['libtiff.so.5', 'libopenslide.so.0'])


class _CImage(ctypes.Structure):
    pass


if os.name == 'nt':
    tiff.TIFFOpenW.restype = ctypes.POINTER(_CImage)
else:
    tiff.TIFFOpen.restype = ctypes.POINTER(_CImage)
tiff.TIFFSetErrorHandler(None)
openslide.openslide_open.restype = ctypes.POINTER(_CImage)
openslide.openslide_get_error.restype = ctypes.c_char_p
openslide.openslide_get_property_value.restype = ctypes.c_char_p


class Color(enum.Enum):
    MINISBLACK = 1
    RGB = 2
    YCBCR = 6


class Codec(enum.Enum):
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

    def __new__(cls, name: str) -> 'TiledImage':
        if not Path(name).exists():
            raise FileNotFoundError(name)
        if cls is not TiledImage:
            return super().__new__(cls)

        for cls in TiledImage.__subclasses__():
            for suf in cls._suffixes:
                if name.endswith(suf):
                    return super().__new__(cls)

        raise ValueError(f'Unknown file format {name}')

    def __init__(self, name: str) -> None:
        self.name = Path(name).as_posix()
        self._lock = threading.RLock()
        self._spec = dict(self._init_spec(self._num_levels))

    def _init_spec(self, num_levels):
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

    def __repr__(self):
        return (f'{type(self).__name__}'
                f"('{self.name}', shape={self.shape}, scales={self.scales})")

    @contextlib.contextmanager
    def _directory(self, level):
        with self._lock:
            yield

    def __getitem__(self, slices):
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


class _TiffImage(TiledImage):
    _suffixes = {'.svs', '.tif', '.tiff'}

    def __init__(self, name: str) -> None:
        if os.name == 'nt':
            self._ptr = tiff.TIFFOpenW(name, b'rm')
        else:
            self._ptr = tiff.TIFFOpen(name.encode(), b'rm')
        assert self._ptr
        weakref.finalize(self, tiff.TIFFClose, self._ptr)

        self._num_levels = tiff.TIFFNumberOfDirectories(self._ptr)
        super().__init__(name)

    def tag(self, type_, tag: int, default=None):
        value = type_()
        res = tiff.TIFFGetField(self._ptr, ctypes.c_uint32(tag),
                                ctypes.byref(value))
        if not res and default is not None:
            return default
        return value.value

    @contextlib.contextmanager
    def _directory(self, level):
        with super()._directory(level):
            tiff.TIFFSetDirectory(self._ptr, level)
            yield
            tiff.TIFFFreeDirectory(self._ptr)

    def _get_spec(self, level) -> dict:
        tiff.TIFFSetDirectory(self._ptr, level)

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

        if tiff.TIFFIsTiled(self._ptr):
            spec['tile'] = [
                self.tag(ctypes.c_uint32, tag) for tag in (323, 322)
            ]

        spec['compression'] = compression = Codec(
            self.tag(ctypes.c_uint16, 259))
        if compression is Codec.JPEG:
            count = ctypes.c_int()
            jpeg_tables = ctypes.c_char_p()
            if tiff.TIFFGetField(
                    self._ptr, 347, ctypes.byref(count),
                    ctypes.byref(jpeg_tables)) and count.value > 4:
                jpt = ctypes.cast(
                    jpeg_tables,
                    ctypes.POINTER(
                        ctypes.c_uint8 * (count.value - 2))).contents
                spec['jpt'] = bytes(jpt)

        return spec

    def _get_tile(self, y, x, **spec):
        if spec['compression'] in (Codec.JPEG2000_RGB, Codec.JPEG2000_YUV):
            return self._get_tile_jpeg(y, x, **spec)

        # ! buggy, incorrect jpeg tables/colormaps
        # if spec['compression'] is Codec.JPEG:
        #     return self._get_tile_jpeg(y, x, **spec)

        return self._get_tile_native(y, x, **spec)

    def _get_tile_native(self, y, x, *, tile, samples_per_pixel, **spec):
        data = np.empty(tile + [samples_per_pixel], dtype=np.uint8)
        isok = tiff.TIFFReadTile(
            self._ptr, data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), x,
            y, 0, 0)
        assert isok
        return data

    def _get_tile_jpeg(self, y, x, *, compression, jpt=None, **spec):
        offset = tiff.TIFFComputeTile(self._ptr, x, y, 0, 0)

        tile_byte_counts = self.tag(ctypes.c_void_p, 325)
        nbytes = ctypes.cast(tile_byte_counts,
                             ctypes.POINTER(ctypes.c_ulonglong))[offset]

        if jpt is not None:
            nbytes += len(jpt)
            data = (ctypes.c_uint8 * nbytes)()
            data_ptr = ctypes.cast(
                ctypes.addressof(data) + len(jpt) - 2,
                ctypes.POINTER(ctypes.c_uint8))
            tiff.TIFFReadRawTile(self._ptr, offset, data_ptr, nbytes)
            data[:len(jpt)] = jpt
        else:
            data = (ctypes.c_uint8 * nbytes)()
            tiff.TIFFReadRawTile(self._ptr, offset, ctypes.byref(data), nbytes)

        return imagecodecs.imread(np.asarray(data))

    def _get_patch(self, box, **spec) -> np.ndarray:
        shape = spec['shape']
        tile = spec['tile']
        samples_per_pixel = spec['samples_per_pixel']

        out = np.zeros(
            [(high - low) for low, high in box] + [samples_per_pixel],
            dtype=np.uint8)

        dy, dx = (start for start, _ in box)

        axes = [
            range(min_ // tile_ * tile_, max_, tile_)
            for min_, max_, tile_ in (
                (*np.clip(s, 0, lim), tile_)
                for s, lim, tile_ in zip(box, shape, tile))
        ]
        for ii in itertools.product(*axes):
            (ty_min, ty_max), (tx_min, tx_max) = (
                (max(min_, i), min(max_, i + tile_, lim))
                for i, [min_, max_], tile_, lim in zip(ii, box, tile, shape))
            iy, ix = ii
            patch = self._get_tile(*ii, **spec)
            out[ty_min - dy:ty_max - dy,
                tx_min - dx:tx_max - dx] = patch[ty_min - iy:ty_max - iy,
                                                 tx_min - ix:tx_max - ix]

        return out


class _OpenslideImage(TiledImage):
    _suffixes = {
        '.bif', '.mrxs', '.ndpi', '.scn', '.svs', '.svslide', '.tif', '.tiff',
        '.vms', '.vmu'
    }

    def __init__(self, name: str) -> None:
        self._ptr = openslide.openslide_open(name.encode())
        err = openslide.openslide_get_error(self._ptr)
        if err:
            raise ValueError(err)
        weakref.finalize(self, openslide.openslide_close, self._ptr)

        bg_color_hex = openslide.openslide_get_property_value(
            self._ptr, 'openslide.background-color')
        self._bg_color = (
            np.full(3, 255, dtype=np.uint8)
            if bg_color_hex is None else bg_color_hex)

        self._num_levels = openslide.openslide_get_level_count(self._ptr)
        super().__init__(name)

    def _get_spec(self, level):
        y, x = ctypes.c_int64(), ctypes.c_int64()
        openslide.openslide_get_level_dimensions(self._ptr, level,
                                                 *map(ctypes.byref, (x, y)))
        return {
            'level': level,
            'shape': [y.value, x.value],
            'tile': True,
        }

    def _get_patch(self, box, step, level, **spec):
        (y_min, y_max), (x_min, x_max) = box

        data = np.empty((y_max - y_min, x_max - x_min, 4), dtype=np.uint8)
        openslide.openslide_read_region(
            self._ptr, data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            x_min * step, y_min * step, level, x_max - x_min, y_max - y_min)

        rgb = data[..., 2::-1]
        opacity = data[..., 3:]
        return np.where(
            opacity, (255 * rgb.astype('u2') / opacity.clip(1)).astype('u1'),
            self._bg_color)
