__all__ = ['Sound']

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from .. import chunked


def _play(
    arr: np.ndarray, rate: int, blocksize: int = 1024, bufsize: int = 20
) -> None:
    """Plays audio from array. Supports interruption via Crtl-C."""
    import sounddevice as sd

    q: Queue[np.ndarray | None] = Queue(bufsize)
    ev = Event()

    def callback(out: np.ndarray, *_) -> None:
        if (data := q.get()) is None:
            raise sd.CallbackAbort

        size = len(data)
        out[:size] = data
        if size < len(out):
            out[size:] = 0
            raise sd.CallbackStop

    stream = sd.OutputStream(
        rate, blocksize, callback=callback, finished_callback=ev.set
    )

    fmt = '{percentage:3.0f}% |{bar}| [{elapsed}<{remaining}]'
    blocks = chunked(arr, blocksize)

    with ExitStack() as s:
        s.enter_context(stream)  # Close stream
        s.callback(ev.wait)  # Wait for completion
        s.callback(q.put, None)  # Close queue

        for data in s.enter_context(
            tqdm(blocks, leave=False, smoothing=0, bar_format=fmt)
        ):
            q.put(data)


@dataclass(repr=False, frozen=True)
class Sound[S: np.number]:
    """Wraps numpy.array to be playable as sound

    Parameters:
    - rate - sample rate to use for playback

    Usage:
    ```
    import numpy as np

    sound = Sound.load('test.flac')

    # Get properties
    rate: int = sound.rate
    dtype: np.dtype = sound.dtype

    # Could be played like:
    import sounddevice as sd
    sd.play(sound, sound.rate)

    # Or like this, if you need Ctrl-C support
    sound.play()

    # Extract underlying array
    raw = sound.raw

    # Same result
    raw = np.array(sound)
    ```
    """

    data: npt.NDArray[S]
    rate: int = 44_100

    def __post_init__(self) -> None:
        if self.data.ndim not in (1, 2):
            raise ValueError(
                f'Sound must be 1d (mono) or 2d array, got {self.data.shape}'
            )
        if self.data.ndim == 1:
            object.__setattr__(self, 'data', self.data[:, None])
        if self.data.shape[-1] not in (1, 2):
            raise ValueError(
                f'Only mono/stereo is supported, got {self.channels} channels'
            )
        if self.data.dtype not in ('i1', 'i2', 'i4', 'f4'):
            raise ValueError(
                'Only int8/int16/int32/float32 sound dtype is supported. '
                f'Got {self.data.dtype}'
            )

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> timedelta:
        return timedelta(seconds=self.data.shape[0] / self.rate)

    def __repr__(self) -> str:
        duration = self.duration
        channels = self.channels
        dtype = self.data.dtype
        return f'{type(self).__name__}({duration=!s}, {channels=}, {dtype=!s})'

    def __array__(self) -> npt.NDArray[S]:
        return self.data

    def play(self, blocksize=1024) -> None:
        """Plays audio from array. Supports interruption via Crtl-C."""
        _play(self.data, self.rate, blocksize=blocksize)

    @classmethod
    def load(cls, path: Path | str) -> 'Sound':
        _check_fmt(path)
        import soundfile

        data, rate = soundfile.read(path)
        return cls(data.astype('f'), rate)

    def save(self, path: Path | str) -> None:
        _check_fmt(path)
        import soundfile

        soundfile.write(path, self.data, self.rate)


def _check_fmt(path: Path | str) -> None:
    fmt = Path(path).suffix.lower()
    if fmt not in ('.flac', '.wav'):
        raise ValueError(f'Only FLAC/WAV is supported, got {fmt}')
