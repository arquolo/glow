from __future__ import annotations

__all__ = ['Sound']

from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np
from tqdm.auto import tqdm

from .. import chunked


def _play(arr: np.ndarray,
          rate: int,
          blocksize: int = 1024,
          bufsize: int = 20):
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
        rate, blocksize, callback=callback, finished_callback=ev.set)

    fmt = '{percentage:3.0f}% |{bar}| [{elapsed}<{remaining}]'
    blocks = chunked(arr, blocksize)

    with ExitStack() as s:
        s.enter_context(stream)  # Close stream
        s.callback(ev.wait)  # Wait for completion
        s.callback(q.put, None)  # Close queue

        for data in s.enter_context(
                tqdm(blocks, leave=False, smoothing=0, bar_format=fmt)):
            q.put(data)


@dataclass(repr=False)
class Sound:
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
    raw: np.ndarray
    rate: int = 44_100
    duration: timedelta = field(init=False)
    channels: int = field(init=False)

    def __post_init__(self):
        assert self.raw.ndim == 2
        assert 0 < self.raw.shape[-1] <= 2
        assert self.raw.dtype in {'int8', 'int16', 'int32', 'float32'}
        num_samples, self.channels = self.raw.shape
        self.duration = timedelta(seconds=num_samples / self.rate)

    def __repr__(self) -> str:
        duration = self.duration
        channels = self.channels
        dtype = self.raw.dtype
        return f'{type(self).__name__}({duration=!s}, {channels=}, {dtype=!s})'

    def __array__(self) -> np.ndarray:
        return self.raw

    def play(self, blocksize=1024) -> None:
        """Plays audio from array. Supports interruption via Crtl-C."""
        _play(self.raw, self.rate, blocksize=blocksize)

    @classmethod
    def load(cls, path: Path | str) -> 'Sound':
        spath = str(path)
        assert spath.endswith('.flac')
        import soundfile

        data, rate = soundfile.read(spath)
        return cls(data.astype('float32'), rate)
