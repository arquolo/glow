import pyaudio
import soundfile
import wrapt


class Sample(wrapt.ObjectProxy):  # pylint: disable=abstract-method
    """Wraps numpy.array to be playable as sound"""

    def __init__(self, data: 'numpy.ndarray', rate=44100):
        assert data.ndim == 2
        assert data.shape[-1] in (1, 2)
        assert data.dtype in ('int8', 'int16', 'int32', 'float32')

        super().__init__(data)
        self._self_rate = rate

    @property
    def rate(self):
        return self._self_rate

    def __repr__(self):
        class_name = type(self).__name__
        array_repr = repr(self.__wrapped__).split('\n')
        array_repr = ('\n' + ' ' * len(class_name) + ' ').join(array_repr)
        return f'{class_name}({array_repr}, rate={self.rate})'

    def play(self, chunk_size=1024) -> None:
        pyaudio_type = f'pa{str(self.dtype).title()}'

        audio = pyaudio.PyAudio()
        stream = audio.open(rate=self.rate,
                            format=getattr(pyaudio, pyaudio_type),
                            channels=self.shape[1], output=True)
        try:
            if len(self.data) > 10 * chunk_size:
                for offset in range(0, len(self), chunk_size):
                    stream.write(self[offset: offset + chunk_size].tobytes())
            else:
                stream.write(self.tobytes())
        finally:
            audio.close(stream)
            audio.terminate()

    @classmethod
    def load(cls, path) -> 'Sample':
        assert str(path).endswith('.flac')
        data, rate = soundfile.read(str(path))
        return Sample(data.astype('float32'), rate)
