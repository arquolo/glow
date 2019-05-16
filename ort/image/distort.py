import numpy as np
from numba import jit


@jit
def dither(image, bits=3, kind='stucki'):
    mat = np.array({
        'jarvis-judice-ninke':
            [[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]],
        'sierra':
            [[0, 0, 0, 5, 3], [2, 4, 5, 4, 2], [0, 2, 3, 2, 0]],
        'stucki':
            [[0, 0, 0, 8, 4], [2, 4, 8, 4, 2], [1, 2, 4, 2, 1]],
    }[kind], dtype='f')
    mat /= mat.sum()

    if image.ndim == 3:
        mat = mat[..., None]
        channel_pad = [(0, 0)]
    else:
        channel_pad = []
    image = np.pad(image, [(0, 2), (2, 2)] + channel_pad, mode='constant')

    dtype = image.dtype
    if dtype == 'uint8':
        max_value = 256
        image = image.astype('i2')
    else:
        max_value = 1
    quant = max_value / 2 ** bits

    for y in range(image.shape[0] - 2):
        for x in range(2, image.shape[1] - 2):
            old = image[y, x]
            new = np.floor(old / quant) * quant
            delta = ((old - new) * mat).astype(image.dtype)
            image[y: y + 3, x - 2: x + 3] += delta
            image[y, x] = new

    image = image[:-2, 2:-2]
    return image.clip(0, max_value - quant).astype(dtype)


def bit_noise(image, keep=4, count=8, seed=None):
    rng = np.random.RandomState(seed)  # pylint: disable=no-member
    residual = image.copy()
    out = np.zeros_like(image)
    for n in range(1, 1 + count):
        thres = .5 ** n
        plane = (residual >= thres).astype(residual.dtype) * thres
        out += (plane if n <= keep
                else rng.choice([0, thres], size=image.shape))
        residual -= plane
    return out
