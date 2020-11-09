# Collection of tools for easier prototyping with deep learning extensions

## `io.TiledImage` - reader for multiscale images (svs, tiff, etc...)
replaces [`torchslide`](https://github.com/arquolo/torchslide) (now deprecated)

```python
from glow.io import TiledImage

slide = TiledImage('test.svs')
shape: 'Tuple[int, int]' = slide.shape
scales: 'Tuple[int, int]' = slide.scales
image: np.ndarray = slide[:2048, :2048].view()  # Get numpy.ndarray
```
