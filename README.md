# Glow Library
Set of functional tools for easier prototyping

## Overview
...

## Installation

For basic installation use:

```bash
pip install glow
```
<details>
<summary>Specific versions with additional requirements</summary>

```bash
pip install glow[io]  # For I/O extras
pip install glow[all]  # For all
```
</details>
Glow is compatible with: Python 3.9+, PyTorch 1.11+.
Tested on ArchLinux, Ubuntu 18.04/20.04, Windows 10/11.

## Structure
- `glow.*` - Core parts, available out the box
- `glow.io.*` - I/O wrappers to access data in convenient formats

## Core features
- `glow.mapped` - convenient tool to parallelize computations
- `glow.memoize` - use if you want to reduce number of calls for any function

## IO features

### `glow.io.Sound` - playable sound wrapper
<details>

```python
from datetime import timedelta

import numpy as np
from glow.io import Sound

array: np.ndarray
sound = Sound(array, rate=44100)  # Wrap np.ndarray
sound = Sound.load('test.flac')  # Load sound into memory from file

# Get properties
rate: int = sound.rate
duration: timedelta = sound.duration
dtype: np.dtype = sound.dtype

 # Plays sound through default device, supports Ctrl-C for interruption
sound.play()
```
</details>
