__all__ = ('get_gpu_state', )

import os
from typing import Iterable


def get_gpu_state():
    from py3nvml.py3nvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
    )
    nvmlInit()
    indices: Iterable[int]
    try:
        indices = map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except KeyError:
        indices = range(nvmlDeviceGetCount())
    devices = [nvmlDeviceGetHandleByIndex(i) for i in indices]
    limit = sum(nvmlDeviceGetMemoryInfo(dev).free for dev in devices)
    nvmlShutdown()
    return (limit // 2 ** 20), len(devices)
