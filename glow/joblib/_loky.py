__all__ = ['get_memmapping_executor']

import numpy as np
from loky.reusable_executor import _ReusablePoolExecutor

from ._reduction import ArrayForwardReducer, reduce_array_backward
from ._resources import TemporaryResourcesManager

_TYPES = (np.ndarray, np.memmap)
_IDLE_WORKER_TIMEOUT = 300
_temp_manager = None


def get_memmapping_executor(n_jobs, initializer=None, initargs=(),
                            env=None, max_nbytes=1e6,
                            **_):
    manager = TemporaryResourcesManager()
    reduce_array_forward = ArrayForwardReducer(max_nbytes, True,
                                               manager.resolve)
    _executor, executor_is_reused = \
        _ReusablePoolExecutor.get_reusable_executor(
            n_jobs,
            timeout=_IDLE_WORKER_TIMEOUT,
            job_reducers=dict.fromkeys(_TYPES, reduce_array_forward),
            result_reducers=dict.fromkeys(_TYPES, reduce_array_backward),
            initializer=initializer,
            initargs=initargs,
            env=env)

    global _temp_manager
    if _temp_manager is None or not executor_is_reused:
        _temp_manager = manager
    return _executor, _temp_manager
