__all__ = ['TemporaryResourcesManager']

import atexit
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from loky.backend import resource_tracker

_SHARED_MEM_FS = Path('/dev/shm')
_SHARED_MEM_FS_MIN_SIZE = int(2e9)

_RM_SUBDIRS_RETRY_TIME = 0.1
_RM_SUBDIRS_N_RETRY = 5


def _get_temp_dir(pool_name: str) -> Path:
    default_pool = Path(tempfile.gettempdir(),
                        pool_name).expanduser().absolute()
    if not _SHARED_MEM_FS.exists():
        return default_pool
    assert sys.platform != 'win32'
    try:
        shm_stats = os.statvfs(_SHARED_MEM_FS)
        if shm_stats.f_bsize * shm_stats.f_bavail <= _SHARED_MEM_FS_MIN_SIZE:
            return default_pool

        pool_folder = _SHARED_MEM_FS / pool_name
        if not pool_folder.exists():
            pool_folder.mkdir(parents=True)
            pool_folder.rmdir()
        return pool_folder.expanduser().absolute()

    except OSError:
        return default_pool


def delete_folder(folder: Path, allow_non_empty=True):
    if not folder.is_dir():
        return
    for errs_left in range(_RM_SUBDIRS_N_RETRY)[::-1]:
        try:
            if not allow_non_empty and [*folder.iterdir()]:
                raise OSError
            shutil.rmtree(folder)
            return
        except OSError:
            if not errs_left:
                raise
        time.sleep(_RM_SUBDIRS_RETRY_TIME)


class TemporaryResourcesManager:
    def __init__(self):
        self._id = uuid4().hex
        self._cached = {}

    def init(self):
        self.set_context(uuid4().hex)

    def set_context(self, context_id):
        self._current_id = context_id
        if context_id in self._cached:
            return

        folder = _get_temp_dir(f'joblib_{os.getpid()}_{self._id}_{context_id}')
        pool_module_name = delete_folder.__module__
        resource_tracker.register(folder.as_posix(), 'folder')

        def _cleanup(allow_non_empty=True):
            delete_folder = __import__(
                pool_module_name, fromlist=['delete_folder']).delete_folder
            try:
                delete_folder(folder, allow_non_empty=allow_non_empty)
                resource_tracker.unregister(folder.as_posix(), 'folder')
            except OSError:
                pass

        self._cached[context_id] = (folder, atexit.register(_cleanup))

    def resolve(self):
        return self._cached[self._current_id][0]

    def unlink(self, context_id):
        folder, cleanup = self._cached.pop(context_id)
        if folder.exists():
            for p in folder.iterdir():
                resource_tracker.maybe_unlink(p.as_posix(), 'file')
            cleanup(False)
            atexit.unregister(cleanup)

    def unlink_all(self):
        for context_id in [*self._cached]:
            self.unlink(context_id)

    def unregister_all(self):
        while self._cached:
            _, (folder, cleanup) = self._cached.popitem()
            if folder.exists():
                for p in folder.iterdir():
                    resource_tracker.unregister(p.as_posix(), 'file')
            cleanup()
            atexit.unregister(cleanup)
