__all__ = ['Parallel', 'delayed', 'cpu_count', 'wrap_non_picklable_objects']

__version__ = '0.14.2.dev0'

import os

from loky import cpu_count, wrap_non_picklable_objects

from .parallel import Parallel, delayed

os.environ.setdefault('KMP_INIT_AT_FORK', 'FALSE')
