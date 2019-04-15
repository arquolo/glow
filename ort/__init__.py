from .debug import sprint, summary, timer, trace_module

from .memory import (CacheAbc, Cache, CacheLRU,
                     sizeof)

from .threads import bufferize, maps, shared_call, threadlocal
from .threads_v2 import map_t, map_in_background

from .utils import as_iter, grouped, once_per_instance, unique


# trace_module('random')
# trace_module('numpy.random')
