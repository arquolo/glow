import functools
import sys
import time
import weakref
from enum import Enum

import numpy as np
import pytest

from glow import sizeof


def test_none():
    assert sizeof(None) == 0


def test_numbers():
    size_int = sizeof(42)
    assert size_int > 0

    size_float = sizeof(3.14)
    assert size_float > 0


def test_strings():
    s = 'hello'
    size = sizeof(s)
    assert size == sys.getsizeof(s)


def test_bytes():
    b = b'hello'
    size = sizeof(b)
    assert size == sys.getsizeof(b)


def test_list():
    lst = [1, 2, 3]
    size = sizeof(lst)

    expected = sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst)
    assert size == expected


def test_nested_list():
    nested = [1, [2, 3], 4]
    size = sizeof(nested)
    assert size > sys.getsizeof(nested)


def test_dict():
    d = {'a': 1, 'b': 2}
    size = sizeof(d)

    expected = (
        sys.getsizeof(d)
        + sum(sys.getsizeof(k) for k in d)
        + sum(sys.getsizeof(v) for v in d.values())
    )
    assert size == expected


def test_tuple():
    t = (1, 2, 3)
    size = sizeof(t)

    expected = sys.getsizeof(t) + sum(sys.getsizeof(x) for x in t)
    assert size == expected


def test_set():
    s = {1, 2, 3}
    size = sizeof(s)

    expected = sys.getsizeof(s) + sum(sys.getsizeof(x) for x in s)
    assert size == expected


def test_simple_class():
    class Simple:
        def __init__(self):
            self.x = 10
            self.y = 'hello'

    obj = Simple()
    size = sizeof(obj)

    expected = sys.getsizeof(obj) + sys.getsizeof(10) + sys.getsizeof('hello')
    assert size >= expected


def test_class_with_slots():
    class Slotted:
        __slots__ = ('a', 'b')

        def __init__(self):
            self.a = 100
            self.b = 'test'

    obj = Slotted()
    size = sizeof(obj)

    assert size > sys.getsizeof(obj)


def test_self_referencing_object():
    class SelfRef:
        def __init__(self):
            self.ref: SelfRef | None = None

    obj = SelfRef()
    obj.ref = obj

    size = sizeof(obj)
    assert size > 0


def test_functools_lru_cache():
    @functools.lru_cache
    def expensive(x):
        return x * 2

    expensive(10)
    size = sizeof(expensive)
    assert size > 0


def test_singleton_types():
    assert sizeof(type) == 0
    assert sizeof(bool) == 0
    assert sizeof(print) == 0  # FunctionType
    assert sizeof(sys) == 0  # ModuleType


def test_property_descriptor():
    class WithProperty:
        @property
        def x(self):
            return 42

    obj = WithProperty()
    size = sizeof(obj)
    assert size == sys.getsizeof(obj) + sys.getsizeof(obj.__dict__)


def test_small_numpy_array():
    arr = np.array([1, 2, 3])
    size = sizeof(arr)
    assert size >= arr.nbytes


def test_large_numpy_array():
    arr = np.zeros((100, 100))
    size = sizeof(arr)

    assert size >= arr.nbytes


def test_numpy_array_view():
    arr = np.zeros((100, 100))
    view = arr[::2, ::2]
    size_view = sizeof(view)
    assert size_view == sys.getsizeof(view) + sys.getsizeof(view.base)


def test_mixed_collection():
    mixed = [42, 'string', {'key': [1, 2, 3]}, (1.5, 2.5)]

    size = sizeof(mixed)
    assert size > 0


def test_deeply_nested():
    deep = []
    current = deep
    for _ in range(100):
        current.append([])
        current = current[-1]

    size = sizeof(deep)
    assert size > 0


def test_cyclic_references():
    lst: list[int | list] = [1, 2, 3]
    lst.append(lst)

    lst2: list[int | list] = [4, 5]
    lst.append(lst2)
    lst2.append(lst)

    size = sizeof(lst)
    assert size > 0


def test_cached_function():
    @functools.cache
    def cached_func(x):
        return x**2

    cached_func(5)
    size = sizeof(cached_func)
    assert size > 0


def test_enum_type():
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    assert sizeof(Color.RED) == sys.getsizeof(Color.RED)


def test_empty_collections():
    assert sizeof([]) == sys.getsizeof([])
    assert sizeof({}) == sys.getsizeof({})
    assert sizeof(set()) == sys.getsizeof(set())


def test_zero_sized_objects():
    class Empty:
        __slots__ = ()

    obj = Empty()
    size = sizeof(obj)
    assert size >= 0


@pytest.mark.parametrize(
    'value', [None, True, False, 0, 0.0, '', b'', [], (), set(), {}]
)
def test_various_empty_values(value):
    size = sizeof(value)
    assert size >= 0


def test_large_collection():
    large_list = list(range(1000))
    size = sizeof(large_list)
    assert size > 0


def test_complete_object_graph():
    class Node:
        def __init__(self, value):
            self.value = value
            self.children = []

        def add_child(self, child):
            self.children.append(child)
            return self

    root = Node(1)
    child1 = Node(2)
    child2 = Node(3)

    root.add_child(child1).add_child(child2)
    child1.add_child(Node(4))

    size = sizeof(root)
    assert size > 0


def test_object_with_weakref():
    class WeakRefHolder:
        def __init__(self):
            self.ref = weakref.ref(self)

    holder = WeakRefHolder()

    size = sizeof(holder)
    assert size > 0


def test_torch_integration():
    try:
        import torch
    except ImportError:
        pytest.skip('PyTorch not installed')

    tensor = torch.zeros(10, 10)
    size = sizeof(tensor)
    assert size > 0


def test_performance_large_object():
    large_dict = {f'key{i}': list(range(100)) for i in range(100)}

    start_time = time.perf_counter()
    size = sizeof(large_dict)
    end_time = time.perf_counter()

    assert size > 0
    assert end_time - start_time < 1.0
