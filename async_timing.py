import asyncio
import time
from collections import abc
from typing import Any, NoReturn, Self

from glow import memoize, time_this

DT = 1.0
FAIL = False


# wall=1, frame=1, susp=0, cpu=1, io=0 (wall=1, process=1, thread=1)
def _cpu_10() -> None:
    # busy += dt
    end = time.perf_counter() + DT
    while time.perf_counter() < end:
        sum(range(1000))


# wall=1, frame=1, susp=0, cpu=0, io=1
def _sleep_01() -> None:
    # idle += dt
    time.sleep(DT)


# wall=1, frame=0, susp=1, cpu=0, io=1
async def _asleep_01() -> None:
    # idle += dt
    await asyncio.sleep(DT)


# wall=2, frame=2, susp=0, cpu=1, io=1
def work_11() -> None:
    _cpu_10()
    _sleep_01()
    if FAIL:
        1 / 0


# wall=3, frame=2, susp=1, cpu=1, io=2
async def awork_12() -> None:
    _cpu_10()
    _sleep_01()
    await _asleep_01()
    if FAIL:
        1 / 0


class _Iter22iter:
    def __init__(self) -> None:
        self.it = iter(range(0, 4, 2))

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> '_SubIter22':  # wall=2, frame=2, cpu=1, io=1
        i = self.it.__next__()  # 0/0/0/0
        work_11()  # 2/2/1/1
        return _SubIter22(i)  # 0/0/0/0

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} at {id(self):x}>'


class _SubIter22:
    def __init__(self, i) -> None:
        self.it = iter(range(i, i + 2))

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:  # wall=2, frame=2, cpu=1, io=1
        i = self.it.__next__()  # 0/0/0/0
        work_11()  # 2/2/1/1
        return i  # 0/0/0/0

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} at {id(self):x}>'


tps = (
    abc.Iterator,
    abc.Generator,  # is iterator
    abc.Awaitable,
    abc.Coroutine,  # is awaitable
    abc.AsyncIterator,
    abc.AsyncGenerator,  # is async iterator
)


def _info[T](obj: T, tag: str = '') -> T:
    print(type(obj).__qualname__, obj, tag)

    bases = [
        tp.__name__
        for tp in tps
        if (
            issubclass(obj, tp)
            if isinstance(obj, type)
            else isinstance(obj, tp)
        )
    ]
    base = {
        *dir(object()),
        *dir(object),
        '__doc__',
        '__module__',
        '__class_getitem__',
        '__del__',
        '__name__',
        '__qualname__',
        '__dict__',
        '__weakref__',
    }
    if bases:
        print('- bases:', *bases)

    props = {k: callable(getattr(obj, k)) for k in sorted({*dir(obj)} - base)}
    if attrs := [k for k, ismethod in props.items() if not ismethod]:
        print('- attrs:', *attrs)
    if methods := [k for k, ismethod in props.items() if ismethod]:
        print('- methods:', *methods)
    return obj


async def adef() -> int:
    try:
        print('<sleep ...>')
        await asyncio.sleep(0.5)
        print('<sleep ok>')
        # async def a2():
        #     await asyncio.sleep(.1)
        #     1 / 0
        #     return 53
        # await asyncio.ensure_future(a2())
    except asyncio.CancelledError:
        print('<sleep err>')
        await asyncio.sleep(0.5)
        print('<sleep 2nd time>')
    except BaseException as e:
        print(f'<sleep base err {e!r}>')
        await asyncio.sleep(0.5)
        print('<sleep 2nd time>')
        raise
    return 42


# -------------------------------- functions ---------------------------------


@time_this
def fn_11() -> None:
    work_11()


@time_this
def fn_11_iter_33_iter() -> abc.Iterator[abc.Iterator[int]]:
    # ! `for xs in fn_11_iter_33_iter()` tracked, but `for ... in xs` not
    work_11()
    return _Iter22iter()


@time_this
def fn_11_gen_33_iter() -> abc.Generator[abc.Iterator[int]]:
    # ! `for xs in fn_11_gen_33_iter()` tracked, but `for ... in xs` not
    work_11()
    return (xs for xs in _Iter22iter())


@time_this
def fn_11_cr_12() -> abc.Coroutine[Any, Any, None]:
    # ! `await fn_11_cr_12()` tracked
    work_11()
    return _asleep_01()


@time_this
def fn_11_fut_12() -> asyncio.Future[None]:
    # ! `await fn_11_fut_12()` tracked
    work_11()
    return asyncio.ensure_future(_asleep_01())


# --------------------------- generator functions ----------------------------


@time_this
async def agen_00() -> abc.AsyncGenerator[int]:
    yield 0
    yield 1
    # return None  # ! SyntaxError
    # raise StopIteration(2)  # ! RuntimeError
    # raise StopAsyncIteration(2)  # ! RuntimeError
    # raise GeneratorExit  # ! ok


@time_this
def gen_30_coros() -> abc.Generator[abc.Awaitable[None]]:
    # ! `for cr in gen_30_coros()` tracked, but `await cr` not
    for _ in range(2):
        _cpu_10()
        yield _asleep_01()
    _cpu_10()


@time_this
def gen_30_futures() -> abc.Generator[asyncio.Future[None]]:
    # ! `for f in gen_30_futures()` tracked, but `await f` not
    for _ in range(2):
        _cpu_10()
        yield asyncio.ensure_future(_asleep_01())
    _cpu_10()


@time_this
def gen_33_iter() -> abc.Generator[abc.Iterator[int]]:
    # ! `for xs in gen_33_iter()` tracked, but `for ... in xs` not
    work_11()
    yield from _Iter22iter()


# --------------------------- coroutine functions ----------------------------


@time_this
async def coro_12() -> int:
    # await _asleep1()
    # await asyncio.to_thread(_cpu1)
    await awork_12()
    return 42


@time_this
async def coro_13_iter() -> abc.Iterator[abc.Iterator[int]]:
    # ! `it = await coro_13_iter()` tracked, but `for ... in it` not
    await awork_12()
    await asyncio.ensure_future(_asleep_01())
    return _Iter22iter()


@time_this
async def coro_12_gen() -> abc.Generator[abc.Iterator[int]]:
    # ! `gen = await coro_12_gen()` tracked, but `for ... in gen` not
    await awork_12()
    return (xs for xs in _Iter22iter())


# ------------------------ async generator functions -------------------------


@time_this
async def agen_02() -> abc.AsyncGenerator[int]:
    await asyncio.sleep(DT)
    yield 0
    time.sleep(DT)
    yield 1
    # raise StopIteration(2)  # ! RuntimeError
    # raise StopAsyncIteration(2)  # ! RuntimeError
    # raise GeneratorExit  # ! ok


@time_this
async def agen_35_iter() -> abc.AsyncGenerator[abc.Iterator[int]]:
    await awork_12()
    await asyncio.gather(
        asyncio.ensure_future(asyncio.sleep(DT)),
        asyncio.sleep(0),
    )
    for xs in _Iter22iter():
        yield xs  # ! `for ... in xs` not tracked


# ----------------------------------------------------------------------------


@memoize(3, batched=True)
async def _batch_aw(xs) -> NoReturn:
    await asyncio.sleep(0.2)
    raise RuntimeError(xs)


async def sleepy(x, t) -> list[Any]:
    await asyncio.sleep(t)
    return await _batch_aw([x])


async def catch() -> None:
    # await func([5, 5])
    await asyncio.gather(sleepy(5, 0.2), sleepy(5, 0.1))


async def main() -> None:  # noqa: PLR0915
    coro = _info(adef())  # -> coroutine: Coroutine & Awaitable

    async def coro2():
        while True:
            try:
                f: asyncio.Future = _info(coro.send(None), '... <- await X')
            except StopIteration as e:
                return _info(e.value, '... <- return X')
            else:
                # _info(f.__await__(), 'X = _.__await__() <- await ...')

                ev = asyncio.Event()
                f.add_done_callback(lambda _, ev=ev: ev.set())

                try:
                    await ev.wait()
                except BaseException as e:
                    coro.throw(e)
                    raise

                # try:
                #     while not f.done():
                #         await asyncio.sleep(0)
                # except BaseException as e:
                #     f.__await__().throw(e)
                #     raise

    t = asyncio.create_task(coro2())
    await asyncio.sleep(0.1)
    t.cancel()

    fn_11()
    print(':: called ::')

    r_iter_iter = fn_11_iter_33_iter()
    print(':: called ::')
    print([x for xs in r_iter_iter for x in xs])

    r_gen_iter1 = fn_11_gen_33_iter()
    print(':: called ::')
    print([x for xs in r_gen_iter1 for x in xs])

    r_cr = fn_11_cr_12()
    print(':: called ::')
    await r_cr
    print(':: awaited ::')

    r_fut = fn_11_fut_12()
    print(':: called ::')
    await r_fut
    print(':: awaited ::')

    r_gen_iter2 = gen_33_iter()
    print(':: called ::')
    print([x for xs in r_gen_iter2 for x in xs])

    r_gen_coro = gen_30_coros()
    print(':: called ::')
    for x in r_gen_coro:
        await x
    print(':: awaited ::')

    r_gen_fut = gen_30_futures()
    print(':: called ::')
    for x in r_gen_fut:
        await x
    print(':: awaited ::')

    r_aw1 = coro_12()
    print(':: called ::')
    _info(await r_aw1)

    r_aw_iter_iter = coro_13_iter()
    print(':: called ::')
    _info(iter_iter := await r_aw_iter_iter)
    print(':: awaited ::')
    print([x for xs in iter_iter for x in xs])

    r_aw_gen_iter = coro_12_gen()
    print(':: called ::')
    _info(gen_iter := await r_aw_gen_iter)
    print(':: awaited ::')
    print([x for xs in gen_iter for x in xs])

    r_agen1 = agen_00()
    print(':: called ::')
    print([x async for x in r_agen1])

    r_agen2 = agen_02()
    print(':: called ::')
    print([x async for x in r_agen2])

    r_agen_iter = agen_35_iter()
    print(':: called ::')
    print([x async for xs in r_agen_iter for x in xs])

    for obj in (
        r_iter_iter,
        r_gen_iter1,
        r_gen_iter2,
        r_aw1,
        r_aw_iter_iter,
        r_aw_gen_iter,
        r_agen1,
        r_agen2,
    ):
        print(obj, type(obj))
        # print(type(o).mro())
        # print(sorted({*dir(o)} - {*dir(object())} - {*dir(object)}
        #              - {'__del__', '__name__', '__qualname__'}))
        print(
            '  mro:',
            [
                tp.__name__
                for tp in (
                    abc.Iterator,
                    abc.Generator,  # is iterator
                    abc.Awaitable,
                    abc.Coroutine,  # is awaitable
                    abc.AsyncIterator,
                    abc.AsyncGenerator,  # is async iterator
                )
                if isinstance(obj, tp)
            ],
        )

    # await catch()


asyncio.run(main())
