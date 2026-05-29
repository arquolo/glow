import asyncio
import time
from collections import abc

from glow import memoize, time_this, timer

DT = 0.01
FAIL = False


def _cpu_wa1_fr1_sus0_cpu1_io0():  # wall=1, frame=1, susp=0, cpu=1, io=0 (wall=1, process=1, thread=1)
    # busy += dt
    end = time.perf_counter() + DT
    while time.perf_counter() < end:
        sum(range(1000))


def _sleep_wa1_fr1_sus0_cpu0_io1():  # wall=1, frame=1, susp=0, cpu=0, io=1
    # idle += dt
    time.sleep(DT)


async def _asleep_wa1_fr0_sus1_cpu0_io1():  # wall=1, frame=0, susp=1, cpu=0, io=1
    # idle += dt
    await asyncio.sleep(DT)


def work_wa2_fr2_sus0_cpu1_io1():  # wall=2, frame=2, susp=0, cpu=1, io=1
    _cpu_wa1_fr1_sus0_cpu1_io0()
    _sleep_wa1_fr1_sus0_cpu0_io1()
    if FAIL:
        1 / 0


async def awork_wa3_fr2_sus1_cpu1_io2():  # wall=3, frame=2, susp=1, cpu=1, io=2
    _cpu_wa1_fr1_sus0_cpu1_io0()
    _sleep_wa1_fr1_sus0_cpu0_io1()
    await _asleep_wa1_fr0_sus1_cpu0_io1()
    if FAIL:
        1 / 0


class _Iter660:
    def __init__(self) -> None:
        self.it = iter(range(0, 4, 2))

    def __iter__(self):
        return self

    def __next__(self):  # wall=2, frame=2, cpu=1, io=1
        i = self.it.__next__()  # 0/0/0/0
        work_wa2_fr2_sus0_cpu1_io1()  # 2/2/1/1
        return _SubIter2211(i)  # 0/0/0/0

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self):x}>'


class _SubIter2211:
    def __init__(self, i) -> None:
        self.it = iter(range(i, i + 2))

    def __iter__(self):
        return self

    def __next__(self):  # wall=2, frame=2, cpu=1, io=1
        i = self.it.__next__()  # 0/0/0/0
        work_wa2_fr2_sus0_cpu1_io1()  # 2/2/1/1
        return i  # 0/0/0/0

    def __repr__(self):
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


@time_this
async def asyncgen0():  # async generator[int]
    yield 0
    yield 1
    # return None  # ! SyntaxError
    # raise StopIteration(2)  # ! RuntimeError
    # raise StopAsyncIteration(2)  # ! RuntimeError
    # raise GeneratorExit  # ! ok


@time_this
async def asyncgen00():  # async generator[int]
    await asyncio.sleep(0.1)
    yield 0
    time.sleep(0.1)
    yield 1
    # raise StopIteration(2)  # ! RuntimeError
    # raise StopAsyncIteration(2)  # ! RuntimeError
    # raise GeneratorExit  # ! ok


@time_this
async def asyncgen771():  # async generator[int]
    await awork_wa3_fr2_sus1_cpu1_io2()
    await asyncio.gather(
        asyncio.ensure_future(asyncio.sleep(DT)),
        asyncio.sleep(0),
    )
    for x in _Iter660():
        yield x


@time_this
def fn_iter770():  # function -> iterator[int]
    work_wa2_fr2_sus0_cpu1_io1()
    return _Iter660()


@time_this
def fn_gen770():  # function -> generator[int]
    work_wa2_fr2_sus0_cpu1_io1()
    return (x for x in _Iter660())


@time_this
def gen770():  # generator[int]
    work_wa2_fr2_sus0_cpu1_io1()
    yield from _Iter660()


@time_this
async def coro111():  # coroutine[int]
    # await _asleep_wa1_fr0_sus1_cpu0_io1()
    # await asyncio.to_thread(_cpu_wa1_fr1_sus0_cpu1_io0)
    await awork_wa3_fr2_sus1_cpu1_io2()
    return 42


@time_this
async def coro_iter771():  # coroutine[iterator[int]]
    await awork_wa3_fr2_sus1_cpu1_io2()
    await asyncio.ensure_future(asyncio.sleep(0.1))
    return _Iter660()


@time_this
async def coro_gen771():  # coroutine[generator[int]]
    await awork_wa3_fr2_sus1_cpu1_io2()
    return (x for x in _Iter660())


@memoize(3, batched=True)
async def func(xs):
    await asyncio.sleep(0.2)
    raise RuntimeError(xs)


async def sleepy(x, t):
    await asyncio.sleep(t)
    return await func([x])


async def catch():
    # await func([5, 5])
    await asyncio.gather(sleepy(5, 0.2), sleepy(5, 0.1))


async def main():
    aw = coro = _info(adef())  # -> coroutine: Coroutine & Awaitable

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

    r_fn_iter = fn_iter770()
    print(':: called ::')
    print([x for xs in r_fn_iter for x in xs])

    r_fn_gen = fn_gen770()
    print(':: called ::')
    print([x for xs in r_fn_gen for x in xs])

    r_gen = gen770()
    print(':: called ::')
    print([x for xs in r_gen for x in xs])

    r_coro = coro111()
    print(':: called ::')
    _info(await r_coro)

    r_coro_iter = coro_iter771()
    print(':: called ::')
    _info(aw := await r_coro_iter)
    print(':: awaited ::')
    print([x for xs in aw for x in xs])

    r_coro_gen = coro_gen771()
    print(':: called ::')
    _info(aw := await r_coro_gen)
    print(':: awaited ::')
    print([x for xs in aw for x in xs])

    r_asyncgen = asyncgen0()
    print(':: called ::')
    print([x async for x in r_asyncgen])

    r_asyncgen = asyncgen00()
    print(':: called ::')
    print([x async for x in r_asyncgen])

    r_asyncgen = asyncgen771()
    print(':: called ::')
    print([x async for xs in r_asyncgen for x in xs])

    for obj in (
        r_fn_iter,
        r_fn_gen,
        r_gen,
        r_coro,
        r_coro_iter,
        r_coro_gen,
        r_asyncgen,
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
