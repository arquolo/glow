import sys
import time
from argparse import ArgumentParser
from collections.abc import Iterator
from typing import TypedDict

from psutil import Process
from tabulate import tabulate  # pip install tabulate


class MemoryMonitor:
    def __init__(self, pid: int) -> None:
        self.proc = Process(pid)

    def __str__(self) -> str:
        summarized = self._summarize()
        table = [
            [
                row_id,
                *(fmt(v) for v in stat.values()),  # type: ignore[arg-type]
                cmd,
            ]
            for row_id, stat, cmd in summarized
        ]
        return tabulate(
            table,
            headers=['PID', *_Stat.__annotations__, 'cmd'],
            missingval='?',
        )

    def _summarize(self) -> Iterator[tuple[str, '_Stat', str]]:
        stat_iter = self._iter_stats()
        summary = _new_stat()

        for pid, stat, cmd in stat_iter:
            summary['uss'] += stat['uss']

            for k in ('pss', 'pshared'):  # Linux only
                if (v := stat[k]) is not None:
                    summary[k] = (summary[k] or 0) + v

            yield pid, stat, cmd

        yield 'total', summary, '-'

    def _iter_stats(self) -> Iterator[tuple[str, '_Stat', str]]:
        todo: list[tuple[int, Process]] = [(0, self.proc)]

        while todo:
            depth, proc = todo.pop()

            with proc.oneshot():
                todo += ((depth + 1, pp) for pp in proc.children())
                cmdline = proc.cmdline()
                m = get_mem_info(proc)

            cmd = ' '.join(f'"{s}"' if ' ' in s else s for s in cmdline)
            yield ('-' * depth + f' {proc.pid}', m, cmd)


def get_mem_info(proc: Process) -> '_Stat':
    s = _new_stat()

    if sys.platform == 'win32':
        mem = proc.memory_full_info()
        s['rss'] += mem.rss  # = private + shared = WSet
        # no PSS
        s['uss'] += mem.uss
        s['shared'] += mem.rss - mem.uss
        # VMS = pagefile = private
    else:
        for m in proc.memory_maps():
            s['rss'] += m.rss
            s['pss'] = (s['pss'] or 0) + m.pss
            s['uss'] += m.private_clean + m.private_dirty
            s['shared'] += m.shared_clean + m.shared_dirty
        s['pshared'] = (s['pss'] or 0) - s['uss']

    return s


def _new_stat() -> '_Stat':
    return _Stat(rss=0, pss=None, uss=0, shared=0, pshared=None)


class _Stat(TypedDict):
    rss: int  # private + shared
    pss: float | None  # private + shared (proportional), linux only
    uss: int  # private
    shared: int  # shared
    pshared: float | None  # shared (proportional), linux only


def fmt(size: float | None) -> str | None:
    if size is None:
        return None
    if size < 99999.5:
        return f'{size:<5.0f}'
    size = size / 1024
    for unit in ('K', 'M', 'G', 'T'):
        if size < 999.5:
            return _fmt(size, unit)
        size /= 1024
    return _fmt(size, 'P')


def _fmt(size: float, unit: str) -> str:
    return f'{size:<4.1f}{unit}' if size < 99.5 else f'{size:<4.0f}{unit}'


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('pid', type=int, help='pid of root process')
    p.add_argument('-t', '--time', type=float, default=5, help='period, s')
    args = p.parse_args()

    m = MemoryMonitor(args.pid)
    while True:
        print(m)
        time.sleep(args.time)
