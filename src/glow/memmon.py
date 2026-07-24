"""
Process memory monitor. Tracks process & all its children.

Usage: python -m glow.memmon PID --time=PERIOD
"""
#!/usr/bin/env python3

import sys
import time
from argparse import ArgumentParser
from collections.abc import Iterator
from dataclasses import dataclass

from prettytable import PrettyTable, TableStyle
from psutil import Process


class MemoryMonitor:
    def __init__(self, pid: int) -> None:
        self.proc = Process(pid)

    def __str__(self) -> str:
        stats = list(self._iter_stats())
        tab = PrettyTable(
            field_names=['PID', *_Stat.__dataclass_fields__, 'cmd'],
            align='l',
        )
        tab.set_style(TableStyle.SINGLE_BORDER)

        summary = _Stat()
        for pid, stat, cmd in stats:
            summary.uss += stat.uss
            if sys.platform != 'win32':
                summary.pss += stat.pss
                summary.pshared += stat.pshared
            tab.add_row([pid, *stat.formatted(), cmd])

        if len(stats) > 1:
            tab.add_divider()
            tab.add_row(['total', *summary.formatted(), '-'])

        return str(tab)

    def _iter_stats(self) -> Iterator[tuple[str, '_Stat', str]]:
        todo: list[tuple[int, Process]] = [(0, self.proc)]

        while todo:
            depth, proc = todo.pop()

            with proc.oneshot():
                todo += ((depth + 1, pp) for pp in proc.children())
                cmdline = proc.cmdline()
                m = get_mem_info(proc)

            cmd = ' '.join(f'"{s}"' if ' ' in s else s for s in cmdline)
            cmd = cmd[:80]
            yield ('-' * depth + f' {proc.pid}', m, cmd)


@dataclass
class _BaseStat:
    rss: int = 0  # private + shared
    uss: int = 0  # private
    shared: int = 0  # shared

    def formatted(self) -> list[str]:
        return [fmt(getattr(self, name)) for name in self.__dataclass_fields__]


if sys.platform == 'win32':

    def get_mem_info(proc: Process) -> '_Stat':
        mem = proc.memory_full_info()
        return _Stat(
            rss=mem.rss,  # = private + shared = WSet
            uss=mem.uss,
            shared=mem.rss - mem.uss,
            # VMS = pagefile = private
        )

    class _Stat(_BaseStat):
        pass
else:

    def get_mem_info(proc: Process) -> '_Stat':
        s = _Stat()
        for m in proc.memory_maps():
            s.rss += m.rss
            s.pss += m.pss
            s.uss += m.private_clean + m.private_dirty
            s.shared += m.shared_clean + m.shared_dirty
        s.pshared = s.pss - s.uss
        return s

    @dataclass
    class _Stat(_BaseStat):
        pss: float = 0  # private + shared (proportional)
        pshared: float = 0  # shared (proportional)


def fmt(size: float) -> str:
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
