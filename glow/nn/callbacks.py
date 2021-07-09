from __future__ import annotations

__all__ = ['Saver']

import heapq
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class Saver:
    """
    Helper for checkpointing. Saves state on each __call__().

    Parameters:
    - mins - names of scalars which must be minimized.
    - maxs - names of scalars which must be maximized.
    - save_n_best - number of best states to save for each scalar.
    """

    folder: Path
    mins: Sequence[str] = ()
    maxs: Sequence[str] = ()
    save_n_best: int = 1
    _step: int = field(default=-1, init=False)
    _saved: dict[str, list[tuple[float, Path]]] = field(
        default_factory=dict, init=False)

    def __post_init__(self):
        assert not {*self.maxs} & {*self.mins}, \
            "'mins' and 'maxs' should have different keys"
        self.folder.mkdir(parents=True, exist_ok=True)

        if (metrics_path := self.folder / '_metrics.json').exists():
            self._load(metrics_path)

    def _load(self, path: Path) -> None:
        state: dict = json.loads(path.read_text())
        self._step = int(state['last'].split('.')[0])
        self._saved = {
            name:
            [(-scalar if name in self.mins else scalar, self.folder / fname)
             for fname, scalar in stored.items()]
            for name, stored in state['best'].items()
        }
        for stored in self._saved.values():
            heapq.heapify(stored)

    def save(self,
             state: Any,
             scalars: dict[str, float],
             step: int | None = None) -> None:
        """Save state to file"""
        ckpt_old = (
            self.folder / f'{self._step}.pth' if self._step >= 0 else None)

        # Update current step
        if step is None:
            self._step += 1
        elif step <= 0:
            raise ValueError("'step' should be non-negative")
        elif step <= self._step:
            raise ValueError("'step' should increase with each call")
        else:
            self._step = step

        # Update files
        ckpt_new = self.folder / f'{self._step}.pth'
        torch.save(state, ckpt_new)
        if ckpt_old is not None:
            ckpt_old.unlink()

        # Make links
        best: dict = {}
        for name, scalar in scalars.items():
            if name not in self.mins and name not in self.maxs:
                continue
            link = self.folder / f'{self._step}.{name}.pth'
            saved = self._saved.setdefault(name, [])

            if name in self.mins:
                # heapq drops smallest items, but we need keep them
                scalar = -scalar

            if len(saved) < self.save_n_best:
                heapq.heappush(saved, (scalar, link))
                ckpt_new.link_to(link)
            else:
                _, popped = heapq.heappushpop(saved, (scalar, link))
                if popped != link:  # `link` is not discarded
                    popped.unlink()
                    ckpt_new.link_to(link)

            best[name] = {
                p.name: (-v if name in self.mins else v) for v, p in saved
            }

        # Dump summary
        (self.folder / '_metrics.json').write_text(
            json.dumps({
                'last': ckpt_new.name,
                'best': best,
            }, indent=2))
