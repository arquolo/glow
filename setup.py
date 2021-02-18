#!/bin/python3
"""
Usage:
    python setup.py sdist
    python setup.py bdist_wheel --plat-name=win-amd64
"""

import io
import sys
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install

_NAME = 'glow'
_URL = ('https://github.com/openslide/openslide-winbuild/releases/download'
        '/v20171122/openslide-win64-20171122.zip')


def _download_deps(cmd, path: Path) -> None:
    if cmd.dry_run or (sys.platform != 'win32') or path.exists():
        return

    from tqdm import tqdm
    cmd.mkpath(path.as_posix())
    try:
        reply = urlopen(_URL)
        buf = io.BytesIO()
        with tqdm(
                desc='Retrieve shared libraries',
                total=int(reply.info().get('Content-Length')),
                unit='B',
                unit_scale=True) as pbar:
            while chunk := reply.read(1024):
                pbar.update(buf.write(chunk))
        with ZipFile(buf) as zf:
            for name in zf.namelist():
                if not name.endswith('.dll'):
                    continue
                with zf.open(name) as f:
                    (path / Path(name).name).write_bytes(f.read())
    except BaseException:  # noqa: B902
        for p in path.glob('*.dll'):
            p.unlink()
        path.unlink()
        raise


class PostInstall(install):
    def run(self):
        _download_deps(self, Path(self.build_lib, _NAME, 'io/libs'))
        install.run(self)


class PostDevelop(develop):
    def run(self):
        _download_deps(self, Path(self.egg_path, _NAME, 'io/libs'))
        develop.run(self)


setuptools.setup(cmdclass={'install': PostInstall, 'develop': PostDevelop})
