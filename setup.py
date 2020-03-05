#!/bin/python3

import io
import os
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlopen

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install

_NAME = 'glow'
_URL = ('https://github.com/openslide/openslide-winbuild/releases/download'
        '/v20171122/openslide-win64-20171122.zip')


def _download_deps(self, path: Path) -> None:
    if self.dry_run or (os.name != 'nt') or path.exists():
        return

    from tqdm import tqdm
    self.mkpath(path.as_posix())
    try:
        r = urlopen(_URL)
        buf = io.BytesIO()
        with tqdm(
                desc='Retrieve shared libraries',
                total=int(r.info().get('Content-Length')),
                unit='B',
                unit_scale=True) as pbar:
            for chunk in iter(lambda: r.read(1024), b''):
                pbar.update(buf.write(chunk))
        with ZipFile(buf) as zf:
            for name in zf.namelist():
                if not name.endswith('.dll'):
                    continue
                with zf.open(name) as f:
                    (path / Path(name).name).write_bytes(f.read())
    except BaseException:
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
