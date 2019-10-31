#!/bin/python3

from pathlib import Path

import setuptools

setuptools.setup(
    name='glow',
    version='0.6.2',
    url='https://github.com/arquolo/glow',
    author='Paul Maevskikh',
    author_email='arquolo@gmail.com',
    description='Optimized RouTines for Python',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    packages=setuptools.find_packages(exclude=['glow.test']),
    python_requires='>=3.6',
    install_requires=[
        "dataclasses ; python_version<'3.7'",
        'loky',
        'numba',
        'numpy>=1.15',
        'psutil',
        'wrapt',
    ],
    extras_require={
        'io': [
            'opencv-python>=4',
            'pyaudio',
            'soundfile',
        ],
        'nn': [
            'future',  # torch.utils.tensorboard dies if missing
            'graphviz',
            'opencv-python>=4',
            'py3nvml',
            'torch>=1.2',
        ],
    },
    dependency_links=[
        'https://download.pytorch.org/whl/torch_stable.html',
    ]
)
