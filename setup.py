#!/bin/python3

from pathlib import Path

import setuptools

setuptools.setup(
    name='glow',
    version='0.8.1',
    url='https://github.com/arquolo/glow',
    author='Paul Maevskikh',
    author_email='arquolo@gmail.com',
    description='Toolset for model training and creation of pipelines',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    packages=setuptools.find_packages(exclude=['glow.test']),
    python_requires='>=3.6',
    install_requires=[
        "dataclasses ; python_version<'3.7'",
        "pickle5 ; python_version<'3.8'",
        'loky',
        'numpy>=1.17',
        'psutil',
        'typing-extensions',
        'wrapt',
    ],
    extras_require={
        'cv': [
            'graphviz',
            'matplotlib',
            'numba',
            'opencv-python>=4',
            'py3nvml',
            'pyaudio',
            'soundfile',
            'torch>=1.4',
        ],
    },
    dependency_links=[
        'https://download.pytorch.org/whl/torch_stable.html',
    ],
)
